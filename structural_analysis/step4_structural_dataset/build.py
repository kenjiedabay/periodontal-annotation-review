"""Construct tooth-level targets from explicit expert confirmations only."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from image_paths import resolve_image
STAGES = {'bone': 'step2_boneline_tooth_correspondence', 'landmark': 'step3_cej_apex_correspondence'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Provenance path escapes repository')
    return path


def verify_bundle(root, data, review, kind):
    fingerprint = data['dataset_id']
    original = {k: v for k, v in data.items() if k not in {'dataset_id', 'sources_unchanged'}}
    calculated = hashlib.sha256(json.dumps(original, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if calculated != fingerprint:
        raise ValueError(f'{kind}: candidate dataset fingerprint mismatch')
    if review.get('candidate_dataset_id' if kind == 'bone' else 'dataset_id') != fingerprint:
        raise ValueError(f'{kind}: expert review belongs to another candidate dataset')
    for relative, expected in data['source_sha256'].items():
        path = safe_path(root, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'{kind}: changed or missing provenance source: {relative}')
    if len({d['id'] for d in review['decisions']}) != len(review['decisions']):
        raise ValueError(f'{kind}: duplicate current decision IDs')


def decision_valid(decision, review):
    if decision.get('status') != 'confirmed' or decision.get('origin') != 'expert_review':
        return False
    if not isinstance(decision.get('reviewer'), str) or not decision['reviewer'].strip():
        return False
    try:
        stamp = datetime.fromisoformat(decision['timestamp_utc'])
        if stamp.tzinfo is None:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    events = [e for e in review['history'] if e.get('id') == decision.get('id')]
    return bool(events) and events[-1] == decision


def construct(root, bundles, input_provenance):
    for kind, (data, review) in bundles.items():
        verify_bundle(root, data, review, kind)
    datasets = {k: {r['key']: r for r in d['records']} for k, (d, _) in bundles.items()}
    trainable, excluded_teeth, excluded_annotations = [], [], []
    source = root / 'DenPAR Radiographs Dataset' / 'Dataset'
    consumed = set()
    teeth_inventory = {}
    for kind, records in datasets.items():
        for key, r in records.items():
            if r['split'] not in {'Training', 'Validation', 'Testing'} or key != f"{r['split']}_{r['image_id']}":
                raise ValueError('Invalid official partition identity')
            for tooth in r['teeth']:
                identity = (key, tooth['tooth_instance_id'])
                if identity in teeth_inventory and teeth_inventory[identity][1]['source'] != tooth['source']:
                    raise ValueError('Conflicting tooth-mask provenance between stages')
                teeth_inventory[identity] = (r, tooth)
    for (key, tooth_id), (r, tooth) in sorted(teeth_inventory.items()):
        prefix = {'key': key, 'image_id': r['image_id'], 'partition': r['split'], 'tooth_instance_id': tooth_id}
        mask_path = safe_path(root, tooth['source'])
        expected_mask = source / r['split'] / 'Masks (Tooth-wise)' / r['image_id'] / f'{tooth_id}.png'
        if mask_path != expected_mask.resolve():
            raise ValueError('Tooth mask source does not match official partition and instance')
        target = {**prefix, 'fdi': None, 'tooth_mask': tooth['source'], 'bbox': None,
                  'cej_points': [], 'apex_points': [], 'bone_lines': [], 'target_provenance': [],
                  'correspondence_status': 'confirmed', 'expert_validated': True}
        issues = list(tooth['issues'])
        image_path = resolve_image(r['split'], r['image_id'], root)
        if not image_path.exists():
            issues.append('missing_original_image')
        selected = []
        for kind, (data, review) in bundles.items():
            selected.extend((kind, d, review) for d in review['decisions'] if d.get('key') == key and d.get('tooth_instance_id') == tooth_id and d.get('status') == 'confirmed')
        if not selected:
            issues.append('no_expert_confirmed_structural_targets')
        if not issues:
            with Image.open(image_path) as img, Image.open(mask_path) as mask:
                fg = np.asarray(mask.convert('L')) != 0
                height, width = fg.shape
                if mask.size != img.size or not fg.any():
                    issues.append('invalid_mask_geometry')
                else:
                    y, x = np.where(fg)
                    target['bbox'] = [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]
                    target['image_size'] = list(img.size)
                    target['image_path'] = str(image_path.relative_to(root))
            if not issues:
                for kind, decision, review in selected:
                    reason = None
                    original_record = datasets[kind].get(key)
                    if not decision_valid(decision, review):
                        reason = 'confirmation_missing_valid_expert_audit_history'
                    elif original_record is None or not any(t['tooth_instance_id'] == tooth_id and not t['issues'] for t in original_record['teeth']):
                        reason = 'target_tooth_unavailable_in_review_dataset'
                    if kind == 'landmark':
                        p = next((p for p in (original_record or {}).get('points', []) if p['point_id'] == decision.get('point_id')), None)
                        if p is None or p['issues'] or p['coordinate'] != decision.get('coordinate') or p['point_type'] != decision.get('point_type'):
                            reason = 'invalid_or_changed_landmark_reference'
                        if any(d.get('status') == 'missing' and d.get('key') == key and d.get('tooth_instance_id') == tooth_id and d.get('point_type') == decision.get('point_type') for d in review['decisions']):
                            reason = 'conflicting_expert_missing_decision'
                        raw_path = source / r['split'] / 'Key Points Annotations' / f"{r['image_id']}.json"
                        field = 'CEJ_Points' if decision.get('point_type') == 'CEJ' else 'Apex_Points'
                        index = p['source_index'] if p else None
                        raw = read(raw_path)
                        if p and (raw.get('Image_id') != f"{r['image_id']}.jpg" or raw.get(field, [])[index] != p['coordinate']):
                            reason = 'original_landmark_provenance_mismatch'
                        if not reason:
                            target['cej_points' if field == 'CEJ_Points' else 'apex_points'].append(p['coordinate'])
                    else:
                        line = next((l for l in (original_record or {}).get('bone_lines', []) if l['bone_line_id'] == decision.get('bone_line_id')), None)
                        if line is None or line['issues'] or decision.get('region'):
                            reason = 'invalid_or_region_only_bone_reference'
                        raw_path = source / r['split'] / 'Bone Level Annotations' / f"{r['image_id']}.json"
                        field, index = 'Bone_Lines', decision.get('bone_line_id')
                        raw = read(raw_path)
                        if line and (raw.get('Image_id') != f"{r['image_id']}.jpg" or raw.get(field, [])[index] != line['points']):
                            reason = 'original_bone_provenance_mismatch'
                        if not reason:
                            target['bone_lines'].append(line['points'])
                    consumed.add((kind, decision['id']))
                    if reason:
                        excluded_annotations.append({**prefix, 'kind': kind, 'decision': decision, 'reason': reason})
                    else:
                        target['target_provenance'].append({'kind': kind, 'source_file': str(raw_path.relative_to(root)),
                            'source_sha256': sha(raw_path), 'json_field': field, 'source_index': index,
                            'review_file': input_provenance[kind]['review_file'], 'review_revision': review['revision'],
                            'candidate_dataset_id': bundles[kind][0]['dataset_id'], 'expert_decision': decision})
        available = {'CEJ': bool(target['cej_points']), 'Apex': bool(target['apex_points']), 'Bone_Lines': bool(target['bone_lines'])}
        if any(available.values()) and not issues:
            target['confirmed_structures'] = available
            target['has_all_required_structures'] = all(available.values())
            target['mask_provenance'] = {'source_file': tooth['source'], 'sha256': sha(mask_path), 'bbox_convention': 'xyxy, maximum edges exclusive; derived from original mask'}
            trainable.append(target)
        else:
            excluded_teeth.append({**prefix, 'tooth_mask': tooth['source'], 'reasons': sorted(set(issues or ['no_usable_confirmed_targets']))})
    for kind, (data, review) in bundles.items():
        states = {c['id']: c for c in data['candidates']}
        states.update({d['id']: d for d in review['decisions']})
        covered = set()
        for item in states.values():
            source_id = item.get('bone_line_id') if kind == 'bone' else item.get('point_id')
            covered.add((item['key'], source_id))
            if (kind, item['id']) not in consumed:
                excluded_annotations.append({'kind': kind, 'key': item['key'], 'annotation_id': source_id,
                    'tooth_instance_id': item.get('tooth_instance_id'), 'status': item['status'],
                    'reason': 'not_confirmed' if item['status'] != 'confirmed' else 'confirmed_target_not_usable_at_tooth_level',
                    'candidate_or_decision_id': item['id'], 'provenance': input_provenance[kind]})
        for r in data['records']:
            for item in r['bone_lines' if kind == 'bone' else 'points']:
                source_id = item['bone_line_id' if kind == 'bone' else 'point_id']
                if (r['key'], source_id) not in covered:
                    excluded_annotations.append({'kind': kind, 'key': r['key'], 'annotation_id': source_id,
                        'reason': 'no_candidate_or_expert_mapping', 'issues': item['issues'], 'provenance': input_provenance[kind]})
    for entry in excluded_annotations:
        kind = entry['kind']
        r = datasets[kind][entry['key']]
        identifier = entry.get('annotation_id')
        if identifier is None and entry.get('decision'):
            identifier = entry['decision'].get('bone_line_id' if kind == 'bone' else 'point_id')
        if kind == 'bone':
            field, index, directory = 'Bone_Lines', identifier, 'Bone Level Annotations'
        else:
            p = next((p for p in r['points'] if p['point_id'] == identifier), None)
            field = ('CEJ_Points' if p['point_type'] == 'CEJ' else 'Apex_Points') if p else None
            index, directory = (p['source_index'] if p else None), 'Key Points Annotations'
        original_path = source / r['split'] / directory / f"{r['image_id']}.json"
        entry['original_provenance'] = {'source_file': str(original_path.relative_to(root)), 'json_field': field, 'source_index': index,
                                        'partition': r['split']}
    summary = {'usable_tooth_records': len(trainable), 'records_with_CEJ': sum(bool(r['cej_points']) for r in trainable),
               'records_with_apex': sum(bool(r['apex_points']) for r in trainable), 'records_with_bone_lines': sum(bool(r['bone_lines']) for r in trainable),
               'records_with_all_required_structures': sum(r['has_all_required_structures'] for r in trainable),
               'required_structures_definition': 'At least one confirmed CEJ point, one confirmed apex point and one confirmed bone line, plus valid original tooth mask and derived bbox. Partial records are allowed with explicit confirmed_structures flags.',
               'unresolved_tooth_records': len(excluded_teeth), 'excluded_annotation_relationships': len(excluded_annotations),
               'unresolved_images': len({r['key'] for r in excluded_teeth+excluded_annotations}),
               'official_partitions': {s: {'trainable': sum(r['partition'] == s for r in trainable), 'excluded_teeth': sum(r['partition'] == s for r in excluded_teeth)} for s in ['Training','Validation','Testing']},
               'exclusion_reason_counts': dict(Counter(r['reason'] for r in excluded_annotations)),
               'expert_review_revisions': {k: r['revision'] for k, (_, r) in bundles.items()},
               'no_confirmations_is_empty_dataset': True, 'source_files_overwritten': False}
    header = {'schema_version': 1, 'path_base': 'repository root', 'input_provenance': input_provenance,
              'policy': 'Explicit expert-confirmed targets only. No index correspondence, FDI inference, disease labels, severity labels or new partitioning.'}
    return {**header, 'records': trainable}, {**header, 'tooth_records': excluded_teeth, 'annotations': excluded_annotations}, summary


def main(output=None):
    output = Path(output) if output is not None else HERE
    output.mkdir(parents=True, exist_ok=True)
    bundles, provenance, snapshots = {}, {}, {}
    for kind, folder in STAGES.items():
        directory = HERE.parent / folder
        candidate = directory / ('candidate_mappings.json' if kind == 'bone' else 'candidate_landmark_mappings.json')
        review = directory / ('expert_review.json' if kind == 'bone' else 'expert_landmark_review.json')
        snapshots[candidate] = sha(candidate)
        snapshots[review] = sha(review)
        bundles[kind] = (read(candidate), read(review))
        provenance[kind] = {'candidate_file': str(candidate.relative_to(ROOT)), 'candidate_sha256': snapshots[candidate],
                            'review_file': str(review.relative_to(ROOT)), 'review_sha256': snapshots[review]}
    trainable, unresolved, summary = construct(ROOT, bundles, provenance)
    if any(sha(path) != value for path, value in snapshots.items()):
        raise RuntimeError('Review inputs changed during construction; rerun')
    for name, data in [('trainable_structural_manifest.json',trainable), ('unresolved_structural_manifest.json',unresolved), ('structural_dataset_summary.json',summary)]:
        temporary = output / (name+'.tmp')
        temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(output/name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
