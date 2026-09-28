"""CEJ review queue, exact blockers, and confirmed-only split readiness."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from landmarks import HERE, ROOT, sha, write

GROUND_TRUTH = HERE.parent / 'step4_structural_ground_truth'


def review_records(data, review):
    """Retain original candidate identity even when an expert reassigns it."""
    decisions = {d['id']: d for d in review['decisions']}
    records = {r['key']: r for r in data['records']}
    queue = []
    for c in data['candidates']:
        if c['point_type'] != 'CEJ':
            continue
        r = records[c['key']]
        t = next(t for t in r['teeth'] if t['tooth_instance_id'] == c['tooth_instance_id'])
        decision = decisions.get(c['id'])
        # A reassignment records a new target decision and a rejected old pair.
        reassigned = [d for d in review['decisions'] if d.get('key') == c['key'] and d.get('point_id') == c['point_id']
                      and d.get('candidate_tooth_instance_id') == c['tooth_instance_id'] and d.get('status') == 'confirmed'
                      and d.get('expert_tooth_instance_id') != c['tooth_instance_id']]
        if reassigned and decision and decision['status'] == 'rejected':
            decision = max(reassigned, key=lambda d: d['timestamp_utc'])
        f = c['features']
        queue.append({'candidate_id': c['id'], 'key': c['key'], 'image_id': c['image_id'], 'partition': r['split'],
                      'cej_point_id': c['point_id'], 'coordinate': c['coordinate'],
                      'candidate_tooth_instance_id': c['tooth_instance_id'], 'candidate_bbox': t['bbox'],
                      'candidate_distance_px': f['min_mask_pixel_center_distance_px'], 'candidate_score': c['candidate_score'],
                      'candidate_reason': f"Mask distance ratio {f['normalized_mask_distance']:.4f}; independent keypoint-box support {f['keypoint_bbox_support']:.4f}. Geometry only; no ordering-based association.",
                      'expert_status': decision['status'] if decision else 'unreviewed',
                      'expert_tooth_instance_id': decision.get('expert_tooth_instance_id', decision['tooth_instance_id']) if decision else None,
                      'expert_comment': decision.get('note', '') if decision else '',
                      'source_file': f"DenPAR Radiographs Dataset/Dataset/{r['split']}/Key Points Annotations/{c['image_id']}.json",
                      'source_field': 'CEJ_Points', 'source_index': int(c['point_id'].split(':')[1]),
                      'tooth_mask': t['source'], 'mask_overlay': t['overlay_url'], 'original_image': r['original_url']})
    return queue


def readiness(data, review, manifest):
    cej_points = {(r['key'], p['point_id']) for r in data['records'] for p in r['points'] if p['point_type'] == 'CEJ'}
    decisions = [d for d in review['decisions'] if d.get('point_type') == 'CEJ']
    included = {(r['key'], f"CEJ:{p['source_index']}") for r in manifest['records'] for p in r['target_provenance']
                if p['kind'] == 'landmark' and p['json_field'] == 'CEJ_Points'}
    splits = {}
    for split in ('Training', 'Validation', 'Testing'):
        records = [r for r in manifest['records'] if r['partition'] == split and r['cej_points']]
        splits[split] = {'images_with_confirmed_CEJ': len({r['image_id'] for r in records}),
                         'teeth_with_confirmed_CEJ': len(records), 'confirmed_CEJ_points': sum(len(r['cej_points']) for r in records)}
    positive = [s for s, values in splits.items() if values['confirmed_CEJ_points']]
    status = 'STATUS A — CEJ DATASET READY' if len(positive) == 3 else ('STATUS B — CEJ DATASET PARTIALLY READY' if positive else 'STATUS C — CEJ DATASET STILL BLOCKED')
    return {'status': status, 'official_partitions': splits, 'expert_review_revision': review['revision'],
            'uncertain_points': len({(d['key'], d['point_id']) for d in decisions if d['status'] == 'uncertain' and d['point_id'] is not None}),
            'rejected_points': len({(d['key'], d['point_id']) for d in decisions if d['status'] == 'rejected' and d['point_id'] is not None}),
            'cannot_determine_points': len({(d['key'], d['point_id']) for d in decisions if d['status'] == 'cannot_determine'}),
            'unresolved_points': len(cej_points-included),
            'unreviewed_points': len(cej_points-{(d['key'], d['point_id']) for d in decisions}),
            'missing_CEJ_cases': sum(d['status'] == 'missing' for d in decisions),
            'multiple_CEJ_per_tooth_cases': [{'key': r['key'], 'tooth_instance_id': r['tooth_instance_id'], 'count': len(r['cej_points'])} for r in manifest['records'] if len(r['cej_points']) > 1],
            'point_status_count_note': 'Unique source points per expert status; counts may overlap when different candidate pairs for one point have different decisions. Unresolved means absent from the confirmed-only manifest. Point totals by split count tooth-target occurrences.',
            'decision_rule': 'A requires usable confirmed targets in all three official partitions (Testing kept separate); B has some usable targets but incomplete split coverage; C has none. No training runs here.',
            'next_action': 'Proceed to CEJ baseline planning with Training/Validation only; reserve Testing until final lock.' if len(positive) == 3 else 'Obtain independent dental-expert decisions and usable images, then regenerate manifests. Keep modeling deferred.',
            'training_performed': False, 'test_evaluation_performed': False}


def refresh():
    candidates = HERE / 'candidate_landmark_mappings.json'
    expert = HERE / 'expert_landmark_review.json'
    hashes = {candidates: sha(candidates), expert: sha(expert)}
    data, review = json.loads(candidates.read_text()), json.loads(expert.read_text())
    if data['dataset_id'] != review['dataset_id']:
        raise ValueError('Candidate/review fingerprint mismatch')
    result = subprocess.run([sys.executable, str(GROUND_TRUTH / 'build.py')], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Confirmed-only manifest construction failed: '+result.stderr[-2000:])
    if any(sha(path) != value for path, value in hashes.items()):
        raise ValueError('Expert review changed during regeneration; regenerate again')
    manifest = json.loads((GROUND_TRUTH / 'trainable_structural_manifest.json').read_text())
    queue = review_records(data, review)
    cej_candidates = [c for c in data['candidates'] if c['point_type'] == 'CEJ']
    cej_decisions = [d for d in review['decisions'] if d.get('point_type') == 'CEJ']
    point_pairs = {(c['key'], c['point_id']) for c in cej_candidates}
    points = [(r, p) for r in data['records'] for p in r['points'] if p['point_type'] == 'CEJ']
    blockers = {'candidate_CEJ_relationships': len(cej_candidates), 'CEJ_points_with_candidates': len(point_pairs),
                'source_CEJ_points': len(points), 'expert_CEJ_decisions': len(cej_decisions),
                'expert_confirmed_CEJ_decisions': sum(d['status'] == 'confirmed' for d in cej_decisions),
                'heuristic_candidate_status_counts': dict(Counter(c['status'] for c in cej_candidates)),
                'A_candidates_exist_but_all_unreviewed': bool(cej_candidates) and not cej_decisions,
                'B_expert_uncertain_decisions': sum(d['status'] == 'uncertain' for d in cej_decisions),
                'C_candidates_never_generated': not cej_candidates,
                'CEJ_points_without_candidates': [{'key': r['key'], 'point_id': p['point_id'], 'issues': p['issues']} for r, p in points if (r['key'], p['point_id']) not in point_pairs],
                'D_exclusion_rule': 'Step 4 requires a current confirmed expert decision with matching history, identity, source hashes and valid geometry. It does not simply read an expert_validated=false flag from candidate records.',
                'E_F_schema_and_manifest_check': 'Candidate/review fingerprints and original hashes verified by successful Step 4 reconstruction. No schema incompatibility prevented recognition of confirmations.',
                'canonical_manifest_directory': str(GROUND_TRUTH.relative_to(ROOT)),
                'legacy_path_note': 'Earlier outputs were under step4_structural_dataset. Requested canonical outputs now use step4_structural_ground_truth; the legacy directory is historical and is not refreshed by this action.',
                'exact_blocker': 'No independent expert CEJ decisions exist; geometric probable/uncertain candidates have not been approved.' if not cej_decisions else 'See readiness and exclusion manifests for current decision eligibility.',
                'missing_original_images_by_partition': dict(Counter(r['split'] for r in data['records'] if not r.get('original_url'))),
                'provenance': {'candidate_file_sha256': hashes[candidates], 'expert_file_sha256': hashes[expert], 'expert_review_revision': review['revision']}}
    ready = readiness(data, review, manifest)
    write(HERE / 'cej_review_records.json', {'schema_version': 1, 'candidate_dataset_id': data['dataset_id'], 'expert_review_revision': review['revision'], 'records': queue})
    write(HERE / 'reports' / 'cej_blocker_audit.json', blockers)
    write(GROUND_TRUTH / 'cej_readiness_report.json', ready)
    return ready


if __name__ == '__main__':
    print(json.dumps(refresh(), indent=2))
