"""Geometric proposals and explicit, separately stored expert decisions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.runtime'))
import cv2
sys.path.insert(0, str(HERE.parent / 'step1_boneline_visualization'))
from build import validate_line

STATUSES = {'confirmed', 'probable', 'uncertain', 'unassigned', 'rejected'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_line(points, spacing=1.0):
    """Sample every segment at <=1 px spacing, including both endpoints."""
    parts = []
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(1, math.ceil(float(np.linalg.norm(b-a)) / spacing))
        parts.append(np.linspace(a, b, n+1)[:-1])
    return np.concatenate([*parts, np.asarray(points[-1:], dtype=float)])


def features(samples, distance, bbox, centroid):
    # Bilinear interpolation of Euclidean distance to foreground pixel centers.
    x, y = samples.T
    h, w = distance.shape
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    fx, fy = x-x0, y-y0
    distances = ((1-fx)*(1-fy)*distance[y0, x0] + fx*(1-fy)*distance[y0, x1]
                 + (1-fx)*fy*distance[y1, x0] + fx*fy*distance[y1, x1])
    left, top, right, bottom = bbox
    dx = np.maximum(np.maximum(left-x, x-right), 0)
    dy = np.maximum(np.maximum(top-y, y-bottom), 0)
    scale = max(1., math.hypot(right-left, bottom-top))
    radius = .05 * scale
    delta = samples.mean(axis=0) - centroid
    minimum = float(distances.min())
    overlap = float(np.mean(distances <= radius))
    score = .7 * math.exp(-minimum / (.05*scale)) + .3 * overlap
    return {'min_mask_distance_px_approx': minimum, 'min_bbox_distance_px_approx': float(np.hypot(dx, dy).min()),
            'mask_distance_normalized': minimum/scale, 'expanded_mask_radius_px': radius,
            'fraction_samples_in_expanded_mask': overlap, 'line_centroid_xy_approx': samples.mean(axis=0).tolist(),
            'line_minus_tooth_centroid_xy': delta.tolist(),
            'relative_horizontal_position': 'left' if delta[0] < -1 else ('right' if delta[0] > 1 else 'aligned'),
            'candidate_score': float(score)}


def pair_id(key, line, tooth=None, region=None):
    return f'{key}|B{line}|T:{tooth}' if tooth is not None else f'{key}|B{line}|R:{region}'


def report(data, review):
    effective = {c['id']: dict(c) for c in data['candidates']}
    for decision in review['decisions']:
        effective[decision['id']] = dict(decision)
    entries = list(effective.values())
    by_line, by_tooth = defaultdict(list), defaultdict(list)
    for c in entries:
        if c['status'] in {'confirmed', 'probable', 'uncertain'}:
            by_line[(c['key'], c['bone_line_id'])].append(c['id'])
            if c.get('tooth_instance_id') is not None:
                by_tooth[(c['key'], c['tooth_instance_id'])].append(c['id'])
    unassigned, no_candidate, ambiguous = [], [], []
    for r in data['records']:
        for line in r['bone_lines']:
            if not by_line[(r['key'], line['bone_line_id'])]:
                unassigned.append({'key': r['key'], 'bone_line_id': line['bone_line_id'],
                                   'reason': line['issues'] or ['no_active_candidate_or_review_mapping']})
        for tooth in r['teeth']:
            if not by_tooth[(r['key'], tooth['tooth_instance_id'])]:
                no_candidate.append({'key': r['key'], 'tooth_instance_id': tooth['tooth_instance_id'], 'issues': tooth['issues']})
        one_many = {str(l['bone_line_id']): by_line[(r['key'], l['bone_line_id'])] for l in r['bone_lines'] if len(by_line[(r['key'], l['bone_line_id'])]) > 1}
        many_one = {t['tooth_instance_id']: by_tooth[(r['key'], t['tooth_instance_id'])] for t in r['teeth'] if len(by_tooth[(r['key'], t['tooth_instance_id'])]) > 1}
        if one_many or many_one:
            ambiguous.append({'key': r['key'], 'line_to_multiple_targets': one_many, 'tooth_to_multiple_lines': many_one})
    counts = Counter(c['status'] for c in entries)
    return {'candidate_dataset_id': data['dataset_id'], 'review_revision': review['revision'],
            'counts': {**{s: counts[s] for s in sorted(STATUSES)}, 'unassigned_bone_lines': len(unassigned),
                       'teeth_with_no_candidate_bone_line': len(no_candidate), 'ambiguous_images': len(ambiguous)},
            'mappings_by_status': {s: [c for c in entries if c['status'] == s] for s in sorted(STATUSES)},
            'unassigned_bone_lines': unassigned, 'teeth_with_no_candidate_bone_line': no_candidate,
            'ambiguous_images': ambiguous,
            'note': 'Probable and uncertain are unconfirmed proposals. Multiplicity is preserved, not resolved automatically. No clinical inference.'}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def empty_review(data):
    return {'schema_version': 1, 'candidate_dataset_id': data['dataset_id'], 'revision': 0, 'decisions': [], 'history': []}


def apply_decision(data, review, payload):
    """Validate human input; never confirm from a heuristic or silently replace another target."""
    if payload.get('revision') != review['revision']:
        raise ValueError('Review changed. Reload before saving.')
    if review['candidate_dataset_id'] != data['dataset_id']:
        raise ValueError('Review belongs to a different candidate dataset.')
    status = payload.get('status')
    if status not in STATUSES:
        raise ValueError('Invalid status')
    reviewer = payload.get('reviewer', '')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('Reviewer name is required')
    record = next((r for r in data['records'] if r['key'] == payload.get('key')), None)
    if record is None:
        raise ValueError('Unknown image')
    line = next((l for l in record['bone_lines'] if l['bone_line_id'] == payload.get('bone_line_id')), None)
    if line is None or line['issues']:
        raise ValueError('Line geometry is unavailable or invalid; resolve source QA separately')
    tooth, region = payload.get('tooth_instance_id'), payload.get('region')
    if (tooth is None) == (region is None):
        raise ValueError('Select exactly one tooth or region')
    if tooth is not None and not any(t['tooth_instance_id'] == tooth and not t['issues'] for t in record['teeth']):
        raise ValueError('Unknown or invalid tooth mask')
    if region is not None and (not isinstance(region, str) or not region.strip() or len(region) > 200):
        raise ValueError('Provide a region description (1–200 characters)')
    if region is not None:
        region = region.strip()
    note = payload.get('note', '')
    if not isinstance(note, str) or len(note) > 5000:
        raise ValueError('Invalid note')
    decision = {'id': pair_id(record['key'], line['bone_line_id'], tooth, region), 'key': record['key'],
                'image_id': record['image_id'], 'bone_line_id': line['bone_line_id'], 'tooth_instance_id': tooth,
                'region': region, 'status': status, 'reviewer': reviewer.strip(), 'note': note,
                'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'origin': 'expert_review'}
    updated = json.loads(json.dumps(review))
    replacement = payload.get('reassign_from')
    if replacement:
        effective = {c['id']: c for c in data['candidates']}
        effective.update({c['id']: c for c in review['decisions']})
        old = effective.get(replacement)
        if not old or old['key'] != decision['key'] or old['bone_line_id'] != decision['bone_line_id'] or replacement == decision['id']:
            raise ValueError('Reassignment must replace a different target on the same line')
        rejected = {**old, 'status': 'rejected', 'origin': 'expert_review', 'reviewer': decision['reviewer'],
                    'timestamp_utc': decision['timestamp_utc'], 'note': f"Reassigned to {decision['id']}. {note}"}
        updated['decisions'] = [d for d in updated['decisions'] if d['id'] != replacement] + [rejected]
        updated['history'].append(rejected)
    updated['decisions'] = [d for d in updated['decisions'] if d['id'] != decision['id']] + [decision]
    updated['history'].append(decision)
    updated['revision'] += 1
    return updated


def build():
    source = ROOT / 'DenPAR Radiographs Dataset' / 'Dataset'
    qa_path = HERE.parent / 'step1_boneline_visualization' / 'qa_summary.json'
    qa = json.loads(qa_path.read_text())
    prior = json.loads((HERE / 'expert_review.json').read_text()) if (HERE / 'expert_review.json').exists() else None
    if prior and prior['history']:
        raise RuntimeError('Existing expert decisions are preserved. Archive this output directory before rebuilding.')
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'review_overlays').mkdir(exist_ok=True)
    hashes = {str(qa_path.relative_to(ROOT)): digest(qa_path)}
    records, candidates = [], []
    for n, previous in enumerate(qa['records']):
        key, image_id, split = previous['key'], previous['image_id'], previous['split']
        r = {'key': key, 'image_id': image_id, 'split': split, 'metadata': previous['metadata'],
             'teeth': [], 'bone_lines': [], 'issues': []}
        records.append(r)
        image_path = source / previous['image_source'] if previous.get('image_source') else source / split / 'Images' / f'{image_id}.jpg'
        base = None
        if image_path.exists():
            try:
                base = Image.open(image_path).convert('RGB')
                hashes[str(image_path.relative_to(ROOT))] = digest(image_path)
            except OSError:
                r['issues'].append('unreadable_image')
        else:
            r['issues'].append('missing_image')
        width, height = base.size if base else (None, None)
        r['dimensions'] = [width, height]
        bone_path = source / previous['bone_source']
        bone = {}
        if bone_path.exists():
            hashes[str(bone_path.relative_to(ROOT))] = digest(bone_path)
            try:
                bone = json.loads(bone_path.read_text(encoding='utf-8-sig'))
                if not isinstance(bone, dict):
                    bone = {}
            except (ValueError, UnicodeError):
                r['issues'].append('unreadable_bone_json')
        raw_lines = bone.get('Bone_Lines', [])
        if not isinstance(raw_lines, list):
            raw_lines = []
            r['issues'].append('malformed_Bone_Lines')
        identity_ok = bone.get('Image_id') == f'{image_id}.jpg'
        if not identity_ok:
            r['issues'].append('bone_image_identity_mismatch')
        if bone.get('Num_of_Bone_Lines') != len(raw_lines):
            r['issues'].append('declared_line_count_mismatch')
        for i, line in enumerate(raw_lines):
            issues = validate_line(line, width, height)
            if base is None:
                issues.append('image_bounds_unavailable')
            if not identity_ok:
                issues.append('bone_image_identity_mismatch')
            r['bone_lines'].append({'bone_line_id': i, 'points': line if not issues else None, 'issues': issues})
        samples = {l['bone_line_id']: sample_line(l['points']) for l in r['bone_lines'] if not l['issues']}
        dest = HERE / 'review_overlays' / key
        if base:
            dest.mkdir(exist_ok=True)
            base.save(dest / 'original.jpg', quality=95)
            r['original_url'] = f'review_overlays/{key}/original.jpg'
            composite = base.convert('RGBA')
        paths = sorted((source / split / 'Masks (Tooth-wise)' / image_id).glob('*.png'))
        if not paths:
            r['issues'].append('missing_tooth_masks')
        for path in paths:
            tooth_id = path.stem
            t = {'tooth_instance_id': tooth_id, 'source': str(path.relative_to(ROOT)), 'issues': []}
            r['teeth'].append(t)
            hashes[str(path.relative_to(ROOT))] = digest(path)
            try:
                with Image.open(path) as mask:
                    fg = np.asarray(mask.convert('L')) != 0
                    t['dimensions'] = list(mask.size)
                yy, xx = np.where(fg)
                if not len(xx):
                    t['issues'].append('empty_mask')
                    continue
                t['bbox_xyxy_pixel_centers'] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
                t['centroid_xy'] = [float(xx.mean()), float(yy.mean())]
                if base is None:
                    t['issues'].append('image_bounds_unavailable')
                elif t['dimensions'] != [width, height]:
                    t['issues'].append('mask_dimension_mismatch')
                if t['issues']:
                    continue
                rgba = np.zeros((height, width, 4), dtype=np.uint8)
                rgba[fg] = [70, 190, 255, 95]
                layer = Image.fromarray(rgba)
                layer.save(dest / f'{tooth_id}.png')
                t['overlay_url'] = f'review_overlays/{key}/{tooth_id}.png'
                composite = Image.alpha_composite(composite, layer)
                distance = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                for line_id, sampled in samples.items():
                    f = features(sampled, distance, t['bbox_xyxy_pixel_centers'], t['centroid_xy'])
                    if f['mask_distance_normalized'] > .15:
                        continue
                    status = 'probable' if f['mask_distance_normalized'] <= .03 and f['fraction_samples_in_expanded_mask'] >= .2 else 'uncertain'
                    candidates.append({'id': pair_id(key, line_id, tooth_id), 'key': key, 'image_id': image_id,
                                       'tooth_instance_id': tooth_id, 'bone_line_id': line_id,
                                       'candidate_score': f.pop('candidate_score'), 'status': status,
                                       'confirmation_status': 'unconfirmed', 'origin': 'geometry_heuristic', 'features': f})
            except OSError as exc:
                t['issues'].append(f'unreadable_mask:{exc}')
        if base:
            draw = ImageDraw.Draw(composite)
            for l in r['bone_lines']:
                if not l['issues']:
                    draw.line([tuple(p) for p in l['points']], fill='yellow', width=3)
                    draw.text(tuple(l['points'][0]), f"B{l['bone_line_id']}", fill='yellow', stroke_width=1, stroke_fill='black')
            composite.convert('RGB').save(dest / 'overview.jpg', quality=90)
        if (n+1) % 100 == 0:
            print(f'{n+1}/{len(qa["records"])} records processed', flush=True)
    data = {'schema_version': 1, 'scope': 'Candidate correspondence only; no clinical inference or model training',
            'method': {'line_sampling_max_spacing_px': 1, 'distance': 'Euclidean distance to foreground pixel centers, bilinear sampled; approximate, not exact continuous-mask distance',
                       'candidate_gate': 'min mask distance / tooth bbox diagonal <= 0.15',
                       'score': '0.7 * exp(-normalized_distance / 0.05) + 0.3 * fraction_in_expanded_mask',
                       'expanded_mask_radius': '0.05 * tooth bbox diagonal',
                       'probable': 'normalized_distance <= 0.03 AND expanded-mask sample fraction >= 0.2',
                       'calibration': 'Heuristic thresholds, not calibrated probabilities or anatomical validation',
                       'identity': 'Split + image ID + original mask filename; bone line ID is zero-based source array index; no FDI assignment'},
            'records': records, 'candidates': candidates, 'source_sha256': hashes}
    data['dataset_id'] = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    data['sources_unchanged'] = all(digest(ROOT / p) == value for p, value in hashes.items())
    if not data['sources_unchanged']:
        raise RuntimeError('Source changed during analysis; rerun from stable inputs')
    atomic_json(HERE / 'candidate_mappings.json', data)
    review = empty_review(data)
    atomic_json(HERE / 'expert_review.json', review)
    result = report(data, review)
    atomic_json(HERE / 'reports' / 'correspondence_summary.json', result)
    print(json.dumps(result['counts'], indent=2))


if __name__ == '__main__':
    build()
