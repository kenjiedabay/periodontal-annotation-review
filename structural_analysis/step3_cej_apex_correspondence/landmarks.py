"""Unordered CEJ/apex geometric candidates and separate expert review."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from image_paths import resolve_image
sys.path.insert(0, str(HERE.parent / 'step1_boneline_visualization'))
from build import valid_point


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def bbox_distance(point, box):
    x, y = point
    a, b, c, d = box
    return math.hypot(max(a-x, x-c, 0), max(b-y, y-d, 0))


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union else 0


def boundary_pixels(fg):
    padded = np.pad(fg, 1)
    interior = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    y, x = np.where(fg & ~interior)
    return np.column_stack((x, y))


def evidence(point, fg, boundary, box, centroid, boxes):
    x, y = point
    px, py = int(round(x)), int(round(y))
    inside = 0 <= py < fg.shape[0] and 0 <= px < fg.shape[1] and bool(fg[py, px])
    distance = math.hypot(x-px, y-py) if inside else float(np.sqrt(((boundary-np.asarray(point))**2).sum(axis=1).min()))
    scale = max(1, math.hypot(box[2]-box[0], box[3]-box[1]))
    box_evidence = [{'bbox_id': b['bbox_id'], 'point_distance_px': bbox_distance(point, b['bbox']),
                     'point_inside': bbox_distance(point, b['bbox']) == 0,
                     'tooth_bbox_iou': iou(box, b['bbox'])} for b in boxes if not b['issues']]
    support = max((b['tooth_bbox_iou'] * math.exp(-b['point_distance_px']/(.05*scale)) for b in box_evidence), default=0)
    centroid_distance = math.dist(point, centroid)
    vertical = (y-box[1])/max(1, box[3]-box[1])
    score = .65*math.exp(-distance/(.05*scale)) + .25*support + .10*math.exp(-centroid_distance/scale)
    return {'inside_mask_nearest_pixel': inside, 'min_mask_pixel_center_distance_px': distance,
            'normalized_mask_distance': distance/scale, 'keypoint_bbox_evidence': box_evidence,
            'keypoint_bbox_support': support, 'distance_to_centroid_px': centroid_distance,
            'point_minus_centroid_xy': [x-centroid[0], y-centroid[1]], 'relative_vertical_position': vertical,
            'image_vertical_region': 'above' if vertical < 0 else 'upper' if vertical < 1/3 else 'middle' if vertical < 2/3 else 'lower' if vertical <= 1 else 'below',
            'candidate_score': score}


def identity(key, point_id, tooth, kind=None):
    return f'{key}|{point_id}|{tooth}' if point_id is not None else f'{key}|missing:{kind}|{tooth}'


def empty_review(data):
    return {'dataset_id': data['dataset_id'], 'revision': 0, 'decisions': [], 'history': []}


def effective(data, review):
    values = {c['id']: c for c in data['candidates']}
    values.update({d['id']: d for d in review['decisions']})
    return list(values.values())


def report(data, review):
    entries = effective(data, review)
    active = [e for e in entries if e['status'] in {'confirmed', 'probable', 'uncertain'} and e.get('point_id') is not None]
    confirmed = [e for e in active if e['status'] == 'confirmed']
    missing = [e for e in entries if e['status'] == 'missing']
    confirmed_points = {(e['key'], e['point_id']) for e in confirmed}
    unresolved = [{'key': r['key'], 'point_id': p['point_id'], 'point_type': p['point_type'], 'issues': p['issues']}
                  for r in data['records'] for p in r['points'] if (r['key'], p['point_id']) not in confirmed_points]
    grouped, point_targets = defaultdict(list), defaultdict(list)
    for e in active:
        grouped[(e['key'], e['tooth_instance_id'], e['point_type'])].append(e)
        point_targets[(e['key'], e['point_id'])].append(e)
    multiples = [{'key': k[0], 'tooth_instance_id': k[1], 'point_type': k[2], 'point_ids': [e['point_id'] for e in v],
                  'confirmed_point_count': sum(e['status'] == 'confirmed' for e in v)} for k, v in grouped.items() if len(v) > 1]
    shared = [{'key': k[0], 'point_id': k[1], 'tooth_instances': [e['tooth_instance_id'] for e in v]} for k, v in point_targets.items() if len(v) > 1]
    reviewed_slots = {(e['key'], e['tooth_instance_id'], e['point_type']) for e in confirmed+missing}
    unreviewed = [{'key': r['key'], 'tooth_instance_id': t['tooth_instance_id'], 'point_type': kind}
                  for r in data['records'] for t in r['teeth'] for kind in ['CEJ', 'Apex']
                  if (r['key'], t['tooth_instance_id'], kind) not in reviewed_slots]
    review_images = sorted({x['key'] for x in unresolved+unreviewed+shared} | {r['key'] for r in data['records'] if r['issues']})
    return {'dataset_id': data['dataset_id'], 'review_revision': review['revision'],
            'counts': {'confirmed_correspondence_count': len(confirmed), 'unresolved_point_count': len(unresolved),
                       'missing_CEJ_count': sum(e['point_type'] == 'CEJ' for e in missing),
                       'missing_apex_count': sum(e['point_type'] == 'Apex' for e in missing),
                       'multiple_point_cases': len(multiples), 'expert_approved_multiple_point_cases': sum(m['confirmed_point_count'] > 1 for m in multiples),
                       'images_requiring_expert_review': len(review_images), 'unreviewed_tooth_type_slots': len(unreviewed)},
            'confirmed_mappings': confirmed, 'unresolved_points': unresolved, 'expert_marked_missing': missing,
            'multiple_point_cases': multiples, 'points_with_multiple_candidate_teeth': shared,
            'unreviewed_tooth_type_slots': unreviewed, 'images_requiring_expert_review': review_images,
            'missing_count_definition': 'Explicit expert-marked missing tooth/type slots only. Unreviewed or no-candidate slots are not asserted missing.'}


def apply_review(data, review, payload):
    if review['dataset_id'] != data['dataset_id'] or payload.get('revision') != review['revision']:
        raise ValueError('Review version changed; reload before saving')
    reviewer = payload.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('Reviewer name is required')
    status = payload.get('status')
    if status not in {'confirmed', 'rejected', 'uncertain', 'cannot_determine', 'missing'}:
        raise ValueError('Invalid review status')
    r = next((r for r in data['records'] if r['key'] == payload.get('key')), None)
    if r is None:
        raise ValueError('Unknown image')
    tooth = payload.get('tooth_instance_id')
    if not any(t['tooth_instance_id'] == tooth and not t['issues'] for t in r['teeth']):
        raise ValueError('Unknown or unavailable tooth')
    point_id = payload.get('point_id')
    if point_id is None:
        kind = payload.get('point_type')
        if kind not in {'CEJ', 'Apex'} or status not in {'missing', 'uncertain'}:
            raise ValueError('Missing review needs a tooth and CEJ/Apex type; uncertain clears a missing decision')
        coordinate = None
    else:
        p = next((p for p in r['points'] if p['point_id'] == point_id), None)
        if p is None or p['issues'] or status == 'missing':
            raise ValueError('Point is invalid/unavailable; mark missing at tooth/type level')
        kind, coordinate = p['point_type'], p['coordinate']
    values = effective(data, review)
    slot = [v for v in values if v['key'] == r['key'] and v['tooth_instance_id'] == tooth and v['point_type'] == kind]
    if status == 'confirmed' and any(v['status'] == 'missing' for v in slot):
        raise ValueError('Clear the missing decision before confirming a point')
    if status == 'missing' and any(v['status'] == 'confirmed' for v in slot):
        raise ValueError('Reject or reassign confirmed points before marking this landmark type missing')
    note = payload.get('note', '')
    if not isinstance(note, str) or len(note) > 5000:
        raise ValueError('Invalid note')
    decision = {'id': identity(r['key'], point_id, tooth, kind), 'key': r['key'], 'image_id': r['image_id'],
                'point_id': point_id, 'point_type': kind, 'coordinate': coordinate, 'tooth_instance_id': tooth,
                'status': status, 'reviewer': reviewer.strip(), 'note': note, 'origin': 'expert_review',
                'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    changes = []
    old_id = payload.get('reassign_from')
    if old_id:
        old = next((v for v in values if v['id'] == old_id), None)
        if point_id is None or status != 'confirmed' or not old or old['key'] != r['key'] or old['point_id'] != point_id or old_id == decision['id']:
            raise ValueError('Reassignment must replace another target for the same point')
        changes.append({**old, 'status': 'rejected', 'reviewer': reviewer.strip(), 'origin': 'expert_review',
                        'note': 'Reassigned to '+tooth+'. '+note, 'timestamp_utc': decision['timestamp_utc'],
                        'candidate_tooth_instance_id': old['tooth_instance_id'], 'expert_tooth_instance_id': tooth,
                        'expert_status': 'rejected', 'expert_comment': 'Reassigned to '+tooth+'. '+note})
    decision.update(candidate_tooth_instance_id=old['tooth_instance_id'] if old_id else tooth,
                    expert_tooth_instance_id=tooth, expert_status=status, expert_comment=note)
    changes.append(decision)
    updated = json.loads(json.dumps(review))
    updated['decisions'] = [d for d in updated['decisions'] if d['id'] not in {c['id'] for c in changes}] + changes
    updated['history'].extend(changes)
    updated['revision'] += 1
    return updated


def build():
    review_path = HERE / 'expert_landmark_review.json'
    if review_path.exists() and json.loads(review_path.read_text())['history']:
        raise RuntimeError('Existing expert history must be archived before rebuilding')
    source = ROOT / 'DenPAR Radiographs Dataset' / 'Dataset'
    qa_path = HERE.parent / 'step1_boneline_visualization' / 'qa_summary.json'
    qa = json.loads(qa_path.read_text())
    hashes = {str(qa_path.relative_to(ROOT)): sha(qa_path)}
    config_path = HERE.parent / 'image_paths.json'
    hashes[str(config_path.relative_to(ROOT))] = sha(config_path)
    records, candidates = [], []
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'overlays').mkdir(exist_ok=True)
    for n, prior in enumerate(qa['records']):
        key, image_id, split = prior['key'], prior['image_id'], prior['split']
        r = {'key': key, 'image_id': image_id, 'split': split, 'metadata': prior['metadata'], 'issues': [], 'teeth': [], 'bboxes': [], 'points': []}
        records.append(r)
        image_path = resolve_image(split, image_id)
        base = None
        if image_path.exists():
            hashes[str(image_path.relative_to(ROOT))] = sha(image_path)
            try:
                with Image.open(image_path) as img:
                    base = img.convert('RGB')
            except OSError:
                r['issues'].append('unreadable_image')
        else:
            r['issues'].append('missing_image')
        width, height = base.size if base else (None, None)
        r['dimensions'] = [width, height]
        kp_path = source / split / 'Key Points Annotations' / f'{image_id}.json'
        kp = {}
        if kp_path.exists():
            hashes[str(kp_path.relative_to(ROOT))] = sha(kp_path)
            try:
                kp = json.loads(kp_path.read_text(encoding='utf-8-sig'))
                if not isinstance(kp, dict):
                    kp = {}
                    r['issues'].append('malformed_keypoint_root')
            except (ValueError, UnicodeError):
                r['issues'].append('unreadable_keypoint_json')
        else:
            r['issues'].append('missing_keypoint_file')
        identity_ok = kp.get('Image_id') == f'{image_id}.jpg'
        if not identity_ok:
            r['issues'].append('keypoint_image_id_mismatch')
        boxes = kp.get('bboxes', [])
        if not isinstance(boxes, list):
            boxes = []
            r['issues'].append('malformed_bbox_array')
        for i, box in enumerate(boxes):
            issues = []
            if not isinstance(box, list) or len(box) != 4 or not valid_point(box[:2]) or not valid_point(box[2:]) or box[0] >= box[2] or box[1] >= box[3]:
                issues.append('malformed_bbox')
            elif base and not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                issues.append('bbox_out_of_bounds')
            if not base:
                issues.append('image_bounds_unavailable')
            if not identity_ok:
                issues.append('keypoint_image_id_mismatch')
            r['bboxes'].append({'bbox_id': i, 'bbox': box if not issues else None, 'issues': issues})
        for kind, field in [('CEJ', 'CEJ_Points'), ('Apex', 'Apex_Points')]:
            points = kp.get(field, [])
            if not isinstance(points, list):
                points = []
                r['issues'].append('malformed_'+field)
            if not points:
                r['issues'].append('no_source_'+field)
            for i, point in enumerate(points):
                issues = []
                if not valid_point(point):
                    issues.append('malformed_coordinate')
                elif base and not (0 <= point[0] < width and 0 <= point[1] < height):
                    issues.append('point_out_of_bounds')
                if not base:
                    issues.append('image_bounds_unavailable')
                if not identity_ok:
                    issues.append('keypoint_image_id_mismatch')
                r['points'].append({'point_id': f'{kind}:{i}', 'point_type': kind, 'source_index': i,
                                    'coordinate': point if valid_point(point) else None, 'issues': issues,
                                    'validation_status': 'invalid_or_unavailable' if issues else 'unconfirmed', 'candidate_teeth': []})
        r['source_counts'] = {'bboxes': len(boxes), 'CEJ': sum(p['point_type'] == 'CEJ' for p in r['points']), 'Apex': sum(p['point_type'] == 'Apex' for p in r['points'])}
        r['source_counts_equal'] = len(set(r['source_counts'].values())) == 1
        dest = HERE / 'overlays' / key
        if base:
            dest.mkdir(exist_ok=True)
            r['original_path'] = str(image_path.relative_to(ROOT))
            r['original_url'] = f'originals/{key}.jpg'
            composite = base.convert('RGBA')
        mask_paths = sorted((source / split / 'Masks (Tooth-wise)' / image_id).glob('*.png'))
        if not mask_paths:
            r['issues'].append('missing_tooth_masks')
        for path in mask_paths:
            hashes[str(path.relative_to(ROOT))] = sha(path)
            tooth = {'tooth_instance_id': path.stem, 'source': str(path.relative_to(ROOT)), 'issues': []}
            r['teeth'].append(tooth)
            try:
                with Image.open(path) as mask:
                    fg = np.asarray(mask.convert('L')) != 0
                    size = mask.size
                y, x = np.where(fg)
                if not len(x):
                    tooth['issues'].append('empty_mask')
                    continue
                box = [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]
                centroid = [float(x.mean()), float(y.mean())]
                tooth.update(bbox=box, centroid_xy=centroid)
                if base is None:
                    tooth['issues'].append('image_bounds_unavailable')
                elif size != base.size:
                    tooth['issues'].append('mask_dimension_mismatch')
                if tooth['issues']:
                    continue
                rgba = np.zeros((*fg.shape, 4), dtype=np.uint8)
                rgba[fg] = [60, 180, 255, 85]
                layer = Image.fromarray(rgba)
                layer.save(dest / f'{path.stem}.png')
                tooth['overlay_url'] = f'overlays/{key}/{path.stem}.png'
                composite = Image.alpha_composite(composite, layer)
                boundary = boundary_pixels(fg)
                for point in r['points']:
                    if point['issues']:
                        continue
                    f = evidence(point['coordinate'], fg, boundary, box, centroid, r['bboxes'])
                    if f['normalized_mask_distance'] > .15 and f['keypoint_bbox_support'] < .25:
                        continue
                    score = f.pop('candidate_score')
                    status = 'probable' if f['normalized_mask_distance'] <= .03 and f['keypoint_bbox_support'] >= .2 else 'uncertain'
                    c = {'id': identity(key, point['point_id'], path.stem), 'key': key, 'image_id': image_id,
                         'point_id': point['point_id'], 'point_type': point['point_type'], 'coordinate': point['coordinate'],
                         'tooth_instance_id': path.stem, 'candidate_score': score, 'status': status,
                         'validation_status': 'unconfirmed', 'origin': 'geometry_heuristic', 'features': f}
                    candidates.append(c)
                    point['candidate_teeth'].append({'tooth_instance_id': path.stem, 'candidate_score': score, 'status': status, 'validation_status': 'unconfirmed'})
            except OSError:
                tooth['issues'].append('unreadable_mask')
        if base:
            draw = ImageDraw.Draw(composite)
            for b in r['bboxes']:
                if not b['issues']:
                    draw.rectangle(b['bbox'], outline='orange', width=2)
            for p in r['points']:
                if not p['issues']:
                    x, y = p['coordinate']
                    color = 'cyan' if p['point_type'] == 'CEJ' else 'magenta'
                    draw.ellipse((x-5, y-5, x+5, y+5), fill=color)
                    draw.text((x+7, y), p['point_id'], fill=color, stroke_width=1, stroke_fill='black')
            composite.convert('RGB').save(dest / 'overview.jpg', quality=92)
        if (n+1) % 100 == 0:
            print(f'{n+1}/{len(qa["records"])}', flush=True)
    data = {'schema_version': 1, 'records': records, 'candidates': candidates, 'source_sha256': hashes,
            'method': {'index_ordering_used_for_correspondence': False, 'ids': 'Source indices identify points/boxes only; mask filenames identify tooth instances; no shared index/FDI mapping.',
                       'distance': 'Euclidean distance to nearest mask foreground pixel center. Inside uses nearest raster pixel.',
                       'bbox_evidence': 'Every point is compared to every valid keypoint box; box-to-tooth support uses geometric bbox IoU, never index.',
                       'score': '0.65 exp(-mask_distance/(0.05 tooth_diagonal)) + 0.25 max(box_IoU exp(-point_box_distance/(0.05 tooth_diagonal))) + 0.10 exp(-centroid_distance/tooth_diagonal)',
                       'gate': 'mask_distance/tooth_diagonal <= 0.15 OR bbox_support >= 0.25',
                       'probable': 'mask_distance/tooth_diagonal <= 0.03 AND bbox_support >= 0.2; otherwise uncertain',
                       'vertical': 'Image top/bottom position relative to tooth bbox is stored with Arch metadata; no crown/root orientation or disease inference is made and vertical position is not a score gate.',
                       'calibration': 'Uncalibrated geometric heuristics; not probabilities or ground truth'}}
    data['dataset_id'] = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if not all(sha(ROOT / p) == value for p, value in hashes.items()):
        raise RuntimeError('Inputs changed during generation')
    if review_path.exists() and json.loads(review_path.read_text())['history']:
        raise RuntimeError('Expert decisions appeared during generation; refusing overwrite')
    write(HERE / 'candidate_landmark_mappings.json', data)
    review = empty_review(data)
    write(review_path, review)
    result = report(data, review)
    write(HERE / 'reports' / 'summary.json', result)
    print(json.dumps(result['counts'], indent=2))


if __name__ == '__main__':
    build()
