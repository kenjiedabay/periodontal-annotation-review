"""Read-only DenPAR Bone_Lines geometry QA and portable layer viewer."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.structural_audit.loader import _read_workbook

LAYERS = {'masks': 'Tooth masks', 'boxes': 'Tooth boxes (mask extents)',
          'bones': 'Bone lines (yellow)', 'cej': 'CEJ (cyan)', 'apex': 'Apex (magenta)'}


def valid_point(p):
    return isinstance(p, list) and len(p) == 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in p)


def validate_line(line, width, height):
    errors = []
    if not isinstance(line, list):
        return ['line_not_list']
    if not line:
        errors.append('empty_line')
    valid = []
    for i, p in enumerate(line):
        if not valid_point(p):
            errors.append(f'point_{i}:malformed_or_nonfinite_pair')
        elif width is not None and height is not None and not (0 <= p[0] < width and 0 <= p[1] < height):
            errors.append(f'point_{i}:out_of_bounds')
        else:
            valid.append(tuple(p))
    if len(valid) < 2:
        errors.append('fewer_than_two_valid_points')
    elif len(set(valid)) < 2:
        errors.append('zero_length_line')
    return errors


def build(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    overlays = output / 'overlays'
    reports = output / 'reports'
    overlays.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    source_hashes = {}
    missing, records = [], []

    def read(path):
        if not path.exists():
            missing.append(str(path.relative_to(dataset)))
            return None
        data = path.read_bytes()
        source_hashes[str(path.relative_to(dataset))] = hashlib.sha256(data).hexdigest()
        try:
            value = json.loads(data.decode('utf-8-sig'))
            return value if isinstance(value, dict) else {'_error': 'JSON root must be object'}
        except (ValueError, UnicodeError) as exc:
            return {'_error': str(exc)}

    workbook = dataset / 'Characteristics of radiographs included.xlsx'
    metadata = _read_workbook(workbook) if workbook.exists() else {}
    if workbook.exists():
        source_hashes[workbook.name] = hashlib.sha256(workbook.read_bytes()).hexdigest()
    else:
        missing.append(workbook.name)
    for split in ('Training', 'Validation', 'Testing'):
        folder = dataset / split
        bone_dir = folder / 'Bone Level Annotations'
        ids = {p.stem for p in bone_dir.glob('*.json') if not p.name.startswith('coco_')}
        ids |= {p.stem for p in (folder / 'Key Points Annotations').glob('*.json')}
        ids |= {p.stem for p in (folder / 'Images').glob('*.jpg')}
        for image_id in sorted(ids):
            key = f'{split}_{image_id}'
            bone_path = bone_dir / f'{image_id}.json'
            bone = read(bone_path)
            kp = read(folder / 'Key Points Annotations' / f'{image_id}.json') or {}
            candidates = [folder / 'Images' / f'{image_id}.jpg', dataset / 'Images' / f'{image_id}.jpg']
            image_path = next((p for p in candidates if p.exists()), None)
            reasons = []
            rec = {'key': key, 'image_id': image_id, 'split': split, 'metadata': metadata.get(image_id, {}),
                   'bone_source': str(bone_path.relative_to(dataset)), 'issues': reasons, 'lines': [], 'tooth_count': 0}
            records.append(rec)
            if image_path is None:
                missing.append(f'{split}/Images/{image_id}.jpg (also absent from shared Images)')
                reasons.append('missing_image')
            lines = (bone or {}).get('Bone_Lines', [])
            if not isinstance(lines, list):
                reasons.append('Bone_Lines_not_list')
                lines = []
            rec['line_count'] = len(lines)
            if not bone or bone.get('_error'):
                reasons.append('missing_or_unreadable_bone_annotation')
            elif bone.get('Image_id') != f'{image_id}.jpg':
                reasons.append('bone_image_id_mismatch')
            if (bone or {}).get('Num_of_Bone_Lines') != len(lines):
                reasons.append('declared_line_count_mismatch')
            if not lines:
                reasons.append('no_bone_lines')
            if not rec['metadata']:
                reasons.append('missing_metadata')
            if image_path is None:
                rec['lines'] = [{'index': i, 'errors': validate_line(line, None, None), 'bounds_checked': False} for i, line in enumerate(lines)]
                continue
            try:
                with Image.open(image_path) as original:
                    base = original.convert('RGB')
            except OSError as exc:
                reasons.append(f'unreadable_image:{exc}')
                rec['lines'] = [{'index': i, 'errors': validate_line(line, None, None), 'bounds_checked': False} for i, line in enumerate(lines)]
                continue
            width, height = base.size
            rec['dimensions'] = [width, height]
            rec['image_source'] = str(image_path.relative_to(dataset))
            dest = overlays / key
            dest.mkdir(exist_ok=True)
            shutil.copyfile(image_path, dest / 'original.jpg')
            layers = {name: Image.new('RGBA', base.size) for name in LAYERS}
            masks = sorted((folder / 'Masks (Tooth-wise)' / image_id).glob('*.png'))
            rec['tooth_count'] = len(masks)
            if not masks:
                missing.append(f'{split}/Masks (Tooth-wise)/{image_id}/*.png')
                reasons.append('missing_tooth_masks')
            for i, path in enumerate(masks):
                try:
                    with Image.open(path) as mask:
                        if mask.size != base.size:
                            reasons.append(f'mask_dimension_mismatch:{path.name}')
                            continue
                        fg = np.asarray(mask.convert('L')) > 0
                    color = [(255, 110, 80), (70, 180, 255), (140, 240, 100), (200, 130, 255)][i % 4]
                    pixels = np.zeros((height, width, 4), dtype=np.uint8)
                    pixels[fg] = (*color, 65)
                    layers['masks'] = Image.alpha_composite(layers['masks'], Image.fromarray(pixels))
                    yy, xx = np.where(fg)
                    if len(xx):
                        draw = ImageDraw.Draw(layers['boxes'])
                        draw.rectangle((int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())), outline=(*color, 255), width=2)
                    else:
                        reasons.append(f'empty_mask:{path.name}')
                except OSError as exc:
                    reasons.append(f'unreadable_mask:{path.name}:{exc}')
            for i, line in enumerate(lines):
                errors = validate_line(line, width, height)
                rec['lines'].append({'index': i, 'errors': errors, 'bounds_checked': True,
                                     'source_points': line if errors else None})
                if errors:
                    reasons.append(f'invalid_bone_line:{i}')
                else:
                    draw = ImageDraw.Draw(layers['bones'])
                    draw.line([tuple(p) for p in line], fill='yellow', width=4)
                    draw.text(tuple(line[0]), f'B{i}', fill='yellow', stroke_width=1, stroke_fill='black')
            for name, field, color in [('cej', 'CEJ_Points', 'cyan'), ('apex', 'Apex_Points', 'magenta')]:
                points = kp.get(field, [])
                if not isinstance(points, list):
                    reasons.append(f'malformed_{field}')
                    continue
                for i, p in enumerate(points):
                    if not valid_point(p) or not (0 <= p[0] < width and 0 <= p[1] < height):
                        reasons.append(f'invalid_{field}:{i}')
                        continue
                    x, y = p
                    ImageDraw.Draw(layers[name]).ellipse((x-5, y-5, x+5, y+5), fill=color, outline='black', width=1)
            if kp.get('_error') or (kp and kp.get('Image_id') != f'{image_id}.jpg'):
                reasons.append('keypoint_parse_or_image_id_error')
            composite = base.convert('RGBA')
            for name, layer in layers.items():
                layer.save(dest / f'{name}.png')
                composite = Image.alpha_composite(composite, layer)
            composite.convert('RGB').save(dest / 'overlay.jpg', quality=92)
            rec['overlay'] = f'overlays/{key}/overlay.jpg'
        print(f'{split}: processed', flush=True)
    available = [r for r in records if 'overlay' in r]
    representatives = {}
    for label, field, value in [('upper_arch', 'Arch', 'Upper'), ('lower_arch', 'Arch', 'Lower'),
                                ('anterior', 'Site', 'Anterior'), ('left_posterior', 'Site', 'Left'), ('right_posterior', 'Site', 'Right')]:
        choices = [r for r in available if r['metadata'].get(field) == value]
        representatives[label] = choices[0]['key'] if choices else None
    if available:
        representatives['low_tooth_count'] = min(available, key=lambda r: r['tooth_count'])['key']
        representatives['high_tooth_count'] = max(available, key=lambda r: r['tooth_count'])['key']
    for label, key in representatives.items():
        if key:
            shutil.copyfile(overlays / key / 'overlay.jpg', overlays / f'representative_{label}.jpg')
    invalid = [{'key': r['key'], **line} for r in records for line in r['lines'] if line['errors']]
    unassessed = sum(not line['bounds_checked'] for r in records for line in r['lines'])
    usable = sum(line['bounds_checked'] and not line['errors'] for r in records for line in r['lines'])
    summary = {'scope': 'Step 1 only; source Bone_Lines, original pixel coordinates',
               'total_images': len(records), 'total_images_with_bone_lines': sum(r['line_count'] > 0 for r in records),
               'total_bone_lines': sum(r['line_count'] for r in records), 'rendered_images': len(available),
               'lines_per_image': {r['key']: r['line_count'] for r in records},
               'line_count_distribution': dict(Counter(r['line_count'] for r in records)),
               'invalid_line_count': len(invalid), 'invalid_lines': invalid, 'coordinate_anomalies': invalid,
               'geometrically_valid_line_count': usable, 'lines_with_unchecked_bounds': unassessed,
               'missing_files': missing, 'images_requiring_manual_review': [r['key'] for r in records if r['issues']],
               'representatives': representatives,
               'geometrically_usable': bool(available) and not invalid and not unassessed,
               'geometry_status': 'partially_usable' if usable and (invalid or unassessed) else ('usable' if usable else 'unavailable'),
               'geometry_note': 'Only complete in-bounds, finite, non-degenerate polylines are drawn. Invalid lines are omitted whole, never repaired or connected across invalid points.',
               'dental_expert_confirmation_required': True,
               'interpretation': 'Anatomical meaning and tooth correspondence remain unconfirmed for every image. Bone lines are not assigned to teeth and do not establish disease or severity. Line length is not a severity measure.',
               'records': records}
    unchanged = all(hashlib.sha256((dataset / p).read_bytes()).hexdigest() == digest for p, digest in source_hashes.items())
    summary['source_annotation_hashes_unchanged'] = unchanged
    (reports / 'source_annotation_sha256.json').write_text(json.dumps(source_hashes, indent=2), encoding='utf-8')
    (output / 'qa_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (reports / 'manual_review.json').write_text(json.dumps([r for r in records if r['issues']], indent=2), encoding='utf-8')
    template = Path(__file__).with_name('viewer.html').read_text(encoding='utf-8')
    (output / 'index.html').write_text(template.replace('/*DATA*/null', json.dumps(summary).replace('<', '\\u003c')), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['records', 'lines_per_image', 'invalid_lines', 'coordinate_anomalies', 'missing_files', 'images_requiring_manual_review']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'DenPAR Radiographs Dataset' / 'Dataset')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error('Output must be outside the original dataset')
    build(args.dataset.resolve(), args.output.resolve())
