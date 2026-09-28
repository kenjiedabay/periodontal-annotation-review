"""Check generated inventory consistency, all layer dimensions and source hashes."""
import hashlib
import json
from pathlib import Path
from PIL import Image

folder = Path(__file__).resolve().parent
data = json.loads((folder / 'qa_summary.json').read_text())
records = data['records']
assert sum(r['line_count'] for r in records) == data['total_bone_lines']
assert data['geometrically_valid_line_count'] + data['invalid_line_count'] + data['lines_with_unchecked_bounds'] == data['total_bone_lines']
rendered = [r for r in records if 'overlay' in r]
assert len(rendered) == data['rendered_images']
for record in rendered:
    location = folder / 'overlays' / record['key']
    for name in ('original.jpg', 'overlay.jpg', 'masks.png', 'boxes.png', 'bones.png', 'cej.png', 'apex.png'):
        with Image.open(location / name) as img:
            assert list(img.size) == record['dimensions'], (record['key'], name)
            img.verify()
for category, key in data['representatives'].items():
    assert key and (folder / 'overlays' / f'representative_{category}.jpg').exists()
dataset = folder.parents[1] / 'DenPAR Radiographs Dataset' / 'Dataset'
hashes = json.loads((folder / 'reports' / 'source_annotation_sha256.json').read_text())
for name, digest in hashes.items():
    assert hashlib.sha256((dataset / name).read_bytes()).hexdigest() == digest, name
print(f'PASS: {len(rendered)} overlay bundles, seven representative categories, {len(hashes)} unchanged source annotations/metadata, and aggregate counts')
