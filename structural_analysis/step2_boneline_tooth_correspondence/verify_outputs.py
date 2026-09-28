"""Check generated proposals, asset coverage, source integrity and review report."""
import json
from collections import Counter
from PIL import Image
from correspondence import HERE, ROOT, atomic_json, digest, report

data = json.loads((HERE / 'candidate_mappings.json').read_text())
review = json.loads((HERE / 'expert_review.json').read_text())
records = {r['key']: r for r in data['records']}
assert len(records) == 1000
assert len({c['id'] for c in data['candidates']}) == len(data['candidates'])
for c in data['candidates']:
    r = records[c['key']]
    line = next(l for l in r['bone_lines'] if l['bone_line_id'] == c['bone_line_id'])
    tooth = next(t for t in r['teeth'] if t['tooth_instance_id'] == c['tooth_instance_id'])
    assert not line['issues'] and not tooth['issues']
    assert c['status'] in {'probable', 'uncertain'} and c['confirmation_status'] == 'unconfirmed'
    assert 0 <= c['candidate_score'] <= 1
    assert c['features']['mask_distance_normalized'] <= .15
assets = 0
for r in records.values():
    if 'original_url' not in r:
        continue
    paths = [r['original_url'], f'review_overlays/{r["key"]}/overview.jpg']
    paths += [t['overlay_url'] for t in r['teeth'] if 'overlay_url' in t]
    for path in paths:
        with Image.open(HERE / path) as img:
            assert list(img.size) == r['dimensions']
            img.verify()
        assets += 1
for path, expected in data['source_sha256'].items():
    assert digest(ROOT / path) == expected, path
expected = report(data, review)
assert expected == json.loads((HERE / 'reports' / 'correspondence_summary.json').read_text())
result = {'status': 'passed', 'candidate_count': len(data['candidates']), 'source_files_unchanged': len(data['source_sha256']),
          'overlay_assets_verified': assets, 'candidate_status_counts': dict(Counter(c['status'] for c in data['candidates'])),
          'expert_review_revision': review['revision'], 'report_consistent': True}
atomic_json(HERE / 'reports' / 'verification.json', result)
print(json.dumps(result, indent=2))
