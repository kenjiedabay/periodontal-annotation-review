import json
from collections import Counter
from PIL import Image
from landmarks import HERE, ROOT, report, sha, write

data=json.loads((HERE/'candidate_landmark_mappings.json').read_text())
review=json.loads((HERE/'expert_landmark_review.json').read_text())
records={r['key']:r for r in data['records']}
assert len({c['id'] for c in data['candidates']})==len(data['candidates'])
for c in data['candidates']:
    r=records[c['key']]
    p=next(p for p in r['points'] if p['point_id']==c['point_id'])
    t=next(t for t in r['teeth'] if t['tooth_instance_id']==c['tooth_instance_id'])
    assert not p['issues'] and not t['issues']
    assert c['coordinate']==p['coordinate'] and c['point_type']==p['point_type']
    assert c['validation_status']=='unconfirmed' and c['status'] in {'probable','uncertain'}
    assert 0 <= c['candidate_score'] <= 1
assets=0
for r in records.values():
    if not r.get('original_url'):
        continue
    paths=[f'overlays/{r["key"]}/overview.jpg']+[t['overlay_url'] for t in r['teeth'] if t.get('overlay_url')]
    if r.get('original_path'):
        with Image.open(ROOT/r['original_path']) as original:
            assert list(original.size)==r['dimensions']
            original.verify()
    else:
        paths.append(r['original_url'])
    for path in paths:
        with Image.open(HERE/path) as image:
            assert list(image.size)==r['dimensions']
            image.verify()
        assets+=1
for path,value in data['source_sha256'].items():
    assert sha(ROOT/path)==value,path
assert report(data,review)==json.loads((HERE/'reports'/'summary.json').read_text())
result={'status':'passed','images':len(records),'available_images':sum(bool(r.get('original_url')) for r in records.values()),
        'source_point_counts':dict(Counter(p['point_type'] for r in records.values() for p in r['points'])),
        'candidate_counts':dict(Counter(c['status'] for c in data['candidates'])),
        'invalid_or_unavailable_points':sum(bool(p['issues']) for r in records.values() for p in r['points']),
        'images_with_unequal_source_counts':sum(not r['source_counts_equal'] for r in records.values()),
        'source_files_unchanged':len(data['source_sha256']),'overlay_assets_verified':assets,'review_revision':review['revision']}
write(HERE/'reports'/'verification.json',result)
print(json.dumps(result,indent=2))
