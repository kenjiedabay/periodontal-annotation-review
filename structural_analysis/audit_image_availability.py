"""Verify official originals, CEJ coordinates and the prior Mask R-CNN provenance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from PIL import Image
from image_paths import ROOT, resolve_image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    source = ROOT / 'DenPAR Radiographs Dataset' / 'Dataset'
    experiment = ROOT / 'models' / 'tooth_instance_maskrcnn_baseline'
    provenance = json.loads((experiment / 'provenance_audit.json').read_text())
    prior = {r['image_id']: r for r in provenance['records']}
    config = json.loads((experiment / 'full_run/config/final_train_config.json').read_text())
    assert config['data']['training_images'] == 'dataset/raw'
    records, totals, original_hashes = [], {}, {}
    for split, expected in [('Training',650),('Validation',150),('Testing',200)]:
        folders = sorted(p for p in (source / split / 'Masks (Tooth-wise)').iterdir() if p.is_dir())
        assert len(folders) == expected
        for folder in folders:
            image_id = folder.name
            path = resolve_image(split,image_id)
            rec = {'key':f'{split}_{image_id}','split':split,'image_id':image_id,'image_path':str(path.relative_to(ROOT)), 'issues':[], 'CEJ_reviewable':0,'CEJ_unavailable':0}
            records.append(rec)
            kp_path = source / split / 'Key Points Annotations' / f'{image_id}.json'
            kp = json.loads(kp_path.read_text())
            original_hashes[str(kp_path.relative_to(ROOT))] = digest(kp_path)
            if not path.exists():
                rec['issues'].append('missing_image')
                rec['CEJ_unavailable'] = len(kp['CEJ_Points'])
                continue
            with Image.open(path) as image:
                size = image.size
                image.verify()
            rec['dimensions'] = list(size)
            rec['sha256'] = digest(path)
            original_hashes[str(path.relative_to(ROOT))] = rec['sha256']
            assert kp['Image_id'] == path.name
            for mask_path in sorted(folder.glob('*.png')):
                with Image.open(mask_path) as mask:
                    assert mask.size == size, mask_path
                original_hashes[str(mask_path.relative_to(ROOT))] = digest(mask_path)
            if split == 'Training':
                previous = prior[image_id]
                assert previous['status'] == 'CONFIRMED'
                assert previous['raw_image']['filename'] == path.name
                assert [previous['raw_image']['width'],previous['raw_image']['height']] == list(size)
                assert previous['raw_image']['sha256'] == rec['sha256']
                rec['matches_maskrcnn_original_sha256'] = True
            for point in kp['CEJ_Points']:
                valid = isinstance(point,list) and len(point)==2 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in point) and 0<=point[0]<size[0] and 0<=point[1]<size[1]
                rec['CEJ_reviewable' if valid else 'CEJ_unavailable'] += 1
        subset = [r for r in records if r['split']==split]
        totals[split] = {'expected':expected, 'found':sum(not r['issues'] for r in subset), 'missing':sum('missing_image' in r['issues'] for r in subset)}
    report = {'partitions':totals,'CEJ_points_now_reviewable':sum(r['CEJ_reviewable'] for r in records),
              'CEJ_points_still_unavailable':sum(r['CEJ_unavailable'] for r in records),
              'Training_originals_matching_MaskRCNN_checksums':sum(r.get('matches_maskrcnn_original_sha256',False) for r in records),
              'path_config':'structural_analysis/image_paths.json',
              'root_cause':'Step 3 used stale Step 1 image discovery, while Step 4 searched only DenPAR folders. Both omitted the recovered dataset/raw Training directory.',
              'path_audit':{'review_UI':'Serves originals directly through /originals/<split>_<id>.jpg using the shared resolver and checked source hash.',
                            'Step_3':'Shared resolver; existing CEJ/Apex JSON coordinates are loaded unchanged. Only derived geometry/candidates and visualization assets refresh.',
                            'Step_4':'Shared resolver; confirmed-only eligibility and official partitions unchanged.'},
              'experiment_evidence':['models/tooth_instance_maskrcnn_baseline/provenance_audit.json','models/tooth_instance_maskrcnn_baseline/full_run/config/final_train_config.json'],
              'source_hashes':original_hashes,'records':records,'training_performed':False}
    destination = ROOT/'structural_analysis/image_availability_report.json'
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'records','source_hashes'}},indent=2))
    return report


if __name__=='__main__':
    audit()
