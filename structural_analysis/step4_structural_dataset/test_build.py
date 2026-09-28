import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from build import construct, sha


def fixture(root, confirmed=False):
    dataset = root/'DenPAR Radiographs Dataset'/'Dataset'/'Validation'
    paths = {'image':dataset/'Images'/'1.jpg', 'mask':dataset/'Masks (Tooth-wise)'/'1'/'mask1.png',
             'landmark':dataset/'Key Points Annotations'/'1.json','bone':dataset/'Bone Level Annotations'/'1.json'}
    for p in paths.values():
        p.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(10,10),'gray').save(paths['image'])
    Image.new('L',(10,10),255).save(paths['mask'])
    paths['landmark'].write_text(json.dumps({'Image_id':'1.jpg','CEJ_Points':[[2,2],[3,2]],'Apex_Points':[[5,8]]}))
    paths['bone'].write_text(json.dumps({'Image_id':'1.jpg','Bone_Lines':[[[1,4],[8,4]]]}))
    points=[{'point_id':pid,'point_type':kind,'source_index':i,'coordinate':xy,'issues':[]} for pid,kind,i,xy in [('CEJ:0','CEJ',0,[2,2]),('CEJ:1','CEJ',1,[3,2]),('Apex:0','Apex',0,[5,8])]]
    record={'key':'Validation_1','image_id':'1','split':'Validation','teeth':[{'tooth_instance_id':'mask1','source':str(paths['mask'].relative_to(root)),'issues':[]}],
            'points':points,'bone_lines':[{'bone_line_id':0,'points':[[1,4],[8,4]],'issues':[]}]}
    bundles, provenance={},{}
    for kind in ['bone','landmark']:
        candidates=[]
        for item in record['bone_lines' if kind=='bone' else 'points']:
            c={'id':f'{kind}:{len(candidates)}','key':'Validation_1','image_id':'1','tooth_instance_id':'mask1','status':'probable'}
            c.update({'bone_line_id':0} if kind=='bone' else {k:item[k] for k in ['point_id','point_type','coordinate']})
            candidates.append(c)
        data={'records':[record],'candidates':candidates,'source_sha256':{str(p.relative_to(root)):sha(p) for p in paths.values()}}
        data['dataset_id']=hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()
        decisions=[{**c,'status':'confirmed','origin':'expert_review','reviewer':'Synthetic expert','timestamp_utc':'2026-09-15T00:00:00+00:00'} for c in candidates] if confirmed else []
        review={'candidate_dataset_id' if kind=='bone' else 'dataset_id':data['dataset_id'],'revision':int(confirmed),'decisions':decisions,'history':list(decisions)}
        bundles[kind]=(data,review)
        provenance[kind]={'review_file':f'{kind}_review.json'}
    return bundles,provenance,paths


class ConstructionTests(unittest.TestCase):
    def test_no_reviews_excludes_all_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundles,prov,_=fixture(root)
            train,excluded,summary=construct(root,bundles,prov)
            self.assertEqual(train['records'],[])
            self.assertEqual(summary['unresolved_tooth_records'],1)
            self.assertEqual(len(excluded['annotations']),4)
            self.assertTrue(all(e['original_provenance']['source_file'] for e in excluded['annotations']))

    def test_confirmed_targets_preserve_multiple_cej_and_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundles,prov,paths=fixture(root,True)
            before={str(p):sha(p) for p in paths.values()}
            train,excluded,summary=construct(root,bundles,prov)
            record=train['records'][0]
            self.assertEqual(record['cej_points'],[[2,2],[3,2]])
            self.assertEqual(record['apex_points'],[[5,8]])
            self.assertEqual(record['partition'],'Validation')
            self.assertEqual(record['bbox'],[0,0,10,10])
            self.assertIsNone(record['fdi'])
            self.assertEqual(summary['records_with_all_required_structures'],1)
            self.assertEqual(len(record['target_provenance']),4)
            self.assertEqual(excluded['annotations'],[])
            self.assertEqual(before,{str(p):sha(p) for p in paths.values()})

    def test_rejection_and_no_history_never_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundles,prov,_=fixture(root,True)
            bundles['bone'][1]['history']=[]
            for d in bundles['landmark'][1]['decisions']:
                d['status']='rejected'
            train,_,summary=construct(root,bundles,prov)
            self.assertEqual(train['records'],[])
            self.assertEqual(summary['usable_tooth_records'],0)

    def test_partial_record_excludes_unconfirmed_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundles,prov,_=fixture(root,True)
            bundles['landmark'][1]['decisions']=[]
            bundles['landmark'][1]['history']=[]
            train,excluded,summary=construct(root,bundles,prov)
            self.assertEqual(train['records'][0]['cej_points'],[])
            self.assertEqual(summary['records_with_bone_lines'],1)
            self.assertEqual(summary['records_with_all_required_structures'],0)
            self.assertEqual(len(excluded['annotations']),3)

    def test_stale_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundles,prov,paths=fixture(root,True)
            bundles['bone'][1]['candidate_dataset_id']='wrong'
            with self.assertRaises(ValueError):
                construct(root,bundles,prov)
            bundles['bone'][1]['candidate_dataset_id']=bundles['bone'][0]['dataset_id']
            paths['bone'].write_text('{}')
            with self.assertRaises(ValueError):
                construct(root,bundles,prov)


if __name__=='__main__':
    unittest.main()
