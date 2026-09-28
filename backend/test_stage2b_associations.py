import copy, json
from pathlib import Path
import numpy as np
import pytest
import stage2b_associations as s

def tooth(tid,mask): return {'tooth_instance_id':tid,'_mask':mask,'_scale':float(np.sqrt(mask.sum()))}

def test_crop_bounds_and_transform_round_trip():
    box, clipped=s.crop_box([2,3,20,30],25,35,s.DEFAULT_CONFIG)
    assert box[0]>=0 and box[1]>=0 and box[2]<=25 and box[3]<=35 and clipped['left']
    p=(12,18); crop=(p[0]-box[0],p[1]-box[1]); assert (crop[0]+box[0],crop[1]+box[1])==p

def test_ambiguous_and_unassigned_are_preserved():
    a=np.zeros((20,20),bool);b=a.copy();a[5:10,2:6]=1;b[5:10,14:18]=1
    proposal=s.associate_point([9.5,7],[tooth('a',a),tooth('b',b)],1.1,.2)
    assert proposal['selected_tooth_instance_id'] is None and proposal['ambiguity_status']=='ambiguous'
    far=s.associate_point([10,19],[tooth('a',a),tooth('b',b)],.01,.001)
    assert far['ambiguity_status']=='unassigned'

def test_multiple_apices_and_no_anatomical_side(tmp_path,monkeypatch):
    # The record format permits repeated neutral apex candidates and only geometric sides.
    assert s.DEFAULT_CONFIG['anatomical_side_policy']=='hypothesis_only'
    proposals=[{'landmark_type':'apex','neutral_landmark_id':f'apex_candidate_{i}'} for i in range(1,4)]
    assert len(proposals)==3 and all('mesial' not in p.values() for p in proposals)

def test_manifest_version_hash_and_exclusion_guards(tmp_path):
    for name,h in s.EXPECTED.items():
        d={'policy_version':'denpar-leakage-remediation-v2','content_hash':h}
        (tmp_path/name).write_text(json.dumps(d))
    with pytest.raises(ValueError,match='hash mismatch'): s.load_manifests(tmp_path)
    d={'policy_version':'denpar-leakage-remediation-v1'};(tmp_path/next(iter(s.EXPECTED))).write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Only v2'): s.load_manifests(tmp_path)

def test_raw_proposal_immutable_on_correction():
    raw={'selected_tooth_instance_id':'t1','original_coordinate':[4,5]}; snapshot=copy.deepcopy(raw)
    correction={'raw_proposal':copy.deepcopy(raw),'review_decision':'reassign','selected_tooth_instance_id':'t2'}
    assert raw==snapshot and correction['raw_proposal']==snapshot

def test_review_correction_keeps_raw_proposal_immutable(monkeypatch):
    from app import association_review as review
    raw={'source_annotation_id':'Training:1:cej:1','candidate_tooth_instance_ids':['Training:1:tooth:01'],'selected_tooth_instance_id':'Training:1:tooth:01'}
    record={'stable_image_id':'Training:1','tooth_instances':[{'tooth_instance_id':'Training:1:tooth:01'}],'landmark_proposals':[raw],'bone_line_proposals':[]}
    monkeypatch.setattr(review,'get_associations',lambda partition,image_id:{**record,'corrections':[]});review.CORRECTIONS.clear()
    payload=review.Correction(reviewer_id='expert',source_annotation_id=raw['source_annotation_id'],decision='reject',raw_proposal=copy.deepcopy(raw))
    saved=review.correct_association('Training','1',payload)
    assert saved['raw_proposal']==raw and record['landmark_proposals'][0]==raw
