import copy, json
from pathlib import Path
import pytest
import stage2c_pilot_v2 as pilot
from app import association_review as review
from fastapi import HTTPException
from pydantic import ValidationError

def first():
 q=json.loads((pilot.OUTPUT/'pilot_queue.json').read_text(encoding='utf-8'));return q,q['entries'][0]

def test_queue_reproducible_and_no_testing_or_exclusions(tmp_path):
 a=pilot.build(tmp_path/'a')['queue'];b=pilot.build(tmp_path/'b')['queue']
 identity=lambda q:[(x['stable_image_id'],x['primary_selection_reason']) for x in q['entries']]
 assert identity(a)==identity(b) and len(a['entries'])==40
 assert {x['source_partition'] for x in a['entries']}=={'Training','Validation'}
 excluded={x['stable_image_id'] for x in pilot.load_manifests()['denpar_exclusion_manifest.json']['entries']}
 assert not excluded.intersection(x['stable_image_id'] for x in a['entries'])

def test_testing_and_partition_changes_rejected(tmp_path,monkeypatch):
 with pytest.raises(HTTPException):review._entry('Testing','1')
 q,e=first();monkeypatch.setattr(review,'REVIEWS',tmp_path)
 payload=review.ExpertAction(reviewer_id='dentist',partition='Validation' if e['source_partition']=='Training' else 'Training',source_annotation_id=e['association_ids'][0],action='accept',finding_type='cej',review_state='expert_confirmed')
 with pytest.raises(HTTPException,match='Partition'):review.save_action(e['source_partition'],e['source_image_id'],payload)

def test_raw_immutable_and_corrections_version(tmp_path,monkeypatch):
 _,e=first();monkeypatch.setattr(review,'REVIEWS',tmp_path);record=review._association(e['source_partition'],e['source_image_id']);raw=copy.deepcopy(record['landmark_proposals'][0]);source=(pilot.STAGE2B/f"{e['source_partition'].lower()}_associations.jsonl").read_bytes()
 base=dict(reviewer_id='dentist-1',partition=e['source_partition'],source_annotation_id=raw['source_annotation_id'],action='accept',finding_type=raw['landmark_type'],review_state='expert_confirmed')
 one=review.save_action(e['source_partition'],e['source_image_id'],review.ExpertAction(**base));two=review.save_action(e['source_partition'],e['source_image_id'],review.ExpertAction(**{**base,'action':'reject','review_state':'expert_rejected','uncertainty_reason':'wrong tooth','parent_action_id':one['action_id']}))
 assert one['version']==1 and two['version']==2 and two['parent_action_id']==one['action_id']
 assert one['reviewer_id']=='dentist-1' and one['created_at'] and record['landmark_proposals'][0]==raw
 assert (pilot.STAGE2B/f"{e['source_partition'].lower()}_associations.jsonl").read_bytes()==source

def test_uncertain_has_no_coordinates_and_multiple_apices_supported(tmp_path,monkeypatch):
 with pytest.raises(ValidationError):review.ExpertAction(reviewer_id='d',partition='Training',action='add',finding_type='apex',review_state='uncertain',landmark_state='uncertain',point=[1,2],uncertainty_reason='overlap')
 _,e=first();monkeypatch.setattr(review,'REVIEWS',tmp_path);r=review._association(e['source_partition'],e['source_image_id']);tooth=r['tooth_instances'][0]['tooth_instance_id']
 for point in ([10,10],[12,12]):
  p=review.ExpertAction(reviewer_id='d',partition=e['source_partition'],action='add',finding_type='apex',review_state='expert_corrected',tooth_instance_id=tooth,selected_tooth_instance_ids=[tooth],point=point,landmark_state='visible')
  review.save_action(e['source_partition'],e['source_image_id'],p)
 assert len(review._reviews(e['stable_image_id']))==2

def test_out_of_bounds_missing_teeth_and_reason_validation(tmp_path,monkeypatch):
 _,e=first();monkeypatch.setattr(review,'REVIEWS',tmp_path)
 with pytest.raises(ValidationError):review.ExpertAction(reviewer_id='d',partition=e['source_partition'],source_annotation_id=e['association_ids'][0],action='reject',finding_type='cej',review_state='expert_rejected')
 p=review.ExpertAction(reviewer_id='d',partition=e['source_partition'],action='add',finding_type='cej',review_state='expert_corrected',tooth_instance_id='missing',point=[-1,3])
 with pytest.raises(HTTPException):review.save_action(e['source_partition'],e['source_image_id'],p)
