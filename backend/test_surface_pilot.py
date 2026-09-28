import hashlib,json
from pathlib import Path
from fastapi.testclient import TestClient
from main import app
from app import pilot_surface_review as pilot

ROOT=Path(__file__).resolve().parents[1]
def phase_a(expert='expert-01'):
 return {'anonymous_expert_id':expert,'started_at':'2026-09-22T00:00:00+00:00','technical_issues':[],'counters':{},'annotation':{'reviewer_id':expert,'partition':'Training','image_id':'978','orientation':{'status':'unverified'},'teeth':[]}}

def test_manifest_training_only_reproducible_and_duplicate_safe():
 manifest=json.loads((ROOT/'artifacts/surface-verification-pilot-v1/pilot_manifest.json').read_text(encoding='utf-8'))
 assert len(manifest['entries'])==20 and [x['review_order'] for x in manifest['entries']]==list(range(1,21))
 assert {x['source_partition'] for x in manifest['entries']}=={'Training'}
 forbidden={hashlib.sha256(p.read_bytes()).hexdigest() for part in ('Validation','Testing') for p in (ROOT/'DenPAR Radiographs Dataset'/'Dataset'/part/'Images').glob('*.jpg')}
 assert not forbidden.intersection(x['image_sha256'] for x in manifest['entries'])

def test_blind_lock_reveal_resume_and_separate_phase_b(monkeypatch,tmp_path):
 monkeypatch.setattr(pilot,'RECORDS',tmp_path/'records');client=TestClient(app)
 case='SVP-001';expert='expert-01'
 initial=client.get(f'/api/surface-pilot/cases/{case}/{expert}').json();assert not initial['phase_a_locked'] and 'model_snapshot' not in initial
 for action in ('start','pause','resume'):
  assert client.post(f'/api/surface-pilot/cases/{case}/session',json={'anonymous_expert_id':expert,'action':action,'notes':''}).status_code==200
 assert client.post(f'/api/surface-pilot/cases/{case}/{expert}/reveal').status_code==403
 locked=client.post(f'/api/surface-pilot/cases/{case}/phase-a-lock',json=phase_a());assert locked.status_code==201
 assert client.post(f'/api/surface-pilot/cases/{case}/phase-a-lock',json=phase_a()).status_code==409
 model=client.post(f'/api/surface-pilot/cases/{case}/{expert}/reveal');assert model.status_code==200
 assisted=client.post(f'/api/surface-pilot/cases/{case}/phase-b',json={'anonymous_expert_id':expert,'decision':'acceptable','notes':'','started_at':'2026-09-22T00:01:00+00:00'});assert assisted.status_code==201
 state=client.get(f'/api/surface-pilot/cases/{case}/{expert}').json()
 assert state['phase_a']['model_information_visible'] is False and state['model_snapshot']['record_type']=='immutable_model_snapshot' and state['phase_b']['phase_a_immutable'] is True

def test_partition_guards_feedback_and_no_metrics(monkeypatch,tmp_path):
 monkeypatch.setattr(pilot,'RECORDS',tmp_path/'records');client=TestClient(app)
 assert client.get('/api/surface-pilot/status/new-expert').json()=={'selected':20,'reviewed':0,'pending':20,'ungradable':0,'metrics_calculated':False}
 ratings={k:3 for k in ('orientation','fdi','mesial_distal','cej','bone_crest','apex','warnings','rbl','usability','clinical_usefulness')}
 body={'anonymous_expert_id':'expert-01','ratings':ratings,'most_difficult':'','unclear_labels':'','incorrect_warnings':'','improvements':'','decision_support_useful':''}
 monkeypatch.setattr(pilot,'PILOT',tmp_path/'pilot')
 # Feedback target is separate; provide only that folder and restore manifest access through direct entry tests above.
 (pilot.PILOT/'feedback').mkdir(parents=True)
 assert client.post('/api/surface-pilot/feedback',json=body).status_code==201
