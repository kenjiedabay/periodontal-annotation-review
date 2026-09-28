"""Blinded, append-only API for the Training-only surface pilot."""
from __future__ import annotations
import hashlib,json,os,threading,uuid
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Literal
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field
from app import surface_verification
from app.surface_verification import VerificationInput,validate_submission

ROOT=Path(__file__).resolve().parents[2]; PILOT=ROOT/'artifacts'/'surface-verification-pilot-v1'; RECORDS=PILOT/'records'; LOCK=threading.RLock()
router=APIRouter(prefix='/api/surface-pilot',tags=['blinded surface pilot'])
def now():return datetime.now(timezone.utc).isoformat()
def load(name):
 p=PILOT/name
 if not p.is_file():raise HTTPException(503,'Frozen pilot package unavailable')
 return json.loads(p.read_text(encoding='utf-8'))
def entry(case_id):
 x=next((x for x in load('pilot_manifest.json')['entries'] if x['pilot_case_id']==case_id),None)
 if not x:raise HTTPException(404,'Pilot case unavailable')
 if x['source_partition']!='Training':raise HTTPException(403,'Pilot is Training-only')
 return x
def folder(case_id,expert):return RECORDS/expert/case_id
def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
def append(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with LOCK:
  try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
  except FileExistsError as error:raise HTTPException(409,'Immutable pilot record already exists') from error
  try:os.write(fd,json.dumps(data,indent=2,ensure_ascii=False).encode())
  finally:os.close(fd)
 return data
def event(case_id,expert,kind,details=None):
 body={'event_id':str(uuid.uuid4()),'case_id':case_id,'anonymous_expert_id':expert,'event_type':kind,'created_at':now(),'details':details or {}}
 return append(folder(case_id,expert)/'events'/f"{body['created_at'].replace(':','-')}_{body['event_id']}.json",body)

class SessionAction(BaseModel):anonymous_expert_id:str=Field(min_length=2,max_length=64);action:Literal['start','pause','resume','complete'];notes:str=''
class PhaseA(BaseModel):anonymous_expert_id:str=Field(min_length=2,max_length=64);annotation:VerificationInput;started_at:str;technical_issues:list[str]=[];counters:dict[str,int]={}
class LockExisting(BaseModel):anonymous_expert_id:str=Field(min_length=2,max_length=64);started_at:str;counters:dict[str,int]={};technical_issues:list[str]=[]
class Assisted(BaseModel):
 model_config=ConfigDict(extra='forbid');anonymous_expert_id:str=Field(min_length=2,max_length=64);decision:Literal['accept','correct','reject','not_assessable','acceptable','partially_acceptable','not_acceptable','ungradable'];notes:str='';assisted_correction:dict[str,Any]|None=None;started_at:str
class Issue(BaseModel):anonymous_expert_id:str=Field(min_length=2,max_length=64);description:str=Field(min_length=1);severity:Literal['minor','blocking']='minor'
class Feedback(BaseModel):
 anonymous_expert_id:str=Field(min_length=2,max_length=64);ratings:dict[str,int];most_difficult:str='';unclear_labels:str='';incorrect_warnings:str='';improvements:str='';decision_support_useful:str=''

@router.get('/config')
def config():return load('pilot_config.json')
@router.get('/manifest')
def manifest():return load('pilot_manifest.json')
@router.get('/status/{expert}')
def status(expert:str):
 entries=load('pilot_manifest.json')['entries'];reviewed=sum((folder(x['pilot_case_id'],expert)/'phase_a.locked.json').is_file() for x in entries)
 ungradable=0
 for x in entries:
  rec=read(folder(x['pilot_case_id'],expert)/'phase_a.locked.json') or {};ungradable+=any(s.get('verification_status')=='ungradable' for t in rec.get('annotation',{}).get('teeth',[]) for s in t.get('surfaces',[]))
 return {'selected':len(entries),'reviewed':reviewed,'pending':len(entries)-reviewed,'ungradable':ungradable,'metrics_calculated':False}
@router.get('/cases/{case_id}/{expert}')
def get_case(case_id:str,expert:str):
 x=entry(case_id);base={'entry':x,'phase_a_locked':(folder(case_id,expert)/'phase_a.locked.json').is_file(),'model_revealed':(folder(case_id,expert)/'model_snapshot.json').is_file(),'draft':read(folder(case_id,expert)/'phase_a.draft.json')}
 if not base['phase_a_locked']:return base
 base['phase_a']=read(folder(case_id,expert)/'phase_a.locked.json');base['phase_b']=read(folder(case_id,expert)/'phase_b.json')
 if base['model_revealed']:base['model_snapshot']=read(folder(case_id,expert)/'model_snapshot.json')
 return base
@router.get('/cases/{case_id}/image')
def image(case_id:str):
 x=entry(case_id); path=(ROOT/x['original_image_reference']).resolve()
 # Older manifests reference the pre-migration DenPAR folder. Use the preserved
 # de-identified cleaned image with the same numeric case ID when that legacy
 # path is absent; never substitute a processed mask/comparison image.
 if not path.is_file():
  fallback=ROOT/'datasets'/'prepared'/'project_dataset'/'raw'/f"{x['source_image_id']}.jpg"
  if not fallback.is_file(): fallback=ROOT/'datasets'/'prepared'/'project_dataset'/'cleaned'/f"{x['source_image_id']}.jpg"
  if fallback.is_file(): path=fallback
 if not path.is_file(): raise HTTPException(404,'Frozen pilot radiograph unavailable')
 return FileResponse(path,media_type='image/jpeg',headers={'Cache-Control':'no-store'})
@router.post('/cases/{case_id}/session')
def session(case_id:str,payload:SessionAction):entry(case_id);return event(case_id,payload.anonymous_expert_id,payload.action,{'notes':payload.notes})
@router.put('/cases/{case_id}/phase-a-draft')
def draft(case_id:str,payload:PhaseA):
 x=entry(case_id)
 if payload.annotation.partition!='Training' or payload.annotation.image_id!=x['source_image_id']:raise HTTPException(422,'Pilot source identity is immutable and Training-only')
 if (folder(case_id,payload.anonymous_expert_id)/'phase_a.locked.json').exists():raise HTTPException(409,'Phase A is locked')
 body={**payload.model_dump(),'record_type':'phase_a_draft','updated_at':now(),'model_information_visible':False}
 path=folder(case_id,payload.anonymous_expert_id)/'phase_a.draft.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(body,indent=2),encoding='utf-8');return body
@router.post('/cases/{case_id}/phase-a-lock',status_code=201)
def lock_phase_a(case_id:str,payload:PhaseA):
 x=entry(case_id)
 if payload.annotation.partition!='Training' or payload.annotation.image_id!=x['source_image_id']:raise HTTPException(422,'Pilot source identity is immutable and Training-only')
 warnings=validate_submission(payload.annotation);completed=now();body={**payload.model_dump(),'record_id':str(uuid.uuid4()),'record_type':'independent_phase_a','locked':True,'locked_at':completed,'phase_seconds':None,'model_information_visible':False,'warnings':warnings,'coordinate_space':'original_image_pixels'}
 return append(folder(case_id,payload.anonymous_expert_id)/'phase_a.locked.json',body)
@router.post('/cases/{case_id}/lock-existing-surface-record',status_code=201)
def lock_existing(case_id:str,payload:LockExisting):
 x=entry(case_id);history=surface_verification._records('Training',x['source_image_id'])
 matching=[r for r in history if r.get('reviewer_id')==payload.anonymous_expert_id]
 if not matching:raise HTTPException(409,'Save an independent surface-verification record with this anonymous expert ID first')
 source=matching[-1];completed=now();body={'record_id':str(uuid.uuid4()),'record_type':'independent_phase_a','anonymous_expert_id':payload.anonymous_expert_id,'started_at':payload.started_at,'locked':True,'locked_at':completed,'source_surface_record_id':source['record_id'],'annotation':source,'counters':payload.counters,'technical_issues':payload.technical_issues,'model_information_visible':False,'coordinate_space':'original_image_pixels'}
 return append(folder(case_id,payload.anonymous_expert_id)/'phase_a.locked.json',body)
@router.post('/cases/{case_id}/{expert}/reveal')
def reveal(case_id:str,expert:str):
 if not (folder(case_id,expert)/'phase_a.locked.json').is_file():raise HTTPException(403,'Lock Phase A before revealing model output')
 if (folder(case_id,expert)/'model_snapshot.json').exists():return read(folder(case_id,expert)/'model_snapshot.json')
 x=entry(case_id);source=next(json.loads(line) for line in (ROOT/'artifacts/denpar-stage2b-associations-v1/training_associations.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line)['source_image_id']==x['source_image_id'])
 body={'record_id':str(uuid.uuid4()),'record_type':'immutable_model_snapshot','created_at':now(),'source_sha256':hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),'model_output':source}
 return append(folder(case_id,expert)/'model_snapshot.json',body)
@router.post('/cases/{case_id}/phase-b',status_code=201)
def phase_b(case_id:str,payload:Assisted):
 if not (folder(case_id,payload.anonymous_expert_id)/'model_snapshot.json').is_file():raise HTTPException(403,'Reveal model output after Phase A first')
 body={**payload.model_dump(),'record_id':str(uuid.uuid4()),'record_type':'assisted_phase_b','created_at':now(),'phase_a_immutable':True,'model_snapshot_immutable':True}
 return append(folder(case_id,payload.anonymous_expert_id)/'phase_b.json',body)
@router.post('/cases/{case_id}/issues',status_code=201)
def issue(case_id:str,payload:Issue):entry(case_id);return event(case_id,payload.anonymous_expert_id,'technical_issue',payload.model_dump(exclude={'anonymous_expert_id'}))
@router.post('/feedback',status_code=201)
def feedback(payload:Feedback):
 required={'orientation','fdi','mesial_distal','cej','bone_crest','apex','warnings','rbl','usability','clinical_usefulness'}
 if set(payload.ratings)!=required or any(v not in range(1,6) for v in payload.ratings.values()):raise HTTPException(422,'All ten ratings must use 1–5')
 body={**payload.model_dump(),'record_id':str(uuid.uuid4()),'record_type':'pilot_feedback','created_at':now()};return append(PILOT/'feedback'/f"{payload.anonymous_expert_id}_{body['record_id']}.json",body)
