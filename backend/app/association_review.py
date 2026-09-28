"""Persistent append-only Stage 2C expert association review API."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
import hashlib, json, os, uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

router=APIRouter(prefix='/api/stage2c',tags=['stage2c'])
legacy_router=APIRouter(prefix='/api/stage2b',tags=['stage2b-compatibility'])
ROOT=Path(__file__).resolve().parents[2]; STAGE2B=ROOT/'artifacts'/'denpar-stage2b-associations-v1'; PILOT=ROOT/'artifacts'/'denpar-stage2c-pilot-v2'; REVIEWS=PILOT/'reviews'; LOCK=Lock()
LABEL='AI-generated anatomical association—awaiting expert review'; NON_COORD={'not_visible','uncertain','not_applicable'}
LEGACY_CORRECTIONS={}
CORRECTIONS=LEGACY_CORRECTIONS
class LegacyCorrection(BaseModel):
 reviewer_id:str;source_annotation_id:str;decision:Literal['accept','reject','reassign','not_visible','uncertain','not_applicable'];selected_tooth_instance_id:str|None=None;uncertainty_reason:str|None=None;raw_proposal:dict
Correction=LegacyCorrection

class ExpertAction(BaseModel):
 reviewer_id:str=Field(min_length=1);partition:Literal['Training','Validation'];source_annotation_id:str|None=None;tooth_instance_id:str|None=None
 action:Literal['accept','reject','reassign','move','add','delete','mark_state','assign_bone_scope'];finding_type:Literal['cej','apex','bone_line']
 review_state:Literal['in_review','expert_confirmed','expert_corrected','expert_rejected','uncertain','needs_second_review']
 selected_tooth_instance_ids:list[str]=[];point:list[float]|None=None;polyline:list[list[float]]|None=None
 landmark_state:Literal['visible','not_visible','uncertain','not_applicable']|None=None;bone_scope:Literal['one_tooth_surface','interproximal_pair','uncertain']|None=None
 uncertainty_reason:str|None=None;expert_note:str='';parent_action_id:str|None=None
 @model_validator(mode='after')
 def validate_action(self):
  if self.review_state in ('expert_rejected','uncertain','needs_second_review') and not (self.uncertainty_reason or '').strip():raise ValueError('A reason is required')
  if self.landmark_state in NON_COORD and (self.point is not None or self.polyline is not None):raise ValueError('Non-coordinate states cannot contain coordinates')
  if self.finding_type in ('cej','apex') and self.polyline is not None:raise ValueError('Landmarks cannot contain polylines')
  if self.finding_type=='bone_line' and self.point is not None:raise ValueError('Bone lines cannot contain a point')
  if self.action!='add' and not self.source_annotation_id:raise ValueError('Correction requires a parent raw association')
  return self

def _queue():
 from stage2c_pilot import canonical_hash
 p=PILOT/'pilot_queue.json'
 if not p.is_file():raise HTTPException(503,'Pilot queue unavailable')
 q=json.loads(p.read_text(encoding='utf-8'))
 if canonical_hash(q)!=q.get('content_hash'):raise HTTPException(503,'Pilot queue hash mismatch')
 return q
def _entry(partition,image_id):
 if partition not in ('Training','Validation'):raise HTTPException(422,'Historical Testing is prohibited')
 x=next((x for x in _queue()['entries'] if x['source_partition']==partition and x['source_image_id']==image_id),None)
 if not x:raise HTTPException(404,'Image is not in the Stage 2C pilot queue')
 return x
def _association(partition,image_id):
 _entry(partition,image_id);p=STAGE2B/f'{partition.lower()}_associations.jsonl'
 for line in p.read_text(encoding='utf-8').splitlines():
  r=json.loads(line)
  if r['source_image_id']==image_id:
   if r['source_partition']!=partition:raise HTTPException(409,'Association partition mismatch')
   return r
 raise HTTPException(404,'Association missing')
def _reviews(stable):
 folder=REVIEWS/stable.replace(':','_')
 return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('*.json'))] if folder.exists() else []

@router.get('/queue')
def get_queue():
 q=_queue();return {**q,'entries':[{**x,'review_count':len(_reviews(x['stable_image_id']))} for x in q['entries']]}
@router.get('/associations/{partition}/{image_id}')
def get_associations(partition:str,image_id:str):
 r=_association(partition,image_id);return {**r,'expert_actions':_reviews(r['stable_image_id'])}
@router.get('/images/{partition}/{image_id}')
def get_image(partition:str,image_id:str):
 r=_association(partition,image_id);return FileResponse((ROOT/r['source_image_path']).resolve(),media_type='image/jpeg')
@router.get('/masks/{partition}/{image_id}/{tooth_instance_id:path}')
def get_mask(partition:str,image_id:str,tooth_instance_id:str):
 r=_association(partition,image_id);t=next((t for t in r['tooth_instances'] if t['tooth_instance_id']==tooth_instance_id),None)
 if not t:raise HTTPException(404,'Tooth instance missing')
 return FileResponse((ROOT/t['reference_mask_path']).resolve(),media_type='image/png')
def _coords(values,width,height):
 if not values or any(len(p)!=2 or not(0<=p[0]<width and 0<=p[1]<height) for p in values):raise HTTPException(422,'Coordinates must be inside the original image')

@router.post('/associations/{partition}/{image_id}/actions',status_code=201)
def save_action(partition:str,image_id:str,payload:ExpertAction):
 if payload.partition!=partition:raise HTTPException(409,'Partition identity cannot be changed')
 r=_association(partition,image_id);raws=r['landmark_proposals']+r['bone_line_proposals'];raw=next((p for p in raws if (p.get('source_annotation_id') or p.get('source_polyline_id'))==payload.source_annotation_id),None)
 if payload.action!='add' and raw is None:raise HTTPException(422,'Parent raw association does not exist')
 teeth={t['tooth_instance_id'] for t in r['tooth_instances']}
 if payload.tooth_instance_id and payload.tooth_instance_id not in teeth:raise HTTPException(422,'Tooth instance does not exist')
 if any(x not in teeth for x in payload.selected_tooth_instance_ids):raise HTTPException(422,'Selected tooth does not exist')
 if payload.bone_scope=='interproximal_pair' and len(payload.selected_tooth_instance_ids)!=2:raise HTTPException(422,'Interproximal pair requires two teeth')
 w,h=r['original_image']['width'],r['original_image']['height']
 if payload.point is not None:_coords([payload.point],w,h)
 if payload.polyline is not None:_coords(payload.polyline,w,h)
 prior=[x for x in _reviews(r['stable_image_id']) if x['reviewer_id']==payload.reviewer_id and x.get('source_annotation_id')==payload.source_annotation_id and x.get('tooth_instance_id')==payload.tooth_instance_id];latest=max(prior,key=lambda x:x['version']) if prior else None
 if payload.parent_action_id not in (None,latest['action_id'] if latest else None):raise HTTPException(409,'Parent must reference latest action')
 now=datetime.now(timezone.utc).isoformat();body={**payload.model_dump(),'action_id':str(uuid.uuid4()),'version':latest['version']+1 if latest else 1,'created_at':now,'stable_image_id':r['stable_image_id'],'parent_action_id':latest['action_id'] if latest else None,'parent_raw_association_id':payload.source_annotation_id,'raw_proposal_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest() if raw else None,'provenance_label':LABEL,'coordinate_space':'original_image_pixels','coordinate_origin':'top_left'}
 folder=REVIEWS/r['stable_image_id'].replace(':','_');folder.mkdir(parents=True,exist_ok=True);target=folder/f"{now.replace(':','-')}_{body['action_id']}.json"
 with LOCK:
  fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
  try:os.write(fd,json.dumps(body,indent=2,ensure_ascii=False).encode())
  finally:os.close(fd)
 return body

@legacy_router.get('/associations/{partition}/{image_id}')
def legacy_get(partition:str,image_id:str):
 if partition not in ('Training','Validation'):raise HTTPException(422,'Only Training and Validation associations are available')
 p=STAGE2B/f'{partition.lower()}_associations.jsonl';r=next((json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if json.loads(x)['source_image_id']==image_id),None)
 if not r:raise HTTPException(404,'Association record not found')
 return {**r,'corrections':LEGACY_CORRECTIONS.get(r['stable_image_id'],[])}

@legacy_router.post('/associations/{partition}/{image_id}/corrections',status_code=201)
def legacy_save(partition:str,image_id:str,payload:LegacyCorrection):
 r=legacy_get(partition,image_id);raw=next((p for p in r['landmark_proposals']+r['bone_line_proposals'] if (p.get('source_annotation_id') or p.get('source_polyline_id'))==payload.source_annotation_id),None)
 if raw is None or raw!=payload.raw_proposal:raise HTTPException(409,'Raw proposal must match immutable stored proposal')
 saved={**payload.model_dump(),'correction_id':str(uuid.uuid4()),'created_at':datetime.now(timezone.utc).isoformat(),'version':len(r['corrections'])+1,'provenance_label':LABEL}
 LEGACY_CORRECTIONS.setdefault(r['stable_image_id'],[]).append(saved);return saved

def correct_association(partition:str,image_id:str,payload:LegacyCorrection):
 """Compatibility entry point retained for Stage 2B tests and callers."""
 r=get_associations(partition,image_id);raw=next((p for p in r['landmark_proposals']+r['bone_line_proposals'] if (p.get('source_annotation_id') or p.get('source_polyline_id'))==payload.source_annotation_id),None)
 if raw is None or raw!=payload.raw_proposal:raise HTTPException(409,'Raw proposal must match immutable stored proposal')
 saved={**payload.model_dump(),'correction_id':str(uuid.uuid4()),'created_at':datetime.now(timezone.utc).isoformat(),'version':len(r.get('corrections',[]))+1,'provenance_label':LABEL}
 CORRECTIONS.setdefault(r['stable_image_id'],[]).append(saved);return saved
