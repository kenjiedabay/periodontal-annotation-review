"""Append-only API for the blinded BRAR verification pilot."""
from __future__ import annotations
import json,os
from pathlib import Path
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from app.brar_review_pilot import BrarReview
ROOT=Path(__file__).resolve().parents[2]; PILOT=ROOT/'artifacts'/'brar-review-pilot-v1'; COMPLETED=PILOT/'completed_reviews'
router=APIRouter(prefix='/api/ground-truth-pilot',tags=['BRAR verification pilot'])
def private():
 path=PILOT/'pilot_manifest.private.json'
 if not path.is_file(): raise HTTPException(503,'BRAR pilot package unavailable')
 return json.loads(path.read_text(encoding='utf-8'))
def entry(case_id):
 value=next((x for x in private()['entries'] if x['pilot_case_id']==case_id),None)
 if not value: raise HTTPException(404,'Pilot case not found')
 return value
@router.get('/manifest')
def manifest():
 path=PILOT/'pilot_manifest.public.json'
 if not path.is_file(): raise HTTPException(503,'BRAR pilot package unavailable')
 return json.loads(path.read_text(encoding='utf-8'))
@router.get('/cases/{case_id}/image')
def image(case_id:str):
 e=entry(case_id); path=(ROOT/e['image_reference']).resolve(); allowed=(ROOT/'BRAR Dataset'/'BRAR-anchored multimodal dataset').resolve()
 if allowed not in path.parents or not path.is_file(): raise HTTPException(404,'Frozen BRAR image unavailable')
 return FileResponse(path,media_type='image/jpeg')
@router.get('/status/{reviewer}')
def status(reviewer:str):
 expected={x['pilot_case_id'] for x in private()['entries']}; folder=COMPLETED/reviewer; done={p.stem for p in folder.glob('*.json')} if folder.is_dir() else set(); return {'reviewer_id':reviewer,'completed':len(expected&done),'total':len(expected),'completed_case_ids':sorted(expected&done)}
@router.post('/reviews',status_code=201)
def submit(review:BrarReview):
 e=entry(review.pilot_case_id)
 if review.reviewer_brar is not None:
  expected=review.bone_loss_percent/e['age']
  if abs(review.reviewer_brar-expected)>.02: raise HTTPException(422,f'BRAR must equal bone-loss percentage divided by age ({expected:.3f})')
  derived=1 if review.reviewer_brar<.25 else 2 if review.reviewer_brar<=1 else 3
  if review.reviewer_level!=derived: raise HTTPException(422,f'Reviewer level must be {derived} for BRAR {review.reviewer_brar:.3f}')
 folder=COMPLETED/review.anonymous_reviewer_id; folder.mkdir(parents=True,exist_ok=True); path=folder/f'{review.pilot_case_id}.json'
 try: descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
 except FileExistsError as error: raise HTTPException(409,'Review already locked') from error
 try: os.write(descriptor,review.model_dump_json(indent=2).encode())
 finally: os.close(descriptor)
 return {'saved':True,'locked':True,'pilot_case_id':review.pilot_case_id}
