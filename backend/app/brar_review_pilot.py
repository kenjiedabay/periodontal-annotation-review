"""Create a blinded BRAR label-verification pilot."""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from collections import defaultdict
from pathlib import Path
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
FDI={f"{q}{t}" for q in range(1,5) for t in range(1,9)}
class Point(BaseModel):
 model_config=ConfigDict(extra='forbid'); x:float=Field(ge=0); y:float=Field(ge=0)
class BrarReview(BaseModel):
 model_config=ConfigDict(extra='forbid'); protocol_version:Literal['1.0']='1.0'; pilot_case_id:str; anonymous_reviewer_id:Literal['reviewer_A','reviewer_B']; image_quality:Literal['acceptable','suboptimal','unreadable']; maximum_loss_fdi:str|None=None; maximum_loss_surface:Literal['mesial','distal']|None=None; cej:Point|None=None; alveolar_crest:Point|None=None; apex:Point|None=None; bone_loss_percent:float|None=Field(default=None,ge=0,le=100); reviewer_brar:float|None=Field(default=None,ge=0); reviewer_level:Literal[1,2,3,'not_assessable']; notes:str=Field(default='',max_length=5000); completed_at_utc:str
 @model_validator(mode='after')
 def complete(self):
  if self.image_quality!='unreadable' and self.reviewer_level!='not_assessable':
   if self.maximum_loss_fdi not in FDI: raise ValueError('A valid FDI tooth is required')
   if any(x is None for x in (self.maximum_loss_surface,self.cej,self.alveolar_crest,self.apex,self.bone_loss_percent,self.reviewer_brar)): raise ValueError('A complete maximum-loss measurement is required')
  return self
def create_package(dataset_root:Path,output_dir:Path,cases:int=45,seed:int=20260923)->dict:
 with (dataset_root/'meta_data.csv').open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
 groups=defaultdict(list)
 for row in rows: groups[row['Level']].append(row)
 rng=random.Random(seed); selected=[]; base,extra=divmod(cases,3)
 for offset,level in enumerate(('1','2','3')): selected.extend(rng.sample(groups[level],base+(offset<extra)))
 rng.shuffle(selected); entries=[]
 for index,row in enumerate(selected,1):
  image=next(dataset_root.rglob(row['File name'])); entries.append({'pilot_case_id':f'BRAR-{index:03d}','file_name':row['File name'],'image_reference':image.relative_to(dataset_root.parents[1]).as_posix(),'age':int(row['Age']),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'published_level':int(row['Level']),'published_bone_resorption':float(row['Bone resorption']),'published_brar':float(row['Bone resorption Age'])})
 public=[{k:v for k,v in e.items() if not k.startswith('published_') and k!='image_reference'} for e in entries]; package={'protocol_version':'1.0','dataset':'BRAR','seed':seed,'case_count':cases,'selection':'class-balanced stratified random sample','published_labels_blinded':True,'entries':entries}
 output_dir.mkdir(parents=True,exist_ok=True); (output_dir/'pilot_manifest.private.json').write_text(json.dumps(package,indent=2),encoding='utf-8'); (output_dir/'pilot_manifest.public.json').write_text(json.dumps({**package,'entries':public},indent=2),encoding='utf-8'); (output_dir/'brar_review.schema.json').write_text(json.dumps(BrarReview.model_json_schema(),indent=2),encoding='utf-8'); (output_dir/'completed_reviews').mkdir(exist_ok=True); return package
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,default=Path('../BRAR Dataset/BRAR-anchored multimodal dataset')); p.add_argument('--output-dir',type=Path,default=Path('../artifacts/brar-review-pilot-v1')); p.add_argument('--cases',type=int,default=45); p.add_argument('--seed',type=int,default=20260923); a=p.parse_args(); result=create_package(a.dataset_root,a.output_dir,a.cases,a.seed); print(json.dumps({'cases':result['case_count'],'levels':{str(i):sum(e['published_level']==i for e in result['entries']) for i in (1,2,3)}},indent=2))
if __name__=='__main__': main()
