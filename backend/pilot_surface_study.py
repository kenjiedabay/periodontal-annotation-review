"""Build the frozen, Training-only supervised surface-verification pilot."""
from __future__ import annotations
import hashlib,json,random,subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; SEED=7312026; COUNT=20
OUTPUT=ROOT/'artifacts'/'surface-verification-pilot-v1'; ASSOCIATIONS=ROOT/'artifacts'/'denpar-stage2b-associations-v1'/'training_associations.jsonl'
DATASET=ROOT/'DenPAR Radiographs Dataset'/'Dataset'; AUDIT=ROOT/'artifacts'/'denpar-stage2a-audit'/'audit_results.json'
SOURCE_HASH='1a8653923669c01b9156277cb65574dfba11600061bfa6ab6a0bf7b9181fcb26'

def sha(path:Path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def git_commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def trait(record):
 fdi=str(record.get('workbook_fdi_values') or '');positions={x.strip()[1:] for x in fdi.split(',') if len(x.strip())==2 and x.strip().isdigit()}
 clipped=any(any(t['clipped_edges'].values()) for t in record['tooth_instances']);ambiguous=any(x['ambiguity_status']!='provisional' for x in record['landmark_proposals']);multi=any(sum(t in p['candidate_tooth_instance_ids'] for p in record['landmark_proposals'] if p.get('landmark_type')=='apex')>1 for t in [x['tooth_instance_id'] for x in record['tooth_instances']])
 return {'partial_boundary':clipped,'unclear_landmark_candidate':ambiguous,'multiple_apex_candidate':multi,'central_incisor_candidate':bool({'1'}&positions),'posterior_candidate':bool({'4','5','6','7','8'}&positions),'potentially_ungradable':clipped and ambiguous}
def build(output:Path=OUTPUT):
 if output.exists():raise FileExistsError(f'Pilot output already exists: {output}')
 records=[json.loads(x) for x in ASSOCIATIONS.read_text(encoding='utf-8').splitlines()]
 audit={x['stable_image_id']:x for x in json.loads(AUDIT.read_text(encoding='utf-8'))['inventory']}
 forbidden={sha(p) for part in ('Validation','Testing') for p in (DATASET/part/'Images').glob('*.jpg')}
 candidates=[]
 for r in records:
  if r['source_partition']!='Training':raise ValueError('Non-Training Stage 2B record')
  image=ROOT/r['source_image_path'];digest=sha(image)
  if digest in forbidden:continue
  fields=(((audit.get(r['stable_image_id']) or {}).get('workbook_metadata_reference') or {}).get('fields') or {})
  candidates.append((r,fields,digest,trait(r)))
 rng=random.Random(SEED);rng.shuffle(candidates);selected=[];coverage=Counter()
 for order in range(1,COUNT+1):
  def score(item):
   r,f,_,t=item;keys=[f"arch:{f.get('Arch','unknown')}",f"site:{f.get('Site','unknown')}"]+[k for k,v in t.items() if v]
   return sum(2/(coverage[k]+1) for k in keys)+rng.random()/1000
  choice=max(candidates,key=score);candidates.remove(choice);r,fields,digest,traits=choice
  keys=[f"arch:{fields.get('Arch','unknown')}",f"site:{fields.get('Site','unknown')}"]+[k for k,v in traits.items() if v]
  coverage.update(keys);primary=min(keys,key=lambda x:coverage[x])
  selected.append({'pilot_case_id':f'SVP-{order:03d}','stable_image_id':r['stable_image_id'],'source_partition':'Training','source_image_id':r['source_image_id'],'original_image_reference':f"DenPAR Radiographs Dataset/Dataset/Training/Images/{r['source_image_id']}.jpg",'image_sha256':digest,'arch':fields.get('Arch','unknown'),'site':fields.get('Site','unknown'),'available_fdi_candidates':r.get('workbook_fdi_values'),'selection_stratum':primary,'inclusion_reason':[k for k,v in traits.items() if v] or ['anatomical_diversity'],'review_order':order})
 payload={'schema_version':'1.0.0','pilot_version':'surface-verification-pilot-v1','selection_seed':SEED,'selection_procedure':'deterministic greedy metadata/structural diversity; no model-correctness or performance filtering','created_at':datetime.now(timezone.utc).isoformat(),'source_partition':'Training','prohibited_partitions':['Validation','Testing'],'case_count':len(selected),'clinical_label_coverage':{'severity':'unavailable—not used for selection','bone_loss_pattern':'unavailable—not used for selection'},'entries':selected}
 payload['content_hash']=hashlib.sha256(canonical(payload)).hexdigest();output.mkdir(parents=True)
 (output/'pilot_manifest.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
 summary={'selected':len(selected),'reviewed':0,'pending':len(selected),'ungradable':0,'metrics_calculated':False,'coverage':dict(coverage)}
 (output/'pilot_status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 return payload
if __name__=='__main__':print(json.dumps(build(),indent=2))
