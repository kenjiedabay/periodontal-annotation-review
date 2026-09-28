"""Balanced, versioned Stage 2C pilot queue; preserves pilot v1."""
from __future__ import annotations
import argparse,json,random
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from typing import Callable
import stage2c_pilot as v1
from stage2b_associations import ROOT,EXPECTED,canonical_hash,load_manifests

OUTPUT=ROOT/'artifacts'/'denpar-stage2c-pilot-v2';SEED=20260922
STAGE2B=v1.STAGE2B
QUOTAS=[('high_confidence_non_clipped',10),('ambiguous_cej_or_apex',10),('multiple_apex_candidates',8),('ambiguous_bone_line',6),('clipped_or_boundary',6)]

def clipped(r):return any(any(t['clipped_edges'].values()) for t in r['tooth_instances'])
def ambiguous_landmark(r):return any(p['ambiguity_status']=='ambiguous' and p['landmark_type'] in ('cej','apex') for p in r['landmark_proposals'])
def multiple_apex(r):
 c=Counter(t for p in r['landmark_proposals'] if p['landmark_type']=='apex' for t in p['candidate_tooth_instance_ids']);return any(n>1 for n in c.values())
def ambiguous_bone(r):return any(p['ambiguity_status']=='ambiguous' for p in r['bone_line_proposals'])
def high_confidence(r):
 p=r['landmark_proposals'];return bool(p) and not clipped(r) and not ambiguous_landmark(r) and sum(x['ambiguity_status']=='provisional' and x['association_confidence']>=.9 for x in p)/len(p)>=.8
RULES:dict[str,Callable]=dict(high_confidence_non_clipped=high_confidence,ambiguous_cej_or_apex=ambiguous_landmark,multiple_apex_candidates=multiple_apex,ambiguous_bone_line=ambiguous_bone,clipped_or_boundary=clipped)

def uncertainty(r):
 items=r['landmark_proposals']+r['bone_line_proposals'];ratio=sum(x['ambiguity_status']!='provisional' for x in items)/max(1,len(items))
 return 'low' if ratio==0 else ('medium' if ratio<=.30 else 'high')
def metadata(r,audit):
 fields=(((audit.get(r['stable_image_id']) or {}).get('workbook_metadata_reference') or {}).get('fields') or {})
 return str(fields.get('Arch') or 'unknown'),str(fields.get('Site') or 'unknown')

def build(output:Path=OUTPUT,seed:int=SEED):
 v1.verify_inputs();manifests=load_manifests();excluded={x['stable_image_id'] for x in manifests['denpar_exclusion_manifest.json']['entries']}
 audit_doc=json.loads((ROOT/'artifacts/denpar-stage2a-audit/audit_results.json').read_text(encoding='utf-8'));audit={x['stable_image_id']:x for x in audit_doc['inventory']}
 records=[]
 for part in ('Training','Validation'):
  for r in v1.load_records(v1.STAGE2B/f'{part.lower()}_associations.jsonl'):
   if r['source_partition']!=part:raise ValueError('Partition mismatch')
   records.append(r)
 rng=random.Random(seed);rng.shuffle(records);selected=[];used=set();diversity=Counter();partition=Counter()
 for primary,quota in QUOTAS:
  candidates=[r for r in records if r['stable_image_id'] not in used and RULES[primary](r)]
  for _ in range(quota):
   if not candidates:break
   def score(r):
    arch,site=metadata(r,audit);unc=uncertainty(r);val_bonus=4 if r['source_partition']=='Validation' and partition['Validation']<10 else 0
    return val_bonus+2/(diversity['arch:'+arch]+1)+2/(diversity['site:'+site]+1)+2/(diversity['uncertainty:'+unc]+1)+1/(diversity['teeth:'+str(len(r['tooth_instances']))]+1)
   best=max(range(len(candidates)),key=lambda i:(score(candidates[i]),-i));r=candidates.pop(best);used.add(r['stable_image_id']);partition[r['source_partition']]+=1
   arch,site=metadata(r,audit);unc=uncertainty(r);reasons=v1.traits(r,audit)
   for x in ('arch:'+arch,'site:'+site,'uncertainty:'+unc,'teeth:'+str(len(r['tooth_instances']))):diversity[x]+=1
   selected.append({'source_partition':r['source_partition'],'source_image_id':r['source_image_id'],'stable_image_id':r['stable_image_id'],'primary_selection_reason':primary,
    'secondary_characteristics':reasons,'uncertainty_stratum':unc,'arch':arch,'site':site,'visible_tooth_count':len(r['tooth_instances']),
    'association_ids':[p.get('source_annotation_id') or p.get('source_polyline_id') for p in r['landmark_proposals']+r['bone_line_proposals']],
    'tooth_instance_ids':[t['tooth_instance_id'] for t in r['tooth_instances']]})
 if len(selected)!=sum(q for _,q in QUOTAS):raise RuntimeError(f'Unmet quota: selected {len(selected)}')
 if len(used)!=len(selected) or any(x in excluded for x in used) or any(x['source_partition']=='Testing' for x in selected):raise RuntimeError('Queue safety failure')
 counts=Counter(x['primary_selection_reason'] for x in selected);unmet={name:quota-counts[name] for name,quota in QUOTAS if counts[name]!=quota}
 payload={'schema_version':'1.0.0','queue_version':'denpar-stage2c-pilot-v2','sampling_seed':seed,'created_at':datetime.now(timezone.utc).isoformat(),
  'primary_group_quotas':dict(QUOTAS),'primary_group_counts':dict(counts),'unmet_quotas':unmet,'allowed_partitions':['Training','Validation'],'prohibited_partition':'Testing',
  'manifest_hashes':{'Training':EXPECTED['denpar_clean_train_manifest.json'],'Validation':EXPECTED['denpar_clean_validation_manifest.json']},'stage2b_artifact_hashes':v1.STAGE2B_HASHES,
  'stage2b_configuration_hash':'cc184a6ecc9899bd1d7184dbe1a63a39a62578bbfef75402c0ff10183b91f196','entries':selected,'overlap_status':'not_created_single_reviewer'}
 payload['content_hash']=canonical_hash(payload);summary={'queue_size':len(selected),'partition_counts':dict(partition),'primary_group_counts':dict(counts),'unmet_quotas':unmet,
  'arch_counts':dict(Counter(x['arch'] for x in selected)),'site_counts':dict(Counter(x['site'] for x in selected)),'uncertainty_counts':dict(Counter(x['uncertainty_stratum'] for x in selected)),
  'visible_tooth_count_distribution':dict(sorted(Counter(str(x['visible_tooth_count']) for x in selected).items())),'acceptance_metrics_calculated':False}
 output.mkdir(parents=True,exist_ok=False);(output/'pilot_queue.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8');(output/'queue_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(output/'README.md').write_text('# Stage 2C pilot v2\n\nBalanced anatomical-association review queue. Pilot v1 is preserved. No acceptance metrics have been calculated.\n',encoding='utf-8');return {'queue':payload,'summary':summary}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['build']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--seed',type=int,default=SEED);a=p.parse_args();print(json.dumps(build(a.output,a.seed)['summary'],indent=2))
