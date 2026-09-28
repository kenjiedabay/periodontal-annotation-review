"""Build the reproducible Stage 2C expert association review queue."""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage2b_associations import ROOT, EXPECTED, canonical_hash, load_manifests

STAGE2B=ROOT/'artifacts'/'denpar-stage2b-associations-v1'
OUTPUT=ROOT/'artifacts'/'denpar-stage2c-pilot-v1'
CONFIG={"queue_version":"denpar-stage2c-pilot-v1","sampling_seed":20260922,"queue_size":40,
 "partition_targets":{"Training":30,"Validation":10},"second_reviewer_available":False,"overlap_fraction":0.20,
 "allowed_partitions":["Training","Validation"],"prohibited_partition":"Testing",
 "selection_strata":["high_confidence","ambiguous_cej","ambiguous_apex","multiple_apex_candidates","ambiguous_bone_line","boundary_tooth","clipped_crop","single_apex_candidate","multiple_apex_candidate","arch","site","visible_tooth_count"]}
STAGE2B_HASHES={
 "training_associations.jsonl":"4596935c464ae46d8ef5b5e40d0230833758741a45befd00987fae58b1afb315",
 "validation_associations.jsonl":"2dd6adddefbe28ce8382e14120b88e5bb1837c371c8d7af17ede82d85c974828",
 "frozen_config.json":"71184c3f712ada220093e1bd4932b2404a6205dc79b3afdf4ef4f9f33e24eae5",
 "summary.json":"113349f2551dbdcc8f4d3886d612748fae52c4e83354d8dbeada856e34cb3ecd"}

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_inputs(stage2b:Path=STAGE2B,manifest_dir:Path|None=None)->dict:
    manifests=load_manifests(manifest_dir or ROOT/'artifacts'/'denpar-leakage-remediation-v2')
    for name,expected in STAGE2B_HASHES.items():
        if sha(stage2b/name)!=expected:raise ValueError(f'Stage 2B artifact hash mismatch: {name}')
    cfg=json.loads((stage2b/'frozen_config.json').read_text(encoding='utf-8'))
    if cfg.get('content_hash')!='cc184a6ecc9899bd1d7184dbe1a63a39a62578bbfef75402c0ff10183b91f196' or canonical_hash(cfg)!=cfg['content_hash']:
        raise ValueError('Stage 2B configuration hash mismatch')
    return manifests

def load_records(path:Path)->list[dict]:return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]

def traits(record:dict,metadata:dict)->list[str]:
    ps=record['landmark_proposals'];bs=record['bone_line_proposals'];ts=record['tooth_instances']; result=[]
    if ps and all(p['ambiguity_status']=='provisional' and p['association_confidence']>=.9 for p in ps):result.append('high_confidence')
    if any(p['landmark_type']=='cej' and p['ambiguity_status']=='ambiguous' for p in ps):result.append('ambiguous_cej')
    if any(p['landmark_type']=='apex' and p['ambiguity_status']=='ambiguous' for p in ps):result.append('ambiguous_apex')
    apex=Counter(x for p in ps if p['landmark_type']=='apex' for x in p['candidate_tooth_instance_ids'])
    result.append('multiple_apex_candidate' if any(v>1 for v in apex.values()) else 'single_apex_candidate')
    if any(b['ambiguity_status']=='ambiguous' for b in bs):result.append('ambiguous_bone_line')
    if any(any(t['clipped_edges'].values()) for t in ts):result+=['boundary_tooth','clipped_crop']
    fields=((metadata.get(record['stable_image_id'],{}).get('workbook_metadata_reference') or {}).get('fields') or {})
    if fields.get('Arch'):result.append('arch:'+str(fields['Arch']))
    if fields.get('Site'):result.append('site:'+str(fields['Site']))
    result.append('visible_tooth_count:'+str(len(ts)))
    return sorted(set(result))

def select(records:list[dict],target:int,seed:int,metadata:dict)->list[dict]:
    rng=random.Random(seed); pool=list(records);rng.shuffle(pool);chosen=[];covered=Counter()
    core=['high_confidence','ambiguous_cej','ambiguous_apex','multiple_apex_candidate','ambiguous_bone_line','boundary_tooth','clipped_crop','single_apex_candidate']
    while pool and len(chosen)<target:
        def score(r):
            tt=traits(r,metadata); rare=sum(5 if covered[x]==0 else 1/(covered[x]+1) for x in tt if x in core); diversity=sum(1/(covered[x]+1) for x in tt if x.startswith(('arch:','site:','visible_tooth_count:')));return rare+diversity
        best=max(range(len(pool)),key=lambda i:(score(pool[i]),-i));rec=pool.pop(best);tt=traits(rec,metadata)
        chosen.append({"source_partition":rec['source_partition'],"source_image_id":rec['source_image_id'],"stable_image_id":rec['stable_image_id'],
          "association_ids":[p.get('source_annotation_id') or p.get('source_polyline_id') for p in rec['landmark_proposals']+rec['bone_line_proposals']],
          "tooth_instance_ids":[t['tooth_instance_id'] for t in rec['tooth_instances']],"selection_reasons":tt})
        covered.update(tt)
    return chosen

def build(output:Path=OUTPUT,stage2b:Path=STAGE2B,config:dict|None=None)->dict:
    cfg={**CONFIG,**(config or {})};manifests=verify_inputs(stage2b); excluded={x['stable_image_id'] for x in manifests['denpar_exclusion_manifest.json']['entries']}
    audit=json.loads((ROOT/'artifacts'/'denpar-stage2a-audit'/'audit_results.json').read_text(encoding='utf-8'));metadata={x['stable_image_id']:x for x in audit['inventory']}
    queue=[]
    for idx,(part,target) in enumerate(cfg['partition_targets'].items()):
        records=load_records(stage2b/f'{part.lower()}_associations.jsonl')
        if any(r['source_partition']!=part for r in records):raise ValueError('Stage 2B partition identity mismatch')
        queue+=select(records,target,cfg['sampling_seed']+idx,metadata)
    if any(q['stable_image_id'] in excluded for q in queue):raise ValueError('Excluded image entered pilot queue')
    if any(q['source_partition']=='Testing' for q in queue):raise ValueError('Historical Testing cannot enter pilot')
    overlap_count=max(8,round(len(queue)*cfg['overlap_fraction'])) if cfg['second_reviewer_available'] else 0
    payload={"schema_version":"1.0.0",**cfg,"created_at":datetime.now(timezone.utc).isoformat(),"entries":queue,
      "overlap_subset":[q['stable_image_id'] for q in queue[:overlap_count]],"overlap_status":"configured" if overlap_count else "not_created_single_reviewer",
      "manifest_hashes":{"Training":EXPECTED['denpar_clean_train_manifest.json'],"Validation":EXPECTED['denpar_clean_validation_manifest.json']},
      "stage2b_artifact_hashes":STAGE2B_HASHES,"stage2b_configuration_hash":"cc184a6ecc9899bd1d7184dbe1a63a39a62578bbfef75402c0ff10183b91f196"}
    payload['content_hash']=canonical_hash(payload);output.mkdir(parents=True,exist_ok=True);(output/'pilot_queue.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    summary={"queue_size":len(queue),"partition_counts":dict(Counter(x['source_partition'] for x in queue)),"selection_reason_counts":dict(Counter(y for x in queue for y in x['selection_reasons'])),"acceptance_metrics_calculated":False}
    (output/'queue_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'README.md').write_text('# Stage 2C expert review pilot\n\nAnatomical association review only. Acceptance metrics remain unavailable until real expert reviews are saved.\n',encoding='utf-8')
    return {"queue":payload,"summary":summary}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['build']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(json.dumps(build(a.output)['summary'],indent=2))
