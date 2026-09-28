"""Stage 2B DenPAR crop metadata and provisional geometric associations.

This module never edits source data and deliberately makes no diagnostic or
mesial/distal claims. Coordinates are original-image pixels with top-left origin.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "artifacts" / "denpar-leakage-remediation-v2"
AUDIT_PATH = ROOT / "artifacts" / "denpar-stage2a-audit" / "audit_results.json"
OUT_DIR = ROOT / "artifacts" / "denpar-stage2b-associations-v1"
LABEL = "AI-generated anatomical association—awaiting expert review"
EXPECTED = {
 "denpar_clean_train_manifest.json":"a58b29ed113a079ddd3399b1b091360bb06a7a95a473888bbda34d678814e09f",
 "denpar_clean_validation_manifest.json":"a6268f608613fbaf0fab8a08048cd3098114f7d247e5915b915cd24490d16a2d",
 "denpar_clean_historical_test_manifest.json":"0dd5f0c9f11c575d023ec8dd94f5bed7f5502f23bdca1fa9e61c21bb06934213",
 "denpar_exclusion_manifest.json":"aa23a920ce3f3c32e83ce3ba1cd1023fc60e102d9cdd668f5cc42b444d5d227b",
 "denpar_duplicate_group_manifest.json":"eee1220816eb6e90d6ea75bb327033449c74d470bd93e61d83aa9026f478c4ec",
 "denpar_near_duplicate_review_queue.json":"eaaf24afe86d5f97944c4bb42aecc242c7197ed9b40055a3c12191e894768619",
}
DEFAULT_CONFIG={"schema_version":"1.0.0","margin_fraction_x":0.25,"margin_fraction_y":0.15,
 "minimum_margin_px":16,"point_max_normalized_distance":0.20,"bone_max_normalized_distance":0.35,
 "ambiguity_delta":0.03,"coordinate_space":"original_image_pixels","coordinate_origin":"top_left",
 "anatomical_side_policy":"hypothesis_only","provenance_label":LABEL,"qa_seed":20260921,"qa_images_per_partition":6}

def canonical_hash(value:dict[str,Any])->str:
    clean={k:v for k,v in value.items() if k!='content_hash'}
    return hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def load_manifests(directory:Path=MANIFEST_DIR)->dict[str,dict]:
    result={}
    for name, expected in EXPECTED.items():
        p=directory/name
        if not p.is_file(): raise ValueError(f"Required v2 manifest missing: {name}")
        data=json.loads(p.read_text(encoding='utf-8'))
        if data.get('policy_version')!='denpar-leakage-remediation-v2': raise ValueError(f"Only v2 manifests accepted: {name}")
        actual=canonical_hash(data)
        if data.get('content_hash')!=expected or actual!=expected: raise ValueError(f"Manifest hash mismatch: {name}")
        result[name]=data
    return result

def crop_box(bbox:list[int], width:int,height:int,cfg:dict)->tuple[list[int],dict]:
    x0,y0,x1,y1=bbox; bw=x1-x0; bh=y1-y0
    mx=max(cfg['minimum_margin_px'],round(bw*cfg['margin_fraction_x']))
    my=max(cfg['minimum_margin_px'],round(bh*cfg['margin_fraction_y']))
    raw=[x0-mx,y0-my,x1+mx,y1+my]
    box=[max(0,raw[0]),max(0,raw[1]),min(width,raw[2]),min(height,raw[3])]
    return box,{"left":raw[0]<0,"top":raw[1]<0,"right":raw[2]>width,"bottom":raw[3]>height}

def mask_info(path:Path)->tuple[np.ndarray,list[int],float]:
    mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if mask is None: raise ValueError(f"Unreadable mask: {path}")
    binary=mask>0; ys,xs=np.where(binary)
    if not len(xs): raise ValueError(f"Empty mask: {path}")
    return binary,[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],math.sqrt(float(binary.sum()))

def point_distance(point:list[float], mask:np.ndarray, distance_map:np.ndarray|None=None)->tuple[bool,float]:
    x,y=point; xi,yi=int(round(x)),int(round(y))
    inside=0<=yi<mask.shape[0] and 0<=xi<mask.shape[1] and bool(mask[yi,xi])
    if inside:return True,0.0
    if distance_map is not None and 0<=yi<mask.shape[0] and 0<=xi<mask.shape[1]:
        return False,float(distance_map[yi,xi])
    ys,xs=np.where(mask)
    return False,float(np.sqrt((xs-x)**2+(ys-y)**2).min())

def associate_point(point:list[float], teeth:list[dict], max_norm:float, ambiguity_delta:float)->dict:
    ranked=[]
    for tooth in teeth:
        inside,d=point_distance(point,tooth['_mask'],tooth.get('_distance_map')); ranked.append((0 if inside else 1,d/tooth['_scale'],d,tooth['tooth_instance_id'],inside))
    ranked.sort(); eligible=[r for r in ranked if r[4] or r[1]<=max_norm]
    candidates=[r[3] for r in eligible]
    ambiguous=len(eligible)>1 and abs(eligible[1][1]-eligible[0][1])<=ambiguity_delta
    selected=None if not eligible or ambiguous else eligible[0][3]
    reason='equidistant_multiple_teeth' if ambiguous else ('outside_distance_threshold' if not eligible else None)
    best=ranked[0] if ranked else (1,float('inf'),float('inf'),None,False)
    return {"candidate_tooth_instance_ids":candidates,"selected_tooth_instance_id":selected,
      "association_method":"inside_mask" if selected and best[4] else ("nearest_mask_boundary" if selected else "unassigned"),
      "distance_px":None if not ranked else round(best[2],4),"normalized_distance":None if not ranked else round(best[1],6),
      "association_confidence":0.0 if not selected else round(max(0.0,1-best[1]/max_norm),4) if not best[4] else 1.0,
      "ambiguity_status":"ambiguous" if ambiguous else ("unassigned" if not selected else "provisional"),"uncertainty_reason":reason}

def build_image(item:dict, manifest_hash:str,cfg:dict)->dict:
    width,height=item['width'],item['height']; teeth=[]
    for index,m in enumerate(sorted(item['tooth_masks'],key=lambda v:v['path']),1):
        mask,bbox,scale=mask_info(ROOT/m['path']); crop,clipped=crop_box(bbox,width,height,cfg)
        tid=f"{item['stable_image_id']}:tooth:{index:02d}"
        teeth.append({"tooth_instance_id":tid,"reference_mask_path":m['path'],"mask_sha256":m['sha256'],
          "mask_provenance":"DenPAR expert-provided tooth-wise mask","original_bbox_xyxy":bbox,"expanded_crop_box_xyxy":crop,
          "crop_margin_config":{"fraction_x":cfg['margin_fraction_x'],"fraction_y":cfg['margin_fraction_y'],"minimum_px":cfg['minimum_margin_px']},
          "clipped_edges":clipped,"crop_to_original":{"x_offset":crop[0],"y_offset":crop[1]},
          "original_to_crop":{"x_offset":-crop[0],"y_offset":-crop[1]},"geometric_sides":["image_left","image_right"],
          "anatomical_side_hypothesis":None,"orientation_status":"unverified","_mask":mask,"_scale":scale,
          "_distance_map":cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)})
    kp=json.loads((ROOT/item['cej_apex_annotation']).read_text(encoding='utf-8')) if item.get('cej_apex_annotation') else {}
    points=[]
    for kind,key in [('cej','CEJ_Points'),('apex','Apex_Points')]:
        for i,point in enumerate(kp.get(key,[]),1):
            proposal=associate_point(point,teeth,cfg['point_max_normalized_distance'],cfg['ambiguity_delta'])
            points.append({"source_annotation_id":f"{item['stable_image_id']}:{kind}:{i}","landmark_type":kind,
              "neutral_landmark_id":f"{kind}_candidate_{i}","original_coordinate":[float(point[0]),float(point[1])],
              "state":"visible","geometric_side":"image_left" if point[0]<width/2 else "image_right",
              "anatomical_side_hypothesis":None,"provenance_label":LABEL,**proposal})
    bone_doc=json.loads((ROOT/item['bone_level_annotation']).read_text(encoding='utf-8')) if item.get('bone_level_annotation') else {}
    bones=[]
    for i,line in enumerate(bone_doc.get('Bone_Lines',[]),1):
        per=[]
        for tooth in teeth:
            values=[point_distance(p,tooth['_mask'],tooth['_distance_map'])[1] for p in line]
            per.append((min(values)/tooth['_scale'],min(values),tooth['tooth_instance_id']))
        per.sort(); eligible=[x for x in per if x[0]<=cfg['bone_max_normalized_distance']]
        ambiguous=len(eligible)>2 or (len(eligible)>1 and abs(eligible[1][0]-eligible[0][0])<=cfg['ambiguity_delta'])
        selected=[] if not eligible or ambiguous else [x[2] for x in eligible[:2]]
        bones.append({"source_polyline_id":f"{item['stable_image_id']}:bone:{i}","original_points":line,
          "candidate_tooth_instance_ids":[x[2] for x in eligible],"selected_tooth_instance_ids":selected,
          "candidate_interproximal_pair":selected if len(selected)==2 else None,"geometric_relationship":"near_mask_boundaries",
          "distances":[{"tooth_instance_id":x[2],"distance_px":round(x[1],4),"normalized_distance":round(x[0],6)} for x in per[:3]],
          "confidence":0.0 if not selected else round(max(0,1-per[0][0]/cfg['bone_max_normalized_distance']),4),
          "ambiguity_status":"ambiguous" if ambiguous else ("unassigned" if not selected else "provisional"),
          "uncertainty_reason":"multiple_similarly_close_teeth" if ambiguous else ('outside_distance_threshold' if not selected else None),"provenance_label":LABEL})
    for t in teeth:t.pop('_mask');t.pop('_scale');t.pop('_distance_map')
    return {"schema_version":"1.0.0","stage":"2B","source_partition":item['official_partition'],"partition_role":"development",
      "source_image_id":item['image_id'],"stable_image_id":item['stable_image_id'],"source_image_path":item['source_path'],
      "source_image_sha256":item['file_sha256'],"original_image":{"width":width,"height":height,"coordinate_space":"original_image_pixels","origin":"top_left"},
      "manifest":{"policy_version":"denpar-leakage-remediation-v2","content_hash":manifest_hash},"workbook_fdi_values":(item.get('workbook_metadata_reference') or {}).get('fields',{}).get('FDI notation of fully/partially visible teeth'),
      "orientation_status":"insufficient_evidence","ordered_fdi_mapping_hypothesis":None,"tooth_instances":teeth,"landmark_proposals":points,"bone_line_proposals":bones,"provenance_label":LABEL}

def summarize(records:list[dict])->dict:
    out={"images":len(records),"tooth_instances":sum(len(r['tooth_instances']) for r in records),"clipped_crops":sum(any(t['clipped_edges'].values()) for r in records for t in r['tooth_instances'])}
    for kind in ('cej','apex'):
        vals=[p for r in records for p in r['landmark_proposals'] if p['landmark_type']==kind]
        out[kind]={s:sum(p['ambiguity_status']==s for p in vals) for s in ('provisional','ambiguous','unassigned')}
        out[kind]['total']=len(vals)
        distances=np.array([p['distance_px'] for p in vals if p['distance_px'] is not None],dtype=float)
        out[kind]['distance_px_distribution']={"minimum":round(float(distances.min()),4),"median":round(float(np.median(distances)),4),"p95":round(float(np.percentile(distances,95)),4),"maximum":round(float(distances.max()),4)} if len(distances) else {}
    vals=[b for r in records for b in r['bone_line_proposals']]
    out['bone_lines']={s:sum(b['ambiguity_status']==s for b in vals) for s in ('provisional','ambiguous','unassigned')};out['bone_lines']['total']=len(vals)
    counts={kind:Counter() for kind in ('cej','apex')}
    for r in records:
      for t in r['tooth_instances']:
       for kind in counts:
        n=sum(t['tooth_instance_id'] in p['candidate_tooth_instance_ids'] for p in r['landmark_proposals'] if p['landmark_type']==kind);counts[kind][str(n)]+=1
    out['candidate_counts_per_tooth']={k:dict(v) for k,v in counts.items()}
    out['uncertainty_reasons']=dict(Counter(x['uncertainty_reason'] for r in records for x in r['landmark_proposals']+r['bone_line_proposals'] if x['uncertainty_reason']))
    return out

def create_qa_panels(records:list[dict],output:Path,count:int)->list[str]:
    """Render a deterministic difficult-case sample; derived PNGs only."""
    qa=output/'qa';qa.mkdir(exist_ok=True)
    ranked=sorted(records,key=lambda r:(-sum(x['ambiguity_status']!='provisional' for x in r['landmark_proposals']+r['bone_line_proposals']),-sum(any(t['clipped_edges'].values()) for t in r['tooth_instances']),r['stable_image_id']))[:count]
    made=[]
    for r in ranked:
        image=cv2.imread(str(ROOT/r['source_image_path']))
        if image is None: continue
        for idx,t in enumerate(r['tooth_instances'],1):
            x0,y0,x1,y1=t['expanded_crop_box_xyxy'];cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,255),2);cv2.putText(image,str(idx),(x0,max(15,y0+15)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        centers={t['tooth_instance_id']:((t['original_bbox_xyxy'][0]+t['original_bbox_xyxy'][2])//2,(t['original_bbox_xyxy'][1]+t['original_bbox_xyxy'][3])//2) for t in r['tooth_instances']}
        for p in r['landmark_proposals']:
            xy=tuple(round(v) for v in p['original_coordinate']);color=(255,0,255) if p['landmark_type']=='cej' else (0,165,255);cv2.circle(image,xy,5,color,-1)
            if p['selected_tooth_instance_id']:cv2.line(image,xy,centers[p['selected_tooth_instance_id']],color,1)
        for b in r['bone_line_proposals']:
            pts=np.array(b['original_points'],np.int32);cv2.polylines(image,[pts],False,(0,255,0),2)
        name=r['stable_image_id'].replace(':','_')+'.png';cv2.imwrite(str(qa/name),image);made.append(f'qa/{name}')
    return made

def build(output:Path=OUT_DIR,manifest_dir:Path=MANIFEST_DIR,audit_path:Path=AUDIT_PATH,cfg:dict|None=None)->dict:
    cfg={**DEFAULT_CONFIG,**(cfg or {})}; manifests=load_manifests(manifest_dir)
    excluded={x['stable_image_id'] for x in manifests['denpar_exclusion_manifest.json']['entries']}
    audit=json.loads(audit_path.read_text(encoding='utf-8')); inventory={x['stable_image_id']:x for x in audit['inventory']}
    output.mkdir(parents=True,exist_ok=True); all_stats={}
    for filename,part in [('denpar_clean_train_manifest.json','Training'),('denpar_clean_validation_manifest.json','Validation')]:
        manifest=manifests[filename]; records=[]
        for entry in manifest['entries']:
            if entry['stable_image_id'] in excluded: raise ValueError(f"Excluded image entered output: {entry['stable_image_id']}")
            item=inventory[entry['stable_image_id']]
            if item['file_sha256']!=entry['file_sha256']: raise ValueError(f"Source hash mismatch: {entry['stable_image_id']}")
            records.append(build_image(item,manifest['content_hash'],cfg))
        (output/f"{part.lower()}_associations.jsonl").write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        all_stats[part]=summarize(records)
        all_stats[part]['qa_panels']=create_qa_panels(records,output,cfg['qa_images_per_partition'])
    config={**cfg,"frozen_at":datetime.now(timezone.utc).isoformat(),"manifest_hashes":EXPECTED}
    config['content_hash']=canonical_hash(config);(output/'frozen_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    report={"schema_version":"1.0.0","stage":"2B","generated_at":datetime.now(timezone.utc).isoformat(),"configuration_hash":config['content_hash'],"statistics":all_stats,"excluded_images_in_output":[],"historical_test_processed":False,"source_files_modified":False}
    report['content_hash']=canonical_hash(report);(output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'README.md').write_text(f"# DenPAR Stage 2B\n\n{LABEL}\n\nMetadata only; no diagnosis, severity, pattern, or bone-loss percentage. Historical Testing was not processed.\n\n```json\n{json.dumps(all_stats,indent=2)}\n```\n",encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['build']);parser.add_argument('--output',type=Path,default=OUT_DIR);args=parser.parse_args();print(json.dumps(build(args.output),indent=2))
