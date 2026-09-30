"""No-training paired RBL, sensitivity, consistency, and review-package audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation import decode_heatmaps
from .geometry import calculate_rbl
from .model import PerioLandmarkNet
from .schema import LANDMARK_NAMES, OBJECT_CLASS_NAMES, TOOTH_CLASSES, rbl_indices
from .train import DEFAULT_MANIFEST, seed_everything
from .training_data import PerioCropDataset

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "artifacts/perio-kpt-landmarks/gate2/pilot_fold0/best.pt"
NEW = ROOT / "artifacts/perio-kpt-landmarks/gate2/pilot_fold0_30epoch_resumable_v1/best.pt"
OUTPUT = ROOT / "artifacts/perio-kpt-landmarks/gate2/measurement_readiness"


def load_model(path: Path, device) -> PerioLandmarkNet:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    model=PerioLandmarkNet(11); model.load_state_dict(payload["model_state_dict"],strict=True)
    return model.to(device).eval()


def predict(model, dataset, device) -> list[dict]:
    loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2,pin_memory=device.type=="cuda")
    rows=[]
    with torch.inference_mode():
        for batch in loader:
            images=batch["image"].to(device); content=batch["valid_content"].to(device)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda"): logits=model(images)
            points,confidence=decode_heatmaps(logits.float().sigmoid(),content)
            for item in range(images.shape[0]):
                rows.append({"record_id":batch["record_id"][item],"class_id":int(batch["class_id"][item]),
                             "target":batch["points"][item].tolist(),
                             "availability":batch["availability"][item].bool().tolist(),
                             "prediction":points[item].cpu().tolist(),"confidence":confidence[item].cpu().tolist()})
    return rows


def points(row: dict, predicted: bool) -> list[tuple[float,float]|None]:
    values=row["prediction"] if predicted else row["target"]
    return [tuple(value) if predicted or available else None for value,available in zip(values,row["availability"])]


def raw_geometry(point_values, indices):
    cej,bone,root=(np.asarray(point_values[i],dtype=float) for i in indices)
    vector=root-cej; length=float(np.linalg.norm(vector))
    if not np.isfinite(length) or length<5:return None
    bone_vector=bone-cej; projection=float(np.dot(bone_vector,vector)/(length**2))
    perpendicular=float(abs(vector[0]*bone_vector[1]-vector[1]*bone_vector[0])/(length**2))
    return {"rbl":100*float(np.linalg.norm(bone_vector))/length,"projection":projection,
            "perpendicular_ratio":perpendicular,"root_length":length}


def summarize(values):
    array=np.asarray(values,dtype=float)
    return {"count":int(array.size),"mean":float(array.mean()) if array.size else None,
            "median":float(np.median(array)) if array.size else None,
            "mae":float(np.abs(array).mean()) if array.size else None,
            "rmse":float(np.sqrt(np.square(array).mean())) if array.size else None}


def paired_analysis(old_rows,new_rows):
    report={}; contributor={}
    for surface in ("mesial","distal"):
        paired=[]; invalid=Counter(); contributions={name:[] for name in ("CEJ","bone_level","root")}
        correlations={name:[[],[]] for name in contributions}; dominant=Counter(); large=0
        for old,new in zip(old_rows,new_rows):
            indices=rbl_indices(old["class_id"],surface); truth_points=points(old,False)
            truth=calculate_rbl(truth_points,surface,indices=indices)
            if truth["status"]!="assessable":continue
            results=[]
            for name,row in (("old",old),("new",new)):
                predicted_points=points(row,True); result=calculate_rbl(predicted_points,surface,indices=indices)
                if result["status"]!="assessable":invalid[f"{name}:{result['reason']}"]+=1
                results.append(result)
            if any(result["status"]!="assessable" for result in results):continue
            old_error=results[0]["rbl_percent"]-truth["rbl_percent"]
            new_error=results[1]["rbl_percent"]-truth["rbl_percent"]
            paired.append({"record_id":old["record_id"],"class_id":old["class_id"],
                           "old_error":old_error,"new_error":new_error})
            for model_name,row,result in (("old",old,results[0]),("new",new,results[1])):
                predicted_points=points(row,True); base=abs(result["rbl_percent"]-truth["rbl_percent"])
                reductions={}
                for role,index in zip(("CEJ","bone_level","root"),indices):
                    component=float(np.linalg.norm(np.asarray(predicted_points[index])-np.asarray(truth_points[index])))
                    correlations[role][0].append(component); correlations[role][1].append(base)
                    replaced=list(predicted_points); replaced[index]=truth_points[index]
                    corrected=calculate_rbl(replaced,surface,indices=indices)
                    reduction=base-abs(corrected["rbl_percent"]-truth["rbl_percent"]) if corrected["status"]=="assessable" else 0.0
                    contributions[role].append({"model":model_name,"reduction":reduction})
                    if model_name=="new":reductions[role]=reduction
                if model_name=="new" and base>=25:
                    large+=1; dominant[max(reductions,key=reductions.get)]+=1
        old_errors=[item["old_error"] for item in paired]; new_errors=[item["new_error"] for item in paired]
        report[surface]={"same_gt_valid_and_both_predictions_assessable":len(paired),
                         "old":summarize(old_errors),"new":summarize(new_errors),
                         "new_better_surface_fraction":float(np.mean(np.abs(new_errors)<np.abs(old_errors))) if paired else None,
                         "invalid_prediction_reasons":dict(invalid)}
        contributor[surface]={"new_large_error_count_ge_25pp":large,"new_dominant_ablation":dict(dominant)}
        for role in contributions:
            for model_name in ("old","new"):
                reductions=[x["reduction"] for x in contributions[role] if x["model"]==model_name]
                contributor[surface][f"{model_name}_{role}_mean_error_reduction_if_ground_truth"] = float(np.mean(reductions))
            x,y=correlations[role]
            contributor[surface][f"{role}_pixel_vs_absolute_rbl_error_correlation"] = float(np.corrcoef(x,y)[0,1])
    return report,contributor


def sensitivity(rows):
    distances=(1,2,5,10); angles=np.linspace(0,2*np.pi,8,endpoint=False); collected=defaultdict(list)
    for row in rows:
        target=points(row,False)
        for surface in ("mesial","distal"):
            indices=rbl_indices(row["class_id"],surface); truth=calculate_rbl(target,surface,indices=indices)
            if truth["status"]!="assessable":continue
            for role,index in zip(("CEJ","bone_level","root"),indices):
                for distance in distances:
                    deltas=[]
                    for angle in angles:
                        changed=list(target); origin=np.asarray(target[index]); changed[index]=tuple(origin+distance*np.asarray((np.cos(angle),np.sin(angle))))
                        result=calculate_rbl(changed,surface,indices=indices)
                        if result["status"]=="assessable":deltas.append(abs(result["rbl_percent"]-truth["rbl_percent"]))
                    collected[(surface,role,distance)].extend(deltas)
    return {surface:{role:{str(distance):{"mean_absolute_rbl_change_pp":float(np.mean(collected[(surface,role,distance)])),
                                         "p90_absolute_rbl_change_pp":float(np.percentile(collected[(surface,role,distance)],90))}
                            for distance in distances} for role in ("CEJ","bone_level","root")}
            for surface in ("mesial","distal")}


def consistency(rows,label):
    output={}
    for class_id in sorted(TOOTH_CLASSES):
        class_rows=[row for row in rows if row["class_id"]==class_id]; summary={}
        for rule in ("existing_0_to_1.25","strict_order_0_to_1","strict_order_plus_axis_0.25"):
            accepted=[];rejected=[];violations=Counter();total=0
            for row in class_rows:
                truth_points=points(row,False);predicted=points(row,True)
                for surface in ("mesial","distal"):
                    indices=rbl_indices(class_id,surface);truth=calculate_rbl(truth_points,surface,indices=indices)
                    if truth["status"]!="assessable":continue
                    total+=1;geometry=raw_geometry(predicted,indices)
                    if geometry is None: accept=False;violations["invalid_root_length"]+=1
                    else:
                        projection=geometry["projection"]
                        violations["before_cej"]+=int(projection<0);violations["beyond_root"]+=int(projection>1)
                        violations["beyond_1.25_root"]+=int(projection>1.25)
                        violations["off_axis_over_0.25"]+=int(geometry["perpendicular_ratio"]>.25)
                        accept=(0<=projection<=1.25) if rule=="existing_0_to_1.25" else (0<=projection<=1)
                        if rule.endswith("axis_0.25"):accept=accept and geometry["perpendicular_ratio"]<=.25
                    raw=geometry["rbl"] if geometry else None
                    error=abs(raw-truth["rbl_percent"]) if raw is not None else None
                    (accepted if accept else rejected).append(error)
            summary[rule]={"gt_valid_surfaces":total,"accepted":len(accepted),"coverage":len(accepted)/total if total else None,
                           "accepted_mae":float(np.mean([v for v in accepted if v is not None])) if accepted else None,
                           "rejected_count":len(rejected),
                           "rejected_raw_mae":float(np.mean([v for v in rejected if v is not None])) if any(v is not None for v in rejected) else None,
                           "violation_counts":dict(violations)}
        output[OBJECT_CLASS_NAMES[class_id]]=summary
    return {label:output}


def review_package(dataset,old_rows,new_rows):
    scored=[]
    groups=((0,3),(1,4,8,9),(2,5,6))
    for index,(old,new) in enumerate(zip(old_rows,new_rows)):
        errors=[float(np.linalg.norm(np.asarray(new["prediction"][i])-np.asarray(new["target"][i]))) for i,a in enumerate(new["availability"]) if a]
        channel={LANDMARK_NAMES[i]:float(np.linalg.norm(np.asarray(new["prediction"][i])-np.asarray(new["target"][i])))
                 for i,a in enumerate(new["availability"]) if a}
        confusion=0
        for group in groups:
            for i in group:
                if i>=len(new["availability"]) or not new["availability"][i]:continue
                alternatives=[j for j in group if j!=i and new["availability"][j]]
                if alternatives:
                    own=channel[LANDMARK_NAMES[i]]; alt=min(np.linalg.norm(np.asarray(new["prediction"][i])-np.asarray(new["target"][j])) for j in alternatives)
                    confusion+=int(alt<own)
        scored.append({"index":index,"mean":float(np.mean(errors)),"channel":channel,"confusion":confusion,"class_id":new["class_id"]})
    chosen=[]
    def add(items,reason,count):
        for item in items:
            if item["index"] not in {x["index"] for x in chosen} and len([x for x in chosen if x.get("reason")==reason])<count:
                chosen.append({**item,"reason":reason})
    ordered=sorted(scored,key=lambda x:x["mean"]);add(ordered,"best",2);add(ordered[len(ordered)//2-2:],"typical",2);add(reversed(ordered),"worst",2)
    add(sorted([x for x in scored if "RL-d" in x["channel"]],key=lambda x:x["channel"].get("RL-d",0),reverse=True),"RL-d",2)
    add(sorted([x for x in scored if "RL-c" in x["channel"]],key=lambda x:x["channel"].get("RL-c",0),reverse=True),"RL-c",2)
    add(sorted(scored,key=lambda x:x["confusion"],reverse=True),"same-family-confusion",2)
    rng=random.Random(42);rng.shuffle(chosen);package=OUTPUT/"expert_review_package";images=package/"images";images.mkdir(parents=True,exist_ok=True)
    key=[];review_rows=[]
    for number,item in enumerate(chosen,1):
        case=f"Case{number:03d}";sample=dataset[item["index"]];pixels=(sample["image"][0].numpy()*255).astype(np.uint8)
        overlay=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
        for landmark,available in enumerate(sample["availability"].bool().tolist()):
            if not available:continue
            x,y=(int(round(v)) for v in sample["points"][landmark].tolist());cv2.circle(overlay,(x,y),4,(0,255,0),-1)
            cv2.putText(overlay,LANDMARK_NAMES[landmark],(min(x+4,220),max(y-4,10)),cv2.FONT_HERSHEY_SIMPLEX,.28,(0,255,255),1)
        cv2.imwrite(str(images/f"{case}.png"),overlay)
        record_id=new_rows[item["index"]]["record_id"]
        key.append({"case_id":case,"record_id":record_id,"selection_reason":item["reason"],"class":OBJECT_CLASS_NAMES[item["class_id"]]})
        review_rows.append([case,OBJECT_CLASS_NAMES[item["class_id"]],"","","","",""])
    with (package/"review_form.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["case_id","root_class","landmark_definition_clear","placement_consistent","uncertain_landmarks","suggested_correction","reviewer_notes"]);writer.writerows(review_rows)
    (package/"blinding_key.json").write_text(json.dumps(key,indent=2),encoding="utf-8")
    (package/"README.md").write_text("# Blinded landmark review\n\nReview green ground-truth landmarks only. Case order hides performance category and source identity. Record definition clarity, placement consistency, uncertainty, and suggested corrections in review_form.csv. Do not edit source labels. blinding_key.json is for analysis after review and should not be given to the blinded reviewer.\n",encoding="utf-8")
    return {"case_count":len(chosen),"path":str(package),"blinding_key":str(package/"blinding_key.json")}


def main():
    seed_everything(42);OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"));fold=manifest["folds"][0]
    dataset=PerioCropDataset(DEFAULT_MANIFEST,fold["validation_image_ids"],256,TOOTH_CLASSES)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_rows=predict(load_model(OLD,device),dataset,device);new_rows=predict(load_model(NEW,device),dataset,device)
    paired,contributors=paired_analysis(old_rows,new_rows)
    report={"scope":"no training; identical fold-0 validation tooth crops","paired_rbl":paired,
            "large_error_contributors":contributors,"ground_truth_sensitivity":sensitivity(old_rows),
            "anatomical_consistency":{**consistency(old_rows,"old_epoch10"),**consistency(new_rows,"new_epoch12")},
            "expert_review_package":review_package(dataset,old_rows,new_rows)}
    (OUTPUT/"measurement_readiness_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
