"""Inference-only fold-0 pilot error analysis using the saved best checkpoint."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

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
PILOT = ROOT / "artifacts/perio-kpt-landmarks/gate2/pilot_fold0"
OUTPUT = PILOT / "error_analysis"
DIAGONAL = float(np.sqrt(2) * 256)


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: None for key in ("count", "mean", "median", "q1", "q3", "iqr", "p90", "p95")}
    q1, median, q3, p90, p95 = np.percentile(array, [25, 50, 75, 90, 95])
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(median),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3-q1),
            "p90": float(p90), "p95": float(p95)}


def metrics(values: list[float]) -> dict:
    result = distribution(values); array = np.asarray(values, dtype=np.float64)
    result.update({"normalized_mean_error": float((array/DIAGONAL).mean()) if array.size else None,
                   "pck_0.02": float((array/DIAGONAL <= .02).mean()) if array.size else None,
                   "pck_0.05": float((array/DIAGONAL <= .05).mean()) if array.size else None,
                   "pck_0.10": float((array/DIAGONAL <= .10).mean()) if array.size else None})
    return result


def render(row: dict, path: Path) -> None:
    pixels = (row["image"][0].numpy()*255).clip(0,255).astype(np.uint8)
    overlay = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    for index, visible in enumerate(row["availability"]):
        if not visible: continue
        target = tuple(int(round(v)) for v in row["target"][index])
        prediction = tuple(int(round(v)) for v in row["prediction"][index])
        cv2.circle(overlay, target, 4, (0,255,0), -1, cv2.LINE_AA)
        cv2.drawMarker(overlay, prediction, (0,0,255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    cv2.putText(overlay, f"mean error {row['mean_error']:.1f}px", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (0,255,255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), overlay)


def evaluate_split(model, dataset, device, split: str) -> tuple[dict, list[dict]]:
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    all_errors, landmarks, classes, visibility_errors = [], defaultdict(list), defaultdict(list), defaultdict(list)
    groups = {"CEJ": (0,3), "bone_level": (1,4,8,9), "root": (2,5,6)}
    failures = {name: Counter() for name in groups}; rows=[]
    rbl_class, rbl_range = defaultdict(list), defaultdict(list); invalid = Counter()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images=batch["image"].to(device); content=batch["valid_content"].to(device)
            with torch.amp.autocast("cuda", enabled=True): logits=model(images)
            predicted, confidence=decode_heatmaps(logits.float().sigmoid(), content)
            for item in range(images.shape[0]):
                class_id=int(batch["class_id"][item]); available=batch["availability"][item].bool().tolist()
                target=batch["points"][item].tolist(); prediction=predicted[item].cpu().tolist()
                valid=batch["valid_content"][item].bool().numpy(); ys,xs=np.where(valid)
                content_bounds=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))
                record_errors=[]; channel_errors={}; record_confusions=0
                for channel, shown in enumerate(available):
                    if not shown: continue
                    error=float(np.linalg.norm(np.asarray(prediction[channel])-np.asarray(target[channel])))
                    all_errors.append(error); record_errors.append(error); landmarks[channel].append(error); classes[class_id].append(error)
                    channel_errors[LANDMARK_NAMES[channel]]=error
                    visibility_errors[int(batch["visibility"][item,channel])].append(error)
                    group=next((name for name,indices in groups.items() if channel in indices),None)
                    if group:
                        bucket="accurate" if error/DIAGONAL<=.02 else "moderate" if error/DIAGONAL<=.10 else "gross"
                        failures[group][bucket]+=1
                        x,y=prediction[channel]; x1,y1,x2,y2=content_bounds
                        failures[group]["prediction_near_content_edge"] += int(min(x-x1,x2-x,y-y1,y2-y)<=4)
                        alternatives=[other for other in groups[group] if other!=channel and available[other]]
                        if alternatives:
                            own=np.linalg.norm(np.asarray(prediction[channel])-np.asarray(target[channel]))
                            alternate=min(np.linalg.norm(np.asarray(prediction[channel])-np.asarray(target[o])) for o in alternatives)
                            confused=int(alternate<own); failures[group]["closer_to_other_same_group_landmark"] += confused
                            record_confusions += confused
                gray=(batch["image"][item,0].numpy()*255).astype(np.uint8)
                row={"record_id":batch["record_id"][item],"class_id":class_id,"image":batch["image"][item],
                     "target":target,"prediction":prediction,"availability":available,
                     "mean_error":float(np.mean(record_errors)),"intensity_std":float(gray.std()),
                     "laplacian_variance":float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                     "channel_errors":channel_errors,"same_family_confusions":record_confusions,
                     "target_near_content_edge":any(shown and min(target[c][0]-content_bounds[0],content_bounds[2]-target[c][0],
                                                    target[c][1]-content_bounds[1],content_bounds[3]-target[c][1])<=4
                                                    for c,shown in enumerate(available))}
                rows.append(row)
                truth_points=[tuple(v) if shown else None for v,shown in zip(target,available)]
                predicted_points=[tuple(v) for v in prediction]
                for surface in ("mesial","distal"):
                    indices=rbl_indices(class_id,surface); truth=calculate_rbl(truth_points,surface,indices=indices)
                    pred=calculate_rbl(predicted_points,surface,indices=indices)
                    if truth["status"]!="assessable": invalid[f"ground_truth:{truth['reason']}"]+=1; continue
                    if pred["status"]!="assessable": invalid[f"prediction:{pred['reason']}"]+=1; continue
                    error=pred["rbl_percent"]-truth["rbl_percent"]
                    rbl_class[(class_id,surface)].append(error)
                    value=truth["rbl_percent"]
                    band="low_<15" if value<15 else "moderate_15_to_33" if value<=33 else "high_>33"
                    rbl_range[(band,surface)].append(error)
    intensity_cut=np.percentile([r["intensity_std"] for r in rows],25)
    blur_cut=np.percentile([r["laplacian_variance"] for r in rows],25)
    association={"low_contrast_bottom_quartile":metrics([r["mean_error"] for r in rows if r["intensity_std"]<=intensity_cut]),
                 "remaining_contrast":metrics([r["mean_error"] for r in rows if r["intensity_std"]>intensity_cut]),
                 "low_sharpness_bottom_quartile":metrics([r["mean_error"] for r in rows if r["laplacian_variance"]<=blur_cut]),
                 "remaining_sharpness":metrics([r["mean_error"] for r in rows if r["laplacian_variance"]>blur_cut]),
                 "target_near_content_edge":metrics([r["mean_error"] for r in rows if r["target_near_content_edge"]]),
                 "target_not_near_content_edge":metrics([r["mean_error"] for r in rows if not r["target_near_content_edge"]]),
                 "visibility_1":metrics(visibility_errors[1]),"visibility_2":metrics(visibility_errors[2])}
    def rbl_summary(values):
        a=np.asarray(values,dtype=float)
        return {"count":int(a.size),"mae":float(np.abs(a).mean()) if a.size else None,
                "rmse":float(np.sqrt(np.square(a).mean())) if a.size else None}
    report={"overall":metrics(all_errors),
            "per_landmark":{LANDMARK_NAMES[i]:metrics(landmarks[i]) for i in range(len(LANDMARK_NAMES))},
            "by_tooth_class":{OBJECT_CLASS_NAMES[i]:metrics(classes[i]) for i in sorted(TOOTH_CLASSES)},
            "failure_categories":{name:dict(counts) for name,counts in failures.items()},
            "associations":association,
            "rbl_by_class":{OBJECT_CLASS_NAMES[c]:{s:rbl_summary(rbl_class[(c,s)]) for s in ("mesial","distal")} for c in sorted(TOOTH_CLASSES)},
            "rbl_by_ground_truth_range":{band:{s:rbl_summary(rbl_range[(band,s)]) for s in ("mesial","distal")}
                                         for band in ("low_<15","moderate_15_to_33","high_>33")},
            "rbl_invalid_reasons":dict(invalid)}
    return report,rows


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir",type=Path,default=PILOT); args=parser.parse_args()
    pilot=args.pilot_dir.resolve(); output=pilot/"error_analysis"
    seed_everything(42); output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8")); fold=manifest["folds"][0]
    train=PerioCropDataset(DEFAULT_MANIFEST,fold["train_image_ids"],256,TOOTH_CLASSES)
    validation=PerioCropDataset(DEFAULT_MANIFEST,fold["validation_image_ids"],256,TOOTH_CLASSES)
    payload=torch.load(pilot/"best.pt",map_location="cpu",weights_only=False)
    model=PerioLandmarkNet(11); model.load_state_dict(payload["model_state_dict"],strict=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    train_report,_=evaluate_split(model,train,device,"train")
    validation_report,rows=evaluate_split(model,validation,device,"validation")
    ordered=sorted(rows,key=lambda row:row["mean_error"]); n=len(ordered)
    chosen={"best":ordered[:3],"typical":ordered[max(0,n//2-1):n//2+2],"worst":ordered[-3:]}
    chosen["rl_d_worst"] = sorted([row for row in rows if "RL-d" in row["channel_errors"]],
                                   key=lambda row:row["channel_errors"]["RL-d"])[-3:]
    chosen["same_family_confusion"] = sorted([row for row in rows if row["same_family_confusions"]],
                                              key=lambda row:(row["same_family_confusions"],row["mean_error"]))[-3:]
    examples={}
    for category,items in chosen.items():
        examples[category]=[]
        for number,row in enumerate(items,1):
            path=output/"examples"/f"{category}_{number}_{row['record_id'].replace(':','_')}.png"
            render(row,path); examples[category].append({"record_id":row["record_id"],"class":OBJECT_CLASS_NAMES[row["class_id"]],
                                                         "mean_error":row["mean_error"],"path":str(path)})
    for row in rows: row.pop("image",None)
    result={"checkpoint":str(pilot/"best.pt"),"checkpoint_epoch":payload.get("epoch",payload.get("completed_epoch")),
            "scope":"inference only; fold-0 tooth crops; no holdout", "train":train_report,
            "validation":validation_report,"examples":examples,
            "rbl_range_definition":"analysis-only bins: low <15%, moderate 15-33%, high >33%; not severity labels"}
    (output/"error_analysis_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
