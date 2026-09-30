"""Validate completed blinded review responses and compare them with current labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path

from .geometry import calculate_rbl
from .schema import LANDMARK_INDEX, OBJECT_CLASS_NAMES, TOOTH_CLASSES, rbl_indices
from .train import DEFAULT_MANIFEST
from .training_data import PerioCropDataset

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "artifacts/perio-kpt-landmarks/gate2/measurement_readiness/expert_review_package"
ALLOWED = {"yes", "no", "uncertain", "not_applicable"}
QUESTIONS = ("cej_correct", "bone_level_correct", "root_landmark_appropriate",
             "mesial_distal_association_correct", "geometry_clinically_plausible",
             "surface_assessable")


def label_status(sample: dict, surface: str) -> dict:
    class_id = int(sample["class_id"]); available = sample["availability"].bool().tolist()
    indices = rbl_indices(class_id, surface)
    target = [tuple(value) if shown else None for value, shown in zip(sample["points"].tolist(), available)]
    geometry = calculate_rbl(target, surface, indices=indices)
    return {"cej_correct": available[indices[0]], "bone_level_correct": available[indices[1]],
            "root_landmark_appropriate": available[indices[2]],
            "mesial_distal_association_correct": True,
            "geometry_clinically_plausible": geometry["status"] == "assessable",
            "surface_assessable": geometry["status"] == "assessable"}


def run(response_path: Path, package: Path = PACKAGE) -> dict:
    key = json.loads((package / "analyst_only/blinding_key.json").read_text(encoding="utf-8"))
    mapping = {item["case_id"]: item for item in key}
    rows = list(csv.DictReader(response_path.open(encoding="utf-8-sig", newline="")))
    expected = {(case_id, surface) for case_id in mapping for surface in ("mesial", "distal")}
    observed = {(row["case_id"], row["surface"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("completed form must contain exactly one mesial and one distal row for every blinded case")
    for row in rows:
        for question in QUESTIONS:
            row[question] = row[question].strip().lower()
            if row[question] not in ALLOWED:
                raise ValueError(f"{row['case_id']} {row['surface']} has invalid {question}: {row[question]!r}")
        if any(row[q] in {"no", "uncertain"} for q in QUESTIONS) and not row["reason_codes"].strip():
            raise ValueError(f"{row['case_id']} {row['surface']} requires at least one reason code")
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8")); fold = manifest["folds"][0]
    dataset = PerioCropDataset(DEFAULT_MANIFEST, fold["validation_image_ids"], 256, TOOTH_CLASSES)
    samples = {annotation.record_id: dataset[index] for index, (_, annotation) in enumerate(dataset.records)}
    counts = {question: Counter() for question in QUESTIONS}; correction_types = Counter(); disagreements = []
    for row in rows:
        hidden = mapping[row["case_id"]]; current = label_status(samples[hidden["record_id"]], row["surface"])
        for question in QUESTIONS:
            expert = row[question]; label_accepts = current[question]
            if expert == "uncertain": category = "expert_uncertain"
            elif expert == "not_applicable": category = "expert_not_applicable"
            elif expert == "yes" and label_accepts: category = "agreement_accept"
            elif expert == "no" and not label_accepts: category = "agreement_reject_or_unavailable"
            elif expert == "no" and label_accepts: category = "expert_rejects_existing_label"
            else: category = "expert_accepts_missing_or_invalid_label"
            counts[question][category] += 1
            if category in {"expert_rejects_existing_label", "expert_accepts_missing_or_invalid_label"}:
                correction = {"cej_correct":"CEJ","bone_level_correct":"bone_level",
                              "root_landmark_appropriate":"root_or_apex",
                              "mesial_distal_association_correct":"mesial_distal_association",
                              "geometry_clinically_plausible":"geometry",
                              "surface_assessable":"surface_exclusion"}[question]
                correction_types[correction] += 1
                disagreements.append({"case_id":row["case_id"],"record_id":hidden["record_id"],
                                      "root_class":hidden["class"],"surface":row["surface"],
                                      "question":question,"expert":expert,"current_label_accepts":label_accepts,
                                      "reason_codes":row["reason_codes"],"comments":row["comments"]})
    report = {"reviewed_surface_count":len(rows),
              "agreement_by_question":{question:dict(value) for question,value in counts.items()},
              "disagreement_count":len(disagreements),"correction_or_exclusion_types":dict(correction_types),
              "disagreements":disagreements,
              "interpretation":"Expert decisions are comparisons only; source labels remain unchanged until separately approved."}
    analyst = package / "analyst_only"
    (analyst / "expert_label_comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (analyst / "disagreements.csv").open("w",newline="",encoding="utf-8") as stream:
        fields=("case_id","record_id","root_class","surface","question","expert","current_label_accepts","reason_codes","comments")
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(disagreements)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses",type=Path,help="completed reviewer_bundle/review_form.csv copy")
    parser.add_argument("--package",type=Path,default=PACKAGE)
    args=parser.parse_args()
    print(json.dumps(run(args.responses,args.package),indent=2))


if __name__=="__main__": main()
