"""Audit BRAR and create a deterministic, leakage-aware split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REQUIRED = {
    "File name", "Age", "Gender", "Bone resorption", "Bone resorption Age",
    "Level", "Number of missing teeth", "Implant", "Residual root",
    "Functional tooth logarithm",
}
FORBIDDEN_PREDICTORS = {"Bone resorption", "Bone resorption Age", "Level"}
SAFE_METADATA = (
    "Age", "Gender", "Number of missing teeth", "Implant", "Residual root",
    "Functional tooth logarithm",
)


def _allocate(items: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in items:
        groups[row["Level"]].append(row)
    result = {"train": [], "validation": [], "test": []}
    for level in sorted(groups):
        rows = sorted(groups[level], key=lambda row: row["File name"])
        rng.shuffle(rows)
        n = len(rows)
        n_test = round(n * 0.15)
        n_validation = round(n * 0.15)
        result["test"].extend(rows[:n_test])
        result["validation"].extend(rows[n_test:n_test + n_validation])
        result["train"].extend(rows[n_test + n_validation:])
    for rows in result.values():
        rows.sort(key=lambda row: row["File name"])
    return result


def prepare(dataset_root: Path, output_dir: Path, seed: int = 42) -> dict:
    csv_path = dataset_root / "meta_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing BRAR metadata: {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != REQUIRED:
        missing = REQUIRED - (set(rows[0]) if rows else set())
        raise ValueError(f"Unexpected BRAR schema; missing columns: {sorted(missing)}")
    if len({row["File name"] for row in rows}) != len(rows):
        raise ValueError("Duplicate file names in BRAR metadata")

    images = {path.name: path for path in dataset_root.rglob("*.jpg")}
    missing_images = sorted(set(row["File name"] for row in rows) - set(images))
    orphan_images = sorted(set(images) - set(row["File name"] for row in rows))
    if missing_images or orphan_images:
        raise ValueError(f"Image/metadata mismatch: {len(missing_images)} missing, {len(orphan_images)} orphan")

    hashes: dict[str, list[str]] = defaultdict(list)
    dimensions: Counter[str] = Counter()
    for name, path in images.items():
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(name)
        with Image.open(path) as image:
            dimensions[f"{image.width}x{image.height}"] += 1
    duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    if duplicate_groups:
        raise ValueError(f"Exact duplicate images detected: {duplicate_groups[:3]}")

    splits = _allocate(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "brar_split_manifest.csv"
    fields = ["file_name", "image_path", "split", "level", *SAFE_METADATA]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in ("train", "validation", "test"):
            for row in splits[split]:
                writer.writerow({
                    "file_name": row["File name"],
                    "image_path": images[row["File name"]].relative_to(dataset_root).as_posix(),
                    "split": split,
                    "level": int(row["Level"]),
                    **{key: row[key] for key in SAFE_METADATA},
                })

    report = {
        "research_only": True,
        "unit_of_prediction": "patient-level panoramic radiograph",
        "localized_disease_supported": False,
        "treatment_planning_supported": False,
        "records": len(rows),
        "images": len(images),
        "missing_values": {key: sum(not row[key].strip() for row in rows) for key in REQUIRED},
        "class_counts": dict(sorted(Counter(row["Level"] for row in rows).items())),
        "split_class_counts": {
            split: dict(sorted(Counter(row["Level"] for row in split_rows).items()))
            for split, split_rows in splits.items()
        },
        "exact_duplicate_groups": 0,
        "image_dimensions": dict(dimensions.most_common()),
        "seed": seed,
        "forbidden_predictors": sorted(FORBIDDEN_PREDICTORS),
        "allowed_metadata_predictors": list(SAFE_METADATA),
        "warning": "Level is derived from Bone resorption Age; never use either bone-resorption field to predict Level.",
    }
    (output_dir / "brar_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("../BRAR Dataset/BRAR-anchored multimodal dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("../dataset/brar"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()

