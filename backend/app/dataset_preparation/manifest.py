"""Manifest serialization for ML-ready dataset records."""

import csv
import json
from pathlib import Path
from typing import Any

MANIFEST_FIELDS = ["image_path", "image_id", "group_id", "label", "affected_tooth", "severity", "annotation_path", "split"]


def write_manifests(records: list[dict[str, Any]], output_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_manifest.csv"
    json_path = output_dir / "dataset_manifest.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps({"config": config, "summary": summary, "records": records}, indent=2), encoding="utf-8")
    return csv_path, json_path
