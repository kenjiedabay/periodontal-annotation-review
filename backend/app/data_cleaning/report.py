"""CSV and JSON report writers for dataset inspection results."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPORT_FIELDS = [
    "image_id",
    "filename",
    "width",
    "height",
    "format",
    "channels",
    "file_size",
    "blur_score",
    "brightness_score",
    "contrast_score",
    "duplicate_status",
    "quality_status",
    "manual_review_required",
    "reason",
]


def write_reports(records: Iterable[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write stable CSV and JSON reports and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    csv_path = output_dir / "dataset_quality_report.csv"
    json_path = output_dir / "dataset_quality_report.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in REPORT_FIELDS} for row in rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "images": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path
