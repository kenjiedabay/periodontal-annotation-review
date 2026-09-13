"""CSV, JSON, and visual report writers."""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .analyzer import ImageRecord

RECORD_FIELDS = [
    "image_id", "filename", "width", "height", "annotation_status", "disease_status",
    "expert_validated", "affected_teeth_count", "affected_teeth", "severity", "findings",
    "completeness_status", "invalid_reason",
]


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_summary_csv(summary: dict[str, Any], output_path: Path) -> None:
    """Write scalar summary metrics as a portable key/value CSV."""
    rows: list[tuple[str, Any]] = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=True, sort_keys=True)
        rows.append((key, value))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def _bar_chart(values: dict[str, int], title: str, path: Path, xlabel: str) -> None:
    labels = list(values.keys()) or ["No data"]
    counts = list(values.values()) or [0]
    width = max(7, min(16, len(labels) * 0.45))
    figure, axis = plt.subplots(figsize=(width, 5))
    axis.bar(labels, counts, color="#4d9687")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Images / annotations")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _completeness_chart(records: list[ImageRecord], path: Path) -> None:
    values = {"Complete": sum(record.completeness_status == "complete" for record in records), "Missing": sum(record.completeness_status == "missing" for record in records), "Invalid": sum(record.completeness_status == "invalid" for record in records), "Incomplete": sum(record.completeness_status == "incomplete" for record in records)}
    _bar_chart(values, "Annotation completeness", path, "Record status")


def write_visual_reports(result: dict[str, Any], output_dir: Path) -> list[Path]:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    records = [ImageRecord(**record) for record in result["records"]]
    outputs = []
    chart_specs = [
        (summary["disease_classification"] if "disease_classification" in summary else summary["class_imbalance"]["disease_classes"], "Disease class distribution", "disease_class_distribution.png", "Disease class"),
        (summary["severity_distribution"], "Severity distribution", "severity_distribution.png", "Severity"),
        (summary["distribution_by_tooth_number"], "Affected tooth distribution", "affected_tooth_distribution.png", "FDI tooth number"),
    ]
    for values, title, filename, xlabel in chart_specs:
        path = charts_dir / filename
        _bar_chart(values, title, path, xlabel)
        outputs.append(path)
    completeness_path = charts_dir / "annotation_completeness.png"
    _completeness_chart(records, completeness_path)
    outputs.append(completeness_path)
    return outputs


def write_reports(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_analysis_records.csv"
    summary_csv_path = output_dir / "dataset_analysis_summary.csv"
    json_path = output_dir / "dataset_analysis_summary.json"
    write_csv(result["records"], csv_path)
    write_summary_csv(result["summary"], summary_csv_path)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    charts = write_visual_reports(result, output_dir)
    return csv_path, json_path, charts
