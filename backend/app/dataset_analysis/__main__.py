"""Command-line entry point for dataset analysis."""

import argparse
from pathlib import Path

from .analyzer import AnalysisConfig, analyze_dataset
from .reports import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze processed periodontal radiographs and expert annotations without modifying labels.")
    parser.add_argument("--processed-dir", type=Path, default=Path("../dataset/processed"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("../dataset/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("../dataset/analysis_reports"))
    parser.add_argument("--minimum-class-samples", type=int, default=10)
    parser.add_argument("--imbalance-ratio-threshold", type=float, default=3.0)
    args = parser.parse_args()
    config = AnalysisConfig(args.processed_dir, args.annotations_dir, args.output_dir, args.minimum_class_samples, args.imbalance_ratio_threshold)
    result = analyze_dataset(config)
    csv_path, json_path, charts = write_reports(result, args.output_dir)
    summary = result["summary"]
    print("Dataset analysis complete")
    for key in ("total_images", "valid_annotations", "expert_validated_annotations", "missing_annotations", "invalid_annotations", "incomplete_annotations"):
        print(f"{key}: {summary[key]}")
    print(f"records_csv_report: {csv_path}")
    print(f"summary_csv_report: {args.output_dir / 'dataset_analysis_summary.csv'}")
    print(f"json_report: {json_path}")
    print(f"charts: {len(charts)}")
    for warning in summary["potential_problems"]:
        print(f"warning: {warning}")
    print(f"feasibility: {summary['feasibility']['recommendation']}")


if __name__ == "__main__":
    main()
