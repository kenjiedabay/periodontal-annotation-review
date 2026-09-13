"""Command line entry point for dataset inspection."""

import argparse
from pathlib import Path

from .scanner import ScanConfig, scan_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and non-destructively clean a radiograph dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("../dataset/raw"), help="Directory containing raw images.")
    parser.add_argument("--cleaned-dir", type=Path, default=Path("../dataset/cleaned"), help="Copy destination for valid images.")
    parser.add_argument("--reports-dir", type=Path, default=Path("../dataset/reports"), help="Directory for CSV and JSON reports.")
    parser.add_argument("--perceptual-distance", type=int, default=6, help="Maximum perceptual hash distance for similar duplicates.")
    args = parser.parse_args()

    config = ScanConfig(
        raw_dir=args.raw_dir,
        cleaned_dir=args.cleaned_dir,
        reports_dir=args.reports_dir,
        perceptual_distance=args.perceptual_distance,
    )
    _, summary, report_paths = scan_dataset(config)
    print("Dataset inspection complete")
    for key, value in summary.as_dict().items():
        print(f"{key}: {value}")
    print(f"csv_report: {report_paths[0]}")
    print(f"json_report: {report_paths[1]}")


if __name__ == "__main__":
    main()
