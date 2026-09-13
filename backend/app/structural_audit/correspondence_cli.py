"""Generate DenPAR correspondence analysis."""

import argparse
import json
from pathlib import Path

from .correspondence import analyze_correspondence
from .loader import audit_validation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DenPAR structural annotation correspondence without inventing pairings.")
    parser.add_argument("--validation-dir", type=Path, default=Path("../DenPAR Radiographs Dataset/Dataset/Validation"))
    parser.add_argument("--output", type=Path, default=Path("../dataset/reports/denpar_correspondence_analysis.json"))
    args = parser.parse_args()
    report = analyze_correspondence(audit_validation_dataset(args.validation_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"correspondence_report: {args.output}")


if __name__ == "__main__":
    main()
