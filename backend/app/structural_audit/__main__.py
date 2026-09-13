"""Generate a DenPAR structural annotation audit report."""

import argparse
import json
from pathlib import Path

from .loader import audit_validation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DenPAR structural annotation correspondence without changing source files.")
    parser.add_argument("--validation-dir", type=Path, default=Path("../DenPAR Radiographs Dataset/Dataset/Validation"))
    parser.add_argument("--output", type=Path, default=Path("../dataset/reports/denpar_structural_audit.json"))
    args = parser.parse_args()
    result = audit_validation_dataset(args.validation_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(json.dumps(result.statistics, indent=2))
    print(f"audit_report: {args.output}")
    for issue in result.issues[:20]:
        print(f"issue: {issue.image_id} [{issue.category}] {issue.message} ({issue.source})")
    if len(result.issues) > 20:
        print(f"issue: ... {len(result.issues) - 20} additional issues in report")


if __name__ == "__main__":
    main()
