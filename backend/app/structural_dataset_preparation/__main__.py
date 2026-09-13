"""CLI for non-diagnostic DenPAR structural dataset preparation."""

import argparse
import json
from pathlib import Path

from .preparer import PreparationConfig, prepare_structural_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare read-only DenPAR structural targets; no disease or severity labels.")
    parser.add_argument("--validation-dir", type=Path, default=Path("../DenPAR Radiographs Dataset/Dataset/Validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("../dataset/structural_prepared"))
    parser.add_argument("--confirmed-correspondence", type=Path, help="Optional independent expert confirmation JSON")
    parser.add_argument("--resize-width", type=int)
    parser.add_argument("--resize-height", type=int)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    report = prepare_structural_dataset(PreparationConfig(**vars(args)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
