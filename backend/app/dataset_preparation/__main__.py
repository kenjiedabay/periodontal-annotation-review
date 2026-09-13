"""Command-line entry point for ML dataset preparation."""

import argparse
from pathlib import Path

from .config import PreparationConfig
from .preparer import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible, leakage-aware ML dataset splits.")
    parser.add_argument("--processed-dir", type=Path, default=Path("../dataset/processed"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("../dataset/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("../dataset"), help="Dataset root containing train, validation, test, and manifests.")
    parser.add_argument("--group-map", type=Path, help="Optional JSON mapping image_id to shared patient/case group ID.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentations-per-image", type=int, default=1)
    parser.add_argument("--include-unvalidated", action="store_true", help="Include valid annotations that are not expert validated.")
    args = parser.parse_args()
    config = PreparationConfig(
        processed_dir=args.processed_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        augmentations_per_training_image=args.augmentations_per_image,
        require_expert_validated=not args.include_unvalidated,
        group_map_path=args.group_map,
    )
    _, summary, manifests = prepare_dataset(config)
    print("ML dataset preparation complete")
    for key, value in summary.as_dict().items():
        print(f"{key}: {value}")
    print(f"manifest_csv: {manifests[0]}")
    print(f"manifest_json: {manifests[1]}")
    print("augmentation_policy: training split only; validation and test remain unaugmented")


if __name__ == "__main__":
    main()
