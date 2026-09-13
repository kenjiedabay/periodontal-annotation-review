"""Command line entry point for preprocessing cleaned radiographs."""

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import process_dataset

DEFAULT_CONFIG = Path(__file__).with_name("default_config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess cleaned radiographs without changing the input files.")
    parser.add_argument("--input-dir", type=Path, default=Path("../dataset/cleaned"), help="Directory containing cleaned images.")
    parser.add_argument("--output-dir", type=Path, default=Path("../dataset/processed"), help="Directory for processed images and metadata.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON preprocessing configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    _, summary = process_dataset(args.input_dir, args.output_dir, config)
    print("Image preprocessing complete")
    for key, value in summary.as_dict().items():
        print(f"{key}: {value}")
    print(f"output_dir: {args.output_dir}")
    print(f"config: {args.config}")


if __name__ == "__main__":
    main()
