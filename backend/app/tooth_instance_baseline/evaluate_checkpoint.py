"""Evaluate a selected experiment checkpoint without retraining."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ToothInstanceDataset, collate, validate_official_split
from .model import create_model
from .train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--training-images-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records, source_report = validate_official_split(Path(args.dataset_root), args.split, Path(args.training_images_dir))
    loader = DataLoader(ToothInstanceDataset(records, 1024), batch_size=1, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = create_model(False, 1024)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    report = evaluate(model.to(device), loader, device, 0.5, 0.5)
    result = {"checkpoint": args.checkpoint, "checkpoint_epoch": checkpoint["epoch"], "split": args.split, "source": {"usable_images": source_report["usable_images"], "total_instances": source_report["total_instances"]}, "aggregate": report["aggregate"], "error_analysis": report["error_analysis"]}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
