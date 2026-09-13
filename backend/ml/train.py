"""Train the research-only disease presence baseline.

This command does not diagnose disease and refuses to train without expert-validated
binary labels in a leakage-checked manifest.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.config import TrainingConfig
from ml.datasets.radiograph import RadiographDataset, read_records
from ml.models.classifier import build_resnet18
from ml.training.engine import evaluate_model, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a non-clinical PyTorch baseline for present/absent labels.")
    parser.add_argument("--manifest", type=Path, default=Path("../dataset/manifests/dataset_manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("ml/checkpoints"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    return TrainingConfig(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
    ).validate()


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    if not config.manifest_path.exists():
        raise SystemExit(f"Training stopped: manifest does not exist: {config.manifest_path}")
    records = read_records(config)
    if not records:
        raise SystemExit("Training stopped: no expert-validated binary annotation records are available.")
    split_records = {split: [record for record in records if record.split == split] for split in ("train", "validation", "test")}
    if any(not split_records[split] for split in split_records):
        raise SystemExit(f"Training stopped: required split is empty: {[split for split, values in split_records.items() if not values]}")
    if len({record.label for record in split_records["train"]}) < 2:
        raise SystemExit("Training stopped: training split contains only one disease class.")
    if len({record.label for record in split_records["validation"]}) < 2:
        print("Warning: validation split contains one class; validation F1 may be unstable.")

    train_dataset = RadiographDataset(records, config, "train")
    validation_dataset = RadiographDataset(records, config, "validation")
    test_dataset = RadiographDataset(records, config, "test")
    loaders = {
        "train": DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers),
        "validation": DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
        "test": DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(pretrained=config.pretrained)
    result = train_model(model, loaders["train"], loaders["validation"], train_dataset.records, config, device)
    checkpoint = config.checkpoint_dir / "best.pt"
    if not checkpoint.exists():
        raise SystemExit("Training stopped: best checkpoint was not created.")
    loss_fn = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate_model(model, loaders["test"], checkpoint, config.checkpoint_dir, loss_fn, device)
    (config.checkpoint_dir / "run_summary.json").write_text(json.dumps({"device": str(device), "config": config.as_dict(), "training": result, "test": test_metrics, "research_only": True}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "best_validation_f1": result["best_validation_f1"], "test_metrics": test_metrics, "checkpoint": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
