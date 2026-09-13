"""Train the expert-label-gated severity classifier."""

import argparse
import json
from collections import Counter
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml.models.classifier import build_resnet18
from .config import SeverityConfig
from .dataset import SeverityDataset, read_severity_records
from .metrics import multiclass_metrics, write_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_backbone(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable if not (name == "fc.weight" or name == "fc.bias") else True


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, labels: list[str], device: torch.device, optimizer: AdamW | None = None) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, target)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * target.size(0)
        targets.extend(target.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / len(targets), multiclass_metrics(targets, predictions, labels)


def train(config: SeverityConfig) -> None:
    config.validate()
    set_seed(config.seed)
    records, labels, observed_counts = read_severity_records(config)
    print(json.dumps({"observed_severity_distribution": observed_counts, "established_labels_used": labels}, indent=2))
    (config.output_dir / "severity_distribution.json").parent.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "severity_distribution.json").write_text(json.dumps({"observed_counts": observed_counts, "labels": labels, "minimum_samples_per_class": config.minimum_samples_per_class}, indent=2), encoding="utf-8")
    if not records:
        raise SystemExit("Severity training stopped: no expert-validated severity labels are available.")
    if len(labels) < 2:
        raise SystemExit("Severity training stopped: fewer than two expert-established severity classes are available.")
    low_support = {label: count for label, count in observed_counts.items() if count < config.minimum_samples_per_class}
    if low_support:
        raise SystemExit(f"Severity training stopped: insufficient samples per class: {low_support}")
    if any(not any(record.split == split for record in records) for split in ("train", "validation", "test")):
        raise SystemExit("Severity training stopped: train, validation, and test splits must all contain severity records.")
    split_label_counts = {split: Counter(record.severity for record in records if record.split == split) for split in ("train", "validation", "test")}
    if any(len(counts) < 2 for counts in split_label_counts.values()):
        print("Warning: at least one split contains fewer than two severity classes; metrics may be unstable.")

    datasets = {split: SeverityDataset(records, config, split) for split in ("train", "validation", "test")}
    loaders = {split: DataLoader(dataset, batch_size=config.batch_size, shuffle=split == "train", num_workers=config.num_workers) for split, dataset in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(labels), pretrained=config.pretrained).to(device)
    if config.freeze_backbone_epochs > 0:
        freeze_backbone(model, False)
    counts = Counter(record.class_index for record in datasets["train"].records)
    weights = torch.tensor([len(datasets["train"].records) / (len(labels) * counts[index]) for index in range(len(labels))], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        if epoch == config.freeze_backbone_epochs + 1:
            freeze_backbone(model, True)
        train_loss, train_metrics = run_epoch(model, loaders["train"], loss_fn, labels, device, optimizer)
        validation_loss, validation_metrics = run_epoch(model, loaders["validation"], loss_fn, labels, device)
        scheduler.step(validation_metrics["f1_macro"])
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": train_loss, "validation_loss": validation_loss, "train_accuracy": train_metrics["accuracy"], "train_precision_macro": train_metrics["precision_macro"], "train_recall_macro": train_metrics["recall_macro"], "train_f1_macro": train_metrics["f1_macro"], "validation_accuracy": validation_metrics["accuracy"], "validation_precision_macro": validation_metrics["precision_macro"], "validation_recall_macro": validation_metrics["recall_macro"], "validation_f1_macro": validation_metrics["f1_macro"]}
        history.append(row)
        checkpoint = {"epoch": epoch, "labels": labels, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config.as_dict(), "validation_metrics": validation_metrics}
        torch.save(checkpoint, config.output_dir / "severity_last.pt")
        if validation_metrics["f1_macro"] > best_f1 + config.min_delta:
            best_f1 = validation_metrics["f1_macro"]
            stale = 0
            torch.save(checkpoint, config.output_dir / "severity_best.pt")
        else:
            stale += 1
        if stale >= config.patience:
            break
    (config.output_dir / "severity_training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = torch.load(config.output_dir / "severity_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    _, test_metrics = run_epoch(model, loaders["test"], loss_fn, labels, device)
    write_metrics(test_metrics, config.output_dir)
    (config.output_dir / "severity_run_summary.json").write_text(json.dumps({"research_only": True, "labels": labels, "best_validation_f1_macro": best_f1, "test": test_metrics}, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation_f1_macro": best_f1, "test": test_metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train severity assessment using only expert-validated labels.")
    parser.add_argument("--manifest", type=Path, default=Path("../dataset/manifests/dataset_manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/checkpoints/severity"))
    parser.add_argument("--minimum-samples-per-class", type=int, default=5)
    parser.add_argument("--labels", help="Optional comma-separated subset of labels observed in expert-validated annotations.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    config = SeverityConfig(manifest_path=args.manifest, dataset_root=args.dataset_root, output_dir=args.output_dir, minimum_samples_per_class=args.minimum_samples_per_class, allowed_labels=tuple(args.labels.split(",")) if args.labels else (), epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, pretrained=not args.no_pretrained)
    train(config)


if __name__ == "__main__":
    main()
