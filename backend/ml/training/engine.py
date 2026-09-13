"""Training, validation, checkpointing, and early stopping."""

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml.config import TrainingConfig
from ml.evaluation.metrics import classification_metrics, write_evaluation


def _run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, optimizer: AdamW | None = None) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    count = len(targets)
    return total_loss / count, classification_metrics(targets, predictions)


def _class_weights(records: list[Any], device: torch.device) -> torch.Tensor:
    counts = [sum(record.label == label for record in records) for label in (0, 1)]
    if not all(counts):
        return torch.ones(2, device=device)
    total = sum(counts)
    return torch.tensor([total / (2 * count) for count in counts], dtype=torch.float32, device=device)


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable if name != "fc.weight" and name != "fc.bias" else True


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, train_records: list[Any], config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    config.validate()
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(train_records, device) if config.use_class_weights else None)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7)
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    model.to(device)
    if config.freeze_backbone_epochs > 0:
        _set_backbone_trainable(model, False)

    for epoch in range(1, config.epochs + 1):
        if epoch == config.freeze_backbone_epochs + 1:
            _set_backbone_trainable(model, True)
        train_loss, train_metrics = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        validation_loss, validation_metrics = _run_epoch(model, validation_loader, loss_fn, device)
        scheduler.step(validation_metrics["f1"])
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
        }
        history.append(row)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config.as_dict(), "validation_metrics": validation_metrics}, checkpoint_dir / "last.pt")
        if validation_metrics["f1"] > best_f1 + config.min_delta:
            best_f1 = validation_metrics["f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config.as_dict(), "validation_metrics": validation_metrics}, checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (checkpoint_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys() if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    return {"best_epoch": best_epoch, "best_validation_f1": best_f1, "epochs_completed": len(history), "history": history}


def evaluate_model(model: nn.Module, loader: DataLoader, checkpoint_path: Path, output_dir: Path, loss_fn: nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, metrics = _run_epoch(model, loader, loss_fn, device)
    metrics["loss"] = test_loss
    write_evaluation(metrics, output_dir)
    return metrics
