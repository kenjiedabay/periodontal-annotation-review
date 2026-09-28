"""Train BRAR image-only or image-plus-safe-metadata severity baselines."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from ml.severity.metrics import multiclass_metrics

META = ("Age", "Gender", "Number of missing teeth", "Implant", "Residual root", "Functional tooth logarithm")


class BrarDataset(Dataset):
    def __init__(self, rows, root: Path, training: bool, means=None, stds=None):
        self.rows, self.root = rows, root
        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.RandomResizedCrop(224, scale=(.9, 1.0)) if training else transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize([.485] * 3, [.229] * 3),
        ])
        values = np.asarray([[float(row[key]) for key in META] for row in rows], dtype=np.float32)
        self.means = values.mean(0) if means is None else means
        self.stds = values.std(0) if stds is None else stds
        self.stds[self.stds == 0] = 1

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as image:
            pixels = self.transform(image.convert("L"))
        meta = (np.asarray([float(row[key]) for key in META], dtype=np.float32) - self.means) / self.stds
        return pixels, torch.from_numpy(meta), torch.tensor(int(row["level"]) - 1)


class Model(nn.Module):
    def __init__(self, multimodal: bool, pretrained: bool):
        super().__init__()
        self.multimodal = multimodal
        self.image = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        width = self.image.fc.in_features
        self.image.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(width + (len(META) if multimodal else 0), 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 3))

    def forward(self, image, metadata):
        features = self.image(image)
        if self.multimodal: features = torch.cat((features, metadata), 1)
        return self.head(features)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None); total = 0.; targets, predictions = [], []
    for image, meta, target in loader:
        image, meta, target = image.to(device), meta.to(device), target.to(device)
        if optimizer: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(image, meta); loss = loss_fn(logits, target)
            if optimizer: loss.backward(); optimizer.step()
        total += loss.item() * len(target); targets += target.cpu().tolist(); predictions += logits.argmax(1).cpu().tolist()
    metrics = multiclass_metrics(targets, predictions, ["level_1", "level_2", "level_3"])
    metrics["ordinal_mae"] = sum(abs(a - b) for a, b in zip(targets, predictions)) / len(targets)
    return total / len(targets), metrics


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with args.manifest.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    train_data = BrarDataset(by_split["train"], args.dataset_root, True)
    datasets = {"train": train_data, **{split: BrarDataset(by_split[split], args.dataset_root, False, train_data.means, train_data.stds) for split in ("validation", "test")}}
    loaders = {key: DataLoader(value, batch_size=args.batch_size, shuffle=key == "train", num_workers=0) for key, value in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args.mode == "multimodal", not args.no_pretrained).to(device)
    counts = np.bincount([int(row["level"]) - 1 for row in by_split["train"]], minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(by_split["train"]) / (3 * counts), dtype=torch.float32, device=device))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True); best, history = -1., []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_loss, val_metrics = run_epoch(model, loaders["validation"], loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss, "train": train_metrics, "validation": val_metrics})
        if val_metrics["f1_macro"] > best:
            best = val_metrics["f1_macro"]
            torch.save({"model": model.state_dict(), "mode": args.mode, "metadata": META, "means": train_data.means, "stds": train_data.stds}, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    _, test_metrics = run_epoch(model, loaders["test"], loss_fn, device)
    summary = {"research_only": True, "mode": args.mode, "best_validation_f1_macro": best, "test": test_metrics, "warning": "Patient-level BRAR grading only; not localized diagnosis or treatment planning."}
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../dataset/brar/brar_split_manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("../BRAR Dataset/BRAR-anchored multimodal dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("image", "multimodal"), default="image")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__": main()

