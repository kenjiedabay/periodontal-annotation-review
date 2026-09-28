"""Repeated stratified validation and proxy-signal controls for BRAR."""

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
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from ml.severity.metrics import multiclass_metrics
from .train import META


def stratified_folds(labels: np.ndarray, folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    result = [[] for _ in range(folds)]
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label).tolist()
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            result[offset % folds].append(index)
    return [sorted(fold) for fold in result]


def image_embeddings(rows, root: Path, cache: Path, batch_size: int) -> np.ndarray:
    if cache.exists():
        payload = np.load(cache)
        if payload["names"].tolist() == [row["file_name"] for row in rows]:
            return payload["embeddings"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity(); model.to(device).eval()
    transform = transforms.Compose([transforms.Grayscale(3), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485] * 3, [.229] * 3)])
    output = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with Image.open(root / row["image_path"]) as image:
                batch.append(transform(image.convert("L")))
        with torch.no_grad(): output.append(model(torch.stack(batch).to(device)).cpu().numpy())
    embeddings = np.concatenate(output).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, names=np.asarray([row["file_name"] for row in rows]), embeddings=embeddings)
    return embeddings


def fit_predict(train_x, train_y, test_x, seed: int, epochs: int) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(train_x, dtype=torch.float32, device=device)
    y = torch.tensor(train_y, dtype=torch.long, device=device)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 3)).to(device)
    counts = np.bincount(train_y, minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train_y) / (3 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        model.train(); order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 64):
            batch = order[start:start + 64]; optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[batch]), y[batch]); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): return model(torch.tensor(test_x, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()


def evaluate(name, features, labels, folds, seed, epochs):
    results = []
    all_indices = np.arange(len(labels))
    for fold_index, test_indices in enumerate(folds):
        test = np.asarray(test_indices); train = np.setdiff1d(all_indices, test)
        mean, std = features[train].mean(0), features[train].std(0); std[std == 0] = 1
        predictions = fit_predict((features[train] - mean) / std, labels[train], (features[test] - mean) / std, seed + fold_index, epochs)
        metrics = multiclass_metrics(labels[test].tolist(), predictions.tolist(), ["level_1", "level_2", "level_3"])
        metrics["ordinal_mae"] = float(np.abs(labels[test] - predictions).mean())
        results.append(metrics)
    keys = ("accuracy", "f1_macro", "recall_macro", "ordinal_mae")
    summary = {key: {"mean": float(np.mean([r[key] for r in results])), "std": float(np.std([r[key] for r in results], ddof=1))} for key in keys}
    summary["per_class_recall"] = {label: {"mean": float(np.mean([r["per_class"][label]["recall"] for r in results])), "std": float(np.std([r["per_class"][label]["recall"] for r in results], ddof=1))} for label in ("level_1", "level_2", "level_3")}
    return {"model": name, "summary": summary, "folds": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../dataset/brar/brar_split_manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("../BRAR Dataset/BRAR-anchored multimodal dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("../models/brar_validation"))
    parser.add_argument("--folds", type=int, default=5); parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle: rows = sorted(csv.DictReader(handle), key=lambda row: row["file_name"])
    labels = np.asarray([int(row["level"]) - 1 for row in rows]); fold_indices = stratified_folds(labels, args.folds, args.seed)
    metadata = np.asarray([[float(row[key]) for key in META] for row in rows], dtype=np.float32)
    images = image_embeddings(rows, args.dataset_root, args.output_dir / "resnet18_embeddings.npz", args.batch_size)
    results = [
        evaluate("frozen_resnet18_image", images, labels, fold_indices, args.seed, args.epochs),
        evaluate("age_only_negative_control", metadata[:, :1], labels, fold_indices, args.seed, args.epochs),
        evaluate("safe_metadata_negative_control", metadata, labels, fold_indices, args.seed, args.epochs),
    ]
    payload = {"research_only": True, "method": "5-fold stratified cross-validation with frozen ImageNet ResNet-18 embeddings and fold-local standardization", "records": len(rows), "seed": args.seed, "results": results, "interpretation_boundary": "Patient-level BRAR grading only; no localized diagnosis or treatment recommendation."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({r["model"]: r["summary"] for r in results}, indent=2))


if __name__ == "__main__": main()

