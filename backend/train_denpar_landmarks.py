"""Image-level CEJ and apex heatmap baseline for DenPAR periapical radiographs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_bone_lines import ROOT, DATA, SmallUNet


def records(split: str):
    items = []
    for path in sorted((DATA / split / "Key Points Annotations").glob("*.json")):
        if not path.stem.isdecimal():
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        image = DATA / split / "Images" / f"{path.stem}.jpg"
        if not image.exists():
            image = DATA / "Images" / f"{path.stem}.jpg"
        if not image.exists() and split == "Training":
            image = ROOT / "dataset" / "raw" / f"{path.stem}.jpg"
        if not image.exists():
            raise FileNotFoundError(image)
        with Image.open(image) as opened:
            width, height = opened.size
        points = []
        valid = True
        for key in ("CEJ_Points", "Apex_Points"):
            group = source.get(key, [])
            if not group:
                valid = False
                break
            for pair in group:
                if len(pair) != 2 or not np.isfinite(pair).all() or not (0 <= pair[0] < width and 0 <= pair[1] < height):
                    valid = False
                    break
            points.append(group)
        if valid:
            items.append({"id": path.stem, "image": image, "points": points, "width": width, "height": height})
        else:
            print(f"Skipping missing or invalid landmarks: {split}/{path.stem}", flush=True)
    return items


class Landmarks(Dataset):
    def __init__(self, items, size=256):
        self.items, self.size = items, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        size = self.size
        scale = min(size / item["width"], size / item["height"])
        width, height = round(item["width"] * scale), round(item["height"] * scale)
        left, top = (size - width) // 2, (size - height) // 2
        with Image.open(item["image"]) as source:
            resized = source.convert("L").resize((width, height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (size, size))
        canvas.paste(resized, (left, top))
        image = np.asarray(canvas, dtype=np.float32) / 255.0
        heatmaps = np.zeros((2, size, size), dtype=np.float32)
        transformed = []
        for channel, group in enumerate(item["points"]):
            group_points = []
            for x, y in group:
                px, py = round(x * scale + left), round(y * scale + top)
                px, py = min(size - 1, px), min(size - 1, py)
                cv2.circle(heatmaps[channel], (px, py), 2, 1.0, -1)
                group_points.append([px, py])
            transformed.append(group_points)
        return torch.from_numpy(image[None].copy()), torch.from_numpy(heatmaps), transformed, item["id"]


def collate(batch):
    images, targets, points, ids = zip(*batch)
    return torch.stack(images), torch.stack(targets), points, ids


def peaks(probability, threshold, limit=30):
    maximum = cv2.dilate(probability, np.ones((9, 9), np.uint8))
    candidates = ((probability >= threshold) & (probability >= maximum - 1e-6)).astype(np.uint8)
    count, labels = cv2.connectedComponents(candidates)
    result = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if len(xs):
            best = np.argmax(probability[ys, xs])
            result.append((int(xs[best]), int(ys[best]), float(probability[ys[best], xs[best]])))
    return sorted(result, key=lambda row: row[2], reverse=True)[:limit]


def evaluate(cached, threshold, radius=8):
    totals = {name: {"tp": 0, "predicted": 0, "reference": 0} for name in ("cej", "apex")}
    for probability, truth in cached:
        for channel, name in enumerate(("cej", "apex")):
            # Caps are the maximum counts observed in Training, not Validation.
            proposed = peaks(probability[channel], threshold, (12, 10)[channel])
            reference = truth[channel]
            used = set()
            totals[name]["predicted"] += len(proposed)
            totals[name]["reference"] += len(reference)
            for x, y, _ in proposed:
                options = [(np.hypot(x - point[0], y - point[1]), i) for i, point in enumerate(reference) if i not in used]
                if options:
                    distance, i = min(options)
                    if distance <= radius:
                        used.add(i)
                        totals[name]["tp"] += 1
    for values in totals.values():
        values["precision"] = values["tp"] / values["predicted"] if values["predicted"] else 0
        values["recall"] = values["tp"] / values["reference"] if values["reference"] else 0
        values["f1"] = 2 * values["tp"] / (values["predicted"] + values["reference"]) if values["predicted"] + values["reference"] else 0
    totals["macro_f1"] = (totals["cej"]["f1"] + totals["apex"]["f1"]) / 2
    return totals


def predictions(model, loader, device):
    model.eval()
    cached = []
    with torch.no_grad():
        for images, _, points, _ in loader:
            outputs = model(images.to(device)).sigmoid().cpu().numpy()
            cached.extend((probability, truth) for probability, truth in zip(outputs, points))
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "denpar_cej_apex_heatmap_256")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--evaluate-validation", action="store_true")
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet()
    model.head = nn.Conv2d(8, 2, 1)
    model.to(device)
    if args.evaluate_test or args.evaluate_validation:
        checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        split = "Testing" if args.evaluate_test else "Validation"
        items = records(split)
        loader = DataLoader(Landmarks(items, checkpoint["size"]), batch_size=args.batch_size, collate_fn=collate)
        cached = predictions(model, loader, device)
        if args.evaluate_validation:
            threshold, metrics = max(((t, evaluate(cached, t)) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)), key=lambda pair: pair[1]["macro_f1"])
        else:
            threshold = checkpoint["threshold"]
            metrics = evaluate(cached, threshold)
        report = {"split": split, "images": len(items), "checkpoint_epoch": checkpoint["epoch"], "threshold_selected_on_validation": threshold, "point_limit_from_training": {"cej": 12, "apex": 10}, "tolerance_pixels_at_model_size": 8, "metrics": metrics}
        (args.output / ("test_report.json" if args.evaluate_test else "validation_postprocess_report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        return
    train, validation = records("Training"), records("Validation")
    train_loader = DataLoader(Landmarks(train, args.size), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Landmarks(validation, args.size), batch_size=args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, _, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            per_pixel = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            positive = (per_pixel * targets).sum() / targets.sum().clamp_min(1)
            negative = (per_pixel * (1 - targets)).sum() / (1 - targets).sum().clamp_min(1)
            loss = positive + negative
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        cached = predictions(model, val_loader, device)
        threshold, metrics = max(((t, evaluate(cached, t)) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)), key=lambda pair: pair[1]["macro_f1"])
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "threshold": threshold, "validation": metrics}
        history.append(entry)
        print(json.dumps({"epoch": epoch, "loss": entry["loss"], "threshold": threshold, "macro_f1": metrics["macro_f1"]}), flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "size": args.size, "threshold": threshold, "epoch": epoch}, args.output / "best.pt")
        (args.output / "training_report.json").write_text(json.dumps({"task": "image_level_cej_and_apex_point_localization", "train_images": len(train), "validation_images": len(validation), "test_used": False, "device": str(device), "radius_pixels_at_model_size": 8, "history": history, "best_validation_macro_f1": best, "limitation": "Points are not linked to tooth instances or bone lines; this is not a bone-loss measurement."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
