"""Research baseline for image-level DenPAR bone-line segmentation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DenPAR Radiographs Dataset" / "Dataset"


def records(split: str) -> list[dict]:
    folder = DATA / split / "Bone Level Annotations"
    result = []
    for annotation_path in sorted(folder.glob("*.json")):
        if not annotation_path.stem.isdecimal():
            continue
        item = json.loads(annotation_path.read_text())
        image_id = annotation_path.stem
        if item.get("Image_id") != f"{image_id}.jpg":
            raise ValueError(f"Image ID mismatch: {annotation_path}")
        image = DATA / split / "Images" / f"{image_id}.jpg"
        if not image.exists():
            image = DATA / "Images" / f"{image_id}.jpg"
        if not image.exists() and split == "Training":
            image = ROOT / "dataset" / "raw" / f"{image_id}.jpg"
        if not image.exists():
            raise FileNotFoundError(image_id)
        lines = item["Bone_Lines"]
        if len(lines) != item["Num_of_Bone_Lines"]:
            raise ValueError(f"Line count mismatch: {annotation_path}")
        with Image.open(image) as source:
            width, height = source.size
        valid = True
        for line in lines:
            if len(line) < 2 or len({tuple(point) for point in line}) < 2:
                valid = False
                break
            if any(len(point) != 2 or not all(np.isfinite(point)) or not (0 <= point[0] < width and 0 <= point[1] < height) for point in line):
                valid = False
                break
        if not valid:
            print(f"Skipping invalid geometry: {split}/{image_id}", flush=True)
            continue
        result.append({"id": image_id, "image": image, "lines": lines, "width": width, "height": height})
    return result


class BoneLines(Dataset):
    def __init__(self, items: list[dict], size: int):
        self.items, self.size = items, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(item["image"]) as source:
            image = source.convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        mask = Image.new("L", (self.size, self.size))
        draw = ImageDraw.Draw(mask)
        for line in item["lines"]:
            points = [(x * self.size / item["width"], y * self.size / item["height"]) for x, y in line]
            draw.line(points, fill=255, width=3, joint="curve")
        return torch.from_numpy(pixels[None].copy()), torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)[None])


class Conv(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1, self.down2, self.bridge = Conv(1, 8), Conv(8, 16), Conv(16, 32)
        self.up2, self.up1 = Conv(48, 16), Conv(24, 8)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(nn.functional.max_pool2d(a, 2))
        c = self.bridge(nn.functional.max_pool2d(b, 2))
        c = self.up2(torch.cat((nn.functional.interpolate(c, size=b.shape[-2:]), b), dim=1))
        c = self.up1(torch.cat((nn.functional.interpolate(c, size=a.shape[-2:]), a), dim=1))
        return self.head(c)


def score(model, loader, device, threshold):
    model.eval()
    matched_pred = pred_total = matched_truth = truth_total = 0
    with torch.no_grad():
        for image, target in loader:
            predicted = (model(image.to(device)).sigmoid() >= threshold).cpu()
            truth = target.bool()
            # Match the baseline's 3-pixel tolerance at 128 px across input sizes.
            radius = max(1, round(3 * image.shape[-1] / 128))
            kernel = 2 * radius + 1
            near_truth = nn.functional.max_pool2d(truth.float(), kernel, stride=1, padding=radius).bool()
            near_pred = nn.functional.max_pool2d(predicted.float(), kernel, stride=1, padding=radius).bool()
            matched_pred += (predicted & near_truth).sum().item()
            pred_total += predicted.sum().item()
            matched_truth += (truth & near_pred).sum().item()
            truth_total += truth.sum().item()
    precision = matched_pred / pred_total if pred_total else 0.0
    recall = matched_truth / truth_total if truth_total else 0.0
    return {"tolerance_precision": precision, "tolerance_recall": recall,
            "tolerance_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "bone_line_unet_baseline")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the locked validation-selected checkpoint once")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.evaluate_test:
        checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
        model = SmallUNet().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        testing = records("Testing")
        loader = DataLoader(BoneLines(testing, checkpoint["size"]), batch_size=args.batch_size, num_workers=0)
        metrics = score(model, loader, device, checkpoint["threshold"])
        report = {"test_images": len(testing), "checkpoint_epoch": checkpoint["epoch"],
                  "threshold_selected_on_validation": checkpoint["threshold"], **metrics,
                  "clinical_interpretation": "None; image-level line localization only"}
        first = testing[0]
        sample_image, sample_target = BoneLines([first], checkpoint["size"])[0]
        with torch.no_grad():
            sample_prediction = (model(sample_image[None].to(device)).sigmoid()[0, 0].cpu().numpy() >= checkpoint["threshold"])
        base = np.asarray(Image.open(first["image"]).convert("L").resize((checkpoint["size"], checkpoint["size"]))).copy()
        overlay = np.stack((base, base, base), axis=2)
        overlay[sample_target[0].numpy().astype(bool)] = (0, 255, 0)
        overlay[sample_prediction] = (255, 80, 80)
        Image.fromarray(overlay).resize((checkpoint["size"] * 4, checkpoint["size"] * 4)).save(args.output / "test_example.png")
        report["example_image_id"] = first["id"]
        (args.output / "test_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    training, validation = records("Training"), records("Validation")
    train_loader = DataLoader(BoneLines(training, args.size), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(BoneLines(validation, args.size), batch_size=args.batch_size, num_workers=0)
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(image)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(75., device=device))
            probability = logits.sigmoid()
            dice_loss = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
            loss = bce + dice_loss
            loss.backward(); optimizer.step()
            losses.append(loss.item())
        candidates = [(threshold, score(model, val_loader, device, threshold)) for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
        threshold, metrics = max(candidates, key=lambda pair: pair[1]["tolerance_f1"])
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "threshold": threshold, **metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["tolerance_f1"] > best:
            best = metrics["tolerance_f1"]
            torch.save({"state_dict": model.state_dict(), "size": args.size, "epoch": epoch, "threshold": threshold}, args.output / "best.pt")
        (args.output / "training_report.json").write_text(json.dumps({"task": "image_level_bone_line_segmentation",
            "train_images": len(training), "validation_images": len(validation), "test_used": False,
            "target": "3-pixel raster of original polylines at resized resolution",
            "metric": "pixel precision/recall/F1 with spatial tolerance scaled from 3 pixels at 128-pixel output",
            "note": "Anatomical line localization only; no disease, severity, or tooth correspondence claim.",
            "device": str(device), "history": history, "best_validation_f1": best}, indent=2))


if __name__ == "__main__":
    main()
