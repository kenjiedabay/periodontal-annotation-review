"""Train a U-Net tooth localizer from the annotated DenPAR radiographs.

This model predicts the *tooth region* only.  DenPAR does not supply expert
periodontal-disease or severity labels, so this script deliberately does not
produce disease, severity, or progression predictions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DENPAR = ROOT / "DenPAR Radiographs Dataset" / "Dataset"


class ToothDataset(Dataset):
    def __init__(self, split: str, size: int, augment: bool = False):
        base = DENPAR / split
        self.images = base / "Images"
        self.masks = base / "Masks (Radiograph-wise)"
        self.items = [p for p in sorted(self.images.glob("*.jpg")) if (self.masks / f"{p.stem}.png").exists()]
        self.size, self.augment = size, augment
        if not self.items:
            raise RuntimeError(f"No image/mask pairs found in {base}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path = self.items[index]
        image = Image.open(image_path).convert("L")
        mask = Image.open(self.masks / f"{image_path.stem}.png").convert("L")
        image = TF.resize(image, [self.size, self.size], antialias=True)
        mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        if self.augment and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        image = TF.to_tensor(image)
        image = (image - 0.5) / 0.5
        return image, (TF.to_tensor(mask) > 0.5).float()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(128, 256)
        self.u3, self.d3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.u2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.u1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        a = self.e1(x); b = self.e2(self.pool(a)); c = self.e3(self.pool(b)); z = self.bridge(self.pool(c))
        z = self.d3(torch.cat([self.u3(z), c], dim=1)); z = self.d2(torch.cat([self.u2(z), b], dim=1))
        return self.out(self.d1(torch.cat([self.u1(z), a], dim=1)))


def dice_score(logits, target):
    pred = (logits.sigmoid() > 0.5).float()
    return ((2 * (pred * target).sum((1, 2, 3)) + 1) / (pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def soft_dice_loss(logits, target):
    pred = logits.sigmoid()
    dice = (2 * (pred * target).sum((1, 2, 3)) + 1) / (pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return 1 - dice.mean()


def evaluate(model, loader, device):
    model.eval(); scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(dice_score(model(x.to(device)), y.to(device)).item())
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "backend" / "models" / "prototype_denpar_tooth_unet.pt")
    args = parser.parse_args()
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(ToothDataset("Training", args.size, True), args.batch_size, shuffle=True, num_workers=args.workers)
    valid = DataLoader(ToothDataset("Validation", args.size), args.batch_size, num_workers=args.workers)
    model = UNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.BCEWithLogitsLoss(); best = -1.0; args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in tqdm(train, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device); optimizer.zero_grad(); logits = model(x)
            loss = loss_fn(logits, y) + soft_dice_loss(logits, y); loss.backward(); optimizer.step(); losses.append(loss.item())
        score = evaluate(model, valid, device)
        print(f"epoch={epoch} train_loss={np.mean(losses):.4f} validation_dice={score:.4f}")
        if score > best:
            best = score; torch.save({"model": model.state_dict(), "size": args.size, "validation_dice": score}, args.output)
    print(f"Saved best model to {args.output} (validation Dice={best:.4f})")


if __name__ == "__main__": main()
