"""Create tooth-region overlays for the unlabeled Images directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as TF

from train_tooth_localizer import ROOT, UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "backend" / "models" / "prototype_denpar_tooth_unet.pt")
    parser.add_argument("--source", type=Path, default=ROOT / "DenPAR Radiographs Dataset" / "Dataset" / "Images")
    parser.add_argument("--output", type=Path, default=ROOT / "backend" / "outputs" / "tooth_localization" / "prototype")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists(): raise FileNotFoundError("Train first: python backend/train_tooth_localizer.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True); size = checkpoint["size"]
    model = UNet().to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    output = args.output; output.mkdir(parents=True, exist_ok=True); results = []
    for source in sorted(args.source.glob("*")):
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
        image = Image.open(source).convert("L"); original_size = image.size
        x = (TF.to_tensor(TF.resize(image, [size, size], antialias=True)) - 0.5) / 0.5
        with torch.no_grad(): mask = (model(x.unsqueeze(0).to(device)).sigmoid()[0, 0] >= args.threshold).float()
        mask = TF.resize(mask.unsqueeze(0), [original_size[1], original_size[0]], interpolation=InterpolationMode.NEAREST)[0]
        rgba = image.convert("RGBA"); overlay = Image.new("RGBA", original_size, (0, 180, 255, 0)); overlay.putalpha(Image.fromarray((mask.cpu().numpy() * 100).astype("uint8")))
        result = Image.alpha_composite(rgba, overlay); destination = output / f"{source.stem}_tooth_overlay.png"; result.save(destination)
        results.append({"image": source.name, "tooth_overlay": str(destination.relative_to(ROOT)), "task": "tooth localization only"})
    (output / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Created {len(results)} overlays in {output}")


if __name__ == "__main__": main()
