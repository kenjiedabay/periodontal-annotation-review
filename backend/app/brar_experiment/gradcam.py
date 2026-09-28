"""Create research-only Grad-CAM overlays for the BRAR image baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .train import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../dataset/brar/brar_split_manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("../BRAR Dataset/BRAR-anchored multimodal dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../models/brar_image/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("../models/brar_image/gradcam"))
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(False, False).to(device); payload = torch.load(args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model"]); model.eval()
    activations, gradients = [], []
    model.image.layer4.register_forward_hook(lambda _m, _i, output: activations.append(output))
    model.image.layer4.register_full_backward_hook(lambda _m, _gi, output: gradients.append(output[0]))
    transform = transforms.Compose([transforms.Grayscale(3), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485] * 3, [.229] * 3)])
    with args.manifest.open(encoding="utf-8", newline="") as handle: rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    selected, records = set(), []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        level = int(row["level"]); image = Image.open(args.dataset_root / row["image_path"]).convert("L")
        tensor = transform(image).unsqueeze(0).to(device); metadata = torch.zeros((1, 6), device=device)
        activations.clear(); gradients.clear(); logits = model(tensor, metadata); predicted = int(logits.argmax(1)) + 1
        if predicted != level or level in selected: continue
        model.zero_grad(set_to_none=True); logits[0, predicted - 1].backward()
        weights = gradients[-1].mean((2, 3), keepdim=True); cam = torch.relu((weights * activations[-1]).sum(1))[0]
        cam -= cam.min(); cam /= cam.max().clamp_min(1e-8)
        cam_image = Image.fromarray((cam.detach().cpu().numpy() * 255).astype(np.uint8)).resize(image.size)
        base = image.convert("RGB"); heat = Image.new("RGB", image.size, (255, 0, 0)); alpha = cam_image.point(lambda value: int(value * .55))
        overlay = Image.composite(heat, base, alpha); output = args.output_dir / f"level_{level}_{Path(row['file_name']).stem}.png"; overlay.save(output)
        records.append({"file_name": row["file_name"], "actual_level": level, "predicted_level": predicted, "output": output.name}); selected.add(level)
        if len(selected) == 3: break
    (args.output_dir / "manifest.json").write_text(json.dumps({"research_only": True, "warning": "Grad-CAM is a qualitative attention visualization, not lesion localization or clinical explanation.", "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__": main()
