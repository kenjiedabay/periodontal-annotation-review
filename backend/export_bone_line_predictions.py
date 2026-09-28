"""Export validation-only bone-line overlays for the structural review UI."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from train_bone_lines import BoneLines, SmallUNet, records, ROOT


def main():
    output = ROOT / "models" / "bone_line_unet_baseline"
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model = SmallUNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    destination = output / "validation_predictions"
    destination.mkdir(parents=True, exist_ok=True)
    exported = []
    with torch.no_grad():
        for item in records("Validation"):
            image, _ = BoneLines([item], checkpoint["size"])[0]
            probability = model(image[None])[0, 0].sigmoid().numpy()
            mask = Image.fromarray((probability >= checkpoint["threshold"]).astype(np.uint8) * 255)
            mask = mask.resize((item["width"], item["height"]), Image.Resampling.NEAREST)
            alpha = np.asarray(mask)
            overlay = np.zeros((item["height"], item["width"], 4), dtype=np.uint8)
            overlay[:, :, :3] = (255, 69, 79)
            overlay[:, :, 3] = (alpha > 0).astype(np.uint8) * 165
            Image.fromarray(overlay, "RGBA").save(destination / f"{item['id']}.png")
            exported.append(item["id"])
    (destination / "manifest.json").write_text(json.dumps({"split": "Validation", "image_ids": exported,
        "checkpoint_epoch": checkpoint["epoch"], "threshold": checkpoint["threshold"],
        "size": checkpoint["size"], "source": "bone_line_unet_baseline/best.pt"}, indent=2))
    print(f"Exported {len(exported)} Validation predictions to {destination}")


if __name__ == "__main__":
    main()
