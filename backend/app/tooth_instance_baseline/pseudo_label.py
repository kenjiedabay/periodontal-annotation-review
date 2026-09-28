"""Generate teacher instance masks for the hidden DenPAR training subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from app.preprocessing.config import load_config
from app.preprocessing.transforms import apply_transforms
from .model import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    args = parser.parse_args()
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    root = Path(args.split_manifest).resolve().parents[2]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = create_model(False, 1024)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    accepted = []
    with torch.inference_mode():
        for index, image_id in enumerate(manifest["pseudo_label_candidate_ids"], 1):
            source = root / manifest["source_image_dir"] / f"{image_id}.jpg"
            raw = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(source)
            processed, _, _ = apply_transforms(raw, load_config())
            tensor = torch.from_numpy(processed.astype(np.float32) / 255).unsqueeze(0).repeat(3, 1, 1).to(device)
            result = model([tensor])[0]
            height, width = raw.shape[:2]
            scale = min(1024 / width, 1024 / height)
            resized_width, resized_height = round(width * scale), round(height * scale)
            left, top = (1024 - resized_width) // 2, (1024 - resized_height) // 2
            masks = []
            for score, predicted in zip(result["scores"], result["masks"]):
                if float(score) < args.score_threshold:
                    continue
                mask = (predicted[0, top:top + resized_height, left:left + resized_width] >= 0.5).byte().cpu().numpy()
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                if int(mask.sum()) < 500:
                    continue
                if any(np.logical_and(mask, old).sum() / max(1, np.logical_or(mask, old).sum()) > 0.5 for old in masks):
                    continue
                masks.append(mask)
                if len(masks) == 8:
                    break
            if masks:
                folder = output / image_id
                folder.mkdir(exist_ok=True)
                for number, mask in enumerate(masks, 1):
                    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"mask{number}.png")
            accepted.append({"image_id": image_id, "instances": len(masks)})
            if index % 25 == 0:
                print(f"Generated {index}/{len(manifest['pseudo_label_candidate_ids'])}", flush=True)
    report = {"checkpoint": args.checkpoint, "score_threshold": args.score_threshold, "accepted_images": sum(item["instances"] > 0 for item in accepted), "accepted_instances": sum(item["instances"] for item in accepted), "per_image": accepted}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
