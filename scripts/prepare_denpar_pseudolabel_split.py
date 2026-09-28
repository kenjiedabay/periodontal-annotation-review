"""Audit DenPAR train pairs and make a reproducible pseudo-label experiment split."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DenPAR Radiographs Dataset" / "Dataset"
OUT = ROOT / "artifacts" / "denpar-pseudolabel" / "split_seed42.json"
SEED = 42


def main():
    images = ROOT / "dataset" / "raw"
    masks = DATA / "Training" / "Masks (Tooth-wise)"
    groups = defaultdict(list)
    missing_images = []
    missing_masks = []
    counts = Counter()
    for folder in sorted(p for p in masks.iterdir() if p.is_dir()):
        image_id = folder.name
        image = images / f"{image_id}.jpg"
        mask_files = list(folder.glob("mask*.png"))
        if not image.is_file():
            missing_images.append(image_id)
            continue
        if not mask_files:
            missing_masks.append(image_id)
            continue
        count = len(mask_files)
        counts[count] += 1
        groups[min(count, 7)].append(image_id)

    if missing_images or missing_masks or sum(counts.values()) != 650:
        raise RuntimeError(f"Incomplete DenPAR training pairs: {len(missing_images)} images and {len(missing_masks)} masks missing; {sum(counts.values())} usable")

    rng = random.Random(SEED)
    labeled, hidden = [], []
    for count, ids in sorted(groups.items()):
        rng.shuffle(ids)
        cutoff = len(ids) // 2
        labeled.extend(ids[:cutoff])
        hidden.extend(ids[cutoff:])
    # Balance the rounding remainder while retaining the mask-count strata.
    while len(labeled) < 325:
        labeled.append(hidden.pop())
    labeled.sort(key=int)
    hidden.sort(key=int)
    result = {
        "purpose": "controlled tooth-instance pseudo-label experiment",
        "seed": SEED,
        "source_image_dir": str(images.relative_to(ROOT)),
        "source_mask_dir": str(masks.relative_to(ROOT)),
        "teacher_training_ids": labeled,
        "supervised_control_training_ids": labeled,
        "pseudo_label_candidate_ids": hidden,
        "validation": "official DenPAR Validation only for selection",
        "test": "official DenPAR Testing; final evaluation only",
        "training_mask_count_distribution": dict(sorted(counts.items())),
        "training_instance_count": sum(k * v for k, v in counts.items()),
        "warning": "Do not use the existing full-650-image checkpoint as teacher; it has seen hidden-subset ground truth.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}: {len(labeled)} labeled, {len(hidden)} hidden, {result['training_instance_count']} masks")


if __name__ == "__main__":
    main()
