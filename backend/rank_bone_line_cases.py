"""Rank DenPAR Validation images by locked bone-line baseline performance."""

import csv
import json

import torch

from train_bone_lines import BoneLines, ROOT, SmallUNet, records, score


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = ROOT / "models" / "bone_line_unet_baseline"
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model = SmallUNet()
    model.load_state_dict(checkpoint["state_dict"])
    dataset = BoneLines(records("Validation"), checkpoint["size"])
    rows = []
    for index, item in enumerate(dataset.items):
        image, target = dataset[index]
        result = score(model, [(image[None], target[None])], torch.device("cpu"), checkpoint["threshold"])
        rows.append({"image_id": item["id"], "split": "Validation", "source_lines": len(item["lines"]),
                     "tolerance_precision": round(result["tolerance_precision"], 6),
                     "tolerance_recall": round(result["tolerance_recall"], 6),
                     "tolerance_f1": round(result["tolerance_f1"], 6),
                     "image_path": str(item["image"].relative_to(ROOT)),
                     "annotation_path": str((ROOT / "DenPAR Radiographs Dataset" / "Dataset" / "Validation" / "Bone Level Annotations" / f"{item['id']}.json").relative_to(ROOT))})
    rows.sort(key=lambda row: (row["tolerance_f1"], row["image_id"]))
    destination = output / "validation_case_ranking.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"ranked_images": len(rows), "lowest_five": rows[:5], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
