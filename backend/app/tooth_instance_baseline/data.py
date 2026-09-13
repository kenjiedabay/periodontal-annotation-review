"""Validated, geometry-preserving DenPAR instance-mask loading."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from app.preprocessing.config import load_config
from app.preprocessing.transforms import apply_transforms
from .provenance_audit import transformed_mask


OFFICIAL_SPLITS = {"train": "Training", "validation": "Validation", "test": "Testing"}


@dataclass(frozen=True)
class Record:
    image_id: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    width: int
    height: int


def binary_mask(image: Image.Image) -> np.ndarray:
    """Safely binarize arbitrary PNG values; nonzero is tooth foreground."""
    return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def box_from_mask(mask: np.ndarray) -> list[float]:
    rows, columns = np.where(mask)
    if not len(rows):
        raise ValueError("empty mask has no bounding box")
    return [float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1)]


def validate_official_split(dataset_root: Path, split: str, training_images_dir: Path | None = None) -> tuple[list[Record], dict[str, Any]]:
    folder = OFFICIAL_SPLITS[split]
    masks_dir = dataset_root / folder / "Masks (Tooth-wise)"
    records, invalid, skipped, instance_counts, dimensions = [], [], [], Counter(), Counter()
    for mask_folder in sorted(path for path in masks_dir.iterdir() if path.is_dir()):
        # DenPAR stores Validation images beneath Validation/Images and Testing
        # images beneath Dataset/Images.  Training images are not substituted
        # from a different partition when absent.
        candidates = ((training_images_dir / f"{mask_folder.name}.jpg",) if split == "train" and training_images_dir else ()) + (dataset_root / folder / "Images" / f"{mask_folder.name}.jpg", dataset_root / "Images" / f"{mask_folder.name}.jpg")
        image_path = next((path for path in candidates if path.exists()), None)
        if image_path is None:
            skipped.append({"image_id": mask_folder.name, "reason": "source_image_missing"})
            continue
        with Image.open(image_path) as source:
            width, height = source.size
        paths = []
        for mask_path in sorted(mask_folder.glob("*.png")):
            try:
                with Image.open(mask_path) as mask_image:
                    if mask_image.size != (width, height):
                        invalid.append({"image_id": mask_folder.name, "mask": mask_path.name, "reason": "dimension_mismatch"})
                        continue
                    mask = binary_mask(mask_image)
                    if not mask.any():
                        invalid.append({"image_id": mask_folder.name, "mask": mask_path.name, "reason": "empty_mask"})
                        continue
                    box_from_mask(mask)
                paths.append(mask_path)
            except (OSError, ValueError) as error:
                invalid.append({"image_id": mask_folder.name, "mask": mask_path.name, "reason": str(error)})
        if not paths:
            skipped.append({"image_id": mask_folder.name, "reason": "no_valid_instance_masks"})
            continue
        records.append(Record(mask_folder.name, image_path, tuple(paths), width, height))
        instance_counts[len(paths)] += 1
        dimensions[f"{width}x{height}"] += 1
    report = {"split": split, "total_images": len(records) + len(skipped), "usable_images": len(records), "total_instances": sum(len(record.mask_paths) for record in records), "valid_masks": sum(len(record.mask_paths) for record in records), "invalid_masks": invalid, "skipped_images": skipped, "masks_per_image": dict(instance_counts), "image_size_distribution": dict(dimensions)}
    return records, report


def _resize(image: Image.Image, masks: list[Image.Image], max_side: int) -> tuple[Image.Image, list[Image.Image], dict[str, Any]]:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    target = (round(width * scale), round(height * scale))
    metadata = {"operation": "identity" if scale == 1 else "resize_longest_side", "source_size": [width, height], "output_size": list(target), "scale_x": target[0] / width, "scale_y": target[1] / height, "image_resample": "bilinear", "mask_resample": "nearest"}
    if scale == 1:
        return image, masks, metadata
    return image.resize(target, Image.Resampling.BILINEAR), [mask.resize(target, Image.Resampling.NEAREST) for mask in masks], metadata


class ToothInstanceDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    def __init__(self, records: list[Record], max_side: int = 1024, augment: bool = False, seed: int = 42):
        self.records, self.max_side, self.augment, self.seed = records, max_side, augment, seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("L").copy()
        masks = []
        for path in record.mask_paths:
            with Image.open(path) as mask:
                masks.append(mask.convert("L").copy())
        # 1024 is the audited project preprocessing: aspect-preserving resize,
        # centered pad, percentile normalization, and CLAHE. Masks receive only
        # the identical geometry with nearest-neighbor resampling.
        if self.max_side == 1024:
            raw = cv2.imread(str(record.image_path), cv2.IMREAD_UNCHANGED)
            processed, _, _ = apply_transforms(raw, load_config())
            rendered_masks, transform = [], None
            for mask in masks:
                rendered, transform = transformed_mask(binary_mask(mask), record.width, record.height, 1024)
                rendered_masks.append(Image.fromarray(rendered))
            image, masks = Image.fromarray(processed), rendered_masks
            transform = {**transform, "preprocessing_version": "1.0.0", "intensity_operations": ["grayscale", "percentile:1-99", "clahe:1.5"]}
        else:
            image, masks, transform = _resize(image, masks, self.max_side)
        # Deterministic per-index augmentation: horizontal flip is acceptable
        # because the target has no laterality or FDI identity semantics.
        generator = torch.Generator().manual_seed(self.seed + index)
        if self.augment and torch.rand((), generator=generator).item() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            masks = [mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for mask in masks]
            transform["augmentation"] = "horizontal_flip"
        else:
            transform["augmentation"] = "none"
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array).unsqueeze(0).repeat(3, 1, 1)
        instance_masks = torch.stack([torch.from_numpy(binary_mask(mask)) for mask in masks]).to(torch.uint8)
        boxes = torch.tensor([box_from_mask(item.numpy().astype(bool)) for item in instance_masks], dtype=torch.float32)
        target = {"boxes": boxes, "labels": torch.ones((len(masks),), dtype=torch.int64), "masks": instance_masks, "image_id": torch.tensor([index]), "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), "iscrowd": torch.zeros((len(masks),), dtype=torch.int64), "metadata": {"image_id": record.image_id, "original_size": [record.width, record.height], "preprocessing": transform}}
        return tensor, target


def collate(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    return tuple(zip(*batch))  # type: ignore[return-value]
