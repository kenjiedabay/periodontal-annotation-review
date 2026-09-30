"""Read-only fold datasets built from Gate 1 geometry and validated annotations."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_image, parse_label, prepare_annotation
from .geometry import gaussian_heatmaps


class PerioCropDataset(Dataset):
    def __init__(self, manifest_path: Path, image_ids: list[str], output_size: int = 256,
                 allowed_classes: set[int] | None = None):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.resolve().parents[3]
        sources = {item["image_id"]: item for item in manifest["sources"]}
        self.records = []
        for image_id in image_ids:
            source = sources[image_id]
            image_path = root / source["image_path"]
            label_path = root / source["label_path"]
            annotations, _ = parse_label(label_path)
            for annotation in annotations:
                if ((allowed_classes is None or annotation.class_id in allowed_classes)
                        and any(point.available for point in annotation.landmarks)):
                    self.records.append((image_path, annotation))
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        image_path, annotation = self.records[index]
        image = load_image(image_path)
        height, width = image.shape
        transform, points = prepare_annotation(annotation, width, height, output_size=self.output_size)
        # A small number of source keypoints lie outside even the prescribed
        # 20%-expanded box. They cannot be represented by this crop and are
        # masked from loss/evaluation rather than clipped to a false location.
        content_x2 = transform.offset_x + transform.resized_width
        content_y2 = transform.offset_y + transform.resized_height
        points = [point if point is not None
                  and transform.offset_x <= point[0] < content_x2
                  and transform.offset_y <= point[1] < content_y2 else None for point in points]
        x1, y1, _, _ = transform.crop_xyxy
        affine = np.asarray([[transform.scale, 0, transform.offset_x - transform.scale * x1],
                             [0, transform.scale, transform.offset_y - transform.scale * y1]], dtype=np.float32)
        crop = cv2.warpAffine(image, affine, (self.output_size, self.output_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        heatmaps, availability = gaussian_heatmaps(points, self.output_size)
        valid_content = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        valid_content[transform.offset_y:content_y2, transform.offset_x:content_x2] = 1.0
        pixels = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
        point_array = np.full((len(points), 2), np.nan, dtype=np.float32)
        for point_index, point in enumerate(points):
            if point is not None:
                point_array[point_index] = point
        return {"image": pixels, "target": torch.from_numpy(heatmaps),
                "availability": torch.from_numpy(availability), "points": torch.from_numpy(point_array),
                "visibility": torch.tensor([point.visibility for point in annotation.landmarks], dtype=torch.int64),
                "valid_content": torch.from_numpy(valid_content),
                "record_id": annotation.record_id, "class_id": annotation.class_id,
                "transform": transform.to_dict()}
