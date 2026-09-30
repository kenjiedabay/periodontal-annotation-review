"""Research-only Mask R-CNN -> Perio-KPT connected inference preview.

This adapter does not alter either checkpoint. Root class is not predicted by
the detector, so RBL is returned as a range across the central-root and
surface-root conventions when both are geometrically valid.
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from app.perio_kpt_landmarks.evaluation import decode_heatmaps
from app.perio_kpt_landmarks.geometry import (calculate_rbl, crop_to_image,
                                              expand_box, make_letterbox_transform)
from app.perio_kpt_landmarks.model import PerioLandmarkNet
from app.perio_kpt_landmarks.schema import LANDMARK_INDEX, LANDMARK_NAMES
from app.tooth_segmentation import service as tooth_service

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts/perio-kpt-landmarks/gate2/pilot_fold0_30epoch_resumable_v1/best.pt"
CONFIDENCE_THRESHOLD = 0.45


class ConnectedPerioService:
    def __init__(self):
        self.lock = Lock(); self.attempted = False; self.model = None; self.load_error = None

    def load(self) -> None:
        if self.attempted:
            return
        self.attempted = True
        try:
            import torch
            payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            if payload.get("completed_epoch") != 12:
                raise ValueError("Expected immutable epoch-12 best checkpoint")
            model = PerioLandmarkNet(11)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device).eval()
        except Exception as error:
            logging.exception("Perio-KPT checkpoint unavailable")
            self.load_error = f"{type(error).__name__}: {error}"

    @staticmethod
    def _measurement(points: list[tuple[float, float]], confidence: list[float], surface: str) -> dict[str, Any]:
        if surface == "mesial":
            cej, bone, surface_root = (LANDMARK_INDEX[x] for x in ("CEJ-m", "BL-m", "RL-m"))
        else:
            cej, bone, surface_root = (LANDMARK_INDEX[x] for x in ("CEJ-d", "BL-d", "RL-d"))
        central_root = LANDMARK_INDEX["RL-c"]
        required = {cej, bone, surface_root, central_root}
        if any(confidence[index] < CONFIDENCE_THRESHOLD for index in required):
            return {"status": "not_assessable", "reason": "low_landmark_confidence",
                    "confidence_threshold": CONFIDENCE_THRESHOLD}
        surface_value = calculate_rbl(points, surface, indices=(cej, bone, surface_root))
        central_value = calculate_rbl(points, surface, indices=(cej, bone, central_root))
        valid = [x["rbl_percent"] for x in (surface_value, central_value) if x["status"] == "assessable"]
        if not valid:
            return {"status": "not_assessable", "reason": "invalid_geometry",
                    "geometry": {"surface_root": surface_value, "central_root": central_value}}
        return {"status": "uncertain", "reason": "root_class_not_predicted",
                "preliminary_rbl_range_percent": [min(valid), max(valid)],
                "surface_root_percent": surface_value.get("rbl_percent"),
                "central_root_percent": central_value.get("rbl_percent"),
                "confidence_threshold": CONFIDENCE_THRESHOLD}

    def predict(self, image_id: str, original: np.ndarray) -> dict[str, Any]:
        detection = tooth_service.predict(image_id, original)
        response: dict[str, Any] = {"image_id": image_id, "task": "connected_perio_kpt_research_preview",
            "model_status": "unavailable", "detector_status": detection["model_status"],
            "detector_version": detection["model_version"], "landmark_model_version": "perio_kpt_fold0_epoch12",
            "width": detection["width"], "height": detection["height"], "instances": [],
            "disclaimer": "Experimental research preview. Bone loss may be present. This is not a definitive diagnosis. Clinical periodontal examination is required.",
            "limitations": ["Connected predicted-crop evaluation has not been completed.",
                            "The tooth detector does not predict root class or FDI tooth number.",
                            "RBL is withheld or represented as a root-convention range, not a clinical measurement."]}
        if detection["model_status"] != "available":
            response["model_error"] = detection.get("model_error", "Tooth detector unavailable")
            return response
        with self.lock:
            self.load()
            if self.model is None:
                response["model_error"] = self.load_error or "Perio-KPT model unavailable"
                return response
            import torch
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
            height, width = gray.shape
            for detected in detection["instances"]:
                try:
                    expanded = expand_box(detected["bbox"], width, height, 0.20)
                    transform = make_letterbox_transform(expanded, width, height, 256)
                    x1, y1, _, _ = transform.crop_xyxy
                    affine = np.asarray([[transform.scale, 0, transform.offset_x-transform.scale*x1],
                                         [0, transform.scale, transform.offset_y-transform.scale*y1]], dtype=np.float32)
                    crop = cv2.warpAffine(gray, affine, (256, 256), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    valid = np.zeros((256, 256), dtype=np.float32)
                    valid[transform.offset_y:transform.offset_y+transform.resized_height,
                          transform.offset_x:transform.offset_x+transform.resized_width] = 1
                    tensor = torch.from_numpy(crop.astype(np.float32)/255).unsqueeze(0).repeat(3,1,1).unsqueeze(0)
                    content = torch.from_numpy(valid).unsqueeze(0)
                    with torch.inference_mode():
                        probabilities = self.model(tensor.to(self.device)).sigmoid().cpu()
                    crop_points, confidence = decode_heatmaps(probabilities, content)
                    image_points = [crop_to_image(point.tolist(), transform) for point in crop_points[0]]
                    conf = confidence[0].tolist()
                    landmarks = [{"type": name, "x": point[0], "y": point[1], "confidence": conf[index],
                                  "confidence_status": "acceptable" if conf[index] >= CONFIDENCE_THRESHOLD else "uncertain"}
                                 for index, (name, point) in enumerate(zip(LANDMARK_NAMES, image_points))]
                    measurements = {surface: self._measurement(image_points, conf, surface)
                                    for surface in ("mesial", "distal")}
                    response["instances"].append({"instance_id": detected["instance_id"],
                        "detector_confidence": detected["confidence"], "bbox": detected["bbox"],
                        "expanded_bbox": list(expanded), "root_class": "unknown", "landmarks": landmarks,
                        "measurements": measurements})
                except Exception as error:
                    response["instances"].append({"instance_id": detected["instance_id"], "bbox": detected["bbox"],
                        "expanded_bbox": None, "root_class": "unknown", "landmarks": [], "measurements": {},
                        "error": f"{type(error).__name__}: {error}"})
            response["model_status"] = "available"
        return response


service = ConnectedPerioService()
