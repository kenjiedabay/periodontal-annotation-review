"""Gate 2.1 read-only heatmap, coordinate, loss, and decoding audit."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import annotation_points_original, load_image, prepare_annotation
from .evaluation import decode_heatmaps
from .geometry import crop_to_image, gaussian_heatmaps, image_to_crop
from .model import PerioLandmarkNet
from .schema import LANDMARK_NAMES
from .train import DEFAULT_MANIFEST, DEFAULT_OUTPUT, seed_everything
from .training_data import PerioCropDataset

OUTPUT = DEFAULT_OUTPUT.parent / "gate2_1_diagnostic"
POSITIVE_THRESHOLD = 0.01


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"minimum": float(array.min()), "mean": float(array.mean()),
            "maximum": float(array.max())} if array.size else {"minimum": None, "mean": None, "maximum": None}


def ground_truth_audit(dataset: PerioCropDataset) -> tuple[dict, dict, list[tuple[int, dict]]]:
    stats = {name: defaultdict(list) for name in LANDMARK_NAMES}
    maximum_roundtrip_error = 0.0
    padding_points, visible_points = 0, 0
    padding_point_records = []
    outside = []
    visual_candidates = []
    for record_index, (image_path, annotation) in enumerate(dataset.records):
        image = load_image(image_path); height, width = image.shape
        transform, raw_crop_points = prepare_annotation(annotation, width, height, output_size=dataset.output_size)
        original_points = annotation_points_original(annotation, width, height)
        prepared = dataset[record_index]
        points = prepared["points"].numpy()
        heatmaps = prepared["target"].numpy()
        availability = prepared["availability"].numpy().astype(bool)
        for channel, (original, raw_crop) in enumerate(zip(original_points, raw_crop_points)):
            if original is None:
                continue
            visible_points += 1
            restored = crop_to_image(image_to_crop(original, transform), transform)
            maximum_roundtrip_error = max(maximum_roundtrip_error,
                                          float(np.linalg.norm(np.asarray(restored) - np.asarray(original))))
            x, y = raw_crop
            in_content = (transform.offset_x <= x < transform.offset_x + transform.resized_width and
                          transform.offset_y <= y < transform.offset_y + transform.resized_height)
            if not in_content:
                padding_points += 1
                padding_point_records.append({"record_id": annotation.record_id,
                                              "landmark": LANDMARK_NAMES[channel],
                                              "crop_xy": [x, y],
                                              "content_xyxy": [transform.offset_x, transform.offset_y,
                                                               transform.offset_x + transform.resized_width,
                                                               transform.offset_y + transform.resized_height]})
            if not (0 <= x < dataset.output_size and 0 <= y < dataset.output_size):
                outside.append({"record_id": annotation.record_id, "landmark": LANDMARK_NAMES[channel],
                                "crop_xy": [x, y], "original_xy": list(original),
                                "crop_xyxy": list(transform.crop_xyxy)})
                continue
            peak_y, peak_x = np.unravel_index(int(heatmaps[channel].argmax()), heatmaps[channel].shape)
            positive = int((heatmaps[channel] >= POSITIVE_THRESHOLD).sum())
            stats[LANDMARK_NAMES[channel]]["peak_value"].append(float(heatmaps[channel, peak_y, peak_x]))
            stats[LANDMARK_NAMES[channel]]["positive_pixels"].append(positive)
            stats[LANDMARK_NAMES[channel]]["peak_coordinate_error"].append(
                float(np.linalg.norm(np.asarray((peak_x, peak_y)) - points[channel])))
        if len(visual_candidates) < 4 and int(availability.sum()) >= 5:
            visual_candidates.append((record_index, prepared))
    channel_report = {}
    pixels = dataset.output_size ** 2
    for name in LANDMARK_NAMES:
        positives = stats[name]["positive_pixels"]
        mean_positive = float(np.mean(positives)) if positives else None
        channel_report[name] = {
            "visible_in_crop_count": len(stats[name]["peak_value"]),
            "peak_value": describe(stats[name]["peak_value"]),
            "positive_pixel_definition": f"target >= {POSITIVE_THRESHOLD}",
            "positive_pixels": describe(positives),
            "mean_background_to_positive_ratio": ((pixels - mean_positive) / mean_positive
                                                   if mean_positive else None),
            "heatmap_peak_to_expected_coordinate_error_pixels": describe(
                stats[name]["peak_coordinate_error"]),
        }
    coordinate_report = {"visible_landmarks": visible_points,
                         "landmarks_in_letterbox_padding": padding_points,
                         "letterbox_padding_landmark_records": padding_point_records,
                         "maximum_forward_inverse_roundtrip_error_pixels": maximum_roundtrip_error,
                         "out_of_crop_landmarks": outside,
                         "xy_convention": "points are (x,y); arrays and OpenCV indexing are [y,x]"}
    return channel_report, coordinate_report, visual_candidates


def render_ground_truth(dataset: PerioCropDataset, candidates: list[tuple[int, dict]]) -> list[str]:
    paths = []
    for number, (_, sample) in enumerate(candidates, 1):
        image = (sample["image"][0].numpy() * 255).astype(np.uint8)
        target = sample["target"].numpy(); points = sample["points"].numpy()
        combined = np.clip(target.max(axis=0) * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(combined, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), .65, color, .35, 0)
        for channel, available in enumerate(sample["availability"].bool().tolist()):
            if not available:
                continue
            y, x = np.unravel_index(int(target[channel].argmax()), target[channel].shape)
            expected = tuple(int(round(value)) for value in points[channel].tolist())
            cv2.circle(overlay, expected, 4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(overlay, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 7, 1)
        path = OUTPUT / "ground_truth_examples" / f"{number:02d}_{sample['record_id'].replace(':', '_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), overlay)
        paths.append(str(path.relative_to(OUTPUT.parents[2])).replace("\\", "/"))
    return paths


def output_and_loss_audit(dataset: PerioCropDataset) -> dict:
    checkpoint = torch.load(DEFAULT_OUTPUT / "best.pt", map_location="cpu", weights_only=False)
    model = PerioLandmarkNet(11); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    logits_values, probabilities_values = [], []
    boundary_argmax, padding_argmax, decoded = 0, 0, 0
    prediction_roundtrip_max = 0.0
    padding_loss, content_loss = 0.0, 0.0
    padding_elements, content_elements = 0, 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images).float(); probabilities = logits.sigmoid()
            target = batch["target"].to(device); available = batch["availability"].to(device).bool()
            points, _ = decode_heatmaps(probabilities)
            logits_values.append(logits.cpu().flatten()); probabilities_values.append(probabilities.cpu().flatten())
            errors = (probabilities - target).square()
            for item in range(images.shape[0]):
                ox = int(batch["transform"]["offset_x"][item]); oy = int(batch["transform"]["offset_y"][item])
                rw = int(batch["transform"]["resized_width"][item]); rh = int(batch["transform"]["resized_height"][item])
                content = torch.zeros((256, 256), dtype=torch.bool, device=device); content[oy:oy+rh, ox:ox+rw] = True
                for channel in torch.where(available[item])[0].tolist():
                    x, y = (int(value) for value in points[item, channel].tolist()); decoded += 1
                    scale = float(batch["transform"]["scale"][item])
                    x1 = float(batch["transform"]["crop_xyxy"][0][item])
                    y1 = float(batch["transform"]["crop_xyxy"][1][item])
                    original_x, original_y = (x - ox) / scale + x1, (y - oy) / scale + y1
                    restored_x = (original_x - x1) * scale + ox
                    restored_y = (original_y - y1) * scale + oy
                    prediction_roundtrip_max = max(prediction_roundtrip_max,
                                                   float(np.hypot(restored_x - x, restored_y - y)))
                    boundary_argmax += int(x <= 3 or x >= 252 or y <= 3 or y >= 252)
                    padding_argmax += int(not bool(content[y, x]))
                    padding_loss += float(errors[item, channel][~content].sum()); padding_elements += int((~content).sum())
                    content_loss += float(errors[item, channel][content].sum()); content_elements += int(content.sum())
    logits = torch.cat(logits_values); probabilities = torch.cat(probabilities_values)
    return {
        "logits": {"minimum": float(logits.min()), "mean": float(logits.mean()),
                   "maximum": float(logits.max()), "std": float(logits.std())},
        "after_sigmoid": {"minimum": float(probabilities.min()), "mean": float(probabilities.mean()),
                          "maximum": float(probabilities.max()), "std": float(probabilities.std())},
        "decoded_visible_channels": decoded,
        "argmax_within_four_pixels_of_canvas_boundary": boundary_argmax,
        "argmax_in_letterbox_padding": padding_argmax,
        "decoded_prediction_crop_original_crop_roundtrip_max_pixels": prediction_roundtrip_max,
        "padding_fraction_of_evaluated_pixels": padding_elements / (padding_elements + content_elements),
        "padding_mse_per_pixel": padding_loss / padding_elements,
        "content_mse_per_pixel": content_loss / content_elements,
        "padding_fraction_of_total_squared_error": padding_loss / (padding_loss + content_loss),
        "decoder": "sigmoid per channel, flattened spatial argmax, x=index%width, y=index//width",
    }


def main() -> None:
    seed_everything(42); OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold"] == 0)
    dataset = PerioCropDataset(DEFAULT_MANIFEST, fold["validation_image_ids"], 256)
    channels, coordinates, candidates = ground_truth_audit(dataset)
    report = {
        "scope": "Gate 2.1 read-only diagnostic; no training or mutation",
        "ground_truth_heatmaps": channels, "coordinates": coordinates,
        "current_objective": {
            "activation": "sigmoid(logits)",
            "loss": "sum(mask * (sigmoid(logit)-target)^2) / (visible_channel_count * H * W)",
            "channel_weighting": "equal weight per available landmark channel; visibility 0 excluded",
            "spatial_weighting": "uniform; target background and letterbox padding are currently included",
            "gaussian_sigma_pixels": 3.0,
        },
        "model_outputs_and_loss": output_and_loss_audit(dataset),
        "ground_truth_visuals": render_ground_truth(dataset, candidates),
    }
    (OUTPUT / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
