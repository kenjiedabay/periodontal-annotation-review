"""Small, conservative image transforms used by the preprocessing pipeline."""

import cv2
import numpy as np

from .config import PreprocessingConfig


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize(image: np.ndarray, config: PreprocessingConfig) -> tuple[np.ndarray, str]:
    target_size = (config.target_width, config.target_height)
    height, width = image.shape[:2]
    if not config.preserve_aspect_ratio:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA), f"resized_to:{config.target_width}x{config.target_height}"

    scale = min(config.target_width / width, config.target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas_shape = (config.target_height, config.target_width) + resized.shape[2:]
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    offset_x = (config.target_width - resized_width) // 2
    offset_y = (config.target_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas, f"resized_with_aspect_ratio_to:{config.target_width}x{config.target_height}"


def _normalize(image: np.ndarray, config: PreprocessingConfig) -> tuple[np.ndarray, str]:
    if config.normalization_method == "none":
        return image, "none"
    if config.normalization_method == "minmax":
        lower, upper = np.min(image, axis=(0, 1)), np.max(image, axis=(0, 1))
        method = "minmax"
    elif config.normalization_method == "percentile":
        lower = np.percentile(image, config.lower_percentile, axis=(0, 1))
        upper = np.percentile(image, config.upper_percentile, axis=(0, 1))
        method = f"percentile:{config.lower_percentile:g}-{config.upper_percentile:g}"
    else:
        raise ValueError(f"Unsupported normalization method: {config.normalization_method}")
    if np.all(upper <= lower):
        return np.zeros_like(image), f"{method}:flat_input"
    safe_range = np.where(upper > lower, upper - lower, 1)
    normalized = np.clip((image.astype(np.float32) - lower) * 255.0 / safe_range, 0, 255)
    return normalized.astype(np.uint8), method


def apply_transforms(image: np.ndarray, config: PreprocessingConfig) -> tuple[np.ndarray, list[str], str]:
    """Apply configured transforms and return image, operation log, and normalization label."""
    operations: list[str] = []
    working = image
    if config.grayscale and working.ndim != 2:
        working = _to_grayscale(working)
        operations.append("grayscale")
    elif config.grayscale:
        operations.append("grayscale_already")

    working, resize_operation = _resize(working, config)
    operations.append(resize_operation)
    working, normalization = _normalize(working, config)
    if normalization != "none":
        operations.append("pixel_intensity_normalization")

    if config.contrast_enhancement:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )
        if working.ndim == 2:
            working = clahe.apply(working)
        else:
            lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            working = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        operations.append(f"clahe_clip_limit:{config.clahe_clip_limit:g}")

    if config.noise_reduction:
        working = cv2.medianBlur(working, config.noise_reduction_kernel)
        operations.append(f"median_noise_reduction_kernel:{config.noise_reduction_kernel}")

    if config.sharpening:
        blurred = cv2.GaussianBlur(working, (0, 0), 1.0)
        working = cv2.addWeighted(working, 1.0 + config.sharpening_amount, blurred, -config.sharpening_amount, 0)
        operations.append(f"unsharp_mask_amount:{config.sharpening_amount:g}")

    return working, operations, normalization
