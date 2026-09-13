"""Configuration for conservative radiograph preprocessing."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    preprocessing_version: str = "1.0.0"
    output_format: str = "PNG"
    grayscale: bool = True
    target_width: int = 1024
    target_height: int = 1024
    preserve_aspect_ratio: bool = True
    normalization_method: str = "percentile"
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    contrast_enhancement: bool = True
    clahe_clip_limit: float = 1.5
    clahe_tile_grid_size: int = 8
    noise_reduction: bool = False
    noise_reduction_kernel: int = 3
    sharpening: bool = False
    sharpening_amount: float = 0.25
    create_comparisons: bool = True

    def validate(self) -> "PreprocessingConfig":
        if self.output_format.upper() != "PNG":
            raise ValueError("Only PNG output is supported by the current pipeline")
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("Target dimensions must be positive")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
        if self.clahe_clip_limit <= 0 or self.clahe_tile_grid_size <= 0:
            raise ValueError("CLAHE settings must be positive")
        if self.noise_reduction_kernel < 1 or self.noise_reduction_kernel % 2 == 0:
            raise ValueError("Noise reduction kernel must be a positive odd number")
        if self.sharpening_amount < 0:
            raise ValueError("Sharpening amount cannot be negative")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | None = None) -> PreprocessingConfig:
    """Load JSON configuration, falling back to conservative defaults."""
    config = PreprocessingConfig()
    if path is not None:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Preprocessing config must be a JSON object")
        config = PreprocessingConfig(**{**config.as_dict(), **values})
    return config.validate()
