"""Configurable severity assessment parameters."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeverityConfig:
    manifest_path: Path = Path("../dataset/manifests/dataset_manifest.json")
    dataset_root: Path = Path("../dataset")
    output_dir: Path = Path("ml/checkpoints/severity")
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 7
    min_delta: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    pretrained: bool = True
    freeze_backbone_epochs: int = 1
    minimum_samples_per_class: int = 5
    allowed_labels: tuple[str, ...] = ()

    def validate(self) -> "SeverityConfig":
        if self.image_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("image_size, batch_size, and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
        if self.patience < 1 or self.minimum_samples_per_class < 1:
            raise ValueError("patience and minimum_samples_per_class must be positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["allowed_labels"] = list(self.allowed_labels)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
