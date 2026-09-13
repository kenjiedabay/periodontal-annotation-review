"""Metadata records for reproducible preprocessing."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ImageMetadata:
    image_id: str
    original_size: str
    processed_size: str
    operations: list[str]
    normalization: str
    preprocessing_version: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def write_metadata(metadata: ImageMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.as_dict(), indent=2), encoding="utf-8")
