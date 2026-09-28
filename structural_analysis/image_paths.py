"""Shared original-image resolution without moving files or changing partitions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_image(split, image_id, root=ROOT):
    if split not in {'Training', 'Validation', 'Testing'} or not str(image_id).isdigit():
        raise ValueError('Invalid official split or image ID')
    config = root / 'structural_analysis' / 'image_paths.json'
    if config.exists():
        directories = json.loads(config.read_text())
        directory = (root / directories[split]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError('Image directory escapes repository')
        return directory / f'{image_id}.jpg'
    # Isolated test fixtures and older standalone datasets.
    source = root / 'DenPAR Radiographs Dataset' / 'Dataset'
    path = source / split / 'Images' / f'{image_id}.jpg'
    return path if path.exists() else source / 'Images' / f'{image_id}.jpg'
