# DenPAR training-radiograph provenance audit

## Result

**STATE A — ORIGINAL TRAINING RADIOGRAPHS FOUND.**

The 650 original Training radiographs are present in `dataset/raw/`, not in `DenPAR Radiographs Dataset/Dataset/Training/Images/`. Each raw JPG has the same image ID as exactly one Training tooth-wise mask folder, and its pixel dimensions match the corresponding masks. `dataset/cleaned/` contains checksum-identical copies. Each `dataset/processed/<id>.png` was then reproduced pixel-for-pixel from the raw source using the versioned preprocessing implementation.

This establishes a legitimate derived training route using either:

- `dataset/raw/<id>.jpg` with the original Training tooth-wise masks, or
- `dataset/processed/<id>.png` with masks transformed by the recorded aspect-ratio resize-and-center-padding transform.

The source files remain unchanged.

## Exact transformation

`backend/app/preprocessing/transforms.py::_resize` applies:

1. grayscale conversion;
2. scale `min(1024 / source_width, 1024 / source_height)`;
3. rounded scaled dimensions;
4. centered zero padding to 1024×1024;
5. percentile normalization (1–99) and CLAHE (clip limit 1.5).

For masks, use nearest-neighbor interpolation with the same scale and integer offsets. The audit generated 12 representative overlays; all were non-empty and within the processed image bounds.

## Limits that remain

The official Training, Validation, and Testing partitions were kept separate. Patient/case identifiers were not available in the supplied dataset metadata; therefore, patient-level independence of the official partitions could not be independently verified. This is a research-only tooth-structure dataset, not a periodontal-disease, severity, progression, or clinical-performance dataset.

The generated machine-readable evidence is in [training_readiness_report.json](../models/tooth_instance_maskrcnn_baseline/training_readiness_report.json), with detailed manifests and QA images in the same directory.
