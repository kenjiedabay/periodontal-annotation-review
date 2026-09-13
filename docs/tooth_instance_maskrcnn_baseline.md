# Tooth-instance Mask R-CNN baseline

The baseline is implemented in `backend/app/tooth_instance_baseline/` and is strictly a one-class structural segmentation experiment. Its only class is `tooth`; it uses no periodontal-disease, severity, FDI, Arch, Site, CEJ/apex, or bone-line targets.

Run from `backend/`:

```text
python -m app.tooth_instance_baseline.train --dataset-root "../DenPAR Radiographs Dataset/Dataset" --output-dir "../models/tooth_instance_maskrcnn_baseline"
```

The loader validates nonempty binary masks, dimensions, image availability, and mask-derived boxes before admitting any record. It uses a recorded longest-side resize with bilinear image and nearest-mask interpolation, a deterministic optional horizontal flip, Mask R-CNN's native RPN/classification/box/mask losses, validation Dice for early stopping, `ReduceLROnPlateau`, best-checkpoint metadata, test Dice/IoU/precision/recall, instance matching at IoU 0.50, per-image output, and qualitative overlays.

## Current execution status

Training is intentionally blocked by the supplied files. `Training/Masks (Tooth-wise)` has 650 annotated image folders but no matching Training radiographs; no substitutions or inferred image pairing are permitted. Validation has 150 usable image-mask records (655 masks) and Testing has 200 usable records (864 masks), but neither may be repurposed as training data under a fabricated split.

The pipeline writes `models/tooth_instance_maskrcnn_baseline/dataset_validation_report.json` and a blocked `evaluation_report.json`. It will begin training only after the missing training radiographs are supplied with their matching IDs. Even then, the official partitions are not proven patient-disjoint because patient/case IDs are unavailable; results remain research-only and non-clinical.
