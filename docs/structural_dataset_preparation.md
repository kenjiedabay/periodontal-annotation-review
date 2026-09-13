# DenPAR structural dataset preparation

Run from `backend/`:

```text
python -m app.structural_dataset_preparation
```

This produces `dataset/structural_prepared/` as a derived dataset. The original DenPAR files are only read.

- Task A contains a binary union tooth mask plus the original per-image tooth-instance mask rasters. `mask1`, `mask2`, and so on remain unpaired local mask identifiers.
- Task B contains COCO tooth boxes, scaled only through the recorded transform.
- Tasks C and D are empty unless `--confirmed-correspondence` is supplied. Neither spatial containment nor equal counts enables a target.
- No disease, severity, FDI, Arch, or Site field is written as a target.
- No case/patient ID is available in the inspected source. The pipeline deliberately writes an `unsplit` pool instead of inventing image-level train/validation/test splits that could leak a case.

By default all images retain their source dimensions. An explicit resize requires both `--resize-width` and `--resize-height`; every manifest record then retains source and output dimensions, scale factors, operation, and interpolation policy. Images use Lanczos; segmentation masks use nearest-neighbor interpolation.

## Optional independent confirmation file

The optional JSON is an independently curated expert record, not inferred from DenPAR. Its top level has an `images` array. Only entries with `status: "confirmed"` are included:

```json
{
  "images": [{
    "image_id": "1002",
    "landmarks": [{"status": "confirmed", "tooth_instance_id": "12", "cej": [100, 200], "apex": [110, 600]}],
    "bone_lines": [{"status": "confirmed", "tooth_instance_id": "12", "anatomical_meaning": "expert-approved description", "points": [[10, 20], [30, 40]]}]
  }]
}
```

The command also writes JSON/CSV manifests, a validation/statistics report, and three box-overlay sample visualizations.
