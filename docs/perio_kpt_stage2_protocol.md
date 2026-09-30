# Perio-KPT Stage 2 protocol

## Scope

This protocol keeps DenPAR and Perio-KPT separate. The locked DenPAR Mask R-CNN remains the Stage 1 tooth-instance model. Perio-KPT supplies Stage 2 tooth-level periodontal landmarks. The models connect only during inference and connected-pipeline evaluation.

Gate 1 is CPU-only dataset validation and deterministic geometry preparation. It does not train a model, modify source data, modify the DenPAR checkpoint, or change an API contract.

## Source and schema

The canonical source is `datasets/sources/perio-kpt/extracted/Periodontal_Keypoint_Dataset`. Training uses only `1_Experiment/standard_box`. The `0_Baseline` directory mixes detection-only and pose rows and is excluded. The rotating-box representation is not used.

A standard pose row contains an object class, normalized center-width-height box, and 11 `(x, y, visibility)` keypoint triplets. The frozen channel order is CEJ-m, BL-m, RL-m, CEJ-d, BL-d, RL-d, RL-c, FA, FBL-m, FBL-d, ARR. Visibility zero means unavailable/not applicable and is excluded from heatmap loss. Visibility one and two are retained as annotated points and must be reported separately during later model evaluation.

The object classes are single-root, double-root, triple-root, ARR region, and PLS region. Tooth classes 0-2 form the connected Mask R-CNN pipeline. ARR can be studied with ground-truth region crops but cannot be proposed by a tooth-only detector. A PLS row without an available landmark supplies no heatmap supervision.

## Frozen splitting

The supplied five folds are preserved. Each contains 140 training and 35 validation radiographs. All crops from one radiograph remain in its radiograph split. The separate 18-image holdout is not used for model or confidence selection. Patient identifiers are unavailable, so patient-level independence cannot be verified and must be reported as a limitation.

Every source image and label in the frozen manifest has a SHA-256 digest. Fold copies must match their canonical image and label bytes. Any malformed record is quarantined rather than repaired. The third row of the holdout `Image24.txt` has 41 values instead of the required 38. `Image190` object 2 also has a box outside normalized image bounds. Both rows are excluded and their source files are preserved unchanged.

## Crop geometry

The normalized ground-truth box is converted to original-image `xyxy` coordinates. Twenty percent of the original box width is added to both horizontal sides and twenty percent of its height to both vertical sides. The expanded box is clipped to image bounds. This intentionally preserves surrounding alveolar bone.

The expanded crop is mapped to a 256x256 canvas with aspect-ratio-preserving resize and centered zero padding. Images must not be stretched to a square. Every transformation stores source dimensions, expanded `xyxy`, scale, resized dimensions, and offsets. Forward and inverse point transforms are tested for numerical round-trip consistency.

## Ground-truth heatmaps

Each prepared object has 11 Gaussian heatmap channels. A visible/annotated point receives a Gaussian target. An unavailable point has an empty channel and availability value zero. The future loss must multiply each channel loss by this availability mask so a missing landmark is not treated as a negative point.

## Ground-truth RBL

RBL is calculated independently by surface when CEJ, bone level, and the appropriate root landmark are all available:

`RBL% = 100 * distance(CEJ, bone level) / distance(CEJ, root landmark)`

For single-root teeth, both surfaces use the central root landmark (`RL-c`) as the apex. Double- and triple-root teeth use the matching surface root landmark (`RL-m` or `RL-d`).

The calculation is not assessable when a required landmark is missing, the root vector is too short or non-finite, or the bone-level projection lies before the CEJ or implausibly beyond the root landmark. These are radiographic geometric measurements only. They are not CAL, probing depth, a definitive diagnosis, or a treatment recommendation.

## Later training gates

Gate 2 will be a one-epoch smoke test using a pretrained ResNet-18 encoder and lightweight U-Net-style 11-channel decoder, input size 256, batch size 4, two workers, AMP, and thermal monitoring. Gate 2 must not begin without explicit approval after review of Gate 1.

After Gate 2 implementation is explicitly approved, its smoke test will be invoked from the repository root with:

```powershell
Set-Location backend
& '.\.venv\Scripts\python.exe' -m app.perio_kpt_landmarks.train --mode smoke --fold 0 --epochs 1 --image-size 256 --batch-size 4 --workers 2 --amp --thermal-log-seconds 30 --thermal-pause-c 75 --thermal-resume-c 68 --thermal-stop-c 80
```

The training module is intentionally absent at Gate 1, so this command is recorded for approval and is not yet runnable.

A later pilot will use one fold only. Full five-fold training will require separate confirmation and will run one job at a time. The locked holdout will remain untouched until the model, confidence policy, RBL validity rules, and malformed-row policy are frozen.

## Intended inference wording

The connected research output will use a temporary detected-tooth index, mesial and distal RBL when assessable, the highest estimated surface, and acceptable/uncertain landmark confidence. Patient-facing wording is limited to "bone loss may be present" and "clinical periodontal examination required." No FDI number is assigned without a validated numbering model.
