# BRAR supplementary severity experiment

BRAR is used only for **patient-level panoramic bone-resorption grading**. It does not provide localized disease targets or treatment labels. The experiment must therefore be reported as supplementary evidence, separate from DenPAR structural localization.

## Step 1: audit and freeze splits

From `backend/`:

```powershell
python -m app.brar_experiment.prepare
```

This verifies the release, rejects exact duplicate images, creates `dataset/brar/brar_audit.json`, and freezes a seeded 70/15/15 class-stratified manifest. The exported manifest deliberately excludes `Bone resorption` and `Bone resorption Age`, because the target `Level` is derived from them.

## Step 2: image-only baseline

```powershell
python -m app.brar_experiment.train --mode image --output-dir ../models/brar_image
```

This is the primary BRAR experiment. Report macro F1, per-class recall, confusion matrix, and ordinal mean absolute error from the locked test split.

## Step 3: safe-metadata comparison

```powershell
python -m app.brar_experiment.train --mode multimodal --output-dir ../models/brar_multimodal
```

The permitted metadata are age, gender, missing-teeth count, implant count, residual-root count, and functional-tooth-pair count. Interpret improvements cautiously: these variables can encode age and accumulated dental status rather than localized periodontal destruction.

## Step 4: interpretation boundary

Compare image-only and multimodal results on the identical frozen split. Do not describe either model as localizing periodontal disease, recommending treatment, predicting progression, or providing clinical validation. DenPAR remains the structural-localization source; expert per-tooth disease and treatment labels remain necessary for the thesis's full target.

## Completed baseline results (seed 42, 20 epochs)

The release audit found 988 complete image/metadata pairs and no exact duplicate images. The frozen stratified split contains 690 training, 149 validation, and 149 locked testing records.

| Model | Test accuracy | Test macro F1 | Ordinal MAE |
|---|---:|---:|---:|
| Image only | 0.564 | 0.511 | 0.483 |
| Image + safe metadata | 0.550 | 0.486 | 0.470 |

The image-only model is the preferred primary baseline because it has higher accuracy and macro F1. Safe metadata did not improve overall classification, although its ordinal error was slightly smaller. This is a single seeded holdout experiment, not a confidence interval or evidence of clinical performance. Repeated stratified cross-validation and external validation would be required for a stronger comparison.

The image-only per-class recalls were 0.500 (Level 1), 0.667 (Level 2), and 0.385 (Level 3). The multimodal per-class recalls were 0.500, 0.690, and 0.282 respectively. Both models struggled particularly with Level 3, so accuracy alone would overstate performance.

Artifacts are stored in `models/brar_image/` and `models/brar_multimodal/`. Each directory contains the validation-selected checkpoint, full epoch history, and locked-test summary.

## Five-fold stability and negative controls

A separate five-fold stratified screen uses fixed ImageNet ResNet-18 embeddings. It is a stability/proxy-signal analysis, not repeated end-to-end fine-tuning. Fold-local normalization prevents held-out information from entering feature scaling.

| Input | Accuracy, mean ± SD | Macro F1, mean ± SD | Ordinal MAE, mean ± SD |
|---|---:|---:|---:|
| Frozen image features | 0.491 ± 0.034 | 0.431 ± 0.040 | 0.555 ± 0.032 |
| Age only | 0.531 ± 0.031 | 0.439 ± 0.049 | 0.557 ± 0.049 |
| Safe metadata | 0.549 ± 0.022 | 0.527 ± 0.023 | 0.534 ± 0.031 |

Safe metadata outperformed frozen image features, and age alone achieved a similar macro F1 to the image representation. This is evidence of substantial proxy signal in the BRAR target. It does not prove that the end-to-end image model uses age, but it requires conservative interpretation: BRAR performance must not be treated as proof that a model learned localized bone-loss measurement.

Level 3 remained difficult: mean recall was 0.353 for frozen image features, 0.131 for age only, and 0.430 for safe metadata. Full fold results are stored in `models/brar_validation/cross_validation.json`.

## Grad-CAM review

One correctly classified locked-test example per level was exported to `models/brar_image/gradcam/`. The maps are broad and inconsistent: the Level 1 example emphasizes much of the lower jaw, the Level 2 example emphasizes a broad central dental region, and the Level 3 example emphasizes substantial non-dental upper-image anatomy. These maps do not establish localized periodontal reasoning. They instead reinforce the need for expert-reviewed tooth/site labels and localization-specific evaluation. Grad-CAM is qualitative and must not be presented as a validated lesion map.
