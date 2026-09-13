# DenPAR tooth-localization training

This workflow uses DenPAR's expert tooth masks to train a U-Net (not YOLO) that localizes tooth regions. It then applies that trained model to the 200 unlabeled radiographs in `Images`.

It does **not** classify periodontal disease, determine severity, or predict progression: those outputs need separate dentist/periodontist-validated labels, which are not present in the supplied folders.

Run from the project root after installing Python 3.10+:

```powershell
python -m pip install -r backend/requirements.txt
python backend/train_tooth_localizer.py --epochs 30
python backend/infer_images.py
```

The best checkpoint is saved to `backend/models/denpar_tooth_unet.pt`; the 200 overlays and their manifest are saved under `backend/outputs/tooth_localization/`. Validation Dice is printed every epoch and is the appropriate localization performance measure for this model.

## Dataset inspection

Run the non-destructive inspection from the `backend` directory:

```powershell
.\.venv\Scripts\python.exe -m app.data_cleaning
```

The command scans `../dataset/raw`, copies valid images to `../dataset/cleaned`, and writes `dataset_quality_report.csv` and `dataset_quality_report.json` under `../dataset/reports`. Raw files are never deleted. Automated checks only flag files for manual review; they do not determine clinical usability or make a diagnosis.

## Image preprocessing

Run the configurable preprocessing pipeline from the `backend` directory:

```powershell
.\.venv\Scripts\python.exe -m app.preprocessing
```

It reads `../dataset/cleaned`, writes standardized PNG images to `../dataset/processed`, and never overwrites cleaned inputs. The default profile is in `app/preprocessing/default_config.json`; pass another JSON file with `--config` to change dimensions or operations without editing Python code.

Each processed image receives a metadata sidecar under `dataset/processed/metadata/`. The pipeline also writes `preprocessing_metadata.json`, `preprocessing_config.json`, and `preprocessing_summary.json`. Side-by-side visual comparisons are saved under `dataset/processed/comparisons/` so researchers can inspect structure preservation. Noise reduction and sharpening are disabled by default and should be enabled only after visual review.

## Annotation storage API

Annotations are validated with Pydantic and stored as one JSON file per image under `dataset/annotations/`:

```text
POST   /annotations
GET    /annotations/{image_id}
PUT    /annotations/{image_id}
DELETE /annotations/{image_id}
```

The stored record contains normalized research labels, affected FDI teeth, findings, region coordinates, expert comments, and an `expert_validated` flag. Updates preserve prior records in `version_history` for auditability. The API rejects unknown fields, and no patient names, addresses, medical record numbers, or other personal identifiers are part of the annotation model.

Example payload:

```json
{
	"image_id": "PERIO-0001",
	"disease_status": "present",
	"affected_teeth": ["36"],
	"severity": "moderate",
	"findings": ["alveolar bone loss"],
	"regions": [{"type": "polygon", "coordinates": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]]}],
	"expert_comment": "",
	"expert_validated": false
}
```

## Dataset analysis

Analyze processed images and annotation files without modifying either input directory:

```powershell
.\.venv\Scripts\python.exe -m app.dataset_analysis
```

The command writes `dataset/analysis_reports/dataset_analysis_records.csv`, `dataset_analysis_summary.csv`, `dataset_analysis_summary.json`, and four PNG charts for disease classes, severity, affected teeth, and annotation completeness. Missing or invalid annotations are reported, not repaired. The JSON includes class-imbalance warnings and feasibility guidance for classification, tooth localization, and severity modeling.

## DenPAR structural audit and viewer

Audit the original Validation annotations without modifying source files:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.structural_audit
```

The report is written to `dataset/reports/denpar_structural_audit.json`. It checks image-to-mask/COCO/key-point/bone/workbook correspondence, dimensions, coordinate bounds, empty masks, polygon validity, duplicates, missing landmarks, and annotation counts. The React shell includes a `Structural audit` button with independent toggles for tooth-wise masks, tooth boxes, CEJ points, apex points, bone lines, and the radiograph-wise mask.

Analyze correspondence separately:

```powershell
.\.venv\Scripts\python.exe -m app.structural_audit.correspondence_cli
```

This writes `dataset/reports/denpar_correspondence_analysis.json`. It confirms only source-proven image/count relationships. It marks mask-to-instance, CEJ-to-box, apex-to-box, bone-line-to-tooth, and FDI-to-instance relationships unresolved when no stable IDs or ordering rules exist. The viewer exposes the same relationships for dental-expert review with statuses `confirmed`, `uncertain`, `missing`, and `not_applicable`.

## ML dataset preparation

Create reproducible train, validation, and test splits from expert-validated annotations:

```powershell
.\.venv\Scripts\python.exe -m app.dataset_preparation
```

The default split is 70% train, 15% validation, and 15% test with seed `42`. Only annotations with `expert_validated: true` and `validation_status: "validated"` are included. Pass `--include-unvalidated` only for an explicitly non-final research experiment.

The command writes images under `dataset/train/`, `dataset/validation/`, and `dataset/test/`, plus CSV/JSON manifests under `dataset/manifests/`. Training images receive one conservative augmentation by default; validation and test images are copied without augmentation. Use `--group-map path.json` to map multiple image IDs to a shared patient/case group before splitting. Without that map, the tool cannot infer patient identity and conservatively treats each image ID as its own group.

## Baseline classification training

The PyTorch baseline is intentionally separate from the preparation step:

```powershell
cd backend
.\.venv\Scripts\python.exe -m ml.train
```

It trains a research-only ResNet-18 transfer-learning classifier for `present` versus `absent`, using only manifest records whose annotation JSON is expert-validated. It performs a leakage check before loading data, uses training-only augmentation, weighted cross-entropy, AdamW, `ReduceLROnPlateau`, validation-F1 checkpointing, early stopping, CSV/JSON logs, and test confusion-matrix metrics. Current execution stops safely because the prepared manifest contains zero expert-validated records; no model is trained or presented as clinically validated.

## Severity assessment

Severity classification is implemented separately and uses only severity labels found in expert-validated annotation JSON files:

```powershell
cd backend
.\.venv\Scripts\python.exe -m ml.severity.train
```

The command reports the observed class distribution before constructing a model. It does not invent `mild`, `moderate`, `severe`, or `cannot_determine`; `cannot_determine` is included only if the dental expert has used it in validated records. Training stops when no validated labels exist, fewer than two observed classes are available, a class has insufficient support, or a required split is empty. The model is a configurable transfer-learning ResNet-18 with weighted loss, validation macro-F1 checkpointing, early stopping, confusion matrix, accuracy, macro precision, macro recall, and macro F1 outputs.
