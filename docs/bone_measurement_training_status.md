# DenPAR bone-level measurement training status

## Trained components

### Bone-line localization

`python backend/train_bone_lines.py --size 256 --epochs 10 --output models/bone_line_unet_256`

- Training: 639 radiographs; 11 rejected for invalid source line geometry.
- Validation: 147 radiographs; 3 rejected.
- Best checkpoint: epoch 10, threshold 0.70 selected on Validation.
- Validation tolerance precision 0.403, recall 0.819, F1 0.541. The tolerance is scaled from 3 pixels at 128-pixel output to 6 pixels at 256-pixel output.
- Locked Testing: 195 radiographs; precision 0.356, recall 0.786, F1 0.490. Five source records were rejected for invalid line geometry.
- The existing 128-pixel baseline scored 0.521 Validation F1 and 0.479 Testing F1 under its corresponding scaled tolerance. The 256-pixel model is a small localization improvement, but the `test_example.png` overlay shows broad false-positive regions on tooth edges. A tolerant pixel F1 does not establish accurate line coordinates or distances.

Artifacts: `models/bone_line_unet_256/best.pt`, `training_report.json`, `test_report.json`, and `test_example.png`.

### Image-level CEJ and apex localization

`python backend/train_denpar_landmarks.py --epochs 12 --output models/denpar_cej_apex_heatmap_balanced_256`

- Training: 649 radiographs with both point sets; one missing/invalid record excluded.
- Validation: 150 radiographs, with 919 CEJ and 661 apex points.
- Best checkpoint: epoch 11. With local maxima capped at the maximum point counts seen in Training (12 CEJ and 10 apex), the validation-selected threshold was 0.70. At an 8-pixel radius in the 256-pixel canvas, CEJ F1 was 0.321 and apex F1 was 0.083; macro F1 was 0.202.
- This result is too weak for derived bone-level distance measurement. Testing was not evaluated for this landmark model.

Artifacts: `models/denpar_cej_apex_heatmap_balanced_256/best.pt`, `training_report.json`, and `validation_postprocess_report.json`. The earlier `models/denpar_cej_apex_heatmap_256` run collapsed to background and was superseded by the balanced-loss run.

## Tooth-level measurement readiness

The project's `structural_analysis/step4_structural_dataset/structural_dataset_summary.json` currently has **zero** trainable tooth records with confirmed line/CEJ/apex correspondence across Training, Validation, and Testing. DenPAR supplies these geometries at image level but does not directly pair every bone line and point to a tooth mask. A distance or bone-loss percentage calculated from unconfirmed nearest-neighbor matches would be a heuristic target, not validated measurement ground truth.

The existing review tools under `structural_analysis/step2_boneline_tooth_correspondence/` and `step3_cej_apex_correspondence/` are the route to confirmed pairings. From the repository root, start the bone-line review with `python structural_analysis/step2_boneline_tooth_correspondence/serve.py` and the CEJ/apex review with `python structural_analysis/step3_cej_apex_correspondence/serve.py`. Their READMEs explain how decisions are saved. Once a meaningful reviewed subset has all required structures, define a per-tooth measurement protocol, calculate reference distances or ratios from confirmed source geometry, then train and validate a measurement model. Do not interpret the current line or point predictions as disease diagnosis, severity, or clinical bone-loss measurements.
