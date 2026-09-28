# DenPAR tooth instance pseudo-label experiment

The 650 DenPAR Training radiographs in `dataset/raw` match 650 tooth-mask folders and contain 2,882 masks. The existing loader's validation report found no missing or invalid image-mask pairs. The official Validation and Testing partitions contain 150 and 200 radiographs, respectively.

`split_seed42.json` fixes a 325-image labeled subset and a disjoint 325-image hidden-label subset. The split was seeded at 42 and stratified approximately by number of tooth masks per image. Regenerate it with `python scripts/prepare_denpar_pseudolabel_split.py`.

## Completed experiment (seed 42)

The supervised control and teacher were the same newly trained Mask R-CNN ResNet-50 FPN model. It used COCO pretrained weights, the 325 labeled Training images, 6 epochs, and 1,950 optimizer updates. The pseudo-label student used identical initialization and the same 325 labeled images plus 324 hidden-subset images accepted at teacher score >= 0.90. It stopped after 1,950 updates. Both runs used learning rate 0.0001, batch size 1, 1024-pixel preprocessing, and the official Validation set for monitoring. No official Testing images were used for training or threshold choice.

The teacher generated 1,390 pseudo instances. A diagnostic audit performed after generation was frozen found 1,347 matching the hidden reference masks at instance IoU >= 0.50, out of 1,432 reference instances. Mean union Dice against hidden references was 0.9465. These hidden labels did not determine which pseudo-labels were accepted.

The final checkpoints at the matched update budget were evaluated with the same evaluator:

| Split / metric | Supervised control | Pseudo-label student |
|---|---:|---:|
| Validation mean union Dice | 0.9224 | 0.9224 |
| Validation mean union IoU | 0.8608 | 0.8614 |
| Validation missed tooth instances | 20 | 21 |
| Validation false-positive instances | 306 | 263 |
| Test mean union Dice | 0.9234 | 0.9256 |
| Test mean union IoU | 0.8629 | 0.8668 |
| Test missed tooth instances | 24 | 29 |
| Test false-positive instances | 459 | 381 |

The small Dice gain comes with five more missed teeth on Testing. One seed and one split do not establish a reliable improvement. The evaluator uses a 0.50 mask cutoff and IoU 0.50 matching; its false-positive counts include the model's returned instances at Torchvision's default detection score threshold. Test was evaluated once for each final checkpoint after the experiment settings were fixed. See `teacher_validation_final.json`, `student_validation_final.json`, `teacher_test_final.json`, and `student_test_final.json` for the complete aggregate metrics.

## Procedure

1. Train a **new** Mask R-CNN teacher on `teacher_training_ids` only. The existing full-run checkpoint has seen all 650 training masks and would invalidate this experiment.
2. Generate instance-mask predictions for `pseudo_label_candidate_ids`, without reading their reference masks during selection. Filter by confidence and inspect a fixed random sample for missed, merged, and split teeth. Choose the threshold on Validation, not Testing.
3. Use the teacher as the supervised control. Train the pseudo-label student on the same 325 labeled images plus accepted pseudo-labels, matching initialization, preprocessing, and total optimizer updates. Track accepted and rejected images and instances.
4. Compare teacher, control, and student on official Validation with instance matching at IoU 0.50, mean matched-instance IoU, missed instances, false-positive instances, and union Dice/IoU. After the protocol is fixed, evaluate the selected models once on official Testing.
5. For a diagnostic audit only, compare pseudo-labels to the hidden 325 reference masks after pseudo-label generation is frozen. This reveals label noise without letting hidden ground truth influence pseudo-label selection.

Patient identifiers are unavailable, so patient-level separation across official partitions cannot be confirmed. This experiment measures tooth segmentation, not disease detection or clinical bone-loss accuracy.
