# Validation missed-tooth review

This review compares the final, equal-step supervised control and pseudo-label student on all 150 official DenPAR Validation radiographs (655 reference tooth masks). It uses the experiment's matching rule: predicted mask pixels >= 0.50 and one-to-one instance match at IoU >= 0.50. Torchvision returns candidate detections down to its default score threshold; this is **not** the app's 0.50 confidence display threshold.

## Findings

| Reference teeth missed | Count |
|---|---:|
| Missed by both models | 20 |
| Missed only by pseudo-label student | 1 |
| Missed only by control | 0 |

The student-only miss is **image 574, mask2**. The control overlaps it at IoU 0.624, but with confidence 0.082; the student has no corresponding overlapping instance. At an app display threshold of 0.50, the control's low-confidence detection would also be hidden. The image has closely spaced, horizontally oriented annotated regions; the second and third masks need source review before calling this purely a model error. See [case_574.png](case_574.png).

Among the student's 21 misses:

- 13 have no candidate with IoU >= 0.05; 8 have a candidate with partial overlap below 0.50. This means lowering a confidence threshold alone cannot recover most misses.
- 5 masks have fewer than 5,000 source pixels, versus 6 of all 655 validation masks. All 3 masks below 1,000 pixels were missed. Small fragments are strongly overrepresented.
- 7 masks touch within 5 pixels of a source image border, versus 70 of all 655 masks. Partial teeth at image edges are overrepresented.
- 6 reference masks have multiple disconnected foreground components after preprocessing. These require annotation review before attributing the disagreement solely to the model.
- Two misses are borderline matches: image 899 mask5 has student best IoU 0.490, and image 935 mask3 has 0.489. Their scores depend on the IoU 0.50 matching rule, but changing that rule would change the evaluation rather than the segmentation.

The [case 31 overlay](case_31.png) shows two very small reference fragments at the right edge that neither model separates. The [case 27 overlay](case_27.png) shows a partial tooth near the right edge. The [case 584 overlay](case_584.png) shows missed masks and uncertain source-mask geometry, consistent with the earlier single-case review. These images are visual evidence for annotation and image-edge review, not clinical judgments about tooth identity or disease.

## Next action

Have an annotator review the 21 missed reference masks, starting with the five tiny missed masks, the seven border-touching masks, image 574, and the disconnected masks. Record whether each mask is a valid visible tooth instance, a fragment that should count under the study's labeling policy, or a mask needing correction. Then run a validation-only confidence sweep and a targeted training experiment for confirmed small or edge teeth. Do not use Testing to choose the threshold or revise labels.

Full per-tooth measurements, including overlap, score, area, and component count, are in [miss_analysis.json](miss_analysis.json).
