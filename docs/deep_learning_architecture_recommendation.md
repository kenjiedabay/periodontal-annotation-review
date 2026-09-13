# Deep-learning architecture recommendation

Study: **Predictive Modeling of Localized Periodontal Disease for Oral Assessment and Treatment Planning**

## Evidence from the prepared dataset

The current prepared dataset was analyzed from `dataset/processed/`, `dataset/annotations/`, and `dataset/manifests/`:

- Processed radiographs: **650**
- Image dimensions: **1024 x 1024** for all 650 images
- Expert-validated annotations: **0**
- Training images: **0**
- Validation images: **0**
- Test images: **0**
- Disease-positive, disease-negative, and uncertain counts: **not available**
- Affected-tooth labels: **not available**
- Severity labels: **not available**
- Region masks or bounding boxes: **not available**
- Patient/case grouping: **not available**

These results mean that no supervised deep-learning model can currently be trained or evaluated without importing expert annotations. The recommendations below are design recommendations for the next labeled dataset stage, not measured results and not clinical claims.

## Tasks currently supported

| Task | Supported now? | Reason |
|---|---:|---|
| Disease detection | No | No validated disease labels are present. |
| Affected tooth/region localization | No | No validated tooth IDs tied to spatial regions are present. |
| Severity assessment | No | No validated severity labels are present. |
| Structural analysis | No supervised task yet | No validated structural masks, landmarks, or measurements are present. |
| Progression estimation | No | The current data are not longitudinal and contain no validated outcome/time-to-event labels. |
| Dataset inventory and preprocessing QA | Yes | Image dimensions and file-level preparation metadata are available. |

The annotation model can represent disease status, FDI teeth, severity, findings, and polygon/freehand/bounding-box regions. Once expert-validated records exist, those fields support the conditional tasks described below.

## Recommended development order

1. **Disease detection** as a patient/case-level or radiograph-level baseline, after class labels and case groups are available.
2. **Affected region localization** using the annotation geometry actually collected: segmentation for polygons, detection for bounding boxes.
3. **Severity assessment** as an ordinal task only after sufficient, consistently defined expert labels exist.
4. **Structural analysis** as a separate segmentation or landmark task, not as an implicit disease label.
5. **Progression estimation** only after longitudinal follow-up images and expert-validated outcomes are collected.

Do not combine all tasks into one multi-head model at the beginning. A modular pipeline makes label requirements, errors, and expert review easier to inspect.

## Architecture recommendations

### 1. Disease detection

**Recommended primary model: DenseNet-121 with transfer learning**

- **Reason for selection:** DenseNet-121 is a reasonable small-to-medium medical-image baseline for limited labeled data because feature reuse can provide strong representation efficiency. Use ImageNet initialization, replace the classifier, and fine-tune conservatively.
- **Expected input:** One 1024×1024 processed radiograph, preferably resized/cropped to a model input such as 224×224 or 320×320. Preserve the original image and document the resampling. A grayscale image can be repeated to three channels for pretrained weights.
- **Expected output:** Probability for `present`, `absent`, and `uncertain`, or a binary output only if the study protocol explicitly excludes uncertain cases. Do not silently discard uncertain labels.
- **Training requirements:** Expert-validated labels, case-level grouping, stratified group splits where possible, class weights or a sampler if imbalance is observed, conservative training augmentation, and a frozen-backbone warm-up followed by limited fine-tuning.
- **Evaluation metrics:** Sensitivity/recall, specificity, balanced accuracy, macro-F1, AUROC, AUPRC, confusion matrix, calibration/Brier score, and confidence intervals at the case level. Report the operating threshold and subgroup counts.
- **Advantages:** Transfer learning is widely available; architecture is moderate in size; feature reuse can help when labels are limited; Grad-CAM-style review may provide a useful inspection aid.
- **Limitations:** A classification heatmap is not a validated lesion boundary; image-level labels do not prove tooth-level localization; 650 images may still be too few once split by case and class.

**Alternative model: ResNet-18 or ResNet-34 with transfer learning**

Use this when compute is limited or a simpler, highly reproducible baseline is preferred. ResNet-18 is especially appropriate as the first baseline. It is easier to tune and less likely to overfit than a larger model, but may represent subtle radiographic patterns less richly.

**Baseline model: Logistic regression or linear SVM on frozen CNN embeddings**

Use frozen ResNet-18/DenseNet features with a linear classifier, plus a non-neural intensity/texture baseline if needed. This establishes whether the dataset contains a learnable signal before investing in end-to-end fine-tuning.

### 2. Bounding-box region localization

**Recommended primary model: Faster R-CNN with a ResNet-50-FPN backbone**

- **Reason for selection:** It is a non-YOLO, established two-stage detector suitable when expert annotations are bounding boxes and region precision matters more than maximum throughput.
- **Expected input:** A processed radiograph, with one or more expert bounding boxes and class labels such as `affected_region` or an explicitly defined tooth/region category.
- **Expected output:** Bounding boxes, class scores, and confidence scores.
- **Training requirements:** A sufficient number of box annotations, consistent box policy, case-grouped splits, multi-scale handling, and careful review of false positives. Start with a frozen or partially frozen backbone.
- **Evaluation metrics:** Box IoU, AP50, AP50:95, recall, precision, and per-tooth/per-region performance where sample counts permit.
- **Advantages:** Strong tooling, interpretable box output, compatible with sparse object annotations, no need to pretend polygon precision exists.
- **Limitations:** Boxes include non-target tissue; performance is sensitive to annotation consistency; a small dataset may overfit; it does not provide anatomical segmentation.

**Alternative model: RetinaNet with ResNet-50-FPN**

A one-stage non-YOLO alternative with focal loss, useful when positive regions are sparse and inference simplicity matters. It generally requires careful tuning and enough boxes to estimate performance reliably.

**Baseline model: Connected-component or rule-based region proposals plus a crop classifier**

This is not a clinical model, but it can establish whether a region proposal/crop workflow is technically viable before training a detector.

### 3. Polygon/freehand region segmentation

**Recommended primary model: DeepLabV3+ with a ResNet-50 or MobileNetV3 backbone**

- **Reason for selection:** This task is justified only when polygon/freehand annotations are converted into reviewed raster masks. DeepLabV3+ provides multi-scale context and boundary-aware decoding without assuming bounding-box labels.
- **Expected input:** A processed radiograph and a binary or multi-class mask rasterized from expert polygons. Keep the rasterization rule and coordinate convention in the manifest.
- **Expected output:** Per-pixel region mask and confidence map.
- **Training requirements:** Enough reviewed masks, consistent boundary policy, class-imbalance handling for foreground/background, spatially safe augmentation, and validation at the case level.
- **Evaluation metrics:** Dice, IoU/Jaccard, precision, recall, 95th-percentile Hausdorff distance where appropriate, boundary F-score, and per-region performance.
- **Advantages:** Better suited than detection when region shape matters; multi-scale context can help with irregular structures; transfer learning is available.
- **Limitations:** Polygon quality and rasterization dominate performance; small foreground regions create severe imbalance; pixel masks do not establish a clinical diagnosis.

**Alternative model: U-Net with a pretrained encoder**

U-Net is justified here only if reviewed polygons are converted into reliable masks and segmentation is truly the target. It is a strong, understandable reference model and often easier to train on small medical datasets, but it should be compared rather than assumed to be best.

**Baseline model: threshold/region proposal baseline or shallow encoder-decoder**

Use a simple mask baseline and report it transparently. This is valuable for checking whether a neural model exceeds a non-learned reference.

### 4. Severity assessment

**Recommended primary model: DenseNet-121 or ResNet-34 ordinal classifier**

- **Reason for selection:** Severity categories have an order. An ordinal objective is preferable to treating mild, moderate, and severe as unrelated nominal classes, provided the experts agree that the categories are ordered for this study.
- **Expected input:** Full radiograph or, preferably, an expert-defined affected-tooth/region crop or image plus region mask.
- **Expected output:** Ordered severity probabilities and an ordinal prediction, with `cannot_determine` handled explicitly rather than forced into the ordinal scale.
- **Training requirements:** Enough examples in every severity level, a written labeling protocol, class and case balance checks, and calibration analysis. Do not train this task before label agreement and sample counts are adequate.
- **Evaluation metrics:** Quadratic-weighted Cohen kappa, macro-F1, balanced accuracy, mean absolute error on ordered levels, ordinal calibration, and confusion matrix.
- **Advantages:** Uses the structure of the target categories; moderate transfer-learning cost; can be run on a localized crop rather than the entire image.
- **Limitations:** Severity labels may be subjective; ordinal assumptions may be wrong; the current dataset has zero severity labels; a 3-class model with very small classes will be unstable.

**Alternative model: EfficientNet-B0/B1 transfer-learning classifier**

EfficientNet can provide a compute-efficient comparison, but its additional complexity is not justified until label volume supports it.

**Baseline model: majority-class and ordinal logistic regression**

Always report a majority-class baseline and a simple linear model on frozen embeddings. A deep model must exceed these baselines on a locked test set to be informative.

### 5. Structural analysis

**Recommended primary model: task-specific segmentation or landmark regression, depending on expert annotations**

- Use DeepLabV3+/U-Net-style segmentation only for reviewed structure masks.
- Use heatmap-based landmark regression with a small ResNet backbone only for explicitly annotated landmarks.
- Do not infer structural measurements from disease labels alone.

**Alternative:** A multi-task encoder with separate segmentation and landmark heads, but only after each label type has independent quality checks and enough samples.

**Baseline:** Expert-defined geometric measurements from masks/landmarks or a simple image-processing measurement pipeline.

**Metrics:** Dice/IoU for masks; point error in pixels and normalized image coordinates for landmarks; measurement MAE; agreement with expert measurements.

**Limitation:** The current dataset contains no structural masks or landmarks, so this task is not currently trainable.

### 6. Possible progression estimation

**Recommendation: do not train this task from the current dataset.**

Progression requires longitudinal images linked to the same case, known time intervals, treatment/exposure information where relevant, and a validated outcome definition. Once collected, candidate designs could include a Siamese/temporal CNN for paired images or a longitudinal transformer, but these would require substantially more data and careful leakage control.

A static radiograph classifier cannot establish future progression. The existing placeholder scenario outputs must not be used as training labels.

## Most realistic architecture

**Today:** No supervised deep-learning architecture is realistically trainable from the prepared dataset because there are 650 images but zero expert-validated annotations and zero split records.

**After a first labeling tranche:** Start with **ResNet-18 transfer learning as the disease-detection baseline**, then compare it with **DenseNet-121 as the primary classifier**. This is the most realistic first experiment because it requires image-level labels rather than dense masks, is computationally manageable, and supports a controlled transfer-learning study.

For localization, choose based on the actual annotation geometry:

- Bounding boxes -> Faster R-CNN ResNet-50-FPN.
- Reviewed polygons/freehand regions -> DeepLabV3+ ResNet-50, with pretrained-encoder U-Net as a comparison.

Do not train a detector and call it segmentation, and do not train a segmentation model from bounding boxes without an explicit reviewed conversion protocol.

## Minimum data and protocol gates before training

- Populate expert-validated JSON annotations for a meaningful subset of images.
- Add a trusted patient/case group identifier or a reviewed group-map file before splitting.
- Confirm that every image in a split has exactly one intended annotation record.
- Review disease and severity class counts; do not rely on augmentation to solve severe label scarcity.
- Measure inter-expert agreement on a subset before treating labels as ground truth.
- Keep a locked, case-disjoint test set untouched until final evaluation.
- Record model version, preprocessing version, seed, split manifest, and threshold selection.
- Report uncertainty and confidence intervals; do not report clinical superiority without external validation.
