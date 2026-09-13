# DenPAR annotation audit

## Scope and method

This audit inspected the actual DenPAR Validation files under:

```text
DenPAR Radiographs Dataset/Dataset/Validation/
```

No model was trained and no annotation file was modified.

## Validation inventory

| Asset | Observed Validation content | Relationship |
|---|---:|---|
| `Images/` | 150 JPG radiographs | Source images, identified by filename such as `1002.jpg` |
| `Masks (Tooth-wise)/` | 150 image-ID folders | Each folder contains one PNG per visible/annotated tooth instance; counts range from 2 to 7 |
| `Masks (Radiograph-wise)/` | 150 PNG masks | One full-radiograph mask per image |
| `Masks (Tooth-wise)/coco_format_instances_valid.json` | 150 COCO images, 655 annotations | Tooth-instance segmentation export |
| `Key Points Annotations/` | 150 JSON files | One JSON per radiograph |
| `Bone Level Annotations/` | 150 image JSON records plus one COCO-format metadata JSON | One JSON per radiograph plus `coco_format_bonelines_val.json` |
| `Characteristics of radiographs included.xlsx` | 1,000 metadata rows | 150 Validation image IDs all have matching metadata rows |

The extra entry in the Bone Level folder is `coco_format_bonelines_val.json`, not an additional radiograph annotation record.

All inspected image/mask pairs preserve the source dimensions. For example, `1002.jpg` is `1171 x 886` RGB, its radiograph-wise mask is `1171 x 886` single-channel, and all four tooth-wise masks are also `1171 x 886` single-channel.

The processed project dataset is separate and currently has 650 resized `1024 x 1024` images. The DenPAR Validation source images retain their original dimensions and should be used when interpreting annotation coordinates.

## Exact annotation formats

### 1. Tooth-wise masks

Example directory:

```text
Masks (Tooth-wise)/1002/mask1.png
Masks (Tooth-wise)/1002/mask2.png
Masks (Tooth-wise)/1002/mask3.png
Masks (Tooth-wise)/1002/mask4.png
```

Observed properties:

- PNG, single-channel (`L`) masks.
- Same width and height as the corresponding radiograph.
- One mask file per annotated tooth instance.
- Mask filenames are positional (`mask1`, `mask2`, ...), not FDI tooth numbers.
- The files do not contain disease, severity, Arch, Site, or FDI labels.
- The mask pixels encode spatial tooth foreground/background; the exact foreground convention should still be verified by a small visual review before training.

Interpretation: tooth-instance segmentation masks, not periodontal-disease masks.

### 2. Radiograph-wise masks

Example:

```text
Masks (Radiograph-wise)/1002.png
```

Observed properties:

- PNG, single-channel (`L`) at the original radiograph dimensions.
- Example unique pixel values are `[0, 255]`.
- One mask per radiograph.
- The directory/file naming indicates a radiograph-wise tooth mask; the inspected COCO category is `tooth`.
- It does not encode a periodontal disease label or severity.

Interpretation: a full-image binary mask of the annotated tooth regions, useful for foreground/tooth localization or segmentation, not disease detection.

### 3. COCO tooth-instance file

File:

```text
Masks (Tooth-wise)/coco_format_instances_valid.json
```

Observed top-level keys:

```json
["images", "annotations", "categories"]
```

Observed contents:

- `images`: 150 records with `id`, `file_name`, `width`, `height`.
- `annotations`: 655 tooth instances.
- `categories`: exactly one category: `{ "id": 1, "name": "tooth", "supercategory": "tooth" }`.
- Each sampled annotation has `image_id`, `category_id: 1`, polygon `segmentation`, COCO `bbox`, `area`, and `iscrowd: 0`.
- `segmentation` is a polygon represented as a list of flattened `[x, y, x, y, ...]` coordinates.
- COCO `bbox` follows `[x, y, width, height]`.
- The COCO image IDs map to the numeric image filename stem, e.g. image ID `16` maps to `16.jpg`.
- Instance counts per Validation image range from 2 to 7 and total 655.

Interpretation: tooth-instance localization/segmentation ground truth. It does not distinguish diseased from non-diseased teeth and does not provide severity.

### 4. Key-point annotations

Example `Key Points Annotations/1002.json`:

```json
{
  "Image_id": "1002.jpg",
  "bboxes": [[26.0, 204.0, 341.0, 736.0]],
  "CEJ_Points": [[108.165, 358.42]],
  "Apex_Points": [[51.22, 736.938]]
}
```

Observed properties across all 150 Validation JSON files:

- Stable keys: `Image_id`, `bboxes`, `CEJ_Points`, `Apex_Points`.
- `Image_id` matches the radiograph filename for all 150 files.
- Coordinates are floating-point image-pixel coordinates.
- Origin is consistent with standard image coordinates: top-left, x increases rightward, y increases downward.
- The sampled `bboxes` are corner coordinates `[x_min, y_min, x_max, y_max]`, not COCO width/height boxes. For `1002.jpg`, `[26, 204, 341, 736]` has a valid lower-right corner; treating the last two values as width/height would exceed the image height.
- `CEJ_Points` and `Apex_Points` are point coordinates in the same image-pixel coordinate system.
- Counts are not consistently one CEJ and one apex per box. In `1002.json`, there are 4 boxes, 5 CEJ points, and 3 apex points. This prevents assuming a direct positional one-to-one pairing without annotation documentation or expert confirmation.
- No tooth IDs, FDI notation, disease labels, severity labels, or class labels are embedded in these JSONs.

Interpretation: tooth-region boxes and anatomical landmark candidates for structural/measurement tasks. They are not disease labels. CEJ and apex points could support landmark localization or derived geometric measurements only after their correspondence rules are confirmed.

### 5. Bone-level annotations

Example `Bone Level Annotations/1002.json`:

```json
{
  "Image_id": "1002.jpg",
  "Num_of_Bone_Lines": 4,
  "Bone_Lines": [
    [[9.349, 445.512], [44.521, 438.813], [96.4416, 430.4381]]
  ]
}
```

Observed properties:

- Stable keys: `Image_id`, `Num_of_Bone_Lines`, `Bone_Lines`.
- `Image_id` matches the image filename for 150 image records.
- Each bone line is a variable-length polyline of `[x, y]` image-pixel coordinates.
- Example `1002.json` has four bone lines with 3, 3, 3, and 6 vertices.
- Validation counts are commonly 2, 3, 4, or 5 bone lines per image.
- The separate `coco_format_bonelines_val.json` is an export/metadata artifact; it must be inspected separately before assuming it is equivalent to the per-image JSON schema.
- The field name says `Bone_Lines`; the files do not state that the lines are a disease label, severity label, or clinical diagnosis.

Interpretation: expert/curated bone-level geometric annotations that may support structural line/landmark analysis or derived measurements. They do not independently establish localized periodontal disease.

### 6. Radiograph characteristics workbook

File:

```text
Characteristics of radiographs included.xlsx
```

Columns:

- `id`
- `Arch`
- `Site`
- `FDI notation of fully/partially visible teeth`

For the 150 Validation images, all IDs matched workbook rows. Validation distributions were:

- Arch: 97 Lower, 53 Upper.
- Site: 75 Right, 53 Left, 22 Anterior.
- FDI visible-teeth strings are metadata lists such as `41,31,32,33,34,35`.

These FDI strings describe fully/partially visible teeth in the radiograph. They are not necessarily disease-positive teeth and are not spatial segmentation labels. They can provide auxiliary tooth-presence metadata or a candidate mapping target, but matching them to `mask1`, `mask2`, key-point boxes, and bone lines requires an explicit ordering rule or expert confirmation.

## What the files do and do not establish

The files explicitly establish geometry and visibility structures:

- Tooth regions/instances.
- Tooth masks and polygons.
- Candidate tooth bounding boxes.
- CEJ and apex point coordinates.
- Bone-line polylines.
- Arch, broad site, and visible FDI tooth metadata.

The files do **not** explicitly establish:

- Localized periodontal disease present/absent.
- Disease-positive tooth or region.
- Mild/moderate/severe periodontal severity.
- Expert clinical diagnosis.
- Treatment need or treatment response.
- Longitudinal progression or outcome.

The word “bone” in `Bone_Lines` does not by itself make the annotation a periodontal-disease label. Disease and severity require a separate expert-approved annotation protocol.

## Task-to-annotation mapping

| Research task | Available DenPAR component | Annotation type | Ground-truth readiness | What it can support | What is still required |
|---|---|---|---|---|---|
| Disease detection | None of the inspected Validation annotations | Image/tooth disease label required | **Not supported now** | No valid supervised disease target | Expert-validated present/absent/uncertain labels, with case grouping and an agreed unit of prediction |
| Affected tooth/region localization | COCO tooth instances, tooth-wise masks, radiograph-wise masks, key-point boxes | Polygon masks, binary masks, and boxes | **Supported for tooth localization only** | Tooth-instance detection/segmentation or tooth-region extraction | Expert confirmation that the task target is tooth presence/localization; FDI-to-instance mapping if tooth identity is required |
| Severity assessment | None | Expert severity category required | **Not supported now** | No severity target can be derived safely | Expert-approved severity definitions and labels; class-distribution and agreement review |
| Structural analysis: tooth shape/extent | Tooth-wise masks and COCO polygons | Instance segmentation | **Supported for exploratory structural localization** | Tooth foreground masks, region shape, tooth-area/extent measurements | Visual QA and confirmation that masks represent the intended structure |
| Structural analysis: CEJ/apex localization | Key-point JSONs | Point landmarks plus boxes | **Partially supported** | Landmark localization and geometric measurements | Confirm point-to-tooth correspondence and missing-point policy |
| Structural analysis: bone-level geometry | Bone-level JSONs | Variable-length polylines | **Partially supported** | Bone-line detection/segmentation or geometric measurements | Confirm exact anatomical meaning, line-to-tooth correspondence, and clinical measurement protocol |
| Progression estimation | None | Longitudinal labels required | **Not supported** | Nothing reliable from these cross-sectional annotations | Repeated images per case, time intervals, treatment/outcomes, and expert-validated progression labels |

## Components that can be trained now

### Trainable now, subject to split and QA checks

1. **Tooth-instance segmentation** from tooth-wise masks or COCO polygons.
2. **Radiograph-level tooth-region segmentation** from radiograph-wise masks.
3. **Tooth-region detection** from COCO or key-point bounding boxes, but note the coordinate-format difference.
4. **CEJ/apex landmark localization**, only as an anatomical landmark task after correspondence and missing-label rules are confirmed.
5. **Bone-line geometry prediction**, only after confirming what `Bone_Lines` represents and how each line maps to a tooth/region.

These are structural/localization tasks, not disease diagnosis tasks.

### Not trainable as supervised tasks from these annotations alone

1. Disease detection.
2. Severity classification.
3. Disease-specific structural finding classification.
4. Treatment planning recommendation.
5. Progression estimation.

## Recommended annotation/validation additions

## Correspondence-analysis module

`backend/app/structural_audit/correspondence.py` performs a read-only, per-image correspondence audit. It confirms only links explicitly keyed by the source image ID. It deliberately does **not** treat equal counts, filename order, spatial containment, or visual proximity as tooth identity evidence.

Run it from `backend/`:

```text
python -m app.structural_audit.correspondence_cli --validation-dir "../DenPAR Radiographs Dataset/Dataset/Validation" --output "../dataset/reports/denpar_correspondence_analysis.json"
```

The report contains complete, partial, and unresolved image counts; missing landmarks; suspicious count/containment findings; and review-only candidates for masks, COCO instances, key-point boxes, CEJ points, apex points, and FDI metadata. Candidate lists are inspection aids, not pairings or landmark-training targets. The Structural Audit viewer exposes the same candidates and records an expert's local `confirmed`, `uncertain`, `missing`, or `not applicable` review status without modifying the original dataset.

Current source-file result: no deterministic CEJ/apex-to-tooth or FDI-to-instance correspondence is encoded. Those annotations must not be used as tooth-specific supervised landmark ground truth unless a qualified expert establishes and records the mapping independently.

Before disease or severity modeling, add a separate expert-validated record for each review unit. At minimum, the protocol should define:

```json
{
  "image_id": "1002.jpg",
  "case_id": "expert-approved-case-id",
  "disease_status": "expert-approved-value",
  "affected_teeth": ["expert-approved-FDI-values"],
  "severity": "expert-approved-value",
  "structural_findings": ["expert-approved-category"],
  "regions": [
    {
      "type": "bounding_box_or_polygon_as_approved",
      "coordinates": []
    }
  ],
  "expert_validated": false,
  "expert_comment": ""
}
```

The category vocabularies, disease unit, severity ordering, region method, and relationship between DenPAR geometry and FDI teeth must be approved by a qualified dental expert. Existing DenPAR annotations should be retained as source structural annotations, not silently relabeled as disease ground truth.

## Split and leakage requirements

- Split by patient/case, not by image filename, whenever case identity is available.
- Do not assume numeric image IDs are patient IDs.
- Keep all images from one case in exactly one split.
- Keep masks, key points, bone lines, workbook metadata, and derived crops attached to their source image ID.
- Do not generate disease labels from mask presence, bone-line presence, Arch, Site, or visible FDI teeth.
- Report missing key points, missing bone lines, inconsistent instance counts, and unresolved FDI-to-mask ordering before training.
