# Stage 1: versioned radiographic review foundation

This is an additive JSON-backed research workflow. Existing `/annotations`,
`/api/annotations`, Mask R-CNN checkpoints, and DenPAR source files are not
migrated or changed. New data live under `data/review_foundation/` by default
(override with `REVIEW_FOUNDATION_DIR`). This directory is ignored by Git because
it can contain uploaded radiographs and expert notes.

## Record schema

Each immutable record has `schema_version: 1`, a UUID `record_id`, one of the
five `record_type` values below, `version`, `image_id`, SHA-256 `image_hash`,
`source_dataset`, `source_partition`, original width/height,
`coordinate_space: original_image_pixels`, `coordinate_origin: top_left`,
`created_at`, `created_by`, `parent_record_id`, optional `model_name` and
`model_version`, `review_status`, and `uncertainty_reasons`.

| Type | Additional fields | Version parent |
| --- | --- | --- |
| `source_record` | `review_mode`, original filename and immutable image reference, DenPAR annotation paths and SHA-256 hashes | None |
| `raw_ai_prediction` | Exact status `AI-generated—awaiting expert review`, untouched Mask R-CNN result without DenPAR ground truth, tooth masks/boxes/confidence, checkpoint SHA-256, preprocessing configuration and model-to-original transform | Source record |
| `independent_expert_annotation` | Snapshot of points, lines, PNG masks, regions, notes, findings and provisional tooth anatomy | Source for v1; previous independent version thereafter |
| `expert_correction` | Same snapshot, plus exact raw prediction ID and independent annotation ID when one exists | Raw prediction for v1; previous correction version thereafter |
| `final_approval` | `approved_record_id`, decision (`approved`, `rejected`, `uncertain`, `needs_revision`), notes, previous approval ID | Exact annotation/correction being decided |

`ToothAnatomy` reserves tooth ID, mask, bounding box, crop box/margin, mesial
and distal CEJ/bone points, visible root apices with anatomical labels, selected
apex indices, confidence, and uncertainty. Each `Landmark` has `state` from
`visible`, `not_visible`, `uncertain`, `not_applicable`. A coordinate is required
only for `visible` and forbidden for all other states. No Stage 1 code infers
mesial/distal orientation or calculates bone loss.

Records are written to UUID-named JSON files using a complete temporary file
and an exclusive hard-link publish step. They have no update or delete endpoint.
The small `source.json` lookup copy and `review_state.json` are written by
atomic replacement. A DenPAR source record references existing source files;
it does not copy or edit them. An uploaded non-DenPAR source is saved as an
immutable, hash-named local image. A claimed DenPAR image is recognized only
when its filename ID and exact SHA-256 match the local official source file.

## API

| Method | Path | Result |
| --- | --- | --- |
| POST | `/api/review/sources` | Register image and choose `independent_evaluation` or `ai_assisted` |
| GET | `/api/review/sources/{image_id}` | Source record |
| GET | `/api/review/sources/{image_id}/image` | Original radiograph |
| GET | `/api/review/sources/{image_id}/state` | Persisted reveal state |
| POST | `/api/review/sources/{image_id}/reveal` | Record AI reveal after independent annotation |
| POST | `/api/review/sources/{image_id}/predictions` | Run existing Mask R-CNN and append raw output |
| GET | `/api/review/sources/{image_id}/predictions/{record_id}` | Raw output, gated while blind |
| POST | `/api/review/sources/{image_id}/independent-annotations` | Append independent snapshot/version |
| POST | `/api/review/sources/{image_id}/corrections` | Append correction snapshot/version |
| GET | `/api/review/sources/{image_id}/annotations/{record_id}` | Exact annotation version |
| POST | `/api/review/sources/{image_id}/approvals` | Append final decision |
| GET | `/api/review/sources/{image_id}/approvals/{record_id}` | Exact decision |
| GET | `/api/review/sources/{image_id}/history` | All visible record versions and review state |

Register a source with multipart form fields `image_id`, `review_mode`,
`created_by`, and `file`. A first independent annotation may be submitted as:

```json
{
  "created_by": "reviewer-01",
  "parent_record_id": "<source-record-uuid>",
  "items": [{
    "item_id": "<item-uuid>", "tooth_instance_id": 1,
    "finding_type": "cej", "annotation_tool": "point",
    "points": [{"x": 120.5, "y": 240.0}], "status": "draft",
    "expert_comment": "Visible proximal contour"
  }],
  "review_status": "draft"
}
```

The response adds all common metadata and `version: 1`. A second independent
save sends the first annotation's `record_id` as `parent_record_id` and receives
`version: 2`; version 1 remains readable. A correction additionally sends
`raw_prediction_record_id` and `independent_annotation_record_id` when present.
An approval sends `created_by`, `approved_record_id`, `decision`, and optional
`notes`; its response references the exact decided record.

## Review workflow and compatibility

At intake, choose **Independent evaluation** or **Routine AI-assisted review**.
Evaluation images open in **Independent Review** with no AI request or overlay.
The backend denies raw-prediction retrieval, structural-audit overlays and the
existing segmentation route for a registered evaluation image until an
independent annotation is saved and reveal is persisted. The guard also checks
the image hash if the same bytes are submitted under another ID. After reveal,
the frontend saves the raw Mask R-CNN result and offers a separate versioned
correction and final decision. Routine images can create a raw prediction and
review it directly. The original spatial drawing tools remain available in
the versioned review; their saved snapshots do not mutate legacy annotations.

The old routes and JSON formats remain readable. There is no authentication or
role authorization yet: a reviewer ID is recorded, but it is self-entered.
Blinding is enforced by the new API and by guarded existing AI endpoints for
registered evaluation images, not by a frontend switch alone. This is a
research prototype and does not produce clinical diagnosis, severity, pattern,
progression or treatment decisions.
