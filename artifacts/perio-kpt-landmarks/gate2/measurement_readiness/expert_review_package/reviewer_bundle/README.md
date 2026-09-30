# Blinded Perio-KPT landmark review

## Review scope

Review only the green ground-truth landmark locations shown in the 12 randomized case images. Do not infer why a case was selected. Do not diagnose disease, assign severity, or edit any source annotation.

Complete both the mesial and distal row for every case in `review_form.csv`. Use only these response values:

- `yes`
- `no`
- `uncertain`
- `not_applicable`

For each surface, answer:

1. Is the CEJ landmark correctly placed?
2. Is the bone-level landmark correctly placed for radiographic bone-loss measurement?
3. Is the selected root landmark/apex appropriate, especially for double- and triple-root teeth?
4. Are the mesial and distal landmark associations correct?
5. Is the CEJ -> bone level -> root geometry clinically plausible?
6. Is the surface assessable from this radiograph?

If any answer is `no` or `uncertain`, enter at least one reason code and explain the issue briefly in `comments`.

Allowed reason codes, separated with `|` when multiple apply:

- `cej_misplaced`
- `bone_level_misplaced`
- `root_or_apex_inappropriate`
- `mesial_distal_reversed`
- `geometry_implausible`
- `landmark_not_visible`
- `overlapping_anatomy`
- `low_image_quality`
- `crop_context_insufficient`
- `surface_not_assessable`
- `definition_ambiguous`
- `other`

Suggested corrections may be described in words. Do not modify the PNG or any dataset file. Save the completed CSV under a new filename such as `completed_review_form.csv`.

The case-to-source mapping and selection categories are intentionally absent from this reviewer bundle.
