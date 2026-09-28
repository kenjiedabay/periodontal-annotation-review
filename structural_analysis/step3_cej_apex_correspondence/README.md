# Step 3: CEJ / apex correspondence validation

Run from the repository root:

```powershell
python structural_analysis/step3_cej_apex_correspondence/serve.py
```

Open **http://127.0.0.1:8767** (or the port printed by `serve.py --port 8768`). Select a point and tooth, inspect the mask, tooth bbox/ID, neighboring masks, independent keypoint boxes and geometric features, then confirm, reject, mark uncertain or choose **Cannot determine**. To reassign, select the old mapping, choose another tooth and click Reassign. This explicitly rejects the old mapping and confirms the new one while retaining both `candidate_tooth_instance_id` and `expert_tooth_instance_id` in separate expert decisions. Other relationships remain available. Any validated point may be manually assigned to any validated tooth, including pairs not proposed by the heuristic.

Multiple CEJ points may be confirmed for one tooth. No exclusivity rule pairs arrays, boxes, masks, CEJ or apex points. Point/box indices are source identifiers only. Tooth IDs are mask filenames, not FDI identifiers.

## Missing landmarks

The missing controls operate on the **selected tooth and landmark type**, not on a point. They save an expert decision with null point ID and coordinate. No point is synthesized. Missing CEJ/apex counts mean **expert-marked missing**; initial zeros do not assert that all landmarks are present. Unreviewed tooth/type slots and unresolved source points are reported separately. Existing confirmed points must be rejected or reassigned before marking their tooth/type missing. Clear a missing decision before confirming a point for that slot.

## Geometric evidence

The generator reads original images, tooth masks and keypoint JSON anew, using the Step 1 inventory and image-resolution paths. It validates dimensions, image identity, finite coordinates and bounds. Missing images and invalid points/masks are retained with reasons and excluded from proposals. Every point record includes its type, source coordinate, candidate teeth, scores and validation status; malformed coordinates are null and traceable via the source file/index.

- Mask containment uses the nearest raster pixel; distance is to the nearest foreground pixel center.
- Every point is evaluated against every valid keypoint box. Point-to-box distance and containment are retained; bbox-to-tooth evidence uses box overlap (IoU), not shared ordering.
- Centroid distance and displacement are recorded.
- Relative vertical position is normalized to the tooth box and labeled above/upper/middle/lower/below. Source Arch is displayed. Image top/bottom does not establish anatomical crown/root orientation; expert interpretation is still required.

Candidates pass when mask distance / tooth-box diagonal ≤ 0.15 **or** bbox support ≥ 0.25. Score is `0.65 exp(-normalized_mask_distance/0.05) + 0.25 bbox_support + 0.10 exp(-normalized_centroid_distance)`. Bbox support is the maximum over all boxes of `tooth_bbox_IoU × exp(-point_box_distance/(0.05 × tooth_diagonal))`. Probable requires mask distance ratio ≤ 0.03 and bbox support ≥ 0.2; other candidates are uncertain. Both remain unconfirmed. These thresholds are heuristics, not calibrated probabilities. Vertical position is descriptive and does not gate the score.

## Files and review integrity

- `candidate_landmark_mappings.json`: original point inventory and unconfirmed geometric proposals, mask geometry, independent boxes, features and input hashes.
- `expert_landmark_review.json`: separate explicit decisions, reviewer attribution, timestamp, revision and decision history. Reviewer identity is entered locally, not authenticated.
- `reports/summary.json`: confirmed mappings, unresolved points, explicit missing counts, multiple-point cases, unreviewed tooth/type slots and image review list. Updated after each save.
- `overlays/<split>_<image>/`: original display copy, individual mask layers and overview with all valid landmarks and keypoint boxes. The browser renders selectable points/boxes directly from validated source coordinates.

Source annotations and earlier steps are read-only. Saved reviews use atomic replacement and revision checks for stale tabs. Generation refuses to overwrite expert history. Multiple-point cases report candidate multiplicity and separately count expert-approved multiple-point cases; multiplicity alone is not an error. Counts being equal is never evidence for pairing.

## Rebuild and test

Requires Python, Pillow and NumPy; no model or OpenCV dependency.

```powershell
python structural_analysis/step3_cej_apex_correspondence/landmarks.py
python -m unittest discover -s structural_analysis/step3_cej_apex_correspondence -p test_landmarks.py
```

No model training, landmark fabrication, disease inference or severity inference is performed. Stop after Step 3.

## CEJ expert-validation preparation

`cej_review_records.json` contains one record per existing plausible CEJ/tooth association, with source coordinates, mask bbox, distance/reason, candidate identity and an independent expert status. A geometric probable/uncertain label never becomes an expert decision. The original candidate mapping file is unchanged. The queue is refreshed after saved decisions; `/api/cej-records` also derives current records directly.

`reports/cej_blocker_audit.json` distinguishes unreviewed candidates, expert uncertainty, absent candidates, manifest gates and schema/provenance checks. Use **Regenerate manifests & CEJ readiness** after expert review to refresh the requested `step4_structural_ground_truth` outputs and CEJ split-readiness report. Until regeneration, previous manifest/readiness files are snapshots; the UI marks them as needing refresh after a save. The same action is available as `python structural_analysis/step3_cej_apex_correspondence/cej_readiness.py`.

Confirmed CEJ counts were zero because no independent expert decisions existed, despite generated associations. Additionally, 650 original Training images were missing from the local source inventory. Training correspondence cannot be reviewed until those originals are available and Steps 1/3 are refreshed; existing expert history must be preserved/migrated deliberately if candidate fingerprints change. No threshold or pseudo-label automatically resolves this blocker.
