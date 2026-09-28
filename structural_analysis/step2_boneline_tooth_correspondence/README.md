# Step 2: Bone-line / tooth candidate correspondence

## Run the review interface

From the repository root:

```powershell
python structural_analysis/step2_boneline_tooth_correspondence/serve.py
```

Open **http://127.0.0.1:8766**. This separate local interface saves decisions directly to `expert_review.json`; a static file preview cannot save reviews. Use `--port` to choose another local port.

Choose an image, bone line and tooth. Click a candidate to inspect its features and isolated mask/line overlay. **Accept mapping** explicitly saves `confirmed`; probable, uncertain, rejected and unassigned are also supported. Reviewer name is required for attribution; this local tool does not authenticate professional credentials.

Selecting any other validated tooth allows an expert-added mapping even when the heuristic proposed none. Choose **Other anatomical region** to describe a region instead. To reassign, choose the old mapping under **Replace mapping**, select the new tooth/region and click **Reassign**. That explicitly rejects the old mapping and confirms the new one. Other candidates are retained; neither the generator nor acceptance enforces one-to-one correspondence.

## Generated data

- `candidate_mappings.json`: image inventory, validated mask extents and centroids, line geometry, unconfirmed proposals, features, thresholds and source hashes.
- `expert_review.json`: current expert decisions, revision and append-only decision history, separate from originals and proposals. Initially empty; no synthetic expert decisions are inserted.
- `reports/correspondence_summary.json`: effective mappings by all five statuses, unassigned bone lines, teeth with no active candidate, and multiplicity by image. Recomputed after each saved review. `/api/report` also computes it from current decisions.
- `review_overlays/<split>_<image>/`: original image display copy, one transparent layer per validated tooth mask, and all-mask/all-valid-line overview. The UI isolates any selected pair dynamically.

Original annotations, Step 1 outputs and models are only read. IDs preserve split + image + mask filename; bone-line IDs are zero-based source array indices. Mask IDs do not assert FDI or COCO correspondence. Missing images, mismatched/empty masks and invalid lines are excluded from candidate generation and retained in the inventory with reasons.

## Geometry and score

Polyline segments are sampled at a maximum 1-pixel interval. An exact Euclidean distance transform to mask foreground pixel centers is bilinearly sampled at these positions, giving an **approximate** minimum line-to-mask distance. Bbox distance is also sampled, not an exact continuous segment distance. The line centroid is approximately arc-length weighted through these samples; mask centroids use all foreground pixels.

All tooth/valid-line pairs are evaluated. Retain every pair whose minimum mask distance is at most **15% of that tooth's bounding-box diagonal**. No top-one selection or exclusivity is imposed. The expanded region consists of locations within **5% of that diagonal** from the mask. Report the sampled fraction within it, bbox proximity, centroid offsets and left/right image position (not anatomical laterality).

Score: `0.7 × exp(-normalized_distance / 0.05) + 0.3 × expanded_region_fraction`.

**Probable** requires normalized distance ≤ 0.03 and expanded-region fraction ≥ 0.2. Other retained pairs are **uncertain**. Both have `confirmation_status: unconfirmed`. Thresholds are heuristic and uncalibrated; scores are not probabilities, clinical assessments or supervision targets.

“Unassigned bone line” means no active confirmed/probable/uncertain target remains. Rejected and unassigned pair decisions do not count as active. A probable proposal does not imply the line has been expert-assigned. “Teeth with no candidate” includes unavailable masks and includes reasons. Multiplicity means more than one active target per line or more than one active line per tooth; it is a review flag, not proof of an error.

## Reproduce and verify

Requires Pillow, NumPy and OpenCV. The repository's existing `.runtime` supplies OpenCV on this machine.

```powershell
python structural_analysis/step2_boneline_tooth_correspondence/correspondence.py
python -m unittest discover -s structural_analysis/step2_boneline_tooth_correspondence -p test_correspondence.py
```

Generation refuses to overwrite any existing expert decision history. Review saves use atomic file replacement and reject stale revisions to avoid overwriting another open tab's decisions. Only explicit review actions can produce confirmed mappings. Dental-expert interpretation remains required. No training or severity inference is performed; the workflow stops after correspondence analysis.
