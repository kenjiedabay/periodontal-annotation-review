# Step 1: DenPAR bone-line visualization and QA

Open `index.html` in a browser. It works directly from disk without a server or network access. Choose any radiograph or use the representative-category buttons. Five independent checkboxes control tooth masks, tooth boxes, bone lines, CEJ and apex. The original stays visible alongside the overlay. Missing-image records remain selectable with their review reasons.

## Reproduce

From the repository root, with Python, Pillow and NumPy installed:

```powershell
python structural_analysis/step1_boneline_visualization/build.py
python -m unittest discover -s structural_analysis/step1_boneline_visualization -p test_geometry.py
```

`--dataset` and `--output` accept alternative directories. Output inside the source dataset is rejected. No models are loaded or trained.

## Sources and display conventions

- Per-image original `Bone Level Annotations/<id>.json` supplies `Bone_Lines`, `Num_of_Bone_Lines` and `Image_id`. COCO bone polygons are not substituted for these polylines.
- Images are resolved from the split's `Images` directory, then the shared dataset `Images` directory. Native dimensions and orientation are retained; no coordinate scaling or clipping is applied.
- Original tooth-wise PNG masks are separate translucent instances. Boxes show each mask's pixel extents; these are derived display geometry, not claimed matches to keypoint boxes, COCO IDs or FDI values.
- Optional original keypoint JSON supplies CEJ and apex points. Spreadsheet Arch, Site and visible FDI notation are displayed as image metadata.
- Bone lines are yellow and labeled with zero-based source line indices, CEJ cyan, apex magenta. Mask colors repeat and have no anatomical meaning.
- Invalid lines are omitted whole. The JSON preserves offending source points and exact point indices. No interpolation, coordinate repair, line-to-tooth matching or disease/severity inference occurs.

## Outputs

- `qa_summary.json`: full inventory, per-image line counts, valid/invalid/unassessed counts, missing files, review reasons and representatives.
- `reports/manual_review.json`: records needing file or geometric review.
- `reports/source_annotation_sha256.json`: hashes of consumed source JSON and metadata, verified unchanged after generation.
- `overlays/<split>_<id>/`: original image copy, five transparent PNG layers and combined JPEG.
- `overlays/representative_*.jpg`: upper/lower arch, anterior, left/right posterior and minimum/maximum available tooth-mask counts. Posterior categories use source Site=Left/Right; no laterality is inferred from pixels. These are representative samples, not expert-certified cases.

Validation requires each coordinate to be a finite numeric pair (excluding booleans), each line to have at least two valid points and two distinct positions, and `0 <= x < width`, `0 <= y < height`. Empty/malformed lines and declared-count/filename mismatches are reported. Missing images permit format checks but leave bounds explicitly unassessed; these lines are not counted as geometrically valid or automatically classified as invalid.

Geometric QA does not establish anatomical correctness. Every image still requires dental-expert confirmation before clinical interpretation or tooth correspondence. Stop here: no subsequent structural-analysis steps are implemented.
