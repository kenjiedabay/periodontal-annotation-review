# Canonical confirmed structural dataset

This is the requested current Step 4 output location. The builder delegates to the existing confirmed-only implementation in `step4_structural_dataset/build.py`, which remains reusable; its earlier JSON outputs are historical snapshots.

Rebuild the dataset, CEJ review queue, blocker audit and split readiness together:

```powershell
python structural_analysis/step3_cej_apex_correspondence/cej_readiness.py
```

Or use **Regenerate manifests & CEJ readiness** in the expert-review UI after saving decisions. A review save alone does not regenerate these manifests. Report revisions identify the exact review snapshot.

Files:

- `trainable_structural_manifest.json`: confirmed targets only, with original mask, derived bbox, coordinates, official partition, expert validation and annotation/review provenance.
- `unresolved_structural_manifest.json`: candidates and unavailable, rejected, uncertain, cannot-determine and missing cases excluded from supervision.
- `structural_dataset_summary.json`: inclusion/exclusion counts.
- `cej_readiness_report.json`: confirmed CEJ images, teeth and target occurrences per official split, expert status counts, unresolved points and multiple-CEJ cases.

No CEJ point count is forced. Empty CEJ arrays and expert missing decisions are not fabricated coordinates or negative supervision. FDI remains unassigned. Testing stays separate and is not evaluated.

Readiness is A when usable expert-confirmed targets exist in all three official splits, B when some are available but split coverage is incomplete, and C when none are available. The A gate is a data-availability gate, not a claim that sample size or landmark completeness has been scientifically validated for training.
