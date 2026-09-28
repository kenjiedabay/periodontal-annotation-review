# Unified structural ground-truth construction

Run from the repository root:

```powershell
python structural_analysis/step4_structural_dataset/build.py
python -m unittest discover -s structural_analysis/step4_structural_dataset -p test_build.py
```

The builder reads the current Step 2 and Step 3 expert-review files and creates:

- `trainable_structural_manifest.json`: one tooth record when at least one usable structural target is explicitly confirmed. Empty arrays mean no included confirmed target, not a negative training label.
- `unresolved_structural_manifest.json`: excluded teeth and annotation relationships, including probable, uncertain, rejected, missing, unavailable and unmapped annotations, with reasons and provenance.
- `structural_dataset_summary.json`: usable/partial/complete counts, exclusions and official partition counts.

The current review files have no expert decisions, so the correct trainable dataset is empty. Original tooth masks alone do not establish validated CEJ/apex/bone correspondence and do not enter this structural-target manifest.

## Inclusion and completeness

A target needs a current `confirmed` expert-review decision, reviewer attribution, timezone-aware timestamp and matching latest audit-history event. Candidate/review fingerprints and original source hashes must match. Invalid or unavailable image/mask/point/line geometry is excluded. A confirmed anatomical-region bone mapping without a tooth target is excluded. Missing landmark decisions do not create coordinates or targets. Contradictory missing and confirmed decisions for a tooth/type are excluded.

Multiple explicitly confirmed CEJ points may be retained for one tooth. No indices are paired and no one-to-one mapping is invented. Targets are re-read from the original annotation field/index. `expert_validated: true` describes included correspondence targets; it does not assert that other structures or FDI were validated.

Partial records are allowed: a valid original mask/bbox plus one or more confirmed CEJ, apex or bone targets. `confirmed_structures` describes available confirmed target types. **All required structures** means a valid mask/bbox and at least one confirmed CEJ point, apex point and bone line. This is a dataset completeness convention, not a clinical claim or a requirement to fabricate absent structures.

`fdi` stays null because neither review system validates tooth-mask-to-FDI correspondence. Boxes are derived from original mask foreground with exclusive maximum edges. File paths are repository-relative. Every included target retains original file/hash, JSON field/index, review file/revision, candidate fingerprint and full expert decision. Exclusions retain original annotation pointers and candidate/review references.

Official Training, Validation and Testing membership is retained from source directories. No new split, target inference, disease/severity label, model training or source-file modification occurs. Construction stops here. Re-run after experts save decisions; do not edit source annotations to force eligibility.
