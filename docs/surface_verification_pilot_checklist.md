# Tooth-surface verification pilot checklist

Use only Training or Validation records. Do not use the official Testing partition and do not calculate acceptance metrics until real expert-reviewed records exist.

## Case coverage

Select a small set covering all four permanent-tooth quadrants, maxillary and mandibular arches, patient left and right, a deliberately flipped display, missing neighbor, partial edge tooth, central incisors, multi-rooted molar, unclear CEJ, unclear bone crest, unclear apex, and an ungradable surface. Record the stable image ID and selection reason without copying or changing the source data.

## Per-case acceptance steps

- Confirm arch, patient side, display orientation, flip status, expert ID, and timestamp.
- Confirm each FDI number; metadata values are candidates, not instance mappings.
- Confirm M/D anatomically and verify the displayed pixel boundary.
- Swap M/D once and confirm that the image and every raw coordinate remain unchanged.
- Place, adjust, and confirm surface-specific CEJ, bone-crest intersection, and apex reference.
- Zoom/resize and confirm landmarks remain aligned; reopen the record and compare raw coordinates.
- Confirm model suggestions remain unchanged after expert correction.
- Confirm RBL appears only after orientation, tooth, surface, and all three landmarks are confirmed.
- For a multi-rooted tooth, select an apex ID and record `expert_selected`, or mark the surface ungradable.
- Confirm unresolved and ungradable surfaces are absent from evaluation output.
- Review every geometry warning; verify that warnings never rewrite expert data.

## Stop conditions

Stop and retain the record as unresolved if orientation conflicts, FDI order is anatomically implausible, a required landmark is not visible, the mask/landmark association is doubtful, or the selected apex is not defensible. Do not infer CAL, probing depth, a definitive diagnosis, or future progression.
