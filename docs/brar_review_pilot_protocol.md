# BRAR label-verification pilot v1.0

This blinded pilot independently verifies the published BRAR patient-level grade on 45 class-balanced radiographs (15 per published level). Published grades and measurements are kept in a private manifest and are never returned by the review API before submission.

For each image, the reviewer records image quality, the FDI tooth and mesial/distal surface with maximum radiographic loss, CEJ/crest/apex coordinates, bone-loss percentage, BRAR (`bone-loss percentage / age`), and Level 1/2/3 (`<0.25`, `0.25–1.0`, `>1.0`). Unreadable cases may be marked not assessable. Reviews are append-only and independently locked.

This verifies patient-level BRAR labels. It does not provide comprehensive per-tooth localization, clinical periodontitis diagnosis, prognosis, or treatment planning.
