# Dataset workspace

Use these directories for anonymized research data only:

- `raw/`: source files received for research processing; keep out of version control.
- `cleaned/`: validated and de-identified inputs.
- `processed/`: derived image data used by experiments.
- `annotations/`: JSON annotations and dataset manifests.

Never store names, medical record numbers, dates of birth, facial images, or other patient-identifiable information here. Real radiographs must remain private and must not be committed to a public GitHub repository.
