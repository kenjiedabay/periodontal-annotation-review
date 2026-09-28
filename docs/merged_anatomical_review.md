# Merged Stage 2C + Surface Workflow

The unified entry point is the existing Surface Workflow pilot with a shared case image and case ID.

This is the canonical workflow for all new tooth identity, mesial/distal, and landmark review. Image Review hands off the current `source_image_id` directly to this workflow. The former standalone surface editor is no longer a write entry point; its historical records remain available through the read-only API and the Advanced / Legacy viewer.

Phase A is the independent surface annotation: orientation, FDI tooth identity, mesial/distal surface, CEJ, alveolar crest, apex, and quality/uncertainty decisions. It is saved and locked before any model output is read.

Phase B is the Stage 2C assisted comparison on that same image. The frozen Stage 2B association proposal is revealed only after the Phase A lock. The expert records an assisted decision separately; Phase A and the model snapshot remain immutable.

Existing Stage 2C and Surface pilot records are historical and remain untouched. The merged route does not ask for patient age or history and does not process Testing data.

Use `?pilot=merged-anatomical-review` or select the unified anatomical review entry in the frontend. Restart the backend after code changes.
