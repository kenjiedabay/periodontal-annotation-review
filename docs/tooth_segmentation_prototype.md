# Tooth segmentation prototype integration

Load radiographs in the existing frontend and click **Tooth segmentation**. Expert validation is not required to run segmentation. Results use the original uploaded image, with independent mask, box, and confidence toggles. Exact pixel matches to available Validation or Testing images enable Model Prediction, Ground Truth, and Overlay Comparison modes.

The original research disclaimer remains visible, together with:

> Research prototype. Tooth segmentation output does not indicate periodontal disease, severity, or treatment need.

## API and geometry

`POST /analysis/tooth-segmentation?image_id=1002`, multipart field `file`.

The cached service loads `models/tooth_instance_maskrcnn_baseline/full_run/best_checkpoint.pt` on first request, strictly checks the state dictionary and epoch 8/input size, and serializes inference with a lock. Failed loading returns `model_status: unavailable` and an empty instance list; restarting the API permits another load attempt. No pretrained download or fallback prediction is used. One model copy is loaded per API worker.

The service calls the same `apply_transforms(original, load_config())` used by the validated training loader: grayscale, aspect-preserving resize and centered 1024-square padding, percentile 1–99 normalization, CLAHE clip limit 1.5/grid 8, then float division by 255 and three repeated channels. Torchvision retains its trained internal normalization. No augmentation is used.

Only tooth class 1 with detection confidence >= 0.5 is returned. Masks use a separate pixel threshold >= 0.5. The historical evaluation code thresholds mask pixels but does not apply this detection confidence filter; API instance counts should therefore not be equated with historical evaluation counts.

`instances` contain local `instance_id`, unrounded `confidence`, `bbox` in original-image XYXY pixels, and `mask_url` as an inline PNG data URL (binary alpha, transparent background). Masks are thresholded in model space, unpadded, and resized by nearest neighbor. Boxes are unpadded and scaled using the actual rounded resize dimensions independently on each axis. The response includes original `width`/`height`, `coordinate_space`, and `bbox_format`. A shared SVG viewBox preserves the image and overlay aspect ratio at any display size. Confidence labels display the backend probability as a percentage rounded to two decimals.

Ground truth is read only from DenPAR Validation/Testing tooth masks after exact decoded-image equality; a matching filename alone is insufficient. No DenPAR source file is written. No disease, severity, progression, or treatment prediction is added.

## Running and verification

Install `backend/requirements.txt`, then run `python -m uvicorn main:app --host 127.0.0.1 --port 8000` from `backend`. Run `npm.cmd run dev` from `frontend` (or `npm run dev` on other shells). The frontend defaults to this API address; `VITE_API_URL` can override it.

Build with `npm.cmd run build`. For the integration check, install `httpx` and `playwright` in the Python environment, ensure Edge is installed, then run `python test_tooth_segmentation_integration.py` from `backend`. Ports 8000 and 5174 must be free. The script starts and stops its own API and static servers.

Verified on 2026-09-12 with the production build and headless Edge: Validation `1002` returned 4 instances and Testing `1` returned 5. All integration assertions passed. Screenshots were visually inspected for source-aligned prediction and comparison overlays. This run used CPU inference.

For this workspace's locally installed runtime dependencies, launch from the project root in PowerShell using `$env:PYTHONPATH="$PWD/.runtime;$PWD/backend"`, then `python -m uvicorn main:app --host 127.0.0.1 --port 8000`. The same environment supports `python backend/test_tooth_segmentation_integration.py`.

The integration script uploads Validation `1002.jpg` and Testing `1.jpg` through the actual frontend, uses the real checkpoint, checks nonempty API predictions, compares preprocessing exactly to the training tensor, verifies original mask dimensions and mask/box extents, compares every SVG box and confidence label to the response, checks equal horizontal/vertical browser scaling at 1440 and 760 pixel viewports, exercises toggles and all comparison modes, checks malformed input and unavailable-model responses, and hashes the used source files before/after. Responses, screenshots, and `verification.json` are saved under `artifacts/tooth-segmentation/`. These checks verify integration; they do not constitute clinical validation or new model selection.
