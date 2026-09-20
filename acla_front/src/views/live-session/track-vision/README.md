# Track Vision

Track Vision runs locally in the Electron desktop app. Open **Live Session → Add Visualization → Track Vision**, select a simulator window, and share it. Segmentation starts enabled and shows the uploaded model name and its labels. **Depth** remains available as an independent, optional detector using the bundled YOLO26n depth model. Enable it to show estimated distances beneath segmentation; the Close and Far controls tune the overlay colors.

## Backend model contract

`GET /ai-model/ultralytics/track-vision` requires the normal JWT and returns the newest uploaded `segment` model, ordered by `createdAt` and `_id`. It returns 404 when none is available. The response contains:

```json
{
  "id": "MongoDB model ID",
  "name": "track-features",
  "task": "segment",
  "classNames": ["track", "curb"],
  "sizeBytes": 123456,
  "sha256": "checkpoint SHA-256",
  "downloadPath": "/ai-model/ultralytics/<id>/file"
}
```

`POST /ai-model/ultralytics` publishes a `.pt` checkpoint with metadata. Array positions in `classNames` are class IDs. The frontend checks the metadata when loading a model and uses the authenticated `GET /ai-model/ultralytics/:id/file` route only if that checkpoint is missing locally. Upload a new segmentation model to make it the version selected on the next model load.

## Local storage and inference

Electron stores backend checkpoints, ONNX exports, and their metadata under:

```text
<app userData>/track-vision/models/<model-id>-<sha256>/
  weights.pt
  weights.onnx
  manifest.json
```

Downloads are checked against the backend byte length and SHA-256 before saving. On first use, the desktop Python environment converts the verified checkpoint to a static, float32, 640px ONNX segmentation model. Export is offline and automatic package installation is disabled. The exporter validates the model task, class count, and output layout. Backend class names are applied in class-ID order and used by both the label list and overlays.

The original checkpoint is retained if conversion fails, so a retry can export it without another download. Verified ONNX exports survive app restarts; damaged exports are rebuilt from the saved checkpoint. Segmentation metadata is requested from the backend when loading, so segmentation requires a backend connection. Depth loads independently from the bundled `public/vision-models/yolo26n-depth.onnx` and needs no backend connection.

Desktop development and packaging install the conversion dependencies from `src/py-scripts/requirements.txt` through the existing `setup:python` lifecycle. After updating an existing checkout, run `npm run setup:python` and restart Electron, or use `npm run start:electron`, which performs setup automatically. `npm run setup:vision` only copies the pinned ONNX Runtime Web assets from node_modules into the app; it never downloads models.

To prepare the bundled depth weights before development or packaging, install `scripts/vision-requirements.txt` in a Python environment and run `npm run setup:vision-models`. The script uses `.venv/track-vision` when present; `VISION_PYTHON` selects another interpreter. This exports only YOLO26n depth, downloading its checkpoint on first use and reusing prepared weights thereafter. The generated ONNX file is ignored by Git and copied into the build with the public assets. Its Ultralytics license is included in `public/vision-models/ULTRALYTICS-LICENSE.txt`.

Inference uses WebGPU by default. **Allow CPU fallback** permits either model to run in a CPU WASM worker if GPU initialization fails. GPU-only failures retry after three seconds. Model retrieval, export, and CPU errors offer a manual Retry button. Screen preview remains available when a model cannot load. Disabling a detector releases its inference session; the other detector and capture continue running.

All frames and inference stay on this device. Captured frames run sequentially at up to 5 FPS. GPU initialization, inference, and disposal share one queue across Track Vision panels.

## Results

`TrackVisionModel.loadBackend(allowCpuFallback = false)` loads segmentation through the desktop cache. `TrackVisionModel.loadBuiltin('depth', allowCpuFallback = false)` loads bundled depth weights. `detect(canvas, confidence)` returns instance masks, normalized boxes, class IDs, inference time, and ordered class names for segmentation, or a copied float32 distance map for depth. Call `dispose()` when finished.

`LiveTrackVision` registers a `TrackVisionHandle` as `visualization:track-vision`. Consumers use `getLatestDetection()` or `subscribeDetection(listener)`. Results include the source timestamp and dimensions, successful segmentation under `detections.segment`, and depth under `detections.depth`. Masks, boxes, and depth maps refer to the letterboxed model input; drawing removes its padding. Results clear on stop, configuration changes, and unmount. Changing the depth range immediately recolors the displayed frame without rerunning inference.
