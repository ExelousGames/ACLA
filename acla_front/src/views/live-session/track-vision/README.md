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

## Screen analysis

The **Screen analysis** panel shows the visible corner, driver position, and
closest supported opponent position. Analysis runs here even when Live Phrases
is closed and no telemetry is available. Each published `TrackVisionDetection`
includes `analysis` alongside the raw detector results. `analysis` is null without
segmentation; uncertain fields remain unset. Live Phrases consumes these fields
only to select sentences. Displayed positions clear when the frame is more than
2 s old, capture stops, or detector configuration changes.

Track Vision must run with a **fixed, forward-facing driving view**. The uploaded
model's existing labels are sufficient; class IDs follow its returned label order.
Label matching ignores case and normalizes whitespace.

| Labels | Use in position analysis |
| --- | --- |
| `track` | Racing surface used to follow the road |
| `left_boundary`, `right_boundary` | Refine the left and right edges inside the detected road region; an absent side falls back to the track mask |
| `car` | Locate an individual opponent relative to track edges at its depth |
| `car pack` | Establish traffic ahead and bridge road occlusion; a group does not supply an individual opponent position |
| `curb`, `grass`, `other`, `fence`, `sand`, `Outfield asphalt road` | Excluded from usable track, even where their masks overlap the track mask |

Legacy track aliases (`road`, `asphalt`, `tarmac`) and individual-car aliases
remain supported, but are not required. `Outfield asphalt road` is never a track
alias. Depth is optional and is not used for position analysis. Position analysis
requires detection confidence of at least 65%.

`track-position-analysis.ts` removes model letterboxing and follows visible track edges
at five image depths. Both edges must curve in the same direction to identify a
left- or right-hand corner. A simple screen offset or slanted straight road does
not establish a corner. Missing or clipped edges and conflicting bends withhold
position estimates. Car occlusions can be bridged between observed track pixels;
kerbs and other non-track masks do not count as road.

The player's position is estimated from an explicitly calibrated **car center**
relative to the foreground track edges. Image center is never a default player
position: a left-seat cockpit camera is offset from the car center.

In Track Vision, use **Car center alignment** for the current capture. Align the
crosshair with the vehicle's own centerline at the marker height (80% down the
captured image), using the visible nose or bonnet, then select **Set car center**.
The car can be anywhere across the track during alignment. Road geometry does
not supply or update this vehicle reference; track edges only classify its
lateral position. Leave alignment unset when the vehicle centerline cannot be
seen. The slider initially shows a draft marker only; positions remain unknown
until alignment is explicitly set.
Changing the slider clears the applied alignment and position estimates.
Stopping capture or changing its resolution clears the reference. Align again
after changing the car, camera, seat position, or field of view. Dynamic
look-to-apex/head movement is not supported by this fixed reference.

The closest supported opponent ahead is located relative
to track edges at its own image depth, including opponents away from the screen
center. Positions are the inside 40%, middle 20%, or outside 40% of the visible
track width. Inside is left in a left-hand corner and right in a right-hand
corner. These are image-based estimates, not measured lateral distances.
