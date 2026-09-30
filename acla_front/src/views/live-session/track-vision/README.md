# Track Vision

Track Vision runs locally in the Electron desktop app. Open **Live Session → Add Visualization → Track Vision**, select a simulator window, and share it. Segmentation and **Depth** start enabled. Segmentation shows the uploaded model name and its labels; depth uses the bundled YOLO26n model. Each captured frame runs through segmentation and depth while retaining the car interior. The final **Reconstructed scene** shows left and right track boundaries plus all accepted cars and car packs in 2D camera image space. The Capture tab shows the clean source frame; filtered depth colors appear in Label depths.

Car interior masks remain available through filtering and label-depth inspection, including detections accepted below the stricter track confidence threshold. They do not cut other masks upstream. Depth receives the full captured image, including cockpit and unlabelled pixels; only letterbox-padding output is discarded. The final scene uses the interior mask to suppress track-outline points on or near cockpit edges. If segmentation is disabled, unavailable or fails, depth waits for a new same-frame result; no previous-frame mask is reused. Empty or interior-only segmentation still permits depth inference.

Use **Expand capture** in the preview to fill most of the app window. Capture, detections, and the optional camera grid continue at the larger size. Choose **Restore capture** or press **Escape** to return to the panel; **Stop capture** is also available in the expanded view.

Use **Display label** in the Segmentation tab to show only one label's masks, boxes, and captions, or choose **All labels** (the default). Changing the selection updates the current preview immediately. Detection, screen analysis and scene reconstruction continue using every label.

## Visual pipeline

The numbered tabs follow one captured frame through six views:

1. **Capture** — the original shared window, without detection overlays.
2. **Camera position** — camera height, angles, FOV and offsets beside the captured frame, with an optional reference ground grid.
3. **Segmentation** — all returned labels over the camera image, a color legend with instance counts, and optional display-label selection.
4. **Filtering** — masks after confidence filtering, depth ordering and supported hidden-mask completion. Accepted car interior masks remain an independent downstream input. The panel explains the applied rules and shows retained counts.
5. **Label depths** — an amber-to-blue heatmap over the retained masks and a table of each label's median, near/far distance and observed depth-pixel count. Distances are estimated optical-axis meters; hidden predictions are excluded from table statistics. Missing or invalid depth remains unavailable rather than becoming zero distance.
6. **Reconstructed scene** — the captured window with track boundaries and labeled car/car-pack boxes overlaid in 2D, with cockpit-adjacent edges removed, plus screen analysis. These overlays require segmentation only.

Capture controls stay available above the tabs. Model settings are collapsible and open automatically on detector errors. Arrow keys, Home and End navigate the tabs. Switching tabs reuses the latest frame, keeps capture and models running, and preserves camera settings. The diagnostic views do not modify published detections or coaching analysis.

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

Inference uses WebGPU by default. **Allow CPU fallback** permits either model to run in a CPU WASM worker if GPU initialization fails. GPU-only failures retry after three seconds. Model retrieval, export, and CPU errors offer a manual Retry button. Screen preview remains available when a model cannot load. Disabling a detector releases its inference session. Capture continues; segmentation can run without depth, while depth requires segmentation masks.

All frames and inference stay on this device. Captured frames run sequentially at up to 5 FPS. GPU initialization, inference, and disposal share one queue across Track Vision panels.

## Results

`TrackVisionModel.loadBackend(allowCpuFallback = false)` loads segmentation through the desktop cache. `TrackVisionModel.loadBuiltin('depth', allowCpuFallback = false)` loads bundled depth weights. `detect(canvas, confidence)` returns instance masks, normalized boxes, class IDs, inference time, and ordered class names for segmentation. Depth requires `detect(canvas, confidence, region)`, where `region` covers the full same-frame segmentation grid, including car interior, and returns a copied float32 distance map with excluded pixels zeroed. Call `dispose()` when finished.

`LiveTrackVision` registers a `TrackVisionHandle` as `visualization:track-vision`. Consumers use `getLatestDetection()` or `subscribeDetection(listener)`. Results include the source timestamp and dimensions, successful segmentation under `detections.segment`, and depth under `detections.depth`. Masks, boxes, and depth maps refer to the letterboxed model input; drawing segmentation removes its padding. Results clear on stop, configuration changes, and unmount. Depth values remain available to detection consumers; the Label depths tab visualizes retained label depths as a heatmap.

The Segmentation tab draws the model's returned masks, boxes and confidence labels
directly, including `car interior`. Only the selected **Display label** controls
which returned instances are drawn; overlapping masks are composited in detection
order. Interior subtraction, analysis confidence thresholds, depth ordering and
hidden-mask completion do not modify the capture overlay.

Filtering uses `amodal-masks.ts` for retained instances, including track, traffic,
roadside and arbitrary uploaded labels. Visible pixels are assigned by depth support,
with smaller silhouettes and confidence breaking ties. Car interior masks are retained
independently without cutting other labels. Raw masks and published depth stay unchanged.

## Screen analysis

The **Screen analysis** panel in the Reconstructed scene tab shows the visible corner, driver position, and
closest supported opponent position. Analysis runs here even when Live Phrases
is closed and no telemetry is available. Each published `TrackVisionDetection`
includes `calibration`, `reconstructedScene`, `reconstruction`, `geometry`, and `analysis` alongside the raw detector results. `analysis` is null without
segmentation; uncertain fields remain unset. Live Phrases consumes these fields
only to select sentences. Displayed positions clear when the frame is more than
2 s old, capture stops, or detector configuration changes.

Track Vision must run with a **fixed, forward-facing driving view**. The uploaded
model's existing labels are sufficient; class IDs follow its returned label order.
Label matching ignores case and normalizes whitespace.

| Labels | Use in position analysis |
| --- | --- |
| `track` | Identifies the racing corridor and anchors left/right boundary construction |
| `car` | Locate an individual opponent relative to track edges at its depth |
| `car pack` | Establish traffic ahead and bridge road occlusion; a group does not supply an individual opponent position |
| `curb`, `grass`, `sand`, `Outfield asphalt road` | Their nearby inner edges refine the track outline; their surfaces remain excluded even where they overlap track |
| `car interior` | Retained through filtering and depth; rejects nearby track outline points in the final 2D scene and excludes bodywork from coaching geometry |
| `other`, `fence` | Excluded obstacles that block boundary expansion |

Legacy track aliases (`road`, `asphalt`, `tarmac`) and individual-car aliases
remain supported, but are not required. `Outfield asphalt road` is never a track
alias. The 2D scene requires segmentation; metric position analysis requires segmentation, depth and applied calibration.
Road geometry requires track labels; opponent analysis additionally requires car or car-pack masks. Position analysis
requires detection confidence of at least 65%.

## Camera position

Enter camera height above the road in meters, pitch (positive down), yaw (positive
right), horizontal field of view, and lateral/forward offsets relative to the car
origin. A left-seat camera has a negative lateral offset. The centered pinhole
camera assumes square pixels and zero roll. Initial values are a draft; select
**Apply camera calibration** to publish metric geometry and positions. The 2D scene is independent of camera settings.

**Enable on capture** shows a reference ground grid on the source image for
checking camera placement. The grid is a calibration aid and supplies no reconstructed geometry.
Editing camera parameters clears applied calibration and coaching geometry. Stopping/restarting
capture or changing its dimensions clears calibration too. Calibration remains available during
model loading or failure. Reapply after changing the car, seat, camera or FOV.

## Reconstructed scene

`reconstructed-scene.ts` traces the outer left and right extents of accepted track masks
in camera image coordinates. It uses the original track coverage before any cockpit cutout,
so a dashboard or pillar does not introduce a new road edge. Points inside or near an accepted
car interior mask are omitted, using a margin scaled to segmentation resolution. Interior
holes do not become track boundaries. Traffic, excluded labels and capture-clipped sides
are not accepted as visible edges. Missing rows and abrupt jumps split the lines; hidden
sections are never joined across a gap. Letterbox padding is removed when mapping to the
source image. Raw detections are not modified.

The published `reconstructedScene` contains image dimensions and separate arrays of
left/right polylines, plus `cars` with class ID, confidence, pack flag and boxes in image pixels.
Every car or car-pack instance meeting the 65% reconstruction confidence threshold is shown,
including traffic outside the coaching analysis region and frames without track boundaries.
Boxes are mapped out of letterbox padding and clipped to the captured image; invalid or
fully offscreen boxes are omitted. The Segmentation tab's display-label filter does not affect them.
The view draws boundaries in green and blue, individual cars in amber and car packs with
dashed purple boxes and confidence labels over the matching captured window frame,
preserving its aspect ratio and alignment with the overlays. The captured
frame remains visible even when segmentation is unavailable. There is no 3D projection,
orbit control, depth requirement or camera-calibration requirement.

The last completed scene and its captured frame stay visible while the next frame processes and are marked stale
after two seconds. New results replace it, including empty scenes. Capture stop and detector
resets clear the display. Camera edits do not renew capture timestamps or change the 2D edges.

## Depth and coaching geometry

Filtering and label depths retain the car interior as a separate input. Other overlapping
labels still use depth ordering and completion between supported visible endpoints. Missing,
invalid and padded depth samples are unavailable; hidden predictions are excluded from
label-depth table statistics.

The `reconstruction` field contains measured vehicle bounds, track boundaries and fitted
`geometry` for coaching consumers. Calibrated depth calculations exclude bodywork and
non-drivable surfaces for metric coaching analysis. Point clouds stay internal to these
calculations; display polygons, road display points and rolling scene memory are not part
of the published result.
