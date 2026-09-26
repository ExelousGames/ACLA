# Track Vision

Track Vision runs locally in the Electron desktop app. Open **Live Session → Add Visualization → Track Vision**, select a simulator window, and share it. Segmentation and **Depth** start enabled. Segmentation shows the uploaded model name and its labels; depth uses the bundled YOLO26n model. Both run on the same captured frame to reconstruct track edges and visible car surfaces in local 3D. Depth does not add a color overlay to the capture preview.

Use **Expand capture** in the preview to fill most of the app window. Capture, detections, and the optional camera grid continue at the larger size. Choose **Restore capture** or press **Escape** to return to the panel; **Stop capture** is also available in the expanded view.

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

`LiveTrackVision` registers a `TrackVisionHandle` as `visualization:track-vision`. Consumers use `getLatestDetection()` or `subscribeDetection(listener)`. Results include the source timestamp and dimensions, successful segmentation under `detections.segment`, and depth under `detections.depth`. Masks, boxes, and depth maps refer to the letterboxed model input; drawing segmentation removes its padding. Results clear on stop, configuration changes, and unmount. Depth values remain available to detection consumers without being rendered as colors.

The capture preview and working map both use `createSegmentationLayers` for overlapping masks. Track coverage is retained independently of cars and car packs, regardless of detection order. The preview composites foreground masks over the track instead of replacing its pixels. Reconstruction uses the retained track for road support while excluding car-covered pixels from road depth sampling. Raw instance masks remain unchanged.

## Screen analysis

The **Screen analysis** panel shows the visible corner, driver position, and
closest supported opponent position. Analysis runs here even when Live Phrases
is closed and no telemetry is available. Each published `TrackVisionDetection`
includes `calibration`, `reconstruction`, `geometry`, and `analysis` alongside the raw detector results. `analysis` is null without
segmentation; uncertain fields remain unset. Live Phrases consumes these fields
only to select sentences. Displayed positions clear when the frame is more than
2 s old, capture stops, or detector configuration changes.

Track Vision must run with a **fixed, forward-facing driving view**. The uploaded
model's existing labels are sufficient; class IDs follow its returned label order.
Label matching ignores case and normalizes whitespace.

| Labels | Use in position analysis |
| --- | --- |
| `track` | Racing surface whose mask defines the left and right track edges |
| `car` | Locate an individual opponent relative to track edges at its depth |
| `car pack` | Establish traffic ahead and bridge road occlusion; a group does not supply an individual opponent position |
| `curb`, `grass`, `other`, `fence`, `sand`, `Outfield asphalt road` | Excluded from usable track, even where their masks overlap the track mask |

Legacy track aliases (`road`, `asphalt`, `tarmac`) and individual-car aliases
remain supported, but are not required. `Outfield asphalt road` is never a track
alias. Reconstruction and position analysis require both segmentation and depth.
Road geometry requires track labels; opponent analysis additionally requires car or car-pack masks. Position analysis
requires detection confidence of at least 65%.

## Camera position

Enter camera height above the road in meters, pitch (positive down), yaw (positive
right), horizontal field of view, and lateral/forward offsets relative to the car
origin. A left-seat camera has a negative lateral offset. The centered pinhole
camera assumes square pixels and zero roll. Initial values are a draft; select
**Apply camera calibration** to publish the reconstruction and positions.

The local 3D preview uses the same camera height, angles, field of view and offsets
shown in the camera controls for both reconstruction and display. It preserves
the capture aspect ratio and updates from draft settings without rerunning
inference. There is no separate display camera or automatic camera override;
invalid settings hide the preview until corrected.
**Enable on capture** shows a reference ground grid on the source image for
checking camera placement. The grid is only a calibration aid; it supplies no
reconstructed geometry. The perspective 3D panel shows track edges, visible car
surface points and their estimated bounds. Car packs are displayed separately
from individual opponents. No top-down image warp or display range controls remain.

The local 3D distance grid follows contours of the detected road's measured depth,
using the same vehicle-forward meters as car labels. It retains road elevation
instead of assuming a flat ground plane, and works without a successful road-edge
fit. Cars, excluded surfaces, missing depth and the boundary cutoff interrupt the
grid; absent or stale road observations show no distance grid. The optional grid
on the capture remains a separate camera-calibration reference.

Editing any camera parameter clears the applied calibration and published geometry.
Stopping/restarting capture or changing its dimensions clears calibration too.
Calibration remains available during model loading or failure. Reapply after
changing the car, seat, camera or FOV. Dynamic camera motion is unsupported.

## Local 3D reconstruction

`depth-projection.ts` samples the estimated optical-axis depth in meters, removes
model letterboxing and unprojects through the calibrated pinhole camera. Camera
pitch, yaw, height and offsets transform points into vehicle coordinates:
**X right, Y forward, Z up**. Elevation is retained; points are not forced onto a
flat road. Mask and depth resolutions may differ. Bilinear depth sampling accepts
only finite positive values up to 200 m supported by the same semantic surface;
padding, sky, background around cars and excluded track surfaces are not mixed in.

The bundled model's calibrated output is used directly. Its metric scale is an
estimate, especially on simulator images; camera settings describe camera pose,
not a learned depth-scale correction. See the
[Ultralytics depth model contract](https://docs.ultralytics.com/tasks/depth/).

Drag the amber **Boundary start** line just above the hood or cockpit obstruction,
or adjust its slider below the capture. The line starts at 75% of the image height;
100% uses the full frame. The shaded area below it is excluded from track-edge
detection and road-edge depth sampling. Moving the line updates the local 3D
preview and published geometry immediately, without rerunning inference or
renewing the frame timestamp. The line also works in expanded capture. Its
position is retained across capture restarts while the panel stays open.
Raw masks, depth maps, and car reconstruction are unchanged.

`TrackVisionFrame.boundaryStartY` stores the normalized source-image cutoff,
independent of model letterboxing; frames without it use the full image.
`track-position-analysis.ts` scans image rows above this cutoff and lifts each visible mask edge
independently using its depth. Cars can bridge an occlusion between road pixels
but cannot supply a road edge. Other surface masks override road labels. Clipped
edges, invalid depths and abrupt changes reject only the affected edge observation.
Scanning resumes beyond gaps, and the boundary arrays can have different lengths.
Individual car and car-pack masks select depth samples
for visible surface point clouds. Robust bounds and median centers are computed
from those samples, without inventing hidden vehicle surfaces or fixed dimensions.
Cars can be reconstructed even when there is no usable road fit.

`reconstruction` contains `leftBoundary`, `rightBoundary`, `cars`, and a nullable
`geometry`. `road-polynomial.ts` fits each boundary's horizontal trace as
`X(Y) = c0 + c1*Y + c2*Y²`, using robust least squares with at least eight observed
samples spanning 10 m. Width, fit error and opposing bends reject uncertain fits.
Raw 3D edges and cars remain visible even when the horizontal road fit is unresolved.

The existing `geometry` and `analysis` fields remain available to consumers.
Geometry includes the 3D boundary points, horizontal boundary/center fits, observed
distance range, fit error, width, lateral offset, heading and curvature. The driver
reference is X = 0 at the nearest jointly observed cross-section (`referenceY`);
geometry is never extrapolated underneath the unseen car. Opponents require road
support below their mask and a reconstructed position inside the fitted road's
observed range. The nearest supported car is selected by depth; a pack indicates
traffic without assigning an individual position. Inside/middle/outside bands are
40% / 20% / 40% of road width, with inside following the corner direction.

Missing or failed segmentation/depth clears the reconstruction and leaves positions
unknown. Capture can continue with either detector alone. Displayed 3D geometry
and positions expire after two seconds; camera edits never renew a capture timestamp.
Live Phrases restarts its hold when applied camera parameters change.
