# Track Vision

Track Vision runs locally in the Electron desktop app. Select and share a simulator window from the **Session Recording** bar, then open **Live Session → Add Visualization → Track Vision** to analyze that stream. The uploaded `track-segments` and `track-position` ONNX models supply segmentation and position in separate GPU inference passes. Optional **Depth** uses the bundled Depth-Anything-V2-Small model. Each captured frame runs through segmentation and, when enabled, depth while retaining the car interior. **Reconstructed scene** shows left and right track boundaries plus all accepted cars and car packs in 2D camera image space. The final **Bird's-eye view** projects that scene onto a flat road and includes your car as the reference point. The Capture tab shows the clean source frame; filtered depth colors appear in Label depths.

Car interior masks remain available through filtering and label-depth inspection, including detections accepted below the stricter track confidence threshold. They do not cut other masks upstream. Depth receives the full captured image, including cockpit and unlabelled pixels; only letterbox-padding output is discarded. The final scene uses the interior mask to suppress track-outline points on or near cockpit edges. If segmentation or enabled depth is unavailable or fails, scene reconstruction waits for those models to be ready; no partial result or previous-frame mask is published. Empty or interior-only segmentation still permits depth inference.

Use **Expand capture** in the preview to fill most of the app window. Capture, detections, and the optional camera grid continue at the larger size. Choose **Restore capture** or press **Escape** to return to the panel; **Stop capture** is also available in the expanded view.

Use **Display label** in the bottom **Settings** section to show only one label's masks, boxes, and captions, or choose **All labels** (the default). Changing the selection updates the current preview immediately. Detection and scene reconstruction continue using every label.

In **Settings → Track models**, **Segmentation** and **Position** each show their ONNX model, fixed training resolution, and independent load/retry status. Depth has an **Input resolution** dropdown with Low (252 × 252), Medium (392 × 392) and High (518 × 518, default) presets. Changing depth resolution reloads only depth while capture continues. **Enable depth model** can disable depth without interrupting segmentation or position.

## Visual pipeline

The numbered tabs follow one captured frame through eight views:

1. **Capture** — the original shared window, without detection overlays.
2. **Camera position** — the captured frame with an optional reference ground grid; adjust camera height, angles, FOV and offsets in the bottom **Settings** section.
3. **Segmentation** — all returned labels over the camera image and a color legend with instance counts. Use **Settings → Display label** to limit the preview to one label.
4. **Filtering** — masks after confidence filtering, depth ordering and supported hidden-mask completion. Accepted car interior masks remain an independent downstream input. The panel explains the applied rules and shows retained counts.
5. **Depth map** — the entire frame's depth as an opaque red-to-blue heatmap, including unlabeled areas and the car interior. The legend uses the full frame's finite positive relative depths, excluding model padding. Hover over the image to read unitless relative depth at the mouse; the reading follows new frames and accounts for resized or expanded previews. Values are comparable only within a frame, not distances in meters. Invalid pixels remain dark and report no valid depth.
6. **Label depths** — all retained masks appear together over the captured frame. Each mask has a numbered label (such as `track #1` and `track #2`) that matches its own table row with median, near/far relative depth and observed depth-pixel count. Numbers identify masks within the current frame. The overlay and legend share one linear red → orange → yellow → green → cyan → blue scale in unitless relative depth, including supported hidden predictions. Hidden predictions are excluded from table statistics. Pixels without valid depth remain transparent, and masks without measured depth are labeled `No depth`.
7. **Reconstructed scene** — the captured window with track ribbons fitted to segmentation edges and labeled car/car-pack boxes overlaid in 2D. Each visible ribbon section has 50 left/right point pairs, with crossbars and a translucent surface. Missing or cockpit-adjacent edges split the ribbon.
8. **Bird's-eye view** — projects the previous reconstructed scene onto a flat road using the applied camera calibration. Left/right boundaries and the middle line retain their gaps. Your car stays at the origin with forward pointing up; detected cars and car packs are placed using their box-bottom centers. The grid and positions are approximate, and traffic marker sizes are schematic. Apply calibration in **Settings → Camera position** first. This projection works with relative depth and is shared with Live Phrases for traffic and corner-relative position inputs.

The bird's-eye view updates with completed frames, retains the latest frame with a stale label after 2 seconds, and clears when calibration or capture is cleared. Rays above the horizon or beyond 200 m are omitted. Traffic whose ground contact cannot be projected (including boxes clipped at the bottom of the capture) is counted as unplaced. Traffic markers describe the current frame, without assuming opponent headings or tracking identities across frames.

With live motion telemetry, the bird's-eye view also remembers recently visible road edges as they move into the cockpit/bodywork blind area. Dashed, faded boundaries indicate these estimates; solid boundaries come from the current frame. Heading changes and body-relative forward/lateral velocity move the old edges into the current car frame. Speed and gear provide a fallback when body velocity is unavailable. Motion is sampled when the image is captured, before inference. No simulator XYZ positions, world velocity, track maps or lap progress are read by this history.

History covers the nearby road (up to 20 m ahead and 5 m behind), keeps at most 24 captures, and expires original observations after 4 seconds. New visible edges take precedence, missing sections remain separate, and moving traffic is never retained. History clears on capture/model reset, calibration/filter changes, telemetry pause/replay/reset, stale motion (over 500 ms), capture gaps (over 1 second), or abrupt heading changes. Without usable motion, the current-frame view remains available. A fixed driving camera and applied calibration are required; motion alignment and flat-road distances remain approximate.

Window selection and capture start/stop controls live in the **Session Recording** bar. Track Vision consumes that shared stream: closing the panel releases its models and detections while capture continues, and reopening it resumes analysis without another window selection. Ending the live session stops capture. Share the game screen before **Start Recording** to include video from the initial telemetry recording start. Telemetry can record without video. Screen selection and sharing start are locked once recording begins, including while paused or after restoring a draft; stopping capture does not allow video to restart in that session. Starting telemetry waits for any pending screen-sharing request to finish or be cancelled. Each session can have at most one video. The app saves silent MP4 video locally beside the telemetry file in its application data folder (`acla-temp`). Stopping either capture or telemetry finalizes the video. **Show local video** opens the saved file location. **Upload Session** includes the saved video by default; uncheck **Include recorded video** for a telemetry-only upload. Upload waits for the final video write, sends video in bounded chunks, and retains the local video after success or failure. **Discard Session** deletes the local telemetry and its video. Restored drafts rediscover the video from the telemetry file path. Multiple local videos block video upload without deleting any files. The backend enforces the one-video limit and stores video chunks on the same session and provides authenticated list/download routes at `GET /racing-session/:id/videos` and `GET /racing-session/:id/videos/:index`. **Settings** is the single configuration section at the bottom of the window. It is collapsible and opens automatically on detector errors. Camera parameters, applied calibration, the reference-grid toggle, display label, both confidence thresholds and depth input resolution are saved automatically in local storage and restored when Track Vision is reopened. Capture sources must still be selected for each new window-sharing session. Its **Filtering confidence** slider defaults to 65% and updates filtered masks, the reconstructed scene and coaching inputs for the current and future frames. The **Segmentation confidence** slider controls which masks the model returns; car interior masks accepted by detection are retained independently of the filtering threshold. Arrow keys, Home and End navigate the tabs. Switching tabs reuses the latest frame, keeps capture and models running, and preserves camera settings. The diagnostic views do not modify published detections or coaching inputs.

## Backend model contract

Track Vision requests `GET /ai-model/ultralytics?name=track-segments` and `GET /ai-model/ultralytics?name=track-position` with the normal JWT. It filters for each exact name, its corresponding `segment` or `position` task, and a `.onnx` filename (case-insensitive), then independently selects the newest ONNX upload by its parsed `createdAt` timestamp. Equal timestamps use descending `_id` as a deterministic tie-breaker. Selection does not depend on catalogue order or version text in a name; missing or invalid upload dates are reported as an error. If a matching model is unavailable, its loader reports an error. A selected segmentation catalogue entry is converted to:

```json
{
  "id": "MongoDB model ID",
  "name": "track-segments",
  "task": "segment",
  "filename": "best.onnx",
  "classNames": ["track", "curb"],
  "sizeBytes": 123456,
  "sha256": "ONNX SHA-256",
  "downloadPath": "/ai-model/ultralytics/<id>/file"
}
```

`POST /ai-model/ultralytics` publishes a `.onnx` model with metadata. For segmentation, array positions in `classNames` are class IDs; position metadata uses the six regression axis names. The frontend checks the metadata when loading a model and uses the authenticated `GET /ai-model/ultralytics/:id/file` route only if verified ONNX bytes are missing locally. Upload a new ONNX model named `track-segments` or `track-position` to update that model on the next load. Cache entries use the selected model ID and checksum, so repeated uploads under the same name remain separate versions.

## Local storage and inference

Electron stores the original backend ONNX bytes under:

```text
<app userData>/track-vision/models/<model-id>-<sha256>/
  weights.onnx
```

Downloads and cached files are checked against the backend byte length and SHA-256 before use. Each downloaded model goes directly into its own ONNX Runtime Web session at its published `metadata.imgsz` training resolution, with no local conversion or Python dependency. `track-segments` takes only the image and returns segmentation predictions and mask prototypes (Ultralytics `output0`/`output1`, in graph output order). Both raw and end-to-end segmentation outputs are supported. `track-position` returns only signed `deltas`; its regression labels are never used for segmentation.

Verified ONNX files survive app restarts; damaged files trigger a fresh download. Both models’ metadata is requested from the backend when loading, so Track Vision requires a backend connection. Depth loads independently from the bundled `public/vision-models/depth-anything-v2-small.onnx` and needs no backend connection.

`npm run setup:vision` copies the pinned ONNX Runtime Web assets from node_modules into the app; it never downloads models.

To prepare all dependencies and bundled depth weights, run `npm run setup:all`. For vision packages alone, run `npm run setup:python -- --mode=vision` to install `scripts/vision-requirements.txt` in `.venv/track-vision`, then run `npm run setup:vision-models`. The model script uses `.venv/track-vision` when present; `VISION_PYTHON` selects another interpreter when running it separately. During setup, it downloads the official `depth-anything/Depth-Anything-V2-Small-hf` pretrained weights at a pinned revision and exports them to ONNX, validating/reusing prepared models thereafter. Static float32 ONNX inputs are `[1, 3, size, size]` for sizes 252, 392 and 518, with RGB ImageNet normalization; outputs are `[1, size, size]` relative disparity. The default 518px export keeps the filename `depth-anything-v2-small.onnx`; the smaller exports use `depth-anything-v2-small-252.onnx` and `depth-anything-v2-small-392.onnx`. These generated files are ignored by Git and copied into the build with the public assets. Its Apache 2.0 license is included in `public/vision-models/DEPTH-ANYTHING-V2-LICENSE.txt`.

Inference requires WebGPU for all models. GPU failures retry after three seconds. Model retrieval and validation errors offer a manual Retry button. Screen preview remains available when a model cannot load, but scene reconstruction waits for segmentation and enabled depth. Missing, loading or failed position inference does not block the scene. Depth can be disabled; there is no CPU fallback.

All frames and inference stay on this device. Captured frames run sequentially at up to 5 FPS. GPU initialization, inference, and disposal share one queue across Track Vision panels.

## Results

### Position model

Track Vision queries the Ultralytics model catalogue once for `track-position`,
filters for that exact name, the `position` task and an ONNX filename, and selects the newest ONNX upload
by `createdAt` with the same `_id` tie-breaker. Electron verifies and caches the original ONNX bytes.
Format/version labels are informational: new, arbitrary and missing labels are accepted.
Compatibility comes from the published graph, input resolution, reference-coordinate semantics and
its executable tensor interface. Changing a version label alone never blocks loading;
changing the network's required inputs still requires their definition and live values.
The runtime warms up the downloaded graph to verify its executable inputs and outputs.
Position runs separately on WebGPU at its saved training resolution.
**Settings → Track models → Position** shows its loading, availability and retry status.
A missing position ONNX model leaves segmentation, depth and screen preview available.

Keep **Live Map** open with its reference lap loaded. For each captured
image, Track Vision samples fresh live `Graphics_normalized_car_position` and asks
that map for the corresponding reference-lap XYZ point.
The lookup interpolates by normalized position, including the start/finish seam.
Map camera zoom, flips and screen coordinates do not affect these inputs. Track Vision
does not download reference data again and never passes driver XYZ to the model.

The model receives the full letterboxed image and reference-lap XYZ
as a `[1, 1, 3]` tensor. Centerline points are not required for position inference.
When its graph requests context, it also receives
`[normalized_position, one-hot game, one-hot map_name]`, using the published category
vocabulary in its saved order. Progress and game come from the capture-time telemetry;
map names use the loaded map's display name or simulator key. Unknown category labels
produce an all-zero one-hot vector. Two-input ONNX graphs run without context.
The final **Bird's-eye view** step plots the reference-lap marker directly
on the top-down scene relative to your car at the origin, alongside its signed XYZ
offset. It uses the output's X/Y values as supplied, with Z as height, without swapping
axes for either game. The view fits the marker, including negative offsets, and can
show it before camera calibration. Stale results are labeled; stopping capture or
resetting telemetry clears the marker. The signed XYZ output triple is shown here
and published as `TrackVisionDetection.position.referenceDelta`,
with `references`, `normalizedPosition`, and `inferenceMs`. It remains in source
coordinates: reference XYZ minus inferred driver XYZ. Missing references, paused or
stale telemetry skip position inference while image-only segmentation continues; resetting telemetry or changing the loaded
map/reference during inference discards the pending position prediction.

The browser loads segmentation predictions and prototypes from its separate
segmentation graph. The standalone `train-position` export supplies position deltas.

### Detector results

`TrackVisionModel.loadBackend()` loads the segmentation graph through the desktop cache. Its `detect(canvas, confidence)` returns instance masks, normalized boxes, class IDs, inference time and ordered class names. `TrackPositionModel.load()` separately loads the position graph; `detect(canvas, references, context?)` returns the signed reference-lap XYZ delta triple and inference time. Position inference runs only with fresh telemetry and reference points. `TrackVisionModel.loadBuiltin('depth', inputSize = 518)` loads bundled depth weights. Depth requires `detect(canvas, confidence, region)`, where `region` covers the full same-frame segmentation grid, including car interior, and returns a copied float32 map with `scale: 'relative'` and excluded pixels zeroed. Disparity is converted to `1 / (1 + disparity)` so smaller values remain nearer and zero disparity remains finite; these values are not meters. All models letterbox the full capture into their square inputs. Segmentation boxes are normalized using the training input size. Call `dispose()` when finished.

`LiveTrackVision` registers a `TrackVisionHandle` as `visualization:track-vision`. Consumers use `getLatestDetection()` or `subscribeDetection(listener)`. Results include the source timestamp and dimensions, successful segmentation under `detections.segment`, and depth under `detections.depth`. Masks, boxes, and depth maps refer to the letterboxed model input; drawing segmentation removes its padding. Results clear on stop, configuration changes, and unmount. Depth values remain available to detection consumers; the Label depths tab visualizes retained label depths as a heatmap.

The Segmentation tab draws the model's returned masks, boxes and confidence labels
directly, including `car interior`. Only the selected **Display label** controls
which returned instances are drawn; overlapping masks are composited in detection
order. Interior subtraction, filtering confidence thresholds, depth ordering and
hidden-mask completion do not modify the capture overlay.

Filtering uses `amodal-masks.ts` for retained instances, including track, traffic,
roadside and arbitrary uploaded labels. Visible pixels are assigned by depth support,
with smaller silhouettes and confidence breaking ties. Car interior masks are retained
independently without cutting other labels. Raw masks and published depth stay unchanged.

## Scene data and Live Phrases

Each published `TrackVisionDetection` includes `calibration`, `reconstructedScene`,
`birdsEyeScene`, `reconstruction`, and `geometry` alongside the raw detector results.
Track Vision does not calculate or publish a separate driver/opponent position summary.

Live Phrases reads `birdsEyeScene`, the same projection rendered in the BEV tab, so
unavailable metric depth does not withhold its inputs. It interprets BEV boundaries and
traffic to determine corner-relative positions and select sentences. The BEV is published
on every completed frame regardless of the selected tab; clearing calibration or capture
clears it, and Live Phrases expires it after 2 s. Remembered BEV points carry their original
`observedAt` timestamp.

Track Vision must run with a **fixed, forward-facing driving view**. The uploaded
model's existing labels are sufficient; class IDs follow its returned label order.
Label matching ignores case and normalizes whitespace.

| Labels | Use in scene reconstruction |
| --- | --- |
| `track` | Identifies the racing corridor and anchors left/right boundary construction |
| `car` | Locate an individual opponent relative to track edges at its depth |
| `car pack` | Establish traffic ahead and bridge road occlusion; a group does not supply an individual opponent position |
| `curb`, `grass`, `sand`, `Outfield asphalt road` | Their nearby inner edges refine the track outline; their surfaces remain excluded even where they overlap track |
| `car interior` | Retained through filtering and depth; rejects nearby track outline points in the final 2D scene and excludes bodywork from coaching geometry |
| `other`, `fence` | Excluded obstacles that block boundary expansion |

Legacy track aliases (`road`, `asphalt`, `tarmac`) and individual-car aliases
remain supported, but are not required. `Outfield asphalt road` is never a track
alias. The 2D scene requires segmentation; metric reconstruction requires segmentation, metric depth and applied calibration. Depth-Anything-V2-Small supplies relative depth, so metric reconstruction and geometry remain unavailable even with camera calibration; 2D reconstruction and the calibrated BEV remain available.
Road geometry requires track labels; vehicle reconstruction additionally requires car or car-pack masks. Scene reconstruction
uses the configured filtering confidence (65% by default), carried on each frame as `filterConfidence`.

## Camera position

Enter camera height above the road in meters, pitch (positive down), yaw (positive
right), horizontal field of view, and lateral/forward offsets relative to the car
origin. A left-seat camera has a negative lateral offset. The centered pinhole
camera assumes square pixels and zero roll. Initial values are a draft; select
**Apply camera calibration** to publish the BEV used by Live Phrases. Metric geometry additionally requires metric depth. The 2D scene is independent of camera settings.

**Enable on capture** shows a reference ground grid on the source image for
checking camera placement. The grid is a calibration aid and supplies no reconstructed geometry.
Editing camera parameters, clearing calibration or changing capture dimensions clears applied
calibration and coaching geometry, including the saved calibration. Stopping capture clears live
results while retaining settings. Applied calibration is restored for captures with matching
dimensions and remains available during model loading or failure. Reapply after changing the car, seat, camera or FOV.

## Reconstructed scene

`reconstructed-scene.ts` collects the outer left and right extents of accepted track masks
as fitting observations in camera image coordinates. It uses the original track coverage before any cockpit cutout,
so a dashboard or pillar does not introduce a new road edge. Points inside or near an accepted
car interior mask are omitted, using a margin scaled to segmentation resolution. Interior
holes do not become track boundaries. Overlapping traffic and other labels do not cut
the original track outline, and sharp bends or width changes do not interrupt tracing.
Capture-clipped sides and missing or cockpit-adjacent pixels split the paired observations;
a new section begins where both edges are available again. Letterbox padding is removed when mapping to the
source image. Raw detections are not modified.

`track-ribbon.ts` constructs exactly 50 paired stations per continuous supported section,
spaced evenly along image height from the farthest to the nearest observed row. Both edges
are fitted at those shared stations with weighted local linear regression. This smooths
mask jaggedness while following perspective taper, multiple bends and changing width.
The middle line is the midpoint of each fitted pair. Isolated rows cannot establish a
ribbon, and fitting never extends beyond a section's observed range or bridges its gaps.
The bird's-eye view projects these fitted edges and middle lines.

The published `reconstructedScene` contains image dimensions, `ribbons` with 50 `pairs`
of `left`/`right` points each, and derived left/right and `centerline` polylines, plus `cars`
with class ID, confidence, pack flag and boxes in image pixels.
Every car or car-pack instance meeting the configured filtering confidence threshold is shown,
including traffic outside the metric reconstruction region and frames without track boundaries.
Boxes are mapped out of letterbox padding and clipped to the captured image; invalid or
fully offscreen boxes are omitted. The Segmentation tab's display-label filter does not affect them.
The view draws a translucent ribbon with paired points and crossbars, boundaries in green
and blue, the middle line in dashed white, individual cars in amber and car packs with
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
non-drivable surfaces for metric coaching inputs. Point clouds stay internal to these
calculations; display polygons, road display points and rolling scene memory are not part
of the published result.
