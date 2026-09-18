# Track Vision

Track Vision is available only in the Electron desktop app, within Live Session.
Open **Live Session → Add Visualization → Track Vision**. The **Ultralytics
detection stack** has three independent switches. Semantic starts enabled;
Depth and Segment start off. Toggle any combination while sharing a game window.
Each detector shows its own loading, ready/running, error, and inference-device state.

| Option | Bundled model | Output |
| --- | --- | --- |
| Semantic | YOLO26n-sem, Cityscapes | Per-pixel scene classes, including road and surrounding objects |
| Depth | YOLO26n-depth | Per-pixel estimated distance, displayed as a warm-to-cool depth map |
| Segment | YOLO11n-seg, COCO | Separate object masks, class labels, boxes, and confidence |

Segment confidence applies only to instance detection. Semantic has no per-object
confidence threshold. Generic COCO weights do not include a racing-track class.
Depth is an estimate from a pretrained model, not simulator telemetry or validated
track geometry. Simulator accuracy depends on view, lighting, and scene content.

Enable **Depth** to show the **Depth range** meter. Adjust **Close** and **Far**
in estimated meters: distances at or below Close use warm colors, distances at
or above Far use cool colors, and the range between them blends smoothly.
The range stays fixed across frames and slider changes recolor the current
preview immediately. Close and Far stay at least 0.5 m apart; the controls cover
0–200 m. **Reset depth range** restores Close to 5 m and Far to 50 m.

## Prepare models once

Inference runs locally in the Electron renderer. Python is used only to export the packaged models:

```sh
python -m venv .venv/track-vision
# Windows:
.venv/track-vision/Scripts/python -m pip install -r scripts/vision-requirements.txt
# macOS/Linux:
.venv/track-vision/bin/python -m pip install -r scripts/vision-requirements.txt
npm run setup:vision-models
npm run setup:vision
```

The model setup script uses that project environment when present, or `python`.
Set `VISION_PYTHON` to choose another interpreter. It downloads official pretrained
weights and exports static 640px, batch-one ONNX models and class-name metadata to
`public/vision-models`. Subsequent runs reuse those files. The pinned Ultralytics
export version is in `scripts/vision-requirements.txt`. Runtime assets come from
the pinned `onnxruntime-web` dependency.

Start/build scripts prepare runtime files and report missing models without
blocking the rest of the app. A detector with missing weights shows an error and
a Retry button; other detectors and screen capture remain usable. For offline
builds, preserve the exported ONNX files and their JSON metadata. ONNX files are
excluded from Git and included in builds through `public`. Exported models use
the upstream Ultralytics license; `ULTRALYTICS-LICENSE.txt` is packaged with them.

## Capture and results

Refresh sources, select the simulator window or screen, then click
**Share game screen**. Capture requires the Electron screen-capture bridge and an
explicit source selection. **Save frame** saves the original frame without overlays.
No audio is captured. All frames and inference stay on this device.

The stack processes one captured frame at a time, sequentially across enabled
models, at no more than 5 FPS. Actual speed depends on the selected models and
hardware. Every overlay matches its captured frame. Models try WebGPU first in
the Electron renderer, requesting the high-performance GPU. All detectors in the
renderer share one GPU queue for initialization, warm-up, inference, and disposal,
so multiple detectors can stay enabled and take turns without overlapping GPU work.
CPU worker operations do not wait on this queue. **Allow CPU fallback**
starts off: failed GPU initialization, warm-up, or live inference automatically
retries after 3 seconds, once any queued model loads finish. Other detectors and
screen capture remain usable. Enable the toggle to allow a CPU WASM worker when
GPU loading fails; GPU is still tried first. Turning it off releases CPU models
and retries them on GPU. Disabling a detector or closing the panel cancels its
retries. Missing weights and CPU failures still require a manual Retry.
Disabling a detector releases its model, clears stale published results and
removes its overlay. Disabling all detectors keeps raw screen capture available.

`TrackVisionModel.loadBuiltin(task, allowCpuFallback = false)` loads `semantic`,
`depth`, or `segment`. GPU failures throw `GpuInferenceError` when fallback is off;
the panel schedules retries.
`detect(canvas, confidence)` returns task-specific masks/maps with processing
time and class names. Call `dispose()` when finished. The semantic decoder handles
baked integer class maps and float logits; depth copies its float output before
tensor disposal; Segment decodes raw YOLO11 outputs with class-aware NMS.

`LiveTrackVision` registers a `TrackVisionHandle` as `visualization:track-vision`.
Consumers call `getLatestDetection()` or `subscribeDetection(listener)` (compatible
with `useSyncExternalStore`). Results contain `capturedAt`, source `width` and
`height`, and a `detections` object keyed by enabled, successful tasks. Each task
includes its own output dimensions and inference time. Masks/maps refer to the
letterboxed 640px input; the preview removes padding when drawing. Segment boxes
are normalized to that model input. Results become `null` on stop, configuration
change, or unmount. Use capture timestamps to assess freshness.

The earlier YOLOP and track-boundary decoding helpers remain available separately;
they are not used by the Track Vision stack.

References: [Ultralytics Semantic](https://docs.ultralytics.com/tasks/semantic/),
[Depth](https://docs.ultralytics.com/tasks/depth/),
[Segment](https://docs.ultralytics.com/tasks/segment/),
[ONNX Runtime WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
