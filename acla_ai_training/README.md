# Local AI training

This folder owns dataset preparation, annotation agents/providers, pipeline
manifests, the Streamlit UI, and model trainers. The Python package is `training`.
It reuses the `app` package from `../acla_ai_service` so training and serving use
the same model architecture, serialization, and feature processing.

Annotation label selection is deterministic. Optional AI annotation and
follow-up requests use Claude CLI, OpenAI, or a configured OpenAI-compatible
endpoint. Training does not install or run a local LLM: there is no llama.cpp
server, GGUF conversion, LoRA adapter loading, or Hugging Face LLM download.
Hugging Face dependencies inherited from the shared runtime requirements remain
for knowledge-base embeddings and speech processing in the live service.

Training orchestration and CLI entrypoints live in `training/pipelines/training`.
Model fitting lives in `training/ml` (classifier, cropper, and opportunity
forecaster) and `training/pipelines/training/transformer_trainer.py`.
Segment classifier targets include behavior and segment-type labels, but exclude
track identities and named track sections. Annotation forms and saved labels use
the same catalog. Track section names and ranges for automatic lap splitting and
map overlays come from the backend circuit map's `centerline_segments`, matched
by game and source track key or circuit name. Backend sections refresh within
one minute; missing maps or cleared sections have no hardcoded fallback.
Removed labels are not copied into annotation metadata.
Storage utilities and dataset adapters live in `training/storage`; cropper
target construction and calibration live in `training/ml/segment_cropper`.
The shared scaler lives in `app/ml/transformer/scaler.py` in the runtime package.
Top-lap reference building and payload serialization live in `training/top_laps`;
the enrichment pipeline publishes its reference payload to the backend. Existing
local reference files live in `storage/top_lap_models`. Serving loads the published
payload into memory without saving another copy.

## Image region and boundary annotation and training

[Labelme + Ultralytics](training/image_segmentation/README.md) adds a separate
image segmentation workflow under `training/image_segmentation`: draw polygons
for track, curbs, grass, cars, and other regions, plus left/right track boundary
polylines using Create LineStrip; convert Labelme JSON to YOLO
segmentation datasets; train locally or in the training container. Labelme and
Ultralytics are included in the training dependencies. After rebuilding and
recreating the training container, launch the editor with
`docker exec -it acla_ai_training_c python3 /app/scripts/open_labelme.py` and open
[Labelme in your browser](http://localhost:6080/vnc.html?autoconnect=true&resize=remote).
The launcher starts a virtual desktop when no display is available; Ctrl+C stops
it. Images and annotations under `/app/storage` persist in the host workspace.
Start training with an explicit annotation directory, for example
`docker exec -it acla_ai_training_c python3 /app/scripts/train_labelme.py /app/storage/annotation_images`.
It recursively reads the selected directory, mixes annotated images from all
folders into an 80/20 training/validation split, and saves reusable sample lists in
`storage/image_segmentation/split.json`. Pass `--prepare-only` to export without
training, `--device 0` to use a configured GPU, or `--rebuild-split` to include new
annotations in a fresh split. See the workflow guide for the full options.
Downloaded pretrained weights live in `storage/image_segmentation/pretrained/`;
training checkpoints and metrics live in `storage/image_segmentation/runs/train*/`.
Successful training exports the saved checkpoint as ONNX and uploads only the
`.onnx` file to `POST /ai-model/ultralytics`; `.pt` checkpoints stay local.
Pass `--no-upload` for offline training or `--upload-name` to set the backend model
name. The workflow guide also describes exporting and publishing existing `.pt`
checkpoints without retraining and the ONNX-only local FastAPI upload endpoint.

## Prepare images, telemetry, and position references

Run the guided preparation command to choose sessions,
frame sampling, and alignment settings:

```bash
# From acla_ai_training/:
python scripts/prepare_position_dataset.py
# Or in the training container:
docker exec -it acla_ai_training_c python3 /app/scripts/prepare_position_dataset.py
```

The prompts query `/racing-session/with-videos` and list the configured account's
video sessions with their creation dates in the machine's local timezone.
Choose **Today**, **Yesterday**, **Last 2 days**, **Last 3 days**,
**Last week**, **Pick a session** (by listed number or ID), or **All sessions with videos**
(regardless of creation date). Last 2/3 days and
last week include today; last week means seven calendar days. Preparation accepts
one or more sessions without assigning splits; training requires at least two sessions per map.
Press Enter to finish picking individual sessions or accept an option's default.
Review the settings before choosing Start; no sessions are downloaded before
then, and Ctrl+C cancels. Preparation downloads the selected sessions, extracts
aligned video frames, and prepares reference-lap/centerline XYZ, game, map name,
normalized lap position, and signed offset targets for `train_position.py`.

For unattended use, pass the existing command-line options instead:

```bash
python scripts/prepare_position_dataset.py \
  --sessions SESSION_ID_A SESSION_ID_B --fps 2
```

`--sessions` accepts multiple backend session IDs or local session/frame folders.
Sessions whose local folders already exist are skipped entirely, even if the
folder is incomplete: they are not downloaded, extracted, or processed again.
Only missing sessions are downloaded and prepared. The command reports
`skipped_sessions` alongside frame counts and
`annotation_directories` to open in Labelme. If every session is skipped, it
exits successfully without creating output. Failures
exit with a nonzero status, remove the failed stage's temporary output, and
preserve completed stages. To retry an incomplete session, move its entire
session folder out of the download directory first.

Preparation is self-contained in `acla_ai_training`: its backend client,
reference-lap reader, and feature definitions are local copies. It does not
import `acla_ai_service` or need that folder on `PYTHONPATH`. For a standalone
environment, install `python -m pip install -r requirements.position.txt`.

The command uses the training-local authenticated `backend_service`. Configure
`BACKEND_SERVER_IP`, `BACKEND_PROXY_PORT`, `AI_SERVICE_USERNAME`, and
`AI_SERVICE_PASSWORD` in the environment or a `.env` file in the working directory,
as for other training commands. No separate FastAPI server or manual access token
is needed. The training container already supplies the dependencies and backend
settings. Video endpoints require the configured
account to own the session; the AI service account cannot download another
user's recordings.

The backend needs an active `top_lap_reference` model containing player XYZ and
a circuit map with `samples.middle_line` for each session's game and track.
The command downloads the reference line and centerline map from the backend
to prepare position inputs and targets.

### Prepared files

Downloads and images are kept in `storage/session_downloads/SESSION_ID/`:

- `telemetry.json`: all telemetry rows, in backend chunk order.
- `video_000.mp4`: the session's MP4, named by backend video index.
- `manifest.json`: session metadata and the video's original name, size,
  timestamp, index, and local filename.
- `frames/video_000/frame_000000.jpg`: images numbered by source frame index.
- `frames/video_000/frame_000000.position.json`: the matching image filename,
  session ID, reference XYZ, game, map name, normalized lap position, and signed offsets.
- `frames/manifest.json`: alignment settings and matched/unmatched frame counts.

The video preparation command creates each image and its `.position.json` file
in the same folder. Each position file contains `session_id`, `image`, `game`,
`map_name`, `normalized_position`, `references`, and `target`.
The `references` and `target` arrays are two XYZ triples, ordered reference lap then
centerline. Images and their matching position files can be moved together.

Training discovers `.position.json` files recursively under the folder passed to
`--manifest`, such as `storage/session_downloads`. Preparation creates no aggregate
`positions.json` index or `position-data` directory.

Preparation looks up both reference XYZ positions at each frame's normalized
lap position and calculates `reference - driver` and `centerline - driver` targets.
Game and map names are copied from the selected centerline map's `game` and
`circuit_name`; normalized lap position is retained as a model input.
Driver XYZ, telemetry rows, and reference-model/map snapshots are not passed to
training. The trainer reads the prepared context, coordinates, targets, and images,
so training works offline without source JSON files.
Frame/telemetry matches are used only in memory during preparation;
no per-image `.telemetry.json` files are written. The original session telemetry
remains in the download cache.
Images remain in their extracted folders and are
referenced by relative paths. Preparation validates the saved pairs with the
training loader through folder discovery. Existing images, position files, and
annotations remain available through folder discovery. A run with only existing
sessions leaves all files untouched.

### Frame alignment

New extraction defaults to every frame; `--fps 2` samples two images per second
using video presentation timestamps relative to the first decoded frame,
including variable-frame-rate recordings. Higher rates never duplicate frames.
Telemetry must have finite, nondecreasing `Graphics_analyst_session_time` in seconds.
Matching uses the nearest complete row without interpolation; ties prefer the
earlier row. Frames outside telemetry coverage or farther than
`--max-delta-ms 100` from the nearest row are skipped.

**Alignment assumption:** telemetry uses analyst elapsed time while video `startedAt`
uses wall-clock time, with no exact shared start marker. The session video's
first frame is aligned to the first telemetry row. Pauses, clock resets, and
startup delays can invalidate this estimate. Check alignment before using these
training targets.

Override a video's start relative to the first telemetry row with repeatable
`--video-offset SESSION_ID=SECONDS`. A positive offset means the video
starts later than that row. Offsets are scoped to a session:

```bash
python scripts/prepare_position_dataset.py \
  --sessions SESSION_ID_A SESSION_ID_B --fps 2 \
  --video-offset SESSION_ID_A=0.25 \
  --video-offset SESSION_ID_B=-0.5
```

Offsets do not correct drift or pauses within a video. The alignment delta is
the distance to the selected telemetry sample under the estimate, not verified
synchronization accuracy. Alignment options apply only to new sessions;
existing session folders are skipped regardless of their saved settings. To
prepare a session with different settings, move its entire session folder first,
preserving any human annotations you need.

### Reference selection and annotation

Track and car come from `Static_track` and `Static_car_model`. Grip is the rounded
average of `Graphics_track_grip_status` over unique aligned telemetry rows, using
the shared reference-model bucketing rule. An exact track/car/grip reference
must exist. Map selection uses the newest matching `source_track_key` within the
session's game. The model's game and map-name inputs come from that map's details.

`Graphics_normalized_car_position` locates both reference XYZ positions. Driver
XYZ is selected by first reading `Graphics_active_cars`, trimming both
`Graphics_car_coordinates` and `Graphics_car_id` to that count, then locating
`Graphics_player_car_id` in the active IDs and reading the corresponding coordinate.
This applies to single-car sessions too. Zero active cars or a missing or ambiguous
player ID match skips the frame; legacy `Graphics_player_pos_x/y/z` fields are not used.
Reference coordinates retain their world axes and scale; curves must cover the lap
and share the telemetry coordinate system. For ACC, preparation aligns the raw
driver Y/Z axes with the stored references, then uses `Physics_heading`,
`Physics_pitch`, and `Physics_roll` (radians) to rotate each reference-minus-driver
vector into car-local XYZ: **right, forward, up**. The position files mark these
targets with `coordinate_frame: car_local`; no world-space targets are saved.
Driver world XYZ is retained only for the annotation map. Training keeps reference
world XYZ as input and uses neither driver XYZ nor rotation as model inputs.
All three rotation fields must be finite. iRacing sessions are currently skipped
because their recorded position and heading fields use different world frames.

After preparation succeeds, annotate the images in each reported directory:

```bash
python scripts/open_labelme.py storage/session_downloads/SESSION_ID_A/frames
python scripts/open_labelme.py storage/session_downloads/SESSION_ID_B/frames
```

Save Labelme annotations as `frame_000000.json` beside each image, alongside its
separate `frame_000000.position.json`. Then train the position extension from the
folder containing the prepared pairs and an existing segmentation
checkpoint:

```bash
python scripts/train_position.py \
  --manifest storage/session_downloads \
  --model storage/image_segmentation/runs/train/weights/best.pt \
  --output storage/image_segmentation/position-runs/run-1 \
  --epochs 100 --imgsz 768 --batch 8 --device 0
```

Folder input includes all `.position.json` files recursively. To train on a subset,
pass a folder containing only those image/position pairs to `--manifest`.

After image extraction, training assigns whole sessions from each
`(game, map_name)` to validation and the rest to training, using
`--val-fraction 0.2` and `--seed 42` by default. Every map appears in both sets.
Each map needs at least two sessions. Validation uses the session combination
whose total frame count is closest to the requested fraction, accounting for
unequal session sizes; the seed randomizes equally balanced choices.
Sessions never cross splits, and all frames are retained. Images from all maps
are shuffled within each split. A single map is sufficient.
The training and validation maps are printed and saved in the run's `split.json`,
along with image paths, target frame fraction, and seed. The same inputs and seed reproduce
the assignments. Position data must use adjacent
`.position.json` files. Indexes containing inline frame data or preset `split`
assignments are rejected; rerun preparation and pass the folder of prepared pairs.

The regression trainer uses images, reference XYZ, game/map names, normalized lap
position, and prepared position targets. Names are one-hot encoded using category
vocabularies saved with the checkpoint for inference. The trainer
does not read Labelme polygons. To incorporate newly drawn region labels into
its image features, train a segmentation checkpoint with those labels first,
then pass that checkpoint to `--model`. The preparation merge leaves that
training behavior unchanged. See the
[regression workflow](training/image_segmentation/README.md#predict-signed-position-differences-from-images)
for model inputs, output definitions, and inference.

After training, the best checkpoint is exported to `best.onnx` with an accompanying
`best.onnx.json` inference manifest. The graph includes position deltas and
segmentation outputs for direct loading by Track Vision. The default upload now
sends ONNX; `--no-upload` still produces the local ONNX artifacts. Until the
backend upload route accepts `.onnx`, use `--no-upload`. See the
[ONNX contract](training/image_segmentation/README.md#track-vision-onnx-contract)
for the backend/frontend handoff and publishing a saved checkpoint without
retraining.

## Extract images from videos

From this folder, extract every frame from the videos directly inside an input
folder as JPEG images:

```bash
python scripts/videos_to_images.py /path/to/videos storage/video_frames
```

To select the input and output folders in a browser, launch the UI in the
training container, then open http://localhost:8501:

```bash
docker exec -it acla_ai_training_c python3 /app/scripts/videos_to_images.py
```

Browse subfolders or enter paths, choose the sampling rate, and click **Extract
images**. The UI browses the container filesystem: host
`acla_ai_training/storage/...` is available at `/app/storage/...`; other host
folders must be bind-mounted into the container. Enter a new output path to
create a folder during extraction. This uses the existing Streamlit dependency
and port mapping; no desktop display is required. Stop the pipeline UI first if
it is already using port 8501. Press Ctrl+C to stop this UI.

For local use, run `python scripts/videos_to_images.py` without both folder
arguments to launch the same browser UI. Passing both paths runs extraction
directly without starting a server.

The output folder is created automatically, with one subfolder per video, such as
`storage/video_frames/session.mp4/frame_000000.jpg`. Frame numbers use the
zero-based position in the source video. Existing nonempty video output folders
are rejected to avoid overwriting images. Input subfolders are not scanned.

To choose an output sampling rate, add `--fps 2` for two images per second or
`--fps 0.5` for one image every two seconds:

```bash
python scripts/videos_to_images.py /path/to/videos storage/video_frames --fps 2
```

Sampling starts with the first frame and uses each video's reported source frame
rate. Rates above the source rate save every frame without duplicating images.
Alternatively, use `--every-n-frames 30` to save the first frame and every 30th
frame after it. `--fps` and `--every-n-frames` cannot be combined.

OpenCV is included through the training dependencies; for standalone use,
install it with `python -m pip install opencv-python`.

## Docker

From the repository root:

```bash
docker compose --env-file .dev.env --env-file .env.secrets \
  -f docker-compose.dev.yaml -f docker-compose.cpu.yaml up -d --build
```

Substitute the NVIDIA or AMD override for GPU training. Training is a separate container,
`acla_ai_training_c`, and is not part of the production stack.

Both AI containers use `BACKEND_SERVER_IP_FOR_AI` and `BACKEND_PORT` from
`.dev.env` to reach the backend, including when it runs on another machine.
The backend uses `AI_SERVICE_URL` for the return connection to the live AI
service. When unset, these values default to `http://backend:7001` and
`http://ai_service:8000` for a stack running on one Docker host. Recreate the
affected containers after changing these settings; a restart retains the old
environment.

The container stays idle until you launch the training UI or a job. Start the UI
manually, then open http://localhost:8501:

```bash
docker exec -it acla_ai_training_c python3 /app/scripts/open_pipeline_management.py
```

Press Ctrl+C in that terminal to close the UI; the container stays running.

Training Dockerfiles use the repository root as their build context. They copy
the runtime package as a shared library. In development,
runtime source is mounted read-only; training source has its own mount. Claude
CLI authentication is mounted only into the training container.

To run a trainer directly:

```bash
docker exec -it acla_ai_training_c \
  python -m training.pipelines.training.entrypoints.train_segment_classifier --help
```

Classifier and cropper training save artifacts locally and upload them as active
models to the backend. Check the training log for upload errors. Restart
`ai_service` after a successful upload to load the new active artifacts.

Training commands require `--annotation-key DATASET_KEY`; there is no default
annotation dataset. Direct classifier and cropper trainer calls must pass
`annotation_cache_key="DATASET_KEY"`. The UI supplies the dataset selected on
the pipeline's training node.

## Existing data

Development Compose retains the existing `ai_models` training volume. Telemetry
datasets, backups, and manifests live under this folder in
`storage/telemetry_lance_store`, `storage/telemetry_lance_backups`, and
`storage/pipelines`. The training source mount makes them available at
`/app/storage/...` inside the container; local Python runs use the same host
directories by default.

`TELEMETRY_STORE_DIR`, `LANCE_BACKUP_DIR`, and `PIPELINE_STORAGE_DIR` can override
these locations. Existing datasets and backups have been moved intact from the
service workspace. Custom absolute dataset paths in pipeline manifests must
point to locations available to the training process.

To remove track and named-section labels from an older annotation dataset:

```bash
python -m training.pipelines.manifest.remove_track_labels DATASET_KEY --dry-run
python -m training.pipelines.manifest.remove_track_labels DATASET_KEY
```

The migration updates labels in place and preserves all other segment fields.

## Local Python

With a Python 3.11+ environment, run from this folder:

```bash
python -m pip install -r requirements.cpu.txt
python -m pip install --no-deps -e ../acla_ai_service
python scripts/open_pipeline_management.py
```

Set the backend address/port and credentials in your shell as needed.
`python -m pytest tests` runs the annotation, training, and UI tests and includes
the sibling runtime package on the test path.

The optional training HTTP/SSE API is separate from the live frontend API:

```bash
python -m uvicorn training.api.app:app --host 127.0.0.1 --port 8002
```

It retains `/annotation/run` and `/annotation/run/stream` and adds
`POST /racing-session/{session_id}/download` for paired telemetry/video downloads
and `POST /models/ultralytics/upload` for publishing binary checkpoints to the backend.
See the [upload instructions](training/image_segmentation/README.md#upload-a-trained-model)
for the multipart fields and metadata format. The manually launched Streamlit UI
invokes annotation in process.
