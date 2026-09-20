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
Start training with
`docker exec -it acla_ai_training_c python3 /app/scripts/train_labelme.py`.
It recursively reads `storage/annotation_images`, mixes annotated images from all
folders into an 80/20 training/validation split, and saves reusable sample lists in
`storage/image_segmentation/split.json`. Pass `--prepare-only` to export without
training, `--device 0` to use a configured GPU, or `--rebuild-split` to include new
annotations in a fresh split. See the workflow guide for the full options.
Downloaded pretrained weights live in `storage/image_segmentation/pretrained/`;
training checkpoints and metrics live in `storage/image_segmentation/runs/train*/`.
Successful training uploads the saved checkpoint to `POST /ai-model/ultralytics`.
Pass `--no-upload` for offline training or `--upload-name` to set the backend model
name. The workflow guide also describes uploading existing `.pt` checkpoints
without retraining and the local FastAPI upload endpoint.

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
`POST /models/ultralytics/upload` for publishing binary checkpoints to the backend.
See the [upload instructions](training/image_segmentation/README.md#upload-a-trained-model)
for the multipart fields and metadata format. The manually launched Streamlit UI
invokes annotation in process.
