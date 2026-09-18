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

## Image polygon annotation and training

[Labelme + Ultralytics](training/image_segmentation/README.md) adds a separate
image segmentation workflow under `training/image_segmentation`: draw polygons
for track, curbs, grass, cars, and other regions; convert Labelme JSON to YOLO
segmentation datasets; train locally or in the training container. Labelme is
installed through Python on a desktop with a display. Ultralytics is included in
the training dependencies.

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

The optional annotation HTTP/SSE API is separate from the live frontend API:

```bash
python -m uvicorn training.api.app:app --host 127.0.0.1 --port 8002
```

It retains `/annotation/run` and `/annotation/run/stream`. The manually launched
Streamlit UI invokes annotation in process.
