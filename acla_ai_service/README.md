# Live AI service

This folder serves the frontend through the NestJS backend: AI chat/voice at
`/voice/stream`, segment classification, live baseline analysis, session review,
and racing guidance. FastAPI listens on port 8000.

All LLM inference (chat, voice conversations, and top-lap guidance) uses the
cloud provider selected by `CHAT_LLM_MODEL`: `openai:<model>` with
`OPENAI_API_KEY`, or `hosted:<model>` with `HOSTED_LLM_BASE_URL` and
`HOSTED_LLM_API_KEY`. The live service does not install or start a local LLM.
Telemetry models, knowledge-base embeddings, and speech processing still run
locally; their CPU/GPU dependencies remain part of the service.
Hugging Face and LlamaIndex support those embeddings and speech models;
LlamaIndex is separate from llama.cpp, which is not used by this service.

`models/` in this folder holds telemetry inference artifacts and speech models
such as Kokoro. Local LLM weights (`*.gguf`), llama-server files, LoRA adapters,
and LLM training datasets do not belong here. The former `app/models/` and
repository-root `models/` folders are unused. Legacy annotation datasets belong
in `../acla_ai_training/storage/llm_datasets/`.

Local annotation, dataset preparation, Streamlit, and trainer entrypoints live in
[`../acla_ai_training`](../acla_ai_training/README.md). The live service never
imports that package. The `app` package also supplies the model formats,
preprocessing, and telemetry types reused by training. Dataset storage and
pipelines belong entirely to the training workspace.

From the repository root, start the development stack using the CPU, NVIDIA,
or AMD override:

```bash
docker compose --env-file .dev.env --env-file .env.secrets \
  -f docker-compose.dev.yaml -f docker-compose.cpu.yaml up -d --build
```

The backend uses `AI_SERVICE_URL=http://ai_service:8000`. Both AI containers reach
the backend at `http://backend:7001` on `ai-network`.

The backend active model store supplies serving artifacts. After training uploads
a new active classifier, restart the live service to hydrate it:

```bash
docker compose --env-file .dev.env --env-file .env.secrets \
  -f docker-compose.dev.yaml restart ai_service
```

Serving caches models in `ai_runtime_models`; local training keeps the existing
`ai_models` volume. The production Compose file runs only this live service.
Top-lap references are loaded from the backend into memory for inference;
the live service does not save them locally. Their builder and saved local
artifacts belong to `acla_ai_training`.

Run tests from this folder with `python -m pytest tests`. The service boundary
tests check imports throughout the runtime source and load the API and model
adapters in fresh processes without the training package. The shared scaler
lives in `app/ml/transformer/scaler.py`, and inference preprocessing lives in
`app/shared/inference_preprocessing.py`. There are no `app/storage` or
`app/pipelines` packages in the live service.
