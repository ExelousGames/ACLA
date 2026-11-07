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

## Phoneme voice curves

`POST /voice/phonemes` accepts `text` and `language` (default `en-us`) and
returns `phonemes` plus `tokens`: `{index, symbol, token_id}`. These indices
address Kokoro's phonetic input symbols, including spaces, stress markers and
punctuation. They are not character positions in the original text; a sound
represented by several symbols can be selected with a range.

`POST /voice/synthesize/prosody` accepts the usual `text`, `voice`, `speed`,
and `language`, plus `prosody: {expected_phonemes, curves}`. Copy
`expected_phonemes` from the preview for the same text and language. Each curve
selects `[start, end)` indices and contains any of these controls:

| Control | Units and bounds | Neutral | Model operation |
| --- | --- | --- | --- |
| `timing` | Duration multiplier, 0.25–4 | 1 | Scale predicted duration before rounding and alignment |
| `energy` | dB adjustment, -12–12 | 0 | Add `value * ln(10) / 20` to predicted log-energy `N` |
| `pitch` | Semitones, -12–12 | 0 | Multiply positive predicted `F0` by `2 ** (value / 12)` |

Each control is a list of `{position, value}` points with strictly increasing
positions from `0` to `1`, including both endpoints. Interpolation is linear.
Timing is sampled at each phoneme's center in the selected index range;
`speed` divides the underlying durations before timing multipliers are applied.
Pitch and energy are sampled at acoustic-frame centers across the range's
resulting duration, so a curve can vary within a single phoneme. Controls outside
selected ranges remain neutral. Ranges cannot overlap; combine controls for the
same range in one object. `curves: []` produces native Kokoro's baseline output.

Example: lengthen the vowel in “smoothly” and give it rising pitch and energy:

```python
from pathlib import Path
import requests

base = "http://localhost:8000/voice"
text = "Brake smoothly."
preview_response = requests.post(f"{base}/phonemes", json={"text": text}, timeout=120)
preview_response.raise_for_status()
preview = preview_response.json()
print(preview["tokens"])  # Use these indices in your curve editor.
target = next(t["index"] for t in preview["tokens"] if t["symbol"] == "u")
response = requests.post(f"{base}/synthesize/prosody", json={
    "text": text,
    "voice": "af_bella",
    "speed": 1.0,
    "language": "en-us",
    "prosody": {
        "expected_phonemes": preview["phonemes"],
        "curves": [{
            "start": target,
            "end": target + 1,
            "timing": [{"position": 0, "value": 1.5}, {"position": 1, "value": 1.5}],
            "energy": [{"position": 0, "value": 0}, {"position": 1, "value": 3}],
            "pitch": [{"position": 0, "value": 0}, {"position": 1, "value": 4}],
        }],
    },
}, timeout=300)
response.raise_for_status()
Path("speech.wav").write_bytes(response.content)
```

The response is a 24 kHz mono PCM16 WAV. Invalid curve structure returns 422;
stale phonemes, excessive duration, or overlong phoneme sequences return 400.
Each request supports at most 510 phonetic symbols and 150 seconds of predicted
duration. Split longer text into requests and preview each separately.

Curve synthesis uses `kokoro==0.9.4` and a pinned revision of the native
`hexgrad/Kokoro-82M` v1.0 weights. Rebuild the AI service image to install the
new dependency. The first preview downloads its small configuration; first
synthesis downloads the weights (about 330 MB) and selected voice into
`KOKORO_MODEL_DIR/native`. Inference uses PyTorch on CUDA/ROCm when available,
otherwise CPU. A separate lazy pool starts with one native engine and grows
on contention. `/voice/health` exposes its engine counts under `prosody_tts`;
these counts include engines that have only loaded the phonemizer.

The existing `/voice/synthesize` and `/voice/stream` retain their ONNX path.
The curve API edits predictions before the decoder; it does not stretch or
pitch-shift finished audio. Energy dB is a change to the model's acoustic
conditioning, not a guaranteed measured loudness change. Extreme edits may
reduce naturalness. No retraining is needed.

Run the focused tests with `python -m pytest tests/test_voice_prosody.py`.
To also download and check real model output against upstream native inference,
run `ACLA_KOKORO_INTEGRATION=1 python -m pytest tests/test_voice_prosody.py`.
