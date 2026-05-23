#!/bin/bash
# Start llama-server hosting the local chat model (Qwen2.5-32B-Instruct GGUF
# by default) and expose an OpenAI-compatible HTTP API on $LLAMA_PORT.
#
# Auto-detects whether the standalone `llama-server` binary is on PATH
# (built from source — see dev.amd.dockerfile) or whether to fall back
# to `python -m llama_cpp.server` from the llama-cpp-python package.
#
# Configuration is read from env vars; all have sensible defaults.

set -e

# --- Configuration -----------------------------------------------------------

MODEL_DIR="${LLAMA_MODEL_DIR:-/app/models/llama_server}"
MODEL_REPO="${LLAMA_MODEL_REPO:-Qwen/Qwen2.5-32B-Instruct-GGUF}"
MODEL_FILE="${LLAMA_MODEL_FILE:-qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-99}"
N_CTX="${LLAMA_N_CTX:-8192}"

MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# --- Download model on first run --------------------------------------------

if [ ! -f "$MODEL_PATH" ]; then
    echo "[llama-server] Model not found at $MODEL_PATH — downloading from $MODEL_REPO..."
    mkdir -p "$MODEL_DIR"
    MODEL_REPO="$MODEL_REPO" MODEL_FILE="$MODEL_FILE" MODEL_DIR="$MODEL_DIR" \
    python - <<'PY'
import os
import re
from huggingface_hub import hf_hub_download, snapshot_download

repo = os.environ["MODEL_REPO"]
filename = os.environ["MODEL_FILE"]
local_dir = os.environ["MODEL_DIR"]
token = os.environ.get("HF_TOKEN")

# Large GGUFs are sharded into "<base>-00001-of-NNNNN.gguf" parts.
# llama-server auto-loads sibling shards when given the first one, so we
# just need to pull every "<base>-*.gguf" file in one shot.
shard = re.match(r"^(.*)-\d{5}-of-\d{5}\.gguf$", filename)
if shard:
    pattern = f"{shard.group(1)}-*.gguf"
    print(f"[llama-server] Downloading sharded {pattern} from {repo} into {local_dir} ...")
    snapshot_download(
        repo_id=repo,
        allow_patterns=[pattern],
        local_dir=local_dir,
        token=token,
    )
    print(f"[llama-server] Sharded download complete.")
else:
    print(f"[llama-server] Downloading {filename} from {repo} into {local_dir} ...")
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )
    print(f"[llama-server] Downloaded to {path}")
PY
fi

# --- Start the server --------------------------------------------------------

echo "[llama-server] Starting on ${LLAMA_HOST}:${LLAMA_PORT} (n_gpu_layers=${N_GPU_LAYERS}, n_ctx=${N_CTX})"
echo "[llama-server] Model: $MODEL_PATH"

if command -v llama-server >/dev/null 2>&1; then
    # Native binary build (e.g. ROCm via dev.amd.dockerfile).
    # The `--jinja` flag enables the model's chat template for native tool calling.
    exec llama-server \
        --model "$MODEL_PATH" \
        --host "$LLAMA_HOST" \
        --port "$LLAMA_PORT" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --ctx-size "$N_CTX" \
        --jinja
else
    # Python-bindings server (llama-cpp-python). `chatml-function-calling`
    # is the format Qwen 2.5 was trained for tool calling.
    exec python -m llama_cpp.server \
        --model "$MODEL_PATH" \
        --host "$LLAMA_HOST" \
        --port "$LLAMA_PORT" \
        --n_gpu_layers "$N_GPU_LAYERS" \
        --n_ctx "$N_CTX" \
        --chat_format chatml-function-calling
fi
