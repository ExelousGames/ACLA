#!/bin/bash
# Prod entrypoint. llama-server is now spawned by the FastAPI lifespan (see
# app/main.py -> _start_chat_sidecar / LlamaServerProcess), so we just run
# uvicorn.

set -e

echo "=============================================="
echo "ACLA AI Service - Prod Startup"
echo "=============================================="

exec uvicorn main:app --host 0.0.0.0 --port 8000
