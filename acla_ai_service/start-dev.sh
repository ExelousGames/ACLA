#!/bin/bash
# Live AI API and chat WebSocket entrypoint.

set -e

echo "=============================================="
echo "ACLA AI Service - Dev Startup"
echo "=============================================="

exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload \
    --reload-exclude "*__pycache__*" \
    --reload-exclude "./models/*"
