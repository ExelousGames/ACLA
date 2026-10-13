#!/bin/bash
# Live AI API and chat WebSocket entrypoint.

set -e

echo "=============================================="
echo "ACLA AI Service - Prod Startup"
echo "=============================================="

exec uvicorn main:app --host 0.0.0.0 --port 8000
