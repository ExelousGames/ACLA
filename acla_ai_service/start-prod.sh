#!/bin/bash
# Prod entrypoint: start llama-server as a background sidecar, then run
# uvicorn in the foreground without hot reload.

set -e

LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_WAIT_SECONDS="${LLAMA_WAIT_SECONDS:-300}"

echo "=============================================="
echo "ACLA AI Service - Prod Startup"
echo "=============================================="

# Start llama-server in the background
bash /app/scripts/start_llama_server.sh &
LLAMA_PID=$!

cleanup() {
    echo "Shutting down — stopping llama-server (pid $LLAMA_PID)..."
    kill -TERM "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for llama-server to become ready
echo "Waiting up to ${LLAMA_WAIT_SECONDS}s for llama-server at http://${LLAMA_HOST}:${LLAMA_PORT}..."
READY=0
for i in $(seq 1 "$LLAMA_WAIT_SECONDS"); do
    if curl -sf "http://${LLAMA_HOST}:${LLAMA_PORT}/v1/models" > /dev/null 2>&1; then
        echo "llama-server is ready after ${i}s"
        READY=1
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: llama-server process exited during startup."
        exit 1
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    echo "WARNING: llama-server did not become ready within ${LLAMA_WAIT_SECONDS}s — starting uvicorn anyway."
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
