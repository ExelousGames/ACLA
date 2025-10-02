#!/bin/bash

# Alternative startup options for development

echo "ACLA AI Service - Development Startup Options"
echo "=============================================="

if [ "$1" = "no-reload" ]; then
    echo "Starting without hot reload (more memory efficient)..."
    exec uvicorn main:app --host 0.0.0.0 --port 8000
elif [ "$1" = "limited-reload" ]; then
    echo "Starting with limited hot reload (watching only app directory)..."
    exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/app
elif [ "$1" = "basic-reload" ]; then
    echo "Starting with basic hot reload (excluding cache directories)..."
    exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload \
        --reload-exclude "*/__pycache__/*" \
        --reload-exclude "*/telemetry_data_cache/*" \
        --reload-exclude "*/models/*" \
        --reload-exclude "*/scripts/debug_output/*"
else
    echo "Starting with default hot reload..."
    echo "If you encounter memory issues, try:"
    echo "  docker exec acla_ai_service_c /app/start-dev.sh no-reload"
    echo "  docker exec acla_ai_service_c /app/start-dev.sh limited-reload"
    echo "  docker exec acla_ai_service_c /app/start-dev.sh basic-reload"
    exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi