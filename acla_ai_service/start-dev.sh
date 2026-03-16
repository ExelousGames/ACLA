#!/bin/bash

# Alternative startup options for development

echo "ACLA AI Service - Development Startup Options"
echo "=============================================="

echo "Starting with basic hot reload (excluding cache directories)..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload \
    --reload-exclude "*__pycache__*" \
    --reload-exclude "./models/*"
