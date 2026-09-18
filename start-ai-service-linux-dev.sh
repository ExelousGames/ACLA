#!/bin/bash

# ACLA AI Service and Training Standalone Startup Script

echo "🤖 Starting ACLA AI Service and Training separately..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Detect GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected."
    COMPOSE_OVERRIDE_ARGS="-f docker-compose.nvidia.yaml"
elif [ -e /dev/kfd ] && [ -e /dev/dri ] && command -v rocminfo &> /dev/null; then
    echo "✅ AMD GPU detected (ROCm)."
    COMPOSE_OVERRIDE_ARGS="-f docker-compose.amd.yaml"
else
    echo "⚠️  No supported GPU detected. Defaulting to CPU profile."
    COMPOSE_OVERRIDE_ARGS="-f docker-compose.cpu.yaml"
fi

# Stop and remove only the AI Service and Training containers
echo "🧹 Stopping existing AI Service and Training containers..."
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS stop ai_service ai_training
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS rm -f ai_service ai_training

# Build and start ai_service and ai_training
echo "🔨 Building and starting AI Service and Training..."
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS up --build -d ai_service ai_training

# Wait for services to start
echo "⏳ Waiting for AI Service and Training to initialize..."
sleep 5

# Check service health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ AI Service is running at http://localhost:8000"
else
    echo "⚠️  AI Service might still be starting. Check logs for details."
fi

if [ "$(docker inspect --format '{{.State.Running}}' acla_ai_training_c 2>/dev/null)" = "true" ]; then
    echo "✅ AI Training workspace is ready. Start the UI manually when needed."
else
    echo "⚠️  AI Training workspace is not running. Check logs for details."
fi

echo ""
echo "🎉 AI Service and Training started independently!"
echo ""
echo "📋 Service URLs:"
echo "   AI Service:    http://localhost:8000"
echo "   AI Training:   http://localhost:8501 (after starting the UI manually)"
echo ""
echo "📖 To start the training UI: docker exec -it acla_ai_training_c python3 /app/scripts/open_pipeline_management.py"
echo ""
echo "📖 To view logs: docker compose -f docker-compose.dev.yaml logs -f ai_service ai_training"
echo "📖 To stop services: docker compose -f docker-compose.dev.yaml stop ai_service ai_training"
echo ""
