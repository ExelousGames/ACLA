#!/bin/bash

# ACLA AI Service Standalone Startup Script

echo "🤖 Starting ACLA AI Service separately..."

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
    echo "⚠️  No supported GPU detected. Defaulting to NVIDIA profile."
    COMPOSE_OVERRIDE_ARGS="-f docker-compose.nvidia.yaml"
fi

# Stop and remove only the ai_service container
echo "🧹 Stopping existing AI Service container..."
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS stop ai_service
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS rm -f ai_service

# Build and start ai_service
echo "🔨 Building and starting AI Service..."
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS up --build -d ai_service

# Wait for service to start
echo "⏳ Waiting for AI Service to initialize..."
sleep 5

# Check service health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ AI Service is running at http://localhost:8000"
else
    echo "⚠️  AI Service might still be starting. Check logs for details."
fi

echo ""
echo "🎉 AI Service started independently!"
echo ""
echo "📋 Service URLs:"
echo "   AI Service:    http://localhost:8000"
echo ""
echo "📖 To view logs: docker compose -f docker-compose.dev.yaml logs -f ai_service"
echo "📖 To stop service: docker compose -f docker-compose.dev.yaml stop ai_service"
echo ""
