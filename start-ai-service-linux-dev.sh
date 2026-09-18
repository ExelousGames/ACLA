#!/bin/bash

# ACLA AI Service and Training Standalone Startup Script

echo "🤖 Which one do you want to start?"
echo "   1) AI Service only"
echo "   2) AI Training only"
echo "   3) Both"

while true; do
    read -r -p "Choose an option [1-3]: " START_CHOICE || exit 1
    case "$START_CHOICE" in
        1)
            SERVICES=(ai_service)
            SERVICE_LABEL="AI Service"
            break
            ;;
        2)
            SERVICES=(ai_training)
            SERVICE_LABEL="AI Training"
            break
            ;;
        3)
            SERVICES=(ai_service ai_training)
            SERVICE_LABEL="AI Service and Training"
            break
            ;;
        *) echo "Please enter 1, 2, or 3." ;;
    esac
done

echo "🤖 Starting $SERVICE_LABEL separately..."

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

# Stop and remove only the selected containers
echo "🧹 Stopping existing $SERVICE_LABEL containers..."
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS stop "${SERVICES[@]}"
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS rm -f "${SERVICES[@]}"

# Build and start the selected services
echo "🔨 Building and starting $SERVICE_LABEL..."
docker compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml $COMPOSE_OVERRIDE_ARGS up --build -d "${SERVICES[@]}"

# Wait for services to start
echo "⏳ Waiting for $SERVICE_LABEL to initialize..."
sleep 5

# Check service health
for SERVICE in "${SERVICES[@]}"; do
    case "$SERVICE" in
        ai_service)
            if curl -f http://localhost:8000/health > /dev/null 2>&1; then
                echo "✅ AI Service is running at http://localhost:8000"
            else
                echo "⚠️  AI Service might still be starting. Check logs for details."
            fi
            ;;
        ai_training)
            if [ "$(docker inspect --format '{{.State.Running}}' acla_ai_training_c 2>/dev/null)" = "true" ]; then
                echo "✅ AI Training workspace is ready. Start the UI manually when needed."
            else
                echo "⚠️  AI Training workspace is not running. Check logs for details."
            fi
            ;;
    esac
done

echo ""
echo "🎉 $SERVICE_LABEL started independently!"
echo ""
echo "📋 Service URLs:"
for SERVICE in "${SERVICES[@]}"; do
    case "$SERVICE" in
        ai_service)
            echo "   AI Service:    http://localhost:8000"
            ;;
        ai_training)
            echo "   AI Training:   http://localhost:8501 (after starting the UI manually)"
            echo ""
            echo "📖 To start the training UI: docker exec -it acla_ai_training_c python3 /app/scripts/open_pipeline_management.py"
            ;;
    esac
done
echo ""
echo "📖 To view logs: docker compose -f docker-compose.dev.yaml logs -f ${SERVICES[*]}"
echo "📖 To stop services: docker compose -f docker-compose.dev.yaml stop ${SERVICES[*]}"
echo ""
