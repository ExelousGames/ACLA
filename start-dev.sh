#!/bin/bash

# ACLA Development Environment Startup Script

echo "🚀 Starting ACLA Development Environment with AI Service..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml down

# Build and start all services
echo "🔨 Building and starting services..."
docker-compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml up --build -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check backend
if curl -f http://localhost:7001 > /dev/null 2>&1; then
    echo "✅ Backend is running at http://localhost:7001"
else
    echo "❌ Backend failed to start"
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is running at http://localhost:3000"
else
    echo "❌ Frontend failed to start"
fi

# Check AI service
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ AI Service is running at http://localhost:8000"
else
    echo "❌ AI Service failed to start"
fi

# Check MongoDB
if curl -f http://localhost:27017 > /dev/null 2>&1; then
    echo "✅ MongoDB is running at http://localhost:27017"
else
    echo "❌ MongoDB failed to start"
fi

# Check Mongo Express
if curl -f http://localhost:8081 > /dev/null 2>&1; then
    echo "✅ Mongo Express is running at http://localhost:8081"
else
    echo "❌ Mongo Express failed to start"
fi

echo ""
echo "🎉 ACLA Development Environment is ready!"
echo ""
echo "📋 Service URLs:"
echo "   Frontend:      http://localhost:3000"
echo "   Backend:       http://localhost:7001"
echo "   AI Service:    http://localhost:8000"
echo "   MongoDB:       http://localhost:27017"
echo "   Mongo Express: http://localhost:8081"
echo ""
echo "📖 To view logs: docker-compose -f docker-compose.dev.yaml logs -f [service_name]"
echo "📖 To stop services: docker-compose -f docker-compose.dev.yaml down"
echo ""
echo "🤖 AI Service Features:"
echo "   - Automatic racing session analysis"
echo "   - Natural language queries about racing data"
echo "   - Performance scoring and recommendations"
echo "   - Pattern detection in racing behavior"
echo "   - Sector-wise performance analysis"
echo ""
echo "📚 See AI_SERVICE_GUIDE.md for detailed documentation"
