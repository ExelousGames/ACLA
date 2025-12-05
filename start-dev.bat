@echo off
REM ACLA Development Environment Startup Script for Windows

echo 🚀 Starting ACLA Development Environment with AI Service...

REM Check WSL memory configuration
if not exist "%USERPROFILE%\.wslconfig" (
    echo ⚠️  WSL memory not optimized. Run 'optimize-memory.bat' first for better performance.
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    exit /b 1
)

REM Detect GPU
set COMPOSE_OVERRIDE_ARGS=-f docker-compose.nvidia.yaml
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ NVIDIA GPU detected.
    set COMPOSE_OVERRIDE_ARGS=-f docker-compose.nvidia.yaml
) else (
    echo ⚠️  No NVIDIA GPU detected. Defaulting to NVIDIA profile.
    set COMPOSE_OVERRIDE_ARGS=-f docker-compose.nvidia.yaml
)

REM Stop any existing containers
echo 🧹 Cleaning up existing containers...
docker-compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml %COMPOSE_OVERRIDE_ARGS% down

REM Build and start all services
echo 🔨 Building and starting services...
docker-compose --env-file .dev.env --env-file .env.secrets -f docker-compose.dev.yaml %COMPOSE_OVERRIDE_ARGS% up --build -d

REM Wait for services to start
echo ⏳ Waiting for services to initialize...
timeout /t 30 >nul

REM Check service health
echo 🔍 Checking service health...

REM Check backend
curl -f http://localhost:7001 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is running at http://localhost:7001
) else (
    echo ❌ Backend failed to start
)

REM Check frontend
curl -f http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend is running at http://localhost:3000
) else (
    echo ❌ Frontend failed to start
)

REM Check AI service
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ AI Service is running at http://localhost:8000
) else (
    echo ❌ AI Service failed to start
)

echo.
echo 🎉 ACLA Development Environment is ready!
echo.
echo 📋 Service URLs:
echo    Frontend:      http://localhost:3000
echo    Backend:       http://localhost:7001
echo    AI Service:    http://localhost:8000
echo    MongoDB:       http://localhost:27017
echo    Mongo Express: http://localhost:8081
echo.
echo 📖 To view logs: docker-compose -f docker-compose.dev.yaml logs -f [service_name]
echo 📖 To stop services: docker-compose -f docker-compose.dev.yaml down
echo.
echo 🤖 AI Service Features:
echo    - Automatic racing session analysis
echo    - Natural language queries about racing data
echo    - Performance scoring and recommendations
echo    - Pattern detection in racing behavior
echo    - Sector-wise performance analysis
echo.
echo.
echo 💾 Memory Usage Tips:
echo    - If WSL uses too much memory, run 'optimize-memory.bat'
echo    - Stop containers when not developing: docker-compose -f docker-compose.dev.yaml down
echo    - Clean up regularly: docker system prune -f

pause
