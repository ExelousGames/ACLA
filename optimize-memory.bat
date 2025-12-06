@echo off
REM WSL Memory Optimization Script for ACLA Development Environment

echo 🔧 Optimizing WSL Memory Usage for ACLA...

REM Stop Docker containers first
echo 📦 Stopping Docker containers...
docker-compose -f docker-compose.dev.yaml down

REM Clean up Docker system
echo 🧹 Cleaning up Docker system...
docker system prune -f
docker volume prune -f
docker network prune -f

REM Compact WSL disk
echo 💾 Compacting WSL disk...
wsl --shutdown
timeout /t 5 >nul

REM Force copy .wslconfig (overwrite existing)
echo 📋 Updating WSL configuration file with aggressive memory limits...
copy ".wslconfig" "%USERPROFILE%\.wslconfig" /Y
echo ✅ WSL configuration file updated at %USERPROFILE%\.wslconfig
echo ⚠️  WSL will be restarted now for changes to take effect...

REM Force restart WSL
wsl --shutdown
timeout /t 10 >nul

pause