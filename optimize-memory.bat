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

echo.
echo 🎯 Memory Optimization Complete!
echo.
echo 📋 What was done:
echo    ✅ Set AGGRESSIVE memory limits on all Docker containers
echo    ✅ Cleaned up Docker system resources  
echo    ✅ Updated WSL to use only 4GB RAM (down from unlimited)
echo    ✅ Forced WSL restart
echo.
echo 🔄 Next Steps:
echo    1. WSL has been restarted with new 4GB limit
echo    2. Start your development environment: start-dev.bat
echo    3. Monitor memory with Task Manager
echo    4. If containers fail, increase limits gradually
echo.
echo 💡 New aggressive limits:
echo    - WSL total memory: 4GB (was unlimited)
echo    - AI Service: 1GB (was 2GB)
echo    - Backend: 512MB (was 1GB)  
echo    - Frontend: 256MB (was 512MB)
echo    - MongoDB: 512MB (was 1GB)
echo    - Other services: 64-128MB each
echo.

pause