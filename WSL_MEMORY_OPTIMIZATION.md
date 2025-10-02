# WSL Memory Optimization Guide

This guide helps you optimize WSL2 memory usage when running the ACLA development environment with Docker.

## Problem
WSL2 can consume excessive memory (16GB+) when running Docker containers, especially with AI services and multiple containers.

## Solutions Implemented

### 1. Container Memory Limits
Added memory limits to all services in `docker-compose.dev.yaml`:
- **AI Service**: 2GB limit (already configured)
- **Backend**: 1GB limit
- **Frontend**: 512MB limit
- **MongoDB**: 1GB limit
- **Nginx Proxy**: 128MB limit
- **Mongo Express**: 256MB limit

### 2. WSL Configuration
Created `.wslconfig` file to limit WSL2 resource usage:
- **Memory**: Limited to 8GB
- **Processors**: Limited to 4 cores
- **Swap**: Limited to 2GB
- **Sparse VHD**: Enabled for better disk management

### 3. Docker Optimization
- Enabled BuildKit for better build performance
- Configured garbage collection
- Set concurrent download/upload limits

## Quick Fix

Run the memory optimization script:
```bash
.\optimize-memory.bat
```

This script will:
1. Stop all containers
2. Clean up Docker system resources
3. Install WSL configuration
4. Provide next steps

## Manual Steps

### 1. Copy WSL Configuration
Copy `.wslconfig` to your Windows user directory:
```bash
copy .wslconfig %USERPROFILE%\.wslconfig
```

### 2. Restart WSL
```bash
wsl --shutdown
# Wait 10 seconds, then restart Docker Desktop
```

### 3. Start Development Environment
```bash
.\start-dev.bat
```

## Monitoring Memory Usage

### Check WSL Memory Usage
```bash
# In PowerShell
Get-Process -Name "Vmmem" | Select-Object ProcessName, WorkingSet64
```

### Check Container Memory Usage
```bash
docker stats
```

## Additional Tips

### Regular Maintenance
- Stop containers when not developing: `docker-compose -f docker-compose.dev.yaml down`
- Clean up weekly: `docker system prune -f`
- Monitor with Task Manager

### Docker Desktop Settings
In Docker Desktop settings:
1. Go to Settings → Resources → Advanced
2. Set Memory limit (e.g., 8GB)
3. Set CPU limit (e.g., 4 cores)
4. Apply & Restart

### Alternative Solutions
If memory issues persist:
1. **Use Docker without WSL2**: Switch to Hyper-V backend
2. **Selective container startup**: Comment out services you don't need
3. **Use external databases**: Use cloud MongoDB instead of local container

## Troubleshooting

### High Memory Usage Persists
1. Check if `.wslconfig` is in the correct location: `%USERPROFILE%\.wslconfig`
2. Ensure WSL was restarted after configuration: `wsl --shutdown`
3. Verify container limits are applied: `docker stats`

### Containers Fail to Start
If containers crash due to memory limits:
1. Increase limits in `docker-compose.dev.yaml`
2. Or run fewer services simultaneously

### WSL Won't Start
If WSL fails after configuration:
1. Remove `.wslconfig` temporarily
2. Restart WSL: `wsl --shutdown`
3. Gradually adjust limits

## Configuration Files

- **`.wslconfig`**: WSL2 resource limits
- **`docker-compose.dev.yaml`**: Container resource limits
- **`optimize-memory.bat`**: Automated optimization script
- **`start-dev.bat`**: Updated startup script with memory tips

## Expected Results

After optimization:
- WSL memory usage should stay under 8GB
- Container startup time may be slightly longer
- System remains responsive during development
- Automatic cleanup prevents memory leaks

## Support

If you continue experiencing memory issues:
1. Check Windows Task Manager for WSL memory usage
2. Verify all configuration files are in place
3. Consider using fewer concurrent containers
4. Monitor individual container memory usage with `docker stats`