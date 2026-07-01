"""
ACLA AI Service - Main application entry point
This file serves as the main entry point and imports the structured application from the app package.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path to find the app module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the main application from the app package
from app.startup.app import app

# Surface basic device info on startup for visibility
try:
    import torch
    _cuda_available = torch.cuda.is_available()
    _device_name = "CPU"
    _backend_type = "CPU"
    
    if _cuda_available:
        _device_name = torch.cuda.get_device_name(0)
        if hasattr(torch.version, 'hip') and torch.version.hip:
            _backend_type = "ROCm (AMD)"
        else:
            _backend_type = "CUDA (NVIDIA)"
            
    print(f"[AI Service] Torch version: {getattr(torch, '__version__', 'unknown')} | Backend: {_backend_type} | Device: {_device_name}")
except Exception as _e:
    print(f"[AI Service] Torch import failed or CUDA check error: {_e}")

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 Starting ACLA AI Service on {host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    # Reference the FastAPI app via its new home in app.startup.app
    uvicorn.run(
        "app.startup.app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
