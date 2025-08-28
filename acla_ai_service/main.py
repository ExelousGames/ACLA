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
from app.main import app

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 Starting ACLA AI Service on {host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    # Fixed: Use app.main:app instead of main:app to properly reference the FastAPI app
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
