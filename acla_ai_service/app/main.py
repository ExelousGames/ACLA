"""
ACLA AI Service - Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core import settings
from app.api import (
    health_router,
    racing_session_router,
)
from app.api.query import router as query_router
from app.api.ml_endpoints import router as ml_router

# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered racing telemetry analysis service for Assetto Corsa Competizione",
    debug=settings.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Include API routers
app.include_router(health_router)
app.include_router(query_router)  # Main query endpoint
app.include_router(racing_session_router)
app.include_router(ml_router)  # Machine Learning endpoints


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print("✅ Using new structured application")
    print(f"🏁 {settings.app_name} v{settings.app_version}")
    print(f"🔧 Backend URL: {settings.backend_url}")
    print(f"🤖 OpenAI API: {'Configured' if settings.openai_api_key else 'Not configured'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print(f"🏁 {settings.app_name} shutting down...")


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true" or settings.debug
    
    print(f"🚀 Starting {settings.app_name} on {host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
