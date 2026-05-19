"""
ACLA AI Service - Main application entry point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core import settings
from app.integrations.backend.client import backend_service
from app.services.llm.llama_health import check_llama_server
from app.api import (
    health_router,
    racing_session_router,
)
from app.api.query import router as query_router
from app.api.voice import router as voice_router


# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("✅ Using new structured application")
    print(f"🏁 {settings.app_name} v{settings.app_version}")
    print(f"🔧 Backend URL: {settings.backend_server_ip}")
    print(f"🤖 OpenAI API: {'Configured (legacy)' if settings.openai_api_key else 'Not configured'}")

    # Check the local llama-server sidecar (canonical LLM backend going forward)
    llama_health = await check_llama_server()
    if llama_health.reachable:
        print(
            f"🦙 llama-server: reachable at {llama_health.base_url} "
            f"({len(llama_health.models)} model(s), {llama_health.latency_ms:.0f}ms)"
        )
    else:
        print(
            f"🦙 llama-server: NOT reachable at {llama_health.base_url} "
            f"({llama_health.error}) — startup script may still be downloading the model"
        )

    # Establish backend connection
    print("🔌 Establishing backend connection...")
    if await backend_service.establish_connection():
        print("✅ Backend connection established successfully")
    else:
        print("⚠️  Backend connection failed - some features may not work")
        print("   Check your backend credentials in environment variables:")
        print("   - AI_SERVICE_USERNAME")
        print("   - AI_SERVICE_PASSWORD")
    
    yield
    
    # Shutdown
    print(f"🏁 {settings.app_name} shutting down...")


# Create FastAPI application with lifespan manager
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered racing telemetry analysis service for Assetto Corsa Competizione",
    debug=settings.debug,
    lifespan=lifespan
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
app.include_router(voice_router)  # Phase 2 — neural TTS (Kokoro)

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
