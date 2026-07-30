"""
ACLA AI Service — FastAPI application (startup wiring).

The llama-server boot lives in ``app.startup.llama``; this module owns the
FastAPI app, its lifespan, CORS, and router wiring. ASGI target: ``app.startup.app:app``.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.chat_llm import resolve_chat_llm_config
from app.infra.config import settings
from app.integrations.backend.client import backend_service
from app.llama.health import check_llama_server
from app.ml.model_hub import hydrate_chatbot_models
from app.api import (
    annotation_router,
    health_router,
    racing_session_router,
)
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
    chat_llm = resolve_chat_llm_config()
    provider_label = (
        f"{chat_llm.provider} ({chat_llm.base_url} / {chat_llm.model})"
        if chat_llm.base_url else f"{chat_llm.provider} ({chat_llm.model})"
    )
    print(f"🤖 LLM: {provider_label}")

    # Chat uses remote providers only. Keep llama health reporting for other
    # features that may depend on a separately managed llama-server.
    print("🦙 chat llama-server sidecar: skipped for remote chat LLM")

    llama_health = await check_llama_server()
    if llama_health.reachable:
        print(
            f"🦙 llama-server: reachable at {llama_health.base_url} "
            f"({len(llama_health.models)} model(s), {llama_health.latency_ms:.0f}ms)"
        )
    else:
        print(
            f"🦙 llama-server: NOT reachable at {llama_health.base_url} "
            f"({llama_health.error})"
        )

    # Establish backend connection
    print("🔌 Establishing backend connection...")
    backend_ok = await backend_service.establish_connection()
    if backend_ok:
        print("✅ Backend connection established successfully")
    else:
        print("⚠️  Backend connection failed - some features may not work")
        print("   Check your backend credentials in environment variables:")
        print("   - AI_SERVICE_USERNAME")
        print("   - AI_SERVICE_PASSWORD")

    # Hydrate chatbot-facing models from the backend active model store. This
    # runs even after the connection probe fails so runtime readiness is reset
    # and no previous local top-lap artifact can become an implicit fallback.
    model_status = await hydrate_chatbot_models()
    for model_name, is_ready in model_status.items():
        if is_ready:
            print(f"{model_name}: ready")
        else:
            print(f"{model_name}: NOT ready (backend active model payload unavailable or invalid)")

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
app.include_router(racing_session_router)
app.include_router(voice_router)  # voice WS = single chat surface (audio + tool-relay)
app.include_router(annotation_router)  # Step 13 — replaces Streamlit's in-process import

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
        "app.startup.app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
