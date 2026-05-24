"""
ACLA AI Service - Main application entry point
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.infra.config import settings
from app.integrations.backend.client import backend_service
from app.llm.chat_model import ensure_chat_gguf
from app.llm.health import check_llama_server
from app.llm.process import LlamaServerConfig, LlamaServerProcess
from app.ml.segment_classifier.bootstrap import ensure_segment_classifier_model
from app.api import (
    annotation_router,
    health_router,
    racing_session_router,
)
from app.api.voice import router as voice_router


# Load environment variables
load_dotenv()


def _start_chat_sidecar() -> LlamaServerProcess:
    """Resolve the chat GGUF (downloading on first boot) and launch llama-server."""
    gguf_path = ensure_chat_gguf(
        model_dir=settings.llama_model_dir,
        model_repo=settings.llama_model_repo,
        model_file=settings.llama_model_file,
        hf_token=settings.hf_token,
    )
    sidecar = LlamaServerProcess(
        LlamaServerConfig(
            model_path=gguf_path,
            host=settings.llama_host,
            port=settings.llama_port,
            n_ctx=settings.llama_n_ctx,
            n_gpu_layers=settings.llama_n_gpu_layers,
            jinja=True,
            startup_timeout_seconds=settings.llama_startup_timeout_seconds,
        )
    )
    sidecar.start_or_attach()
    return sidecar


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("✅ Using new structured application")
    print(f"🏁 {settings.app_name} v{settings.app_version}")
    print(f"🔧 Backend URL: {settings.backend_server_ip}")
    print(f"🤖 OpenAI API: {'Configured (legacy)' if settings.openai_api_key else 'Not configured'}")

    # Bring up the chat sidecar (downloads model on first boot — may take minutes).
    # Run in an executor so the event loop isn't blocked during the wait.
    chat_sidecar: LlamaServerProcess | None = None
    try:
        chat_sidecar = await asyncio.get_running_loop().run_in_executor(
            None, _start_chat_sidecar,
        )
    except Exception as exc:  # noqa: BLE001 — we want uvicorn to keep starting
        print(
            f"⚠️  llama-server failed to start: {exc} — continuing without it. "
            f"/query/health will report degraded status."
        )

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

    # Hydrate segment classifier from backend if local artifacts are missing
    # (e.g. fresh container / named volume). Skipped when backend is down —
    # the classifier endpoint will surface "Model not trained or found".
    if backend_ok:
        if await ensure_segment_classifier_model():
            print("🧩 segment_classifier: ready")
        else:
            print("🧩 segment_classifier: NOT ready (no local artifacts and no active backend payload)")

    yield

    # Shutdown
    print(f"🏁 {settings.app_name} shutting down...")
    if chat_sidecar is not None:
        chat_sidecar.stop()


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
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
