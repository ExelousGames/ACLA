"""
API routers initialization
"""

from .health import router as health_router
from .datasets import router as datasets_router
from .ai import router as ai_router
from .racing_session import router as racing_session_router
from .telemetry import router as telemetry_router
from .backend import router as backend_router
from .models import router as models_router

__all__ = [
    "health_router",
    "datasets_router", 
    "ai_router",
    "racing_session_router",
    "telemetry_router",
    "backend_router",
    "models_router"
]
