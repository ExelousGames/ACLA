"""
API routers initialization
"""

from .health import router as health_router
from .racing_session import router as racing_session_router


__all__ = [
    "health_router", 
    "racing_session_router",
]
