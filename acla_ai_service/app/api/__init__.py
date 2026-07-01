"""
API routers initialization
"""

from .annotation import router as annotation_router
from .health import router as health_router
from .racing_session import router as racing_session_router


__all__ = [
    "annotation_router",
    "health_router",
    "racing_session_router",
]
