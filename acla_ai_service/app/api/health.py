"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime
from app.models import HealthResponse
from app.services.backend_service import backend_service
from app.core import settings

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with backend connection status"""
    backend_status = backend_service.get_connection_status()
    
    return HealthResponse(
        status="healthy",
        service="ACLA AI Service",
        version=settings.app_version,
        timestamp=datetime.utcnow().isoformat(),
        backend_connection=backend_status
    )
