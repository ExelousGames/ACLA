"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime
from app.api.schemas.query import HealthResponse
from app.integrations.backend.client import backend_service
from app.infra.config import settings

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
