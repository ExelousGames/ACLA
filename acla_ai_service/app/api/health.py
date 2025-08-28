"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime
from app.models import HealthResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="ACLA AI Service",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )
