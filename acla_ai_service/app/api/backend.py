"""
Backend integration endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.models import BackendCallRequest

router = APIRouter(prefix="/backend", tags=["backend"])

@router.post("/call")
async def call_backend_function(request: BackendCallRequest):
    """Call backend functions"""
    # Implementation would go here
    return {"message": "Backend call endpoint - to be implemented"}
