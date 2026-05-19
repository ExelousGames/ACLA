"""HTTP request / response DTOs.

These shapes are the public HTTP boundary of the AI service — they cross
the wire between React and FastAPI. They are NOT used for internal
function-to-function calls; for that, use the domain types in app/domain/.

Moved from app/models/api_models.py in refactor/hexagonal-v1, Step 3.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DatasetInfo(BaseModel):
    """Dataset information model"""
    id: str
    name: str
    size: int
    columns: List[str]
    upload_time: str


class QueryRequest(BaseModel):
    """Natural language query request model"""
    query: str
    dataset_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AnalysisResult(BaseModel):
    """Analysis result model"""
    dataset_id: str
    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str


class BackendCallRequest(BaseModel):
    """Backend function call request model"""
    function_name: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    version: str
    timestamp: str
    backend_connection: Optional[Dict[str, Any]] = None


__all__ = [
    "DatasetInfo",
    "QueryRequest",
    "AnalysisResult",
    "BackendCallRequest",
    "HealthResponse",
]
