"""
Pydantic models for the ACLA AI Service
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class DatasetInfo(BaseModel):
    """Dataset information model"""
    id: str
    name: str
    size: int
    columns: List[str]
    upload_time: str


class AnalysisRequest(BaseModel):
    """Analysis request model"""
    dataset_id: str
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None


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
