"""Wire-format DTOs for talking to the Node backend (acla_backend).

These are the shapes serialized over HTTP to a system we don't own.
They sit in app/integrations/ because they are NOT the public API of
this service — they are private to the backend HTTP adapter.

Moved from app/models/api_models.py in refactor/hexagonal-v1, Step 3.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class AiModelDto(BaseModel):
    """Payload used when persisting AI models to the backend."""
    modelType: str
    modelData: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    isActive: bool


class ActiveModelData(BaseModel):
    """Complete active model data structure returned by getCompleteActiveModelData"""
    modelType: str
    isActive: bool
    metadata: Dict[str, Any]
    modelData: Dict[str, Any]


__all__ = [
    "AiModelDto",
    "ActiveModelData",
]
