"""Publish trained Ultralytics checkpoints through the local training API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError, field_validator

from training.model_publication import upload_ultralytics_model

router = APIRouter(prefix="/models/ultralytics", tags=["models"])
MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024


class UltralyticsMetadata(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    task: Literal["detect", "segment", "classify", "pose", "obb"]
    class_names: list[str] = Field(alias="classNames", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value.strip()

    @field_validator("class_names")
    @classmethod
    def validate_class_names(cls, value: list[str]) -> list[str]:
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError("classNames must contain unique, non-empty names in class-ID order")
        return value


def get_backend_service():
    from app.integrations.backend.client import backend_service

    return backend_service


@router.post("/upload", status_code=201)
async def upload_model(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    backend_service=Depends(get_backend_service),
) -> dict[str, Any]:
    """Upload a .pt file and JSON metadata; return the backend's saved record."""
    if not file.filename or Path(file.filename).suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail="Model file must use the .pt extension")
    if not file.size:
        raise HTTPException(status_code=400, detail="Model file must not be empty")
    if file.size > MAX_MODEL_BYTES:
        raise HTTPException(status_code=413, detail="Model file exceeds 512 MiB")
    if len(metadata.encode("utf-8")) > MAX_METADATA_BYTES:
        raise HTTPException(status_code=413, detail="Model metadata exceeds 1 MiB")
    try:
        model_metadata = UltralyticsMetadata.model_validate_json(metadata)
        payload = model_metadata.model_dump(by_alias=True)
        json.dumps(payload, allow_nan=False)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        return await upload_ultralytics_model(
            backend_service, file.file, file.filename, payload,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Backend model upload timed out") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        raise HTTPException(
            status_code=status if status in (400, 413, 422) else 502,
            detail=f"Backend model upload failed (HTTP {status})",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Backend model upload failed") from exc
