"""Publish trained model artifacts to the backend model store."""

import logging
import json
from typing import Any, BinaryIO, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


async def upload_ultralytics_model(
    backend_service,
    model_file: BinaryIO,
    filename: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Stream a checkpoint to the backend's binary Ultralytics model store."""
    if not await backend_service.ensure_connection():
        raise ConnectionError("Failed to establish backend connection")

    url = f"{backend_service.base_url}:{backend_service.base_port}/ai-model/ultralytics"
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=180.0, pool=180.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(2):
            model_file.seek(0)
            response = await client.post(
                url,
                headers=backend_service.get_auth_headers(),
                data={"metadata": json.dumps(metadata, allow_nan=False)},
                files={"file": (filename, model_file, "application/octet-stream")},
            )
            # A rejected JWT cannot create a record. Other failures must not be
            # retried automatically because each upload creates a new record.
            if response.status_code == 401 and attempt == 0:
                if not await backend_service.establish_connection():
                    raise ConnectionError("Failed to refresh backend authentication")
                continue
            response.raise_for_status()
            return response.json()


async def save_ai_model(
    backend_service,
    model_type: str,
    model_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    is_active: bool = True,
) -> Dict[str, Any]:
    """Save AI model results using the backend client's chunked transfer.

    Args:
        backend_service: Backend client used for authenticated uploads
        model_type: Type of the AI model (e.g., "tire_grip_analysis", "top_lap_reference")
        model_data: The serialized model data payload
        metadata: Optional metadata containing model info and timestamps
        is_active: Whether this model should be set as active
    """
    metadata = metadata or {}

    print(f"[INFO] Saving AI model results to backend: {model_type}")
    logger.info(f"Saving AI model results to backend: {model_type}")

    structured_data = {
        "modelType": model_type,
        "modelData": model_data,
        "metadata": metadata,
        "isActive": is_active,
    }

    try:
        response = await backend_service.send_chunked_data(
            data=structured_data,
            endpoint="ai-model/save",
            chunk_size=512 * 1024,  # 512KB chunks
        )

        if not response.get("success", False):
            raise Exception(f"Backend rejected data: {response.get('message', 'Unknown error')}")

    except Exception as e:
        logger.error(f"❌ Failed to save AI model results: {str(e)}")
        raise
    return {"success": True}
