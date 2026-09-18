"""Publish trained model artifacts to the backend model store."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
