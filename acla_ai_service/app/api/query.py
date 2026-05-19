"""
Natural language query processing endpoints with AI model integration
OpenAI generates the main answers while trained telemetry AI models provide supporting data
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from app.services.ai_service import AIService
from app.services.llm.llama_health import check_llama_server

router = APIRouter(tags=["query"])

# Pydantic models for request validation
class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class PredictionQueryRequest(BaseModel):
    query: str
    model_id: str
    current_telemetry: Dict[str, Any]
    user_id: str

ai_service = AIService()

@router.post("/naturallanguagequery")
async def process_query(request: QueryRequest):

    """
    Main endpoint for processing natural language queries
    OpenAI generates intelligent answers using trained AI models as supporting tools
    """
    try:
        
        # Add any additional context from the request

            
        # Process the query with OpenAI, which can call telemetry AI models as needed
        try:
            result = await ai_service.process_natural_language_query(
                request.question,
                request.context
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"process_natural_language_query() AI query processing error: {str(e)}")

        return {
            "success": True,
            "query": request.question,
            "payload": result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.post("/naturallanguagequery/stream")
async def process_query_stream(request: QueryRequest):
    """Phase 2.5 — Streaming variant of /naturallanguagequery.

    Returns a Server-Sent Events stream. Each event's `data:` field is a
    JSON object with a `type` discriminator. See
    `app/voice/stream_events.py` for the protocol.

    Event types: token, audio, tool_start, tool_end, done, error.

    The frontend renders tokens into the chat bubble as they arrive and
    queues audio events for gapless sentence-by-sentence playback. Total
    time-to-first-audio target: ~500ms.
    """
    async def event_source():
        async for sse_chunk in ai_service.stream_natural_language_query(
            request.question,
            request.context,
        ):
            yield sse_chunk

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            # Disable any reverse-proxy buffering so events flush immediately.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/query/predict")
async def query_predict(request: PredictionQueryRequest):
    """
    Process natural language queries about making predictions via backend controller
    Example: "What lap time will I get with this current telemetry data?"
    """
    try:
        enhanced_query = f"""
        {request.query}
        
        Prediction Parameters:
        - Model ID: {request.model_id}
        - User ID: {request.user_id}
        - Current telemetry data provided
        """
        
        context = {
            "user_id": request.user_id,
            "prediction_request": {
                "model_id": request.model_id,
                "current_telemetry": request.current_telemetry
            },
            "query_type": "prediction"
        }
        
        # Process the query, OpenAI will call predict_with_telemetry_model which uses backend
        result = await ai_service.process_natural_language_query(
            enhanced_query, 
            context
        )
        
        return {
            "success": True,
            "query": request.query,
            "payload": result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction query failed: {str(e)}")

@router.get("/query/available-functions")
async def get_available_functions():
    """Get list of available functions the active LLM can call"""
    try:
        functions = ai_service.get_available_functions()

        return {
            "success": True,
            "available_functions": functions,
            "total_functions": len(functions),
            "ai_model_functions": [f for f in functions if
                                 f["name"].startswith("train_") or
                                 f["name"].startswith("predict_") or
                                 f["name"].startswith("evaluate_") or
                                 f["name"].startswith("get_model_")],
            "telemetry_functions": [f for f in functions if
                                  "telemetry" in f["name"] or
                                  "performance" in f["name"] or
                                  "session" in f["name"]],
            # Active LLM backend status (llama-server is canonical from Phase 1)
            "llm_provider": ai_service.llm_provider,
            "chat_model": ai_service.chat_model,
            # Legacy fallback availability — kept for any consumers still reading it
            "openai_available": bool(ai_service.openai_client),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get functions: {str(e)}")


@router.delete("/query/models/{model_id}")
async def delete_model(model_id: str, user_id: str = Body(...)):
    """Delete a user's AI model"""
    try:
        result = await ai_service.delete_user_model(model_id, user_id)
        
        return {
            "success": result.get("success", False),
            "message": "Model deleted successfully" if result.get("success") else "Failed to delete model",
            "model_id": model_id,
            "error": result.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

# Health check
@router.get("/query/health")
async def health_check():
    """Health check for the query processing system.

    Reports whether each LLM backend is reachable:
      - llama_server: local Qwen-via-llama.cpp sidecar (canonical, Phase 1+).
      - openai_configured: legacy OpenAI client (being phased out).
    Overall status is "healthy" if at least one backend is usable.
    """
    llama_health = await check_llama_server()

    openai_configured = bool(ai_service.openai_client)

    # The chat endpoint is "ready" iff the active provider's backend is up.
    if ai_service.llm_provider == "llama":
        active_backend_ready = llama_health.reachable
    else:  # "openai"
        active_backend_ready = openai_configured

    return {
        "status": "healthy" if active_backend_ready else "degraded",
        "llm_provider": ai_service.llm_provider,
        "chat_model": ai_service.chat_model,
        "llama_server": llama_health.to_dict(),
        "openai_configured": openai_configured,
        "telemetry_service": "available",
        "ai_model_training": "enabled",
        "query_processing": "operational",
        "integration_mode": (
            "llama_server" if ai_service.llm_provider == "llama" and llama_health.reachable
            else "openai_legacy" if ai_service.llm_provider == "openai" and openai_configured
            else "degraded"
        ),
    }


@router.get("/query/llama-health")
async def llama_health_check():
    """Dedicated llama-server reachability probe (also used by start scripts)."""
    health = await check_llama_server()
    return health.to_dict()
