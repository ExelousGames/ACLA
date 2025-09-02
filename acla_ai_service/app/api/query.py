"""
Natural language query processing endpoints with AI model integration
OpenAI generates the main answers while trained telemetry AI models provide supporting data
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from app.services.ai_service import AIService

router = APIRouter(tags=["query"])

# Pydantic models for request validation
class QueryRequest(BaseModel):
    question: str
    dataset_id: Optional[str] = None  # session_id
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class PredictionQueryRequest(BaseModel):
    query: str
    model_id: str
    current_telemetry: Dict[str, Any]
    user_id: str

ai_service = AIService()

@router.post("/query")
async def process_query(request: QueryRequest):
    """
    Main endpoint for processing natural language queries
    OpenAI generates intelligent answers using trained AI models as supporting tools
    """
    try:
        context = {
            "session_id": request.dataset_id,
            "user_id": request.user_id
        }
        
        # Add any additional context from the request
        if request.context:
            context.update(request.context)

        # Process the query with OpenAI, which can call telemetry AI models as needed
        result = await ai_service.process_natural_language_query(
            request.question, 
            context
        )
        
        return {
            "success": True,
            "query": request.question,
            "answer": result.get("answer"),
            "function_calls": result.get("function_calls", []),
            "context": result.get("context"),
            "has_openai": bool(ai_service.openai_client),
            "error": result.get("error"),
            "fallback": result.get("fallback"),
            "processing_steps": result.get("processing_steps", []),
            "ai_models_used": [fc["function"] for fc in result.get("function_calls", []) 
                             if fc["function"].startswith("train_") or fc["function"].startswith("predict_")]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

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
            "answer": result.get("answer"),
            "prediction_result": result.get("function_calls"),
            "backend_integration": "ai-model controller",
            "processing_type": "ai_model_prediction_via_backend"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction query failed: {str(e)}")

@router.post("/query/basic")
async def process_basic_query(request: QueryRequest):
    """Fallback endpoint for basic query processing without OpenAI"""
    try:
        context = {
            "session_id": request.dataset_id,
            "user_id": request.user_id
        }
        
        result = await ai_service._fallback_query_processing(
            request.question, 
            context
        )
        
        return {
            "success": True,
            "query": request.question,
            "answer": result.get("answer"),
            "data": result.get("data"),
            "suggestion": result.get("suggestion"),
            "available_functions": result.get("available_functions"),
            "mode": "basic_fallback"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Basic query processing failed: {str(e)}")

@router.get("/query/available-functions")
async def get_available_functions():
    """Get list of available functions that OpenAI can call"""
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
            "openai_available": bool(ai_service.openai_client)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get functions: {str(e)}")

@router.get("/query/user-models/{user_id}")
async def get_user_models(user_id: str):
    """Get all AI models trained by a user"""
    try:
        result = await ai_service.get_user_models(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "models": result.get("data", []),
            "total_models": len(result.get("data", [])) if result.get("success") else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user models: {str(e)}")

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
    """Health check for the query processing system"""
    return {
        "status": "healthy",
        "openai_configured": bool(ai_service.openai_client),
        "telemetry_service": "available",
        "ai_model_training": "enabled",
        "query_processing": "operational",
        "integration_mode": "openai_with_ai_model_support"
    }
