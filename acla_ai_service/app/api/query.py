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
