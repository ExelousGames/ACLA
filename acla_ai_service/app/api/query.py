"""
Natural language query processing endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from app.models import QueryRequest
from app.services.ai_service import AIService

router = APIRouter(tags=["query"])

ai_service = AIService()

@router.post("/query")
async def process_query(request: QueryRequest):
    """Main endpoint for processing natural language queries with AI function calling"""
    try:
        context = {
            "session_id": request.dataset_id,
            "user_id": request.user_id
        }
        
        # Add any additional context from the request
        if hasattr(request, 'context') and request.context:
            context.update(request.context)
        
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
            "fallback": result.get("fallback")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

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
            "mode": "basic"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Basic query processing failed: {str(e)}")
