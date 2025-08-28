"""
AI-powered analysis endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from app.models import QueryRequest, AnalysisRequest
from app.core import settings
from app.services.ai_service import AIService

router = APIRouter(prefix="/ai", tags=["ai"])

ai_service = AIService()

@router.post("/query")
async def intelligent_query(request: QueryRequest):
    """Process natural language queries with OpenAI function calling"""
    try:
        context = {
            "session_id": request.dataset_id,
            "user_id": request.user_id,
            "type": getattr(request, 'context', {}).get('type', 'general')
        }
        
        # Add any additional context
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
            "error": result.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.post("/conversation")
async def ai_conversation(request: QueryRequest):
    """AI-powered conversation about racing data"""
    try:
        context = {
            "session_id": request.dataset_id,
            "user_id": request.user_id,
            "conversation_mode": True
        }
        
        result = await ai_service.process_natural_language_query(
            request.question, 
            context
        )
        
        return {
            "success": True,
            "response": result.get("answer"),
            "data": result.get("function_calls"),
            "context": context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

@router.post("/explain-data")
async def explain_data(request: AnalysisRequest):
    """AI explanation of racing data patterns"""
    try:
        # First get the data analysis
        from app.services.telemetry_service import TelemetryService
        telemetry_service = TelemetryService()
        
        analysis_result = await telemetry_service.analyze_racing_performance(
            request.dataset_id,
            request.analysis_type,
            request.parameters.get("focus_areas", []) if request.parameters else []
        )
        
        # Then get AI explanation
        explanation = await ai_service.explain_data_patterns(analysis_result)
        
        return {
            "success": True,
            "explanation": explanation.get("explanation"),
            "data_analysis": analysis_result,
            "error": explanation.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data explanation failed: {str(e)}")

@router.post("/coach-advice")
async def coach_advice(request: QueryRequest):
    """AI coaching advice based on racing data"""
    try:
        # Get performance data first
        from app.services.telemetry_service import TelemetryService
        telemetry_service = TelemetryService()
        
        performance_data = await telemetry_service.analyze_racing_performance(
            request.dataset_id,
            "overall",
            []
        )
        
        # Generate coaching advice
        skill_level = getattr(request, 'context', {}).get('skill_level', 'intermediate') if hasattr(request, 'context') else 'intermediate'
        advice = await ai_service.generate_coaching_advice(performance_data, skill_level)
        
        return {
            "success": True,
            "coaching_advice": advice.get("coaching_advice"),
            "skill_level": skill_level,
            "performance_data": performance_data,
            "error": advice.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coaching advice failed: {str(e)}")

@router.post("/ask-about-session")
async def ask_about_session(request: QueryRequest):
    """Ask specific questions about a racing session"""
    try:
        context = {
            "session_id": request.dataset_id,
            "user_id": request.user_id,
            "type": "racing_session"
        }
        
        result = await ai_service.process_natural_language_query(
            request.question,
            context
        )
        
        return {
            "success": True,
            "session_id": request.dataset_id,
            "question": request.question,
            "answer": result.get("answer"),
            "analysis_performed": result.get("function_calls", []),
            "error": result.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session query failed: {str(e)}")

@router.post("/model-operations")
async def ai_model_operations(request: Dict[str, Any]):
    """Handle AI model training and prediction requests via natural language"""
    try:
        query = request.get("query", "")
        context = {
            "user_id": request.get("user_id"),
            "operation_type": "model_operations"
        }
        
        result = await ai_service.process_natural_language_query(query, context)
        
        return {
            "success": True,
            "operation_result": result.get("answer"),
            "function_calls": result.get("function_calls", []),
            "error": result.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model operation failed: {str(e)}")

@router.get("/capabilities")
async def get_ai_capabilities():
    """Get information about AI service capabilities"""
    openai_configured = bool(settings.openai_api_key)
    
    capabilities = {
        "openai_configured": openai_configured,
        "available_functions": ai_service.get_available_functions() if openai_configured else [],
        "supported_queries": [
            "Performance analysis questions",
            "Improvement suggestions",
            "Session comparisons",
            "Telemetry insights",
            "AI model training requests",
            "Prediction requests"
        ],
        "fallback_mode": not openai_configured,
        "api_version": "2.0"
    }
    
    return capabilities
