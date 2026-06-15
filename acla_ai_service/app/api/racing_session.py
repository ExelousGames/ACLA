"""
Racing session analysis endpoints for AI model training and analysis
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from httpx import request
from pydantic import BaseModel
import asyncio
import pandas as pd
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from app.racing_engineer.expert_actions import predict_expert_actions
from app.ml.segment_classifier.service import segment_classifier
from app.ml.opportunity_forecaster import opportunity_forecaster
from app.services.user_session_analysis import analyze_user_sessions
from app.shared.label_hierarchy import build_track_area_segments


router = APIRouter(prefix="/racing-session", tags=["racing-session"])

# Pydantic models for request/response validation
class TrainingRequest(BaseModel):
    session_id: str
    telemetry_data: List[Dict[str, Any]]
    target_variable: str = "lap_time"
    model_type: str = "lap_time_prediction"
    preferred_algorithm: Optional[str] = None
    user_id: Optional[str] = None
    existing_model_data: Optional[str] = None

class MultipleTrainingRequest(BaseModel):
    session_id: str
    telemetry_data: List[Dict[str, Any]]

    '''#example of models_config
        {
            "config_id": "rf_model",
            "target_variable": "lap_time", 
            "model_type": "lap_time_prediction",
            "preferred_algorithm": "random_forest",
            "existing_model_data": data
        }
    '''
    models_config: List[Dict[str, Any]]  # List of model configurations to train
    user_id: Optional[str] = None
    parallel_training: bool = True  # Whether to train models in parallel or sequentially

class PredictionRequest(BaseModel):
    telemetry_data: Dict[str, Any]
    model_data: str  # Base64 encoded model data from database
    model_type: Optional[str] = "lap_time_prediction"
    use_river: bool = True  # Whether to use River ML or legacy scikit-learn
    user_id: Optional[str] = None

class ImitationPredictRequest(BaseModel):
    current_telemetry: Dict[str, Any]
    human_request: Optional[str] = None
    delay_seconds: Optional[float] = 0.0
    track_name: str
    car_name: str   
    user_id: Optional[str] = None

class OpportunityForecastRequest(BaseModel):
    telemetry_data: List[Dict[str, Any]]
    horizon_seconds: Optional[float] = 10.0
    top_k: Optional[int] = 3

class TrackCornerKnowledgeRequest(BaseModel):
    track_name: str
    corner_name: str
    normalized_position: Optional[float] = None
    trigger_position: Optional[float] = None
    current_telemetry: Optional[Dict[str, Any]] = None

class AnalyzeUserSessionsRequest(BaseModel):
    user_id: str

class SegmentClassificationRequest(BaseModel):
    session_id: Optional[str] = None
    telemetry_data: List[Dict[str, Any]]
    track_name: Optional[str] = None
    car_name: Optional[str] = None
    
# Initialize telemetry service
telemetryMLService = Full_dataset_TelemetryMLService()


@router.post("/imitation-learning-guidance")
async def get_imitation_learning_expert_guidance(request: ImitationPredictRequest) -> Dict[str, Any]:
    """
    Get expert driving guidance using imitation learning model
    Provides recommendations based on expert driving behavior analysis
    """
    try:
        # Validate guidance_type parameter
        try:
            # Call the telemetryMLService to get expert guidance
            result = await predict_expert_actions(
                telemetryMLService,
                telemetry_dict=request.current_telemetry,
                user_request=request.human_request,
            )

        except Exception as e:
            print(f"[ERROR] Exception in expert guidance service: \n {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in expert guidance service: {str(e)}")
        
        return {
            "message": "Expert guidance generated successfully",
            "guidance_result": result,
            "timestamp": result.get("timestamp"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expert guidance failed: {str(e)}")


@router.post("/opportunity-forecast")
async def get_opportunity_forecast(request: OpportunityForecastRequest) -> Dict[str, Any]:
    try:
        if not request.telemetry_data:
            raise HTTPException(status_code=400, detail="telemetry_data is required")

        result = opportunity_forecaster.forecast(
            request.telemetry_data,
            horizon_seconds=request.horizon_seconds or 10.0,
            top_k=request.top_k or 3,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Opportunity forecast failed: {str(e)}")


@router.post("/track-corner-knowledge")
async def get_track_corner_knowledge(request: TrackCornerKnowledgeRequest) -> Dict[str, Any]:
    try:
        track_name = (request.track_name or "").strip()
        corner_name = (request.corner_name or "").strip()
        if not track_name:
            raise HTTPException(status_code=400, detail="track_name is required")
        if not corner_name:
            raise HTTPException(status_code=400, detail="corner_name is required")

        from app.external_knowledge_base import track as track_lookup

        track_key = track_name.lower().replace(" ", "_")
        result = track_lookup(track_key, corner=corner_name)
        if result is None:
            raise HTTPException(status_code=404, detail=f"track '{track_name}' not in corpus")
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "track_knowledge": result,
            "normalized_position": request.normalized_position,
            "trigger_position": request.trigger_position,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Track corner knowledge failed: {str(e)}")


@router.post("/segment-classification")
async def classify_session_segments(request: SegmentClassificationRequest) -> Dict[str, Any]:
    try:
        if not request.telemetry_data:
            raise HTTPException(status_code=400, detail="telemetry_data is required")

        dataframe = pd.DataFrame(request.telemetry_data)
        predicted_segments = segment_classifier.scan_telemetry_data(dataframe)
        raw_segments = []

        for segment in predicted_segments:
            segment_dict = segment.to_dict() if hasattr(segment, "to_dict") else dict(segment)
            raw_segments.append({
                "id": segment_dict.get("id"),
                "labels": segment_dict.get("labels", []),
                "start_index": segment_dict.get("start_index"),
                "end_index": segment_dict.get("end_index"),
            })

        segments = build_track_area_segments(
            raw_segments,
            request.telemetry_data,
            request.track_name,
        )

        return {
            "status": "success",
            "session_id": request.session_id,
            "samples_analyzed": len(request.telemetry_data),
            "segment_count": len(segments),
            "segments": segments,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segment classification failed: {str(e)}")


@router.post("/analyze-user-sessions")
async def analyze_all_user_sessions(request: AnalyzeUserSessionsRequest) -> Dict[str, Any]:
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        session_analysis = await analyze_user_sessions(request.user_id)
        return {
            "status": "success",
            "sessionAnalysis": session_analysis,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User session analysis failed: {str(e)}")
