"""
Racing session analysis endpoints for AI model training and analysis
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from httpx import request
from pydantic import BaseModel
import asyncio
from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService


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
            result = await telemetryMLService.predict_expert_actions(
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