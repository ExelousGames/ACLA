"""
Racing session analysis endpoints for AI model training and analysis
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.services.telemetry_service import TelemetryService

router = APIRouter(prefix="/racing-session", tags=["racing-session"])

# Pydantic models for request/response validation
class TrainingRequest(BaseModel):
    session_id: str
    telemetry_data: List[Dict[str, Any]]
    target_variable: str = "lap_time"
    model_type: str = "lap_time_prediction"
    user_id: Optional[str] = None
    existing_model_data: Optional[str] = None
    session_metadata: Optional[Dict[str, Any]] = None

class BatchTrainingRequest(BaseModel):
    training_sessions: List[Dict[str, Any]]
    target_variable: str = "lap_time"
    model_type: str = "lap_time_prediction"
    existing_model_data: Optional[str] = None

# Request model for retraining
class RetrainModelRequest(BaseModel):
    model_data: str
    new_telemetry_data: List[Dict[str, Any]]
    target_variable: str = "lap_time"
    model_type: str = "lap_time_prediction"
    session_metadata: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    telemetry_data: Dict[str, Any]
    model_data: str
    model_type: str = "lap_time_prediction"

class ModelEvaluationRequest(BaseModel):
    model_data: str
    test_telemetry_data: List[Dict[str, Any]]
    target_variable: str = "lap_time"
    model_type: str = "lap_time_prediction"

class AnalysisRequest(BaseModel):
    session_id: str
    analysis_type: str = "overall"
    focus_areas: Optional[List[str]] = None
    telemetry_data: Optional[List[Dict[str, Any]]] = None
    use_ai_model: bool = False
    model_data: Optional[str] = None

class ImprovementSuggestionsRequest(BaseModel):
    session_id: str
    skill_level: str = "intermediate"
    focus_area: Optional[str] = None
    telemetry_data: Optional[List[Dict[str, Any]]] = None
    model_data: Optional[str] = None

# Initialize telemetry service
telemetry_service = TelemetryService()

@router.post("/train-model")
async def train_ai_model(request: TrainingRequest) -> Dict[str, Any]:
    """
    Train AI model on telemetry data for performance prediction, optional existingmodel data
    Returns trained model data for backend storage - no persistent data in AI service
    """
    try:
        result = await telemetry_service.train_ai_model(
            telemetry_data=request.telemetry_data,
            target_variable=request.target_variable,
            model_type=request.model_type,
            existing_model_data=request.existing_model_data,
            user_id=request.user_id,
            session_metadata=request.session_metadata
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Training failed"))
        
        return {
            "message": "Model training completed successfully",
            "trained_model": result,
            "instructions": "Save the model_data field to your database for future use"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/batch-train-model")
async def batch_train_ai_model(request: BatchTrainingRequest) -> Dict[str, Any]:
    """
    Train AI model on multiple telemetry sessions in batch
    Supports incremental learning if existing model provided
    """
    try:
        result = await telemetry_service.batch_train_model(
            training_sessions=request.training_sessions,
            target_variable=request.target_variable,
            model_type=request.model_type,
            existing_model_data=request.existing_model_data
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Batch training failed"))
        
        return {
            "message": "Batch model training completed successfully",
            "trained_model": result,
            "sessions_processed": len(request.training_sessions),
            "instructions": "Save the model_data field to your database for future use"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch training failed: {str(e)}")


@router.post("/evaluate-model")
async def evaluate_model(request: ModelEvaluationRequest) -> Dict[str, Any]:
    """
    Evaluate trained model performance on test data
    """
    try:
        result = await telemetry_service.evaluate_model_performance(
            model_data=request.model_data,
            test_telemetry_data=request.test_telemetry_data,
            target_variable=request.target_variable,
            model_type=request.model_type
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Evaluation failed"))
        
        return {
            "message": "Model evaluation completed successfully",
            "evaluation_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


@router.post("/validate-telemetry")
async def validate_telemetry_data(telemetry_data: List[Dict[str, Any]] = Body(...)) -> Dict[str, Any]:
    """
    Validate telemetry data quality and completeness for training
    """
    try:
        # Convert to dict format expected by validation method
        data_dict = {}
        if telemetry_data:
            # Combine all telemetry records into columns
            for record in telemetry_data:
                for key, value in record.items():
                    if key not in data_dict:
                        data_dict[key] = []
                    data_dict[key].append(value)
        
        result = telemetry_service.validate_telemetry_data(data_dict)
        
        return {
            "message": "Telemetry data validation completed",
            "validation_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


