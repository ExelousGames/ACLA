"""
Racing session analysis endpoints for AI model training and analysis
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from httpx import request
from pydantic import BaseModel
import asyncio
from app.services.scikit_ml_service import TelemetryMLService
from app.services.telemetry_service import TelemetryService

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
    track_name: str
    car_name: str   
    guidance_type: str = "both"  # "actions", "behavior", or "both"
    user_id: Optional[str] = None
    
# Initialize telemetry service
telemetry_service = TelemetryService()
telemetryMLService = TelemetryMLService()

@router.post("/train-model")
async def train_ai_model(request: TrainingRequest) -> Dict[str, Any]:
    """
    Train AI model on telemetry data for performance prediction, optional existing model data
    Returns trained model data for backend storage - no persistent data in AI service
    """
    try:
        result = await telemetry_service.train_ai_model(
            telemetry_data=request.telemetry_data,
            target_variable=request.target_variable,
            model_type=request.model_type,
            preferred_algorithm=request.preferred_algorithm,
            existing_model_data_from_db=request.existing_model_data,
            user_id=request.user_id,
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Training failed"))
        
        return {
            "message": "Model training completed successfully",
            "trained_model": result,
            "instructions": "Save the model_data field to your database, and use it by this AI service"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/train-multiple-models")
async def train_multiple_ai_models(request: MultipleTrainingRequest) -> Dict[str, Any]:
    """
    Train multiple AI models simultaneously on telemetry data with different configurations
    Returns trained model data for all successful trainings - no persistent data in AI service
    """
    try:
        if not request.models_config:
            raise HTTPException(status_code=400, detail="No model configurations provided")
        
        training_results = {}
        failed_trainings = {}
        successful_count = 0
        
        import asyncio
        
        async def train_single_model(model_config: Dict[str, Any], config_id: str):
            """Train a single model with given configuration"""
            try:
                # Validate required fields in model_config
                target_variable = model_config.get("target_variable")
                model_type = model_config.get("model_type")
                preferred_algorithm = model_config.get("preferred_algorithm")
                existing_model_data = model_config.get("existing_model_data")
                
                result = await telemetry_service.train_ai_model(
                    telemetry_data=request.telemetry_data,
                    target_variable=target_variable,
                    model_type=model_type,
                    preferred_algorithm=preferred_algorithm,
                    existing_model_data_from_db=existing_model_data,
                    user_id=request.user_id,
                )
                
                return config_id, result, None
                
            except Exception as e:
                return config_id, None, str(e)
        
        # Prepare training tasks
        training_tasks = []
        for i, model_config in enumerate(request.models_config):
            config_id = model_config.get("config_id", f"model_{i+1}")
            training_tasks.append(train_single_model(model_config, config_id))
        
        # Execute training (parallel or sequential based on request parameter)
        if request.parallel_training:
            # Train all models in parallel
            results = await asyncio.gather(*training_tasks, return_exceptions=True)
        else:
            # Train models sequentially
            results = []
            for task in training_tasks:
                result = await task
                results.append(result)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions from asyncio.gather
                config_id = "unknown"
                failed_trainings[config_id] = f"Unexpected error: {str(result)}"
                continue
                
            config_id, training_result, error = result
            
            if error:

                failed_trainings[config_id] = error


            else:
                # Save the model_data field from each successful training to your database
                training_results[config_id] = training_result
                successful_count += 1
        
        # Prepare response
        response = {
            "message": f"Multiple model training completed. {successful_count} successful, {len(failed_trainings)} failed.",
            "session_id": request.session_id,
            "total_models_requested": len(request.models_config),
            "successful_trainings": successful_count,
            "failed_trainings": len(failed_trainings),
            "training_results": training_results,
            "instructions": "Save the model_data field from each successful training to your database"
        }
        
        # Include failed trainings info if any
        if failed_trainings:
            response["failed_training_details"] = failed_trainings
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multiple model training failed: {str(e)}")


@router.post("/imitation-learning-guidance")
async def get_imitation_learning_expert_guidance(request: ImitationPredictRequest) -> Dict[str, Any]:
    """
    Get expert driving guidance using imitation learning model
    Provides recommendations based on expert driving behavior analysis
    """
    try:
        # Validate guidance_type parameter
        valid_guidance_types = ["actions", "behavior", "both"]
        if request.guidance_type not in valid_guidance_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid guidance_type. Must be one of: {valid_guidance_types}"
            )
        
        try:
            # Call the telemetryMLService to get expert guidance
            result = await telemetryMLService.get_imitation_learning_expert_guidance(
                current_telemetry=request.current_telemetry,
                trackName=request.track_name,
                carName=request.car_name,
                guidance_type=request.guidance_type,
            )
        
            print(f"Imitation learning guidance result: {result}")
        except Exception as e:
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