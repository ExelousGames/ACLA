"""
Racing session analysis endpoints for AI model training and analysis
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio
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


class PredictionRequest(BaseModel):
    telemetry_data: Dict[str, Any]
    model_data: str
    model_type: str = "lap_time_prediction"

class ModelEvaluationRequest(BaseModel):
    model_data: str
    test_telemetry_data: List[Dict[str, Any]]
    target_variable: str = "lap_time"
    model_type: str = "lap_time_prediction"

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

# Initialize telemetry service
telemetry_service = TelemetryService()

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
            existing_model_data=request.existing_model_data,
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
                    existing_model_data=existing_model_data,
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


@router.get("/algorithms/available")
async def get_available_algorithms() -> Dict[str, Any]:
    """
    Get all available algorithms and supported prediction tasks
    """
    try:
        from app.models.ml_algorithms import AlgorithmConfiguration
        algorithm_config = AlgorithmConfiguration()
        
        return {
            "message": "Available algorithms and tasks retrieved successfully",
            "supported_tasks": algorithm_config.get_supported_tasks(),
            "task_descriptions": {
                task: algorithm_config.get_task_description(task) 
                for task in algorithm_config.get_supported_tasks()
            },
            "algorithm_options": {
                task: {
                    "primary": algorithm_config.algorithm_configs[task]["primary"],
                    "alternatives": algorithm_config.algorithm_configs[task]["alternatives"],
                    "description": algorithm_config.algorithm_configs[task]["description"]
                }
                for task in algorithm_config.get_supported_tasks()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithms: {str(e)}")


@router.get("/algorithms/{model_type}")
async def get_algorithm_info(model_type: str) -> Dict[str, Any]:
    """
    Get detailed information about algorithms for a specific task
    """
    try:
        from app.models.ml_algorithms import AlgorithmConfiguration
        algorithm_config = AlgorithmConfiguration()
        
        if model_type not in algorithm_config.get_supported_tasks():
            raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not supported")
        
        task_config = algorithm_config.algorithm_configs[model_type]
        alternatives = algorithm_config.get_algorithm_alternatives(model_type)
        
        algorithm_details = {}
        for algo_name in alternatives:
            if algo_name in algorithm_config.algorithms:
                algo_info = algorithm_config.algorithms[algo_name]
                algorithm_details[algo_name] = {
                    "type": algo_info["type"],
                    "incremental_learning": algo_info["incremental"],
                    "supports_feature_importance": algorithm_config.get_feature_importance_method(algo_name) is not None,
                    "supports_prediction_confidence": algorithm_config.get_prediction_confidence_method(algo_name) is not None
                }
        
        return {
            "model_type": model_type,
            "description": task_config["description"],
            "primary_algorithm": task_config["primary"],
            "alternative_algorithms": task_config["alternatives"],
            "algorithm_details": algorithm_details,
            "recommended_features": algorithm_config.get_recommended_features(model_type)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm info: {str(e)}")


@router.post("/compare-algorithms")
async def compare_algorithms(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Compare performance of different algorithms on the same dataset
    """
    try:
        telemetry_data = request.get("telemetry_data", [])
        target_variable = request.get("target_variable", "lap_time")
        model_type = request.get("model_type", "lap_time_prediction")
        algorithms_to_compare = request.get("algorithms", None)
        
        if not telemetry_data:
            raise HTTPException(status_code=400, detail="No telemetry data provided")
        
        from app.models.ml_algorithms import AlgorithmConfiguration
        algorithm_config = AlgorithmConfiguration()
        
        # Get algorithms to compare
        if algorithms_to_compare is None:
            algorithms_to_compare = algorithm_config.get_algorithm_alternatives(model_type)
        
        comparison_results = {}
        
        for algorithm_name in algorithms_to_compare:
            try:
                # Train model with this algorithm
                result = await telemetry_service.train_ai_model(
                    telemetry_data=telemetry_data,
                    target_variable=target_variable,
                    model_type=model_type,
                    preferred_algorithm=algorithm_name
                )
                
                if result.get("success", False):
                    comparison_results[algorithm_name] = {
                        "success": True,
                        "metrics": result["training_metrics"],
                        "algorithm_type": result["algorithm_type"],
                        "feature_count": result["feature_count"],
                        "training_samples": result["training_samples"],
                        "supports_incremental": result["supports_incremental"]
                    }
                else:
                    comparison_results[algorithm_name] = {
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    }
                    
            except Exception as e:
                comparison_results[algorithm_name] = {
                    "success": False,
                    "error": f"Failed to train with {algorithm_name}: {str(e)}"
                }
        
        # Find best performing algorithm
        best_algorithm = None
        best_score = -float('inf')
        
        for algo_name, result in comparison_results.items():
            if result.get("success", False):
                metrics = result.get("metrics", {})
                # Use R2 score for regression, accuracy for classification
                score = metrics.get("test_r2", metrics.get("test_accuracy", 0))
                if score > best_score:
                    best_score = score
                    best_algorithm = algo_name
        
        return {
            "message": "Algorithm comparison completed",
            "model_type": model_type,
            "target_variable": target_variable,
            "comparison_results": comparison_results,
            "best_algorithm": best_algorithm,
            "best_score": best_score,
            "algorithms_tested": len(comparison_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Algorithm comparison failed: {str(e)}")


