"""
River-based Online Machine Learning Service for Racing Telemetry
Handles streaming data and incremental learning using River library
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import base64
import json
from datetime import datetime, timezone
from river import metrics, compose, preprocessing
import io

from app.models.telemetry_models import TelemetryFeatures, FeatureProcessor
from app.models.ml_algorithms import AlgorithmConfiguration


class RiverMLService:
    """Service for online machine learning using River library"""
    
    def __init__(self):
        self.telemetry_features = TelemetryFeatures()
        self.algorithm_config = AlgorithmConfiguration()
        
        # Model types supported for different prediction tasks
        self.model_types = {
            "lap_time_prediction": "regression",
            "sector_time_prediction": "regression", 
            "performance_classification": "classification", 
            "setup_recommendation": "classification",
            "tire_strategy": "classification",
            "fuel_consumption": "regression",
            "brake_performance": "regression",
            "overtaking_opportunity": "classification",
            "racing_line_optimization": "regression",
            "weather_adaptation": "regression",
            "consistency_analysis": "regression",
            "damage_prediction": "classification"
        }
    
    async def train_online_model(self, 
                                telemetry_data: List[Dict[str, Any]], 
                                target_variable: str,
                                model_type: str = "lap_time_prediction",
                                preferred_algorithm: Optional[str] = None,
                                existing_model_data: Optional[str] = None,
                                user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Train or update an online model using River's streaming approach
        
        Args:
            telemetry_data: List of telemetry data dictionaries
            target_variable: The variable to predict
            model_type: Type of model to train
            preferred_algorithm: Override the default algorithm for this task
            existing_model_data: Base64 encoded existing model for incremental training
            user_id: User identifier for tracking
         
        Returns:
            Dict containing trained model data and metrics
        """
        try:
            
            # Convert telemetry data to DataFrame
            df = pd.DataFrame(telemetry_data)
            
            if df.empty:
                return {
                    "success": False,
                    "error": "No telemetry data provided",
                    "model_type": model_type,
                }
            
            feature_processor = FeatureProcessor(df)

            # Clean data
            processed_df = feature_processor.prepare_for_analysis()
            
            # Load existing model or create new one
            if existing_model_data:
                model, model_type, target_name, feature_names, algorithm_name, metrics_tracker = self._load_model(existing_model_data)
            else:
                # Get optimal algorithm configuration for creating this new task
                algorithm_config = self.algorithm_config.get_algorithm_for_task(model_type, preferred_algorithm)
                model, metrics_tracker = self._create_model(algorithm_config)

            # Prepare features data and target value for online learning
            cleanedFeatureData, target_values, feature_names = self._prepare_online_features_and_target(
                processed_df, target_variable, model_type
            )
            
            if not cleanedFeatureData:
                return {
                    "success": False,
                    "error": f"Could not prepare features data",
                    "model_type": model_type,
                    "algorithm_used": algorithm_config["name"]
                }
            
            
            if not target_values:
                return {
                    "success": False,
                    "error": f"Could not prepare target values for '{target_variable}'",
                    "model_type": model_type,
                    "algorithm_used": algorithm_config["name"]
                }
            
            
            # Train the model incrementally
            training_results = self._train_incrementally(
                model, metrics_tracker, cleanedFeatureData, target_values, feature_names
            )
            
            # Serialize the trained model
            model_data = self._serialize_river_model(model, feature_names, algorithm_config["name"])
            
            return {
                "success": True,
                "model_data": model_data,
                "model_type": model_type,
                "algorithm_used": algorithm_config["name"],
                "algorithm_type": algorithm_config["type"],
                "target_variable": target_variable,
                "training_metrics": training_results["metrics"],
                "samples_processed": len(cleanedFeatureData),
                "features_count": len(feature_names),
                "feature_names": feature_names,
                "algorithm_description": self.algorithm_config.get_algorithm_description(algorithm_config["name"]),
                "algorithm_strengths": self.algorithm_config.get_algorithm_strengths(algorithm_config["name"]),
                "training_time": training_results["training_time"],
                "data_quality_score": self._calculate_data_quality_score(processed_df),
                "recommendations": self._generate_training_recommendations(
                    training_results["metrics"], algorithm_config
                ),
                "model_version": self._get_model_version(existing_model_data),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}",
                "model_type": model_type,
                "algorithm_used": algorithm_config.get("name", "unknown") if 'algorithm_config' in locals() else "unknown"
            }
    
    def _prepare_online_features_and_target(self, df: pd.DataFrame, target_variable: str, 
                                          model_type: str) -> Tuple[List[Dict], List, List[str]]:
        """
        Prepare data related features and target for online learning
        
        Args:
            df: DataFrame with telemetry data
            target_variable: Target variable name
            model_type: Type of prediction task
        
        Returns:
            cleanedFeatureData: List of cleaned data about selected features or River
            target_values: List of target values
            feature_names: List of feature values
        """
        try:
            
            # Get recommended features for this model type
            recommended_features = self.algorithm_config.get_recommended_features(model_type)
   
            # select only those features that are in the recommended_features that are also present in the DataFrame
            #for each col in recommended_features, if the col is in df.columns, then use col
            available_features = [col for col in recommended_features if col in df.columns]
            
            if not available_features or target_variable not in df.columns:
                raise RuntimeError("Insufficient features or target variable missing")

            # Drop rows with missing target values
            df_clean = df.dropna(subset=[target_variable])
            
            if df_clean.empty:
                raise RuntimeError("No valid rows available for training")
            
            # Convert to River format (list of dictionaries)
            cleanedFeatureData = []
            target_values = []
            invalid_samples = 0
            
            for idx, row in df_clean.iterrows():
                # Create feature dictionary for this sample

                feature_dict = {}
                valid_sample = True
                
                for feature in available_features:
                    #check each available features in the current row

                    # value contains telemetry data of the current row with selected feature name
                    value = row[feature]

                    if pd.isna(value):
                        # If value is missing, mark current row as invalid
                        valid_sample = False
                        invalid_samples += 1
                        break

                    # If value is valid, add to feature dictionary
                    feature_dict[feature] = float(value)
                
                # if its a valid sample add to features and target lists
                if valid_sample:
                    cleanedFeatureData.append(feature_dict)
                    target_values.append(float(row[target_variable]))
            
            return cleanedFeatureData, target_values, available_features
            
        except Exception as e:
            return [], [], []
    

    def _create_model(self, algorithm_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Load existing model or create new one with metrics tracker
        
        Args:
            algorithm_config: Algorithm configuration
            existing_model_data: Base64 encoded existing model
        
        Returns:
            model: River model instance
            metrics_tracker: Metrics tracker
        """  

        model = self.algorithm_config.create_algorithm_instance(algorithm_config)
        
        # Create metrics tracker
        metrics_tracker = self.algorithm_config.create_metrics_tracker(algorithm_config["type"])

        # Additional model information
        return model,metrics_tracker
    
    
    def _load_model(self, existing_model_data: str) -> Tuple[Any, Any]:
        """
        Load existing model or create new one with metrics tracker
        
        Args:
            algorithm_config: Algorithm configuration
            existing_model_data: Base64 encoded existing model
        
        Returns:
            model: River model instance
            metrics_tracker: Metrics tracker
        """  

        try:
            model, modelType, target_name, feature_names, algorithm_name = self._deserialize_river_model(existing_model_data)
        except Exception as e:
            raise Exception("Failed to load existing model") from e
        
        algorithm_config = self.algorithms[algorithm_name].copy()
        algorithm_config["name"] = algorithm_name
        algorithm_config["task_description"] = self.algorithm_config.get(modelType, {}).get("description", "Unknown task")
        
        # Create metrics tracker
        metrics_tracker = self.algorithm_config.create_metrics_tracker(algorithm_config["type"])

        # Additional model information
        return model,modelType, target_name, feature_names, algorithm_name, metrics_tracker
  
    def _train_incrementally(self, model: Any, metrics_tracker: Dict[str, Any], 
                           featuresData: List[Dict], target_values: List, 
                           feature_names: List[str]) -> Dict[str, Any]:
        """
        Train the model incrementally using River's online learning
        
        Args:
            model: River model instance
            metrics_tracker: Metrics tracker
            featuresData: List of feature dictionaries contains each feature's data
            target_values: List of target values
            feature_names: List of feature names
        
        Returns:
            Training results
        """

        start_time = datetime.now()
        prediction_count = 0
        successful_predictions = 0
        failed_predictions = 0
        
        # Train incrementally
        for i, (x, y) in enumerate(zip(featuresData, target_values)):
            # Make prediction first (for metrics)
            if hasattr(model, 'predict_one'):
                try:
                    y_pred = model.predict_one(x)
                    prediction_count += 1
                    
                    if y_pred is not None:
                        successful_predictions += 1
                        # Update each metric in the tracker
                        for metric_name, metric_obj in metrics_tracker.items():
                            try:
                                metric_obj.update(y, y_pred)
                            except Exception as metric_e:
                                pass
                    else:
                        failed_predictions += 1

                except Exception as e:
                    failed_predictions += 1
            
            # Learn from this sample
            try:
                model.learn_one(x, y)
            except Exception as e:
                pass
        
        training_time = (datetime.now() - start_time).total_seconds()

        # Extract metrics
        final_metrics = {}
        
        # Extract metrics from the dictionary of metric objects
        for metric_name, metric_obj in metrics_tracker.items():
            try:
                if hasattr(metric_obj, 'get'):
                    metric_value = metric_obj.get()
                    final_metrics[metric_name.lower()] = metric_value
            except Exception as e:
                continue
       
        return {
            "metrics": final_metrics,
            "training_time": training_time,
            "samples_trained": len(featuresData)
        }
    
    def _serialize_river_model(self, model: Any, modelType:str, feature_names: List[str], target_name: str,
                              algorithm_name: str) -> str:
        """
        Serialize River model to base64 string
        
        Args:
            model: Trained River model
            feature_names: List of feature names
            algorithm_name: Name of the algorithm
        
        Returns:
            Base64 encoded model data
        """
        try:
            model_data = {
                "model": model,
                "modelType": modelType,
                "targetName": target_name,
                "feature_names": feature_names,
                "algorithm_name": algorithm_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "river_version": "0.21.0"
            }
            
            
            # Serialize using pickle
            buffer = io.BytesIO()
            pickle.dump(model_data, buffer)
            buffer.seek(0)
        
            # Encode to base64
            encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return encoded_data
            
        except Exception as e:
            raise Exception(f"Failed to serialize model: {str(e)}")
    
    def _deserialize_river_model(self, model_data: str) -> Tuple[Any, List[str]]:
        """
        Deserialize River model from base64 string
        
        Args:
            model_data: Base64 encoded model data
        
        Returns:
            model: River model instance
            target_name: Name of the target variable
            feature_names: List of feature names
            algorithm_name: Name of the algorithm
        """
        
        try:
            # Decode from base64
            decoded_data = base64.b64decode(model_data.encode('utf-8'))
            
            # Deserialize using pickle
            buffer = io.BytesIO(decoded_data)
            model_dict = pickle.load(buffer)
            
            model = model_dict["model"]
            modelType = model_dict["modelType"]
            target_name = model_dict["targetName"]
            feature_names = model_dict["feature_names"]
            algorithm_name = model_dict["algorithm_name"]
            return model,modelType,target_name, feature_names,algorithm_name
            
        except Exception as e:
            raise Exception(f"Failed to deserialize model: {str(e)}")
    
    async def predict_online(self, 
                           telemetry_data: Dict[str, Any],
                           model_data: str,
                           model_type: str) -> Dict[str, Any]:
        """
        Make predictions using a trained River model
        
        Args:
            telemetry_data: Current telemetry data for prediction
            model_data: Base64 encoded model data
            model_type: Type of model
        
        Returns:
            Prediction results
        """
        try:
            # Deserialize the model
            model, feature_names = self._deserialize_river_model(model_data)
            
            # Prepare features for prediction
            feature_dict = {}
            missing_features = []
            
            for feature_name in feature_names:
                if feature_name in telemetry_data:
                    feature_dict[feature_name] = float(telemetry_data[feature_name])
                else:
                    missing_features.append(feature_name)
            
            if missing_features:
                return {
                    "success": False,
                    "error": f"Missing required features: {missing_features}",
                    "required_features": feature_names
                }
            
            # Make prediction
            algorithm_type = self.model_types.get(model_type, "regression")
            
            if algorithm_type == "regression":
                prediction = model.predict_one(feature_dict)
                confidence = None
            else:
                # For classification, get probabilities
                if hasattr(model, 'predict_proba_one'):
                    probabilities = model.predict_proba_one(feature_dict)
                    prediction = max(probabilities, key=probabilities.get) if probabilities else None
                    confidence = max(probabilities.values()) if probabilities else None
                else:
                    prediction = model.predict_one(feature_dict)
                    confidence = None
            
            return {
                "success": True,
                "prediction": prediction,
                "confidence": confidence,
                "model_type": model_type,
                "features_used": feature_names,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "model_type": model_type
            }
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-100)"""
        try:
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            missing_score = max(0, 100 - (missing_ratio * 100))
            
            # Check for duplicate rows
            duplicate_ratio = df.duplicated().sum() / len(df)
            duplicate_score = max(0, 100 - (duplicate_ratio * 100))
            
            # Check for outliers (using IQR method)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_ratio = 0
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_ratio += outliers / len(df)
                outlier_ratio = outlier_ratio / len(numeric_cols)
            
            outlier_score = max(0, 100 - (outlier_ratio * 100))
            
            # Average the scores
            overall_score = (missing_score + duplicate_score + outlier_score) / 3
            return round(overall_score, 2)
            
        except Exception:
            return 75.0  # Default moderate score
    
    def _generate_training_recommendations(self, metrics: Dict[str, Any], 
                                         algorithm_config: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on training metrics and algorithm"""
        recommendations = []
        algorithm_name = algorithm_config.get("name", "unknown")
        algorithm_type = algorithm_config.get("type", "regression")
        
        # Performance-based recommendations
        if algorithm_type == "regression":
            mae = metrics.get("mae", float('inf'))
            rmse = metrics.get("rmse", float('inf'))
            r2 = metrics.get("r2", 0)
            
            if r2 < 0.6:
                recommendations.append("Consider feature engineering or trying a different algorithm")
            if mae > 5.0:
                recommendations.append("High prediction error - check data quality and feature relevance")
                
        else:
            accuracy = metrics.get("accuracy", 0)
            if accuracy < 0.7:
                recommendations.append("Low classification accuracy - consider feature selection or different algorithm")
        
        # Algorithm-specific recommendations
        if algorithm_name == "linear_regression" and metrics.get("r2", 0) < 0.6:
            recommendations.append("Linear model may not capture complex patterns - try tree-based algorithms")
        elif algorithm_name in ["hoeffding_tree", "hoeffding_tree_classifier"]:
            recommendations.append("Tree model adapts to concept drift - good for evolving racing conditions")
        elif algorithm_name.startswith("adaptive_random_forest"):
            recommendations.append("Ensemble model provides robust predictions - excellent choice for complex telemetry")
        
        if not recommendations:
            recommendations.append("Model training completed successfully - monitor performance over time")
        
        return recommendations
    
    def _get_model_version(self, existing_model_data: Optional[str]) -> int:
        """Get the next model version number"""
        if existing_model_data:
            return 2  # Increment version for updated model
        else:
            return 1  # First version
    
    async def get_model_insights(self, 
                               model_data: str,
                               feature_importance_top_n: int = 10) -> Dict[str, Any]:
        """
        Get insights about a trained River model
        
        Args:
            model_data: Base64 encoded model data
            feature_importance_top_n: Number of top features to return
        
        Returns:
            Model insights
        """
        try:
            model, feature_names = self._deserialize_river_model(model_data)
            
            # Extract model information
            model_info = {
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "model_type": type(model).__name__,
                "supports_online_learning": True,
                "memory_efficient": True
            }
            
            # Try to extract feature importance
            feature_importance = self.algorithm_config.extract_feature_importance(
                model, feature_names, type(model).__name__.lower()
            )
            
            if feature_importance:
                # Sort by importance and take top N
                sorted_importance = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:feature_importance_top_n]
                model_info["feature_importance"] = dict(sorted_importance)
            
            return {
                "success": True,
                "insights": model_info,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract insights: {str(e)}"
            }
