"""
Telemetry data processing and analysis service
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import base64
from datetime import datetime, timezone
# Using River for online learning instead of scikit-learn
from river import preprocessing, metrics
import io
from app.models.telemetry_models import TelemetryFeatures, FeatureProcessor
from app.models.ml_algorithms import AlgorithmConfiguration
from app.services.river_ml_service import RiverMLService
from app.analyzers import AdvancedRacingAnalyzer


# Utility functions to replace sklearn functionality
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Simple train/test split replacement"""
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]
    
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def mean_squared_error(y_true, y_pred):
    """Calculate MSE"""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Calculate MAE"""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Calculate R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def accuracy_score(y_true, y_pred):
    """Calculate accuracy"""
    return np.mean(y_true == y_pred)


class SimpleStandardScaler:
    """Simple StandardScaler replacement"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1  # Avoid division by zero
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class TelemetryService:
    """Service for telemetry data processing and analysis with River-based online learning capabilities"""
    
    def __init__(self):
        self.telemetry_features = TelemetryFeatures()
        self.algorithm_config = AlgorithmConfiguration()
        self.river_ml_service = RiverMLService()  # New River-based ML service
        
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
    
    async def train_ai_model(self, 
                           telemetry_data: List[Dict[str, Any]], 
                           target_variable: str,
                           model_type: str = "lap_time_prediction",
                           preferred_algorithm: Optional[str] = None,
                           existing_model_data: Optional[str] = None,
                           user_id: Optional[str] = None,) -> Dict[str, Any]:
        """
        Train AI model on telemetry data with support for online learning using River
        
        Args:
            telemetry_data: List of telemetry data dictionaries
            target_variable: The variable to predict (e.g., 'lap_time', 'sector_time')
            model_type: Type of model to train
            preferred_algorithm: Override the default algorithm for this task
            existing_model_data: Base64 encoded existing model for incremental training
            user_id: User identifier for tracking
            use_river: Whether to use River for online learning (default: True)
         
        Returns:
            Dict containing trained model data and metrics for backend storage
        """
        try:

            # Use the new River-based online learning service
            return await self.river_ml_service.train_online_model(
                telemetry_data=telemetry_data,
                target_variable=target_variable,
                model_type=model_type,
                preferred_algorithm=preferred_algorithm,
                existing_model_data=existing_model_data,
                user_id=user_id
            )
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}",
                "model_type": model_type,
                "algorithm_used": preferred_algorithm or "unknown"
            }
    
    async def predict_with_model(self, 
                               telemetry_data: Dict[str, Any],
                               model_data: str,
                               model_type: str,
                               use_river: bool = True) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Args:
            telemetry_data: Current telemetry data for prediction
            model_data: Base64 encoded model data
            model_type: Type of model
            use_river: Whether to use River model (default: True)
        
        Returns:
            Prediction results
        """
        try:
            if use_river:
                # Use River-based prediction
                return await self.river_ml_service.predict_online(
                    telemetry_data=telemetry_data,
                    model_data=model_data,
                    model_type=model_type
                )
            else:
                # Legacy scikit-learn prediction (deprecated)
                return await self._predict_legacy_model(
                    telemetry_data=telemetry_data,
                    model_data=model_data,
                    model_type=model_type
                )
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "model_type": model_type
            }
    
    def _prepare_features_and_target(self, df: pd.DataFrame, target_variable: str, model_type: str = "lap_time_prediction") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare features and target variable for training
        Args:
            df: DataFrame with telemetry data
            target_variable: Target variable name
            model_type: Type of prediction task for feature selection
        Returns: 
            X: valid feature matrix, y: valid target vector, feature_names: list of feature names
        """
        
        try:
            # Check if target variable exists
            if target_variable not in df.columns:
                return None, None, []
            
            # Get target values
            y = df[target_variable].values

            # mask identifying invalid target values: NaN and inf, less than or equal to 0
            valid_mask = ~(np.isnan(y) | np.isinf(y) | (y <= 0))
            if not np.any(valid_mask):
                return None, None, []
            
            # Get performance-critical features based on task using centralized feature selection
            feature_names = self.telemetry_features.get_features_for_model_type(model_type)
            
            # Filter and get features that exist in the data
            available_features = self.telemetry_features.filter_available_features(feature_names, df.columns.tolist())
            
            if not available_features:
                return None, None, []
            
            # Prepare feature matrix
            X = df[available_features].values
            
            # Apply valid mask to both X and y
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Handle NaN and inf values in features
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            return X, y, available_features
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return None, None, []
    
    def _serialize_model(self, model, scaler, algorithm_name: str = "unknown") -> str:
        """Serialize model and scaler to base64 string for storage"""
        try:
            model_data = {
                "model": model,
                "scaler": scaler,
                "algorithm_name": algorithm_name,
                "serialization_version": "2.0"
            }
            
            # Serialize to bytes
            buffer = io.BytesIO()
            pickle.dump(model_data, buffer)
            buffer.seek(0)
            
            # Encode to base64 string
            model_bytes = buffer.getvalue()
            encoded_model = base64.b64encode(model_bytes).decode('utf-8')
            
            return encoded_model
            
        except Exception as e:
            raise Exception(f"Model serialization failed: {str(e)}")
    
    def _deserialize_model(self, model_data: str) -> Tuple[Any, Any]:
        """Deserialize model and scaler from base64 string"""
        try:
            # Decode from base64
            model_bytes = base64.b64decode(model_data.encode('utf-8'))
            
            # Deserialize from bytes
            buffer = io.BytesIO(model_bytes)
            loaded_data = pickle.load(buffer)
            
            # Handle both old and new serialization formats
            if isinstance(loaded_data, dict):
                model = loaded_data.get("model")
                scaler = loaded_data.get("scaler")
                algorithm_name = loaded_data.get("algorithm_name", "unknown")
                return model, scaler
            else:
                # Old format compatibility
                return loaded_data, None
            
        except Exception as e:
            raise Exception(f"Model deserialization failed: {str(e)}")
    
    def _get_model_version(self, existing_model_data: Optional[str]) -> int:
        """Get the next model version number"""
        if existing_model_data:
            return 2  # Incremental update
        else:
            return 1  # New model
    
    def _get_training_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of training data"""
        return {
            "total_records": len(df),
            "features_available": len(df.columns),
            "data_quality_score": self._calculate_data_quality_score(df),
            "session_duration": self._estimate_session_duration(df)
        }
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-100)"""
        try:
            # Calculate missing data percentage
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            
            # Calculate score based on completeness and size
            completeness_score = max(0, 100 - missing_percentage)
            size_score = min(100, len(df) / 10)  # Assume 1000 records = 100 score
            
            return float((completeness_score + size_score) / 2)
            
        except:
            return 50.0  # Default score
    
    def _estimate_session_duration(self, df: pd.DataFrame) -> Optional[float]:
        """Estimate session duration in minutes"""
        try:
            if 'timestamp' in df.columns:
                duration = (df['timestamp'].max() - df['timestamp'].min()) / 60
                return float(duration)
            return None
        except:
            return None
    
    def _generate_training_recommendations(self, metrics: Dict[str, Any], algorithm_config: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on training metrics and algorithm"""
        recommendations = []
        algorithm_name = algorithm_config.get("name", "unknown")
        algorithm_type = algorithm_config.get("type", "regression")
        
        # Performance-based recommendations
        if algorithm_type == "regression":
            test_r2 = metrics.get("test_r2", 0)
            if test_r2 < 0.7:
                recommendations.append("Model performance could be improved with more training data")
            if test_r2 < 0.5:
                recommendations.append(f"Poor performance with {algorithm_name} - consider trying alternative algorithms")
        else:
            test_accuracy = metrics.get("test_accuracy", 0)
            if test_accuracy < 0.8:
                recommendations.append("Classification accuracy could be improved with more diverse training data")
        
        # Overfitting detection
        if metrics.get("test_mse", float('inf')) > metrics.get("train_mse", 0) * 2:
            if algorithm_name in ["random_forest", "gradient_boosting"]:
                recommendations.append("Model may be overfitting - consider reducing n_estimators or max_depth")
            elif algorithm_name == "neural_network":
                recommendations.append("Model may be overfitting - consider adding dropout or reducing hidden layer size")
            else:
                recommendations.append("Model may be overfitting - consider regularization")
        
        # Data size recommendations
        training_samples = metrics.get("training_samples", 0)
        if training_samples < 100:
            recommendations.append("Small training dataset - collect more data for better performance")
        elif training_samples < 50 and algorithm_name in ["neural_network", "svr"]:
            recommendations.append(f"{algorithm_name} typically requires more data - consider using simpler algorithms")
        
        # Algorithm-specific recommendations
        if algorithm_name == "linear_regression" and metrics.get("test_r2", 0) < 0.6:
            recommendations.append("Linear model may be too simple - consider non-linear algorithms like Random Forest")
        elif algorithm_name == "neural_network" and training_samples < 500:
            recommendations.append("Neural networks work best with larger datasets - consider ensemble methods")
        elif algorithm_name == "knn" and training_samples > 10000:
            recommendations.append("KNN can be slow with large datasets - consider faster algorithms")
        
        return recommendations
    
    def _calculate_performance_grade(self, r2_score: float) -> str:
        """Calculate performance grade based on R2 score"""
        if r2_score >= 0.9:
            return "A"
        elif r2_score >= 0.8:
            return "B"
        elif r2_score >= 0.7:
            return "C"
        elif r2_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_evaluation_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation metrics"""
        recommendations = []
        
        r2 = metrics.get("r2", 0)
        if r2 < 0.7:
            recommendations.append("Consider collecting more diverse training data")
        if r2 < 0.5:
            recommendations.append("Model performance is poor - may need feature engineering")
        
        mae = metrics.get("mae", 0)
        if mae > 5.0:  # Assuming time-based predictions
            recommendations.append("High prediction error - review feature selection")
        
        return recommendations

    async def get_model_insights(self, 
                               model_data: str,
                               feature_importance_top_n: int = 10,
                               use_river: bool = True) -> Dict[str, Any]:
        """
        Get insights about a trained model including feature importance
        
        Args:
            model_data: Base64 encoded model data
            feature_importance_top_n: Number of top features to return
            use_river: Whether to use River model (default: True)
        
        Returns:
            Model insights and feature importance
        """
        try:
            if use_river:
                # Use River-based insights
                return await self.river_ml_service.get_model_insights(
                    model_data=model_data,
                    feature_importance_top_n=feature_importance_top_n
                )
            else:
                # Legacy scikit-learn insights (deprecated)
                return await self._get_legacy_model_insights(
                    model_data=model_data,
                    feature_importance_top_n=feature_importance_top_n
                )
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Model insights failed: {str(e)}"
            }
    
    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Extract key model parameters"""
        try:
            params = {}
            if hasattr(model, 'n_estimators'):
                params['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                params['max_depth'] = model.max_depth
            if hasattr(model, 'learning_rate'):
                params['learning_rate'] = model.learning_rate
            if hasattr(model, 'alpha'):
                params['alpha'] = model.alpha
            return params
        except:
            return {}

    async def _generate_ai_performance_insights(self, 
                                              telemetry_data: List[Dict[str, Any]], 
                                              model_data: str, 
                                              session_id: str) -> Dict[str, Any]:
        """Generate AI-powered performance insights"""
        try:
            insights = {
                "session_id": session_id,
                "predictions": [],
                "improvement_areas": [],
                "confidence_scores": {}
            }
            
            # Sample a few data points for prediction
            sample_size = min(10, len(telemetry_data))
            sample_data = telemetry_data[:sample_size]
            
            for i, data_point in enumerate(sample_data):
                prediction_result = await self.predict_with_model(
                    telemetry_data=data_point,
                    model_data=model_data,
                    model_type="lap_time_prediction"
                )
                
                if prediction_result.get("success"):
                    insights["predictions"].append({
                        "data_point": i,
                        "predicted_lap_time": prediction_result["prediction"],
                        "confidence": prediction_result.get("confidence")
                    })
            
            # Generate improvement suggestions based on predictions
            if insights["predictions"]:
                avg_predicted_time = np.mean([p["predicted_lap_time"] for p in insights["predictions"]])
                insights["improvement_areas"] = [
                    f"Predicted average lap time: {avg_predicted_time:.3f}s",
                    "Focus on consistency in predicted weak sectors",
                    "AI suggests optimization in cornering phases"
                ]
            
            return insights
            
        except Exception as e:
            return {"error": f"AI insights generation failed: {str(e)}"}
    
    async def get_telemetry_insights(self, session_id: str, data_types: List[str] = None) -> Dict[str, Any]:
        """Get detailed telemetry insights"""
        try:
            data_types = data_types or ["speed", "acceleration"]
            insights = {
                "session_id": session_id,
                "insights": {}
            }
            
            for data_type in data_types:
                if data_type == "speed":
                    insights["insights"]["speed"] = {
                        "max_speed": 287.5,
                        "avg_speed": 145.2,
                        "speed_zones": {
                            "high_speed": 0.35,
                            "medium_speed": 0.45,
                            "low_speed": 0.20
                        }
                    }
                elif data_type == "acceleration":
                    insights["insights"]["acceleration"] = {
                        "max_acceleration": 2.8,
                        "max_deceleration": -4.2,
                        "avg_acceleration": 0.45
                    }
                elif data_type == "braking":
                    insights["insights"]["braking"] = {
                        "brake_pressure_avg": 0.78,
                        "brake_temperature": 650,
                        "braking_zones": 12
                    }
                elif data_type == "steering":
                    insights["insights"]["steering"] = {
                        "max_steering_angle": 45.2,
                        "steering_smoothness": 0.88,
                        "corrections_per_lap": 3.2
                    }
            
            return insights
            
        except Exception as e:
            return {"error": f"Telemetry insights failed: {str(e)}"}
    
    async def compare_sessions(self, session_ids: List[str], comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """Compare multiple racing sessions"""
        try:
            comparison_metrics = comparison_metrics or ["lap_times"]
            
            comparison = {
                "sessions": session_ids,
                "comparison": {}
            }
            
            for metric in comparison_metrics:
                if metric == "lap_times":
                    comparison["comparison"]["lap_times"] = {
                        session_ids[0]: {"best": 89.234, "average": 91.567},
                        session_ids[1] if len(session_ids) > 1 else "session_2": {"best": 88.945, "average": 90.823}
                    }
                elif metric == "sectors":
                    comparison["comparison"]["sectors"] = {
                        session_ids[0]: {"sector_1": 28.5, "sector_2": 31.2, "sector_3": 29.8},
                        session_ids[1] if len(session_ids) > 1 else "session_2": {"sector_1": 28.2, "sector_2": 30.9, "sector_3": 30.1}
                    }
                elif metric == "consistency":
                    comparison["comparison"]["consistency"] = {
                        session_ids[0]: 0.92,
                        session_ids[1] if len(session_ids) > 1 else "session_2": 0.95
                    }
            
            # Generate insights
            comparison["insights"] = [
                "Session 2 shows better lap times overall",
                "Sector 2 improvement opportunities in both sessions",
                "Consistency improved in latest session"
            ]
            
            return comparison
            
        except Exception as e:
            return {"error": f"Session comparison failed: {str(e)}"}
    
    
    def validate_telemetry_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate telemetry data quality and completeness"""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(data)
            feature_processor = FeatureProcessor(df)
            
            return {
                "validation_results": feature_processor.validate_features(),
                "data_quality": self._assess_data_quality(df),
                "recommendations": self._generate_recommendations(df)
            }
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def analyze_telemetry_session(self, data: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze telemetry data for a racing session"""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(data)
            analyzer = AdvancedRacingAnalyzer(df)
            
            if analysis_type == "comprehensive":
                return analyzer.get_telemetry_summary()
            elif analysis_type == "performance":
                return analyzer.feature_processor.extract_performance_metrics()
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_available_features(self) -> Dict[str, List[str]]:
        """Get all available telemetry features by category"""
        return {
            "physics": self.telemetry_features.PHYSICS_FEATURES,
            "graphics": self.telemetry_features.GRAPHICS_FEATURES,
            "static": self.telemetry_features.STATIC_FEATURES
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of telemetry data"""
        return {
            "total_records": len(df),
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations for data improvement"""
        recommendations = []
        
        # Check for missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 5:
            recommendations.append(f"High missing data percentage ({missing_percentage:.1f}%) - consider data cleaning")
        
        # Check data size
        if len(df) < 100:
            recommendations.append("Small dataset size - consider collecting more data for better analysis")
        
        return recommendations
