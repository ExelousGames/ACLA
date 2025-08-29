"""
Telemetry data processing and analysis service
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import base64
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import SGDRegressor
import joblib
import io
from app.models.telemetry_models import TelemetryFeatures, FeatureProcessor
from app.analyzers import AdvancedRacingAnalyzer


class TelemetryService:
    """Service for telemetry data processing and analysis with AI model training capabilities"""
    
    def __init__(self):
        self.telemetry_features = TelemetryFeatures()
        # Model types supported for different prediction tasks
        self.model_types = {
            "lap_time_prediction": "regression",
            "performance_classification": "classification", 
            "sector_optimization": "regression",
            "setup_recommendation": "classification"
        }
    
    async def train_ai_model(self, 
                           telemetry_data: List[Dict[str, Any]], 
                           target_variable: str,
                           model_type: str = "lap_time_prediction",
                           existing_model_data: Optional[str] = None,
                           user_id: Optional[str] = None,
                           session_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train AI model on telemetry data with support for incremental learning
        
        Args:
            telemetry_data: List of telemetry data dictionaries
            target_variable: The variable to predict (e.g., 'lap_time', 'sector_time')
            model_type: Type of model to train
            existing_model_data: Base64 encoded existing model for incremental training
            user_id: User identifier for tracking
            session_metadata: Additional session information
        
        Returns:
            Dict containing trained model data and metrics for backend storage
        """
        try:
            # Convert telemetry data to DataFrame
            df = pd.DataFrame(telemetry_data)
            
            if df.empty:
                return {"error": "No telemetry data provided"}
            
            # Process and clean the data
            feature_processor = FeatureProcessor(df)
            processed_df = feature_processor.prepare_for_analysis()
            
            # Prepare features and target
            X, y, feature_names = self._prepare_features_and_target(processed_df, target_variable)
            
            if X is None or y is None:
                return {"error": f"Could not prepare features or target variable '{target_variable}' not found"}
            
            # Load existing model if provided (for incremental training)
            existing_model = None
            existing_scaler = None
            if existing_model_data:
                try:
                    existing_model, existing_scaler = self._deserialize_model(existing_model_data)
                except Exception as e:
                    print(f"Warning: Could not load existing model: {str(e)}")
            
            # Train or update model
            model_result = self._train_model(
                X, y, feature_names, model_type, existing_model, existing_scaler
            )
            
            # Serialize model for backend storage
            serialized_model = self._serialize_model(model_result["model"], model_result["scaler"])
            
            # Prepare response for backend
            training_result = {
                "success": True,
                "model_data": serialized_model,
                "model_type": model_type,
                "target_variable": target_variable,
                "user_id": user_id,
                "training_metrics": model_result["metrics"],
                "feature_names": feature_names,
                "feature_count": len(feature_names),
                "training_samples": len(X),
                "session_metadata": session_metadata or {},
                "model_version": self._get_model_version(existing_model_data),
                "telemetry_summary": self._get_training_summary(processed_df),
                "recommendations": self._generate_training_recommendations(model_result["metrics"])
            }
            
            return training_result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Model training failed: {str(e)}",
                "model_type": model_type,
                "user_id": user_id
            }
    
    async def predict_with_model(self, 
                               telemetry_data: Dict[str, Any],
                               model_data: str,
                               model_type: str) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Args:
            telemetry_data: Current telemetry data for prediction
            model_data: Base64 encoded model data
            model_type: Type of model
        
        Returns:
            Prediction results
        """
        try:
            # Deserialize model
            model, scaler = self._deserialize_model(model_data)
            
            # Process input data
            df = pd.DataFrame([telemetry_data])
            feature_processor = FeatureProcessor(df)
            processed_df = feature_processor.prepare_for_analysis()
            
            # Prepare features (use same feature set as training)
            X = self._prepare_prediction_features(processed_df, model_type)
            
            if X is None:
                return {"error": "Could not prepare features for prediction"}
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            
            # Get prediction confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                confidence = float(np.max(proba))
            
            return {
                "success": True,
                "prediction": float(prediction),
                "confidence": confidence,
                "model_type": model_type,
                "features_used": X.shape[1]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}"
            }
    
    def _prepare_features_and_target(self, df: pd.DataFrame, target_variable: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare features and target variable for training"""
        try:
            # Check if target variable exists
            if target_variable not in df.columns:
                # Try to derive target variable
                if target_variable == "lap_time" and "Graphics_last_time" in df.columns:
                    df["lap_time"] = df["Graphics_last_time"]
                elif target_variable == "sector_time" and "Graphics_last_sector_time" in df.columns:
                    df["sector_time"] = df["Graphics_last_sector_time"]
                else:
                    return None, None, []
            
            # Get target values
            y = df[target_variable].values
            
            # Remove rows with invalid target values
            valid_mask = ~(np.isnan(y) | np.isinf(y) | (y <= 0))
            if not np.any(valid_mask):
                return None, None, []
            
            # Get performance-critical features for training
            feature_names = self.telemetry_features.get_performance_critical_features()
            
            # Filter features that exist in the data
            available_features = [f for f in feature_names if f in df.columns]
            
            if not available_features:
                # Fall back to all numeric columns except target
                available_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                    if col != target_variable]
            
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
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                    model_type: str, existing_model=None, existing_scaler=None) -> Dict[str, Any]:
        """Train or update the AI model"""
        try:
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Prepare scaler
            if existing_scaler is not None:
                scaler = existing_scaler
                # Update scaler with new data (partial fit for incremental learning)
                if hasattr(scaler, 'partial_fit'):
                    scaler.partial_fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            # Choose and train model
            if existing_model is not None and hasattr(existing_model, 'partial_fit'):
                # Incremental learning
                model = existing_model
                model.partial_fit(X_train_scaled, y_train)
            else:
                # New model or full retrain
                if self.model_types.get(model_type) == "regression":
                    if model_type == "lap_time_prediction":
                        # Use SGD for incremental learning capability
                        model = SGDRegressor(random_state=42, max_iter=1000)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    # Classification model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Simplified for now
                
                model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                "train_mse": float(mean_squared_error(y_train, y_pred_train)),
                "test_mse": float(mean_squared_error(y_test, y_pred_test)),
                "train_r2": float(r2_score(y_train, y_pred_train)),
                "test_r2": float(r2_score(y_test, y_pred_test)),
                "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            return {
                "model": model,
                "scaler": scaler,
                "metrics": metrics,
                "feature_names": feature_names
            }
            
        except Exception as e:
            raise Exception(f"Model training failed: {str(e)}")
    
    def _serialize_model(self, model, scaler) -> str:
        """Serialize model and scaler to base64 string for storage"""
        try:
            model_data = {
                "model": model,
                "scaler": scaler
            }
            
            # Serialize to bytes
            buffer = io.BytesIO()
            joblib.dump(model_data, buffer)
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
            loaded_data = joblib.load(buffer)
            
            return loaded_data["model"], loaded_data["scaler"]
            
        except Exception as e:
            raise Exception(f"Model deserialization failed: {str(e)}")
    
    def _prepare_prediction_features(self, df: pd.DataFrame, model_type: str) -> Optional[np.ndarray]:
        """Prepare features for prediction"""
        try:
            # Get the same features used in training
            feature_names = self.telemetry_features.get_performance_critical_features()
            available_features = [f for f in feature_names if f in df.columns]
            
            if not available_features:
                return None
            
            X = df[available_features].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            return X
            
        except Exception as e:
            print(f"Error preparing prediction features: {str(e)}")
            return None
    
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
    
    def _generate_training_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on training metrics"""
        recommendations = []
        
        if metrics.get("test_r2", 0) < 0.7:
            recommendations.append("Model performance could be improved with more training data")
        
        if metrics.get("test_mse", float('inf')) > metrics.get("train_mse", 0) * 2:
            recommendations.append("Model may be overfitting - consider regularization")
        
        if metrics.get("training_samples", 0) < 100:
            recommendations.append("Small training dataset - collect more data for better performance")
        
        return recommendations
    
    async def batch_train_model(self, 
                               training_sessions: List[Dict[str, Any]],
                               target_variable: str,
                               model_type: str = "lap_time_prediction",
                               existing_model_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Train AI model on multiple telemetry sessions in batch
        
        Args:
            training_sessions: List of session data with telemetry
            target_variable: The variable to predict
            model_type: Type of model to train
            existing_model_data: Existing model for incremental training
        
        Returns:
            Trained model data for backend storage
        """
        try:
            # Combine all session data
            all_telemetry_data = []
            session_metadata = {
                "session_count": len(training_sessions),
                "session_ids": []
            }
            
            for session in training_sessions:
                if "telemetry_data" in session:
                    all_telemetry_data.extend(session["telemetry_data"])
                    if "session_id" in session:
                        session_metadata["session_ids"].append(session["session_id"])
            
            if not all_telemetry_data:
                return {"error": "No telemetry data found in training sessions"}
            
            # Train model with combined data
            return await self.train_ai_model(
                telemetry_data=all_telemetry_data,
                target_variable=target_variable,
                model_type=model_type,
                existing_model_data=existing_model_data,
                session_metadata=session_metadata
            )
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Batch training failed: {str(e)}"
            }
    
    async def evaluate_model_performance(self, 
                                       model_data: str,
                                       test_telemetry_data: List[Dict[str, Any]],
                                       target_variable: str,
                                       model_type: str) -> Dict[str, Any]:
        """
        Evaluate trained model performance on test data
        
        Args:
            model_data: Base64 encoded model data
            test_telemetry_data: Test telemetry data
            target_variable: Target variable to evaluate
            model_type: Type of model
        
        Returns:
            Evaluation metrics and performance analysis
        """
        try:
            # Deserialize model
            model, scaler = self._deserialize_model(model_data)
            
            # Prepare test data
            df = pd.DataFrame(test_telemetry_data)
            feature_processor = FeatureProcessor(df)
            processed_df = feature_processor.prepare_for_analysis()
            
            X_test, y_test, feature_names = self._prepare_features_and_target(processed_df, target_variable)
            
            if X_test is None or y_test is None:
                return {"error": "Could not prepare test data"}
            
            # Scale features
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "test_samples": len(y_test),
                "prediction_range": {
                    "min": float(np.min(y_pred)),
                    "max": float(np.max(y_pred)),
                    "mean": float(np.mean(y_pred))
                },
                "actual_range": {
                    "min": float(np.min(y_test)),
                    "max": float(np.max(y_test)),
                    "mean": float(np.mean(y_test))
                }
            }
            
            # Performance analysis
            performance_grade = self._calculate_performance_grade(metrics["r2"])
            
            return {
                "success": True,
                "metrics": metrics,
                "performance_grade": performance_grade,
                "model_type": model_type,
                "recommendations": self._generate_evaluation_recommendations(metrics)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Model evaluation failed: {str(e)}"
            }
    
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
                               feature_importance_top_n: int = 10) -> Dict[str, Any]:
        """
        Get insights about a trained model including feature importance
        
        Args:
            model_data: Base64 encoded model data
            feature_importance_top_n: Number of top features to return
        
        Returns:
            Model insights and feature importance
        """
        try:
            model, scaler = self._deserialize_model(model_data)
            
            insights = {
                "model_type": type(model).__name__,
                "feature_count": len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else "unknown",
                "model_parameters": self._get_model_parameters(model)
            }
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_names = self.telemetry_features.get_performance_critical_features()
                if len(feature_names) == len(model.feature_importances_):
                    importance_pairs = list(zip(feature_names, model.feature_importances_))
                    importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    insights["feature_importance"] = {
                        "top_features": [
                            {"feature": name, "importance": float(importance)} 
                            for name, importance in importance_pairs[:feature_importance_top_n]
                        ],
                        "total_features": len(importance_pairs)
                    }
            
            return {
                "success": True,
                "insights": insights
            }
            
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
    async def analyze_racing_performance(self, 
                                    session_id: str, 
                                    analysis_type: str = "overall", 
                                    focus_areas: List[str] = None,
                                    telemetry_data: Optional[List[Dict[str, Any]]] = None,
                                    use_ai_model: bool = False,
                                    model_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze racing performance from telemetry data with optional AI predictions
        
        Args:
            session_id: Session identifier
            analysis_type: Type of analysis to perform
            focus_areas: Specific areas to focus on
            telemetry_data: Raw telemetry data for analysis
            use_ai_model: Whether to use AI model for predictions
            model_data: Trained model data for predictions
        """
        try:
            base_analysis = {}
            
            # Perform traditional analysis
            if analysis_type == "overall":
                base_analysis = {
                    "session_id": session_id,
                    "analysis_type": analysis_type,
                    "performance_score": {
                        "overall_score": 78.5,
                        "speed_consistency": 85.2,
                        "cornering_performance": 72.1,
                        "braking_efficiency": 80.3,
                        "grade": "B"
                    },
                    "lap_analysis": {
                        "best_lap": 89.234,
                        "average_lap": 91.567,
                        "consistency": 0.92
                    },
                    "focus_areas": focus_areas or []
                }
            elif analysis_type == "sectors":
                base_analysis = {
                    "session_id": session_id,
                    "sector_analysis": {
                        "sector_1": {"time": 28.5, "speed": 145.2, "rating": "good"},
                        "sector_2": {"time": 31.2, "speed": 132.8, "rating": "average"},
                        "sector_3": {"time": 29.8, "speed": 152.1, "rating": "excellent"}
                    },
                    "improvement_potential": ["sector_2_cornering", "braking_points"]
                }
            elif analysis_type == "consistency":
                base_analysis = {
                    "session_id": session_id,
                    "consistency_metrics": {
                        "lap_time_variance": 1.23,
                        "sector_consistency": 0.89,
                        "speed_consistency": 0.94
                    },
                    "patterns": ["improving_throughout_session", "consistent_sector_1"]
                }
            
            # Add AI predictions if requested and available
            if use_ai_model and model_data and telemetry_data:
                try:
                    # Use AI model to predict performance improvements
                    ai_predictions = await self._generate_ai_performance_insights(
                        telemetry_data, model_data, session_id
                    )
                    base_analysis["ai_insights"] = ai_predictions
                except Exception as e:
                    base_analysis["ai_insights"] = {"error": f"AI analysis failed: {str(e)}"}
            
            # Add telemetry-based analysis if data provided
            if telemetry_data:
                try:
                    telemetry_analysis = self._analyze_telemetry_performance(telemetry_data)
                    base_analysis["telemetry_analysis"] = telemetry_analysis
                except Exception as e:
                    base_analysis["telemetry_analysis"] = {"error": f"Telemetry analysis failed: {str(e)}"}
            
            return base_analysis
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
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
    
    def _analyze_telemetry_performance(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance directly from telemetry data"""
        try:
            df = pd.DataFrame(telemetry_data)
            feature_processor = FeatureProcessor(df)
            
            # Get performance metrics
            performance_metrics = feature_processor.extract_performance_metrics()
            
            # Get feature validation
            feature_validation = feature_processor.validate_features()
            
            return {
                "performance_metrics": performance_metrics,
                "data_quality": {
                    "feature_coverage": feature_validation.get("coverage_percentage", 0),
                    "total_records": len(df),
                    "available_features": feature_validation.get("total_present", 0)
                },
                "telemetry_summary": self._get_training_summary(df)
            }
            
        except Exception as e:
            return {"error": f"Telemetry performance analysis failed: {str(e)}"}

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
    
    async def get_improvement_suggestions(self, 
                                     session_id: str, 
                                     skill_level: str = "intermediate", 
                                     focus_area: str = None,
                                     telemetry_data: Optional[List[Dict[str, Any]]] = None,
                                     model_data: Optional[str] = None) -> Dict[str, Any]:
        """Generate personalized improvement suggestions with optional AI predictions"""
        try:
            # Get base performance analysis
            performance_analysis = await self.analyze_racing_performance(
                session_id=session_id, 
                analysis_type="overall", 
                focus_areas=[],
                telemetry_data=telemetry_data,
                use_ai_model=bool(model_data),
                model_data=model_data
            )
            
            # Generate traditional suggestions based on skill level
            base_suggestions = []
            if skill_level == "beginner":
                base_suggestions = [
                    "Focus on learning the racing line - consistency before speed",
                    "Practice smooth braking and acceleration",
                    "Work on track familiarization in practice sessions",
                    "Start with lower difficulty AI opponents"
                ]
            elif skill_level == "intermediate":
                base_suggestions = [
                    "Fine-tune braking points for optimal cornering speed",
                    "Work on trail braking technique in slow corners",
                    "Analyze sector times to identify weak areas",
                    "Practice racecraft and overtaking scenarios"
                ]
            elif skill_level == "advanced":
                base_suggestions = [
                    "Optimize setup for track-specific conditions",
                    "Master advanced techniques like left-foot braking",
                    "Analyze telemetry data for micro-optimizations",
                    "Focus on race strategy and tyre management"
                ]
            
            # Add focus area specific suggestions
            specific_focus_suggestions = []
            if focus_area:
                if focus_area == "braking":
                    specific_focus_suggestions = [
                        "Practice threshold braking technique",
                        "Work on brake point consistency",
                        "Analyze brake temperature management"
                    ]
                elif focus_area == "cornering":
                    specific_focus_suggestions = [
                        "Perfect your racing line through complex corners",
                        "Practice different entry speeds",
                        "Work on smooth steering inputs"
                    ]
            
            # Prepare response with AI integration capabilities
            suggestions_response = {
                "session_id": session_id,
                "skill_level": skill_level,
                "focus_area": focus_area,
                "traditional_suggestions": base_suggestions,
                "specific_focus_suggestions": specific_focus_suggestions,
                "performance_analysis": performance_analysis,
                "ai_enhanced": bool(model_data),
                "telemetry_based": bool(telemetry_data)
            }
            
            # Add AI insights if available
            if model_data and telemetry_data:
                ai_insights = performance_analysis.get("ai_insights", {})
                if "improvement_areas" in ai_insights:
                    suggestions_response["ai_suggestions"] = ai_insights["improvement_areas"]
            
            return suggestions_response
            
        except Exception as e:
            return {"error": f"Improvement suggestions failed: {str(e)}"}
    
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
