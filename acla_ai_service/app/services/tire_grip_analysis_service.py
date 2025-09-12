"""
Tire Grip and Friction Circle Analysis Service for Assetto Corsa Competizione

This service provides comprehensive tire grip analysis and friction circle utilization
estimation using machine learning models. It extracts features related to:

- Tire grip estimation based on physics telemetry
- Friction circle utilization (how close the car is to the limit)  
- Weight transfer analysis
- Predictive load calculations
- Tire performance degradation
- Optimal grip windows

The extracted features are designed to be inserted back into telemetry data for enhanced AI analysis.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import math
import asyncio
import pickle
import base64
import io
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import zscore

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Import backend service and models
from .backend_service import backend_service
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor, _safe_float

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


class TireGripFeatures:
    """Data class for tire grip analysis features"""
    
    def __init__(self):
        # Friction circle utilization (0-1 scale)
        self.friction_circle_utilization = 0.0
        self.friction_circle_utilization_front = 0.0
        self.friction_circle_utilization_rear = 0.0
        
        # Tire grip estimation (0-1 scale where 1 is optimal grip)
        self.estimated_tire_grip_fl = 0.0
        self.estimated_tire_grip_fr = 0.0
        self.estimated_tire_grip_rl = 0.0
        self.estimated_tire_grip_rr = 0.0
        self.overall_tire_grip = 0.0
        
        # Weight transfer analysis
        self.longitudinal_weight_transfer = 0.0  # -1 to 1 (rear to front)
        self.lateral_weight_transfer = 0.0       # -1 to 1 (right to left)
        self.dynamic_weight_distribution = 0.0   # How much weight is shifting
        
        # Predictive load on each tire
        self.predictive_load_fl = 0.0
        self.predictive_load_fr = 0.0
        self.predictive_load_rl = 0.0
        self.predictive_load_rr = 0.0
        
        # Tire performance metrics
        self.tire_saturation_level = 0.0     # How close tires are to saturation (0-1)
        self.optimal_grip_window = 0.0       # Whether in optimal temperature/pressure window (0-1)
        self.tire_degradation_factor = 0.0   # Estimated tire degradation (0-1)
        
        # Advanced grip metrics
        self.slip_angle_efficiency = 0.0     # How efficiently slip angles are being used
        self.slip_ratio_efficiency = 0.0     # How efficiently slip ratios are being used
        self.cornering_grip_utilization = 0.0 # Specific to cornering forces
        self.braking_grip_utilization = 0.0   # Specific to braking forces
        self.acceleration_grip_utilization = 0.0  # Specific to acceleration forces


class TireGripAnalysisService:
    """
    Machine Learning Service for Tire Grip and Friction Circle Analysis
    
    This service analyzes telemetry data to estimate:
    - Tire grip levels and utilization
    - Friction circle utilization
    - Weight transfer dynamics
    - Predictive tire loading
    - Performance optimization opportunities
    """
    
    def __init__(self, models_directory: str = "models/tire_grip"):
        """
        Initialize the tire grip analysis service
        
        Args:
            models_directory: Directory to save/load trained models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        self.telemetry_features = TelemetryFeatures()
        self.trained_models = {}
        self.scalers = {}
        self.model_metadata = {}  # Store model training metadata
        
        # Backend service integration
        self.backend_service = backend_service
        
        # Initialize models cache
        self._models_cache = {}
        
        print(f"[INFO] TireGripAnalysisService initialized with models directory: {self.models_directory}")

    def _calculate_friction_circle_utilization(self, row: pd.Series) -> float:
        """
        Calculate friction circle utilization based on G-forces
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Friction circle utilization (0-1)
        """
        try:
            # Get G-forces (assuming peak performance around 2-3G)
            g_x = _safe_float(row.get('Physics_g_force_x', 0))
            g_y = _safe_float(row.get('Physics_g_force_y', 0))
            
            # Calculate total G-force magnitude
            total_g = math.sqrt(g_x**2 + g_y**2)
            
            # Normalize to typical racing car limits (assume ~2.5G peak)
            max_g = 2.5
            utilization = min(total_g / max_g, 1.0)
            
            return utilization
            
        except Exception:
            return 0.0

    def _calculate_weight_transfer(self, row: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate weight transfer based on G-forces and suspension travel
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Tuple of (longitudinal_transfer, lateral_transfer, dynamic_distribution)
        """
        try:
            # Longitudinal weight transfer (braking/acceleration)
            g_x = _safe_float(row.get('Physics_g_force_x', 0))
            long_transfer = np.tanh(g_x / 1.5)  # Normalize and bound to [-1, 1]
            
            # Lateral weight transfer (cornering)
            g_y = _safe_float(row.get('Physics_g_force_y', 0))
            lat_transfer = np.tanh(g_y / 1.5)  # Normalize and bound to [-1, 1]
            
            # Dynamic weight distribution (how much weight is shifting)
            total_transfer = math.sqrt(g_x**2 + g_y**2)
            dynamic_dist = min(total_transfer / 2.0, 1.0)
            
            return long_transfer, lat_transfer, dynamic_dist
            
        except Exception:
            return 0.0, 0.0, 0.0

    def _estimate_tire_temperatures_optimal_window(self, row: pd.Series) -> float:
        """
        Estimate if tires are in optimal temperature window
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Optimal window factor (0-1)
        """
        try:
            # Get tire core temperatures
            temp_fl = _safe_float(row.get('Physics_tyre_core_temp_front_left', 70))
            temp_fr = _safe_float(row.get('Physics_tyre_core_temp_front_right', 70))
            temp_rl = _safe_float(row.get('Physics_tyre_core_temp_rear_left', 70))
            temp_rr = _safe_float(row.get('Physics_tyre_core_temp_rear_right', 70))
            
            avg_temp = (temp_fl + temp_fr + temp_rl + temp_rr) / 4
            
            # Optimal temperature window (typically 80-110°C for racing tires)
            optimal_min = 80
            optimal_max = 110
            
            if optimal_min <= avg_temp <= optimal_max:
                # In optimal window
                window_factor = 1.0
            elif avg_temp < optimal_min:
                # Too cold
                window_factor = max(0, avg_temp / optimal_min)
            else:
                # Too hot
                excess_heat = avg_temp - optimal_max
                window_factor = max(0, 1 - (excess_heat / 50))  # Degrade over 50°C excess
            
            return window_factor
            
        except Exception:
            return 0.7  # Default moderate value

    def _calculate_slip_efficiency(self, row: pd.Series) -> Tuple[float, float]:
        """
        Calculate slip angle and slip ratio efficiency
        
        Args:
            row: Single telemetry data row
            
        Returns:
            Tuple of (slip_angle_efficiency, slip_ratio_efficiency)
        """
        try:
            # Get slip angles (front axle average)
            slip_angle_fl = _safe_float(row.get('Physics_slip_angle_front_left', 0))
            slip_angle_fr = _safe_float(row.get('Physics_slip_angle_front_right', 0))
            avg_slip_angle = abs((slip_angle_fl + slip_angle_fr) / 2)
            
            # Get slip ratios (all wheels average)
            slip_ratio_fl = _safe_float(row.get('Physics_slip_ratio_front_left', 0))
            slip_ratio_fr = _safe_float(row.get('Physics_slip_ratio_front_right', 0))
            slip_ratio_rl = _safe_float(row.get('Physics_slip_ratio_rear_left', 0))
            slip_ratio_rr = _safe_float(row.get('Physics_slip_ratio_rear_right', 0))
            avg_slip_ratio = abs((slip_ratio_fl + slip_ratio_fr + slip_ratio_rl + slip_ratio_rr) / 4)
            
            # Calculate efficiency (optimal slip angles around 5-8 degrees, slip ratios around 10-15%)
            optimal_slip_angle = 6.5  # degrees
            optimal_slip_ratio = 0.125  # 12.5%
            
            # Efficiency curves (inverted bell curves centered on optimal values)
            slip_angle_eff = max(0, 1 - abs(avg_slip_angle - optimal_slip_angle) / optimal_slip_angle)
            slip_ratio_eff = max(0, 1 - abs(avg_slip_ratio - optimal_slip_ratio) / optimal_slip_ratio)
            
            return slip_angle_eff, slip_ratio_eff
            
        except Exception:
            return 0.5, 0.5  # Default moderate values

    def _prepare_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model training
        
        Args:
            df: Processed telemetry DataFrame
            
        Returns:
            DataFrame with training features
        """
        print(f"[INFO] Preparing training features from {len(df)} telemetry records")
        
        # Create feature DataFrame
        features_df = pd.DataFrame()
        
        # Basic physics features
        physics_features = [
            'Physics_speed_kmh', 'Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z',
            'Physics_brake', 'Physics_gas', 'Physics_steer_angle',
            'Physics_slip_angle_front_left', 'Physics_slip_angle_front_right',
            'Physics_slip_angle_rear_left', 'Physics_slip_angle_rear_right',
            'Physics_slip_ratio_front_left', 'Physics_slip_ratio_front_right',
            'Physics_slip_ratio_rear_left', 'Physics_slip_ratio_rear_right',
            'Physics_wheel_slip_front_left', 'Physics_wheel_slip_front_right',
            'Physics_wheel_slip_rear_left', 'Physics_wheel_slip_rear_right',
            'Physics_suspension_travel_front_left', 'Physics_suspension_travel_front_right',
            'Physics_suspension_travel_rear_left', 'Physics_suspension_travel_rear_right',
            'Physics_tyre_core_temp_front_left', 'Physics_tyre_core_temp_front_right',
            'Physics_tyre_core_temp_rear_left', 'Physics_tyre_core_temp_rear_right',
            'Physics_brake_temp_front_left', 'Physics_brake_temp_front_right',
            'Physics_brake_temp_rear_left', 'Physics_brake_temp_rear_right',
            'Physics_wheel_pressure_front_left', 'Physics_wheel_pressure_front_right',
            'Physics_wheel_pressure_rear_left', 'Physics_wheel_pressure_rear_right'
        ]
        
        # Add available physics features
        for feature in physics_features:
            if feature in df.columns:
                features_df[feature] = df[feature].apply(_safe_float)
        
        # Calculate derived features
        features_df['total_g_force'] = np.sqrt(
            features_df.get('Physics_g_force_x', 0)**2 + 
            features_df.get('Physics_g_force_y', 0)**2
        )
        
        features_df['avg_tire_temp'] = (
            features_df.get('Physics_tyre_core_temp_front_left', 70) +
            features_df.get('Physics_tyre_core_temp_front_right', 70) +
            features_df.get('Physics_tyre_core_temp_rear_left', 70) +
            features_df.get('Physics_tyre_core_temp_rear_right', 70)
        ) / 4
        
        features_df['avg_slip_angle'] = (
            abs(features_df.get('Physics_slip_angle_front_left', 0)) +
            abs(features_df.get('Physics_slip_angle_front_right', 0)) +
            abs(features_df.get('Physics_slip_angle_rear_left', 0)) +
            abs(features_df.get('Physics_slip_angle_rear_right', 0))
        ) / 4
        
        features_df['avg_slip_ratio'] = (
            abs(features_df.get('Physics_slip_ratio_front_left', 0)) +
            abs(features_df.get('Physics_slip_ratio_front_right', 0)) +
            abs(features_df.get('Physics_slip_ratio_rear_left', 0)) +
            abs(features_df.get('Physics_slip_ratio_rear_right', 0))
        ) / 4
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(0)
        
        print(f"[INFO] Prepared {len(features_df.columns)} training features")
        return features_df

    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for supervised learning
        
        Args:
            df: Processed telemetry DataFrame
            
        Returns:
            DataFrame with target variables
        """
        targets_df = pd.DataFrame()
        
        # Calculate friction circle utilization as primary target
        for i, row in df.iterrows():
            friction_util = self._calculate_friction_circle_utilization(row)
            targets_df.loc[i, 'friction_circle_utilization'] = friction_util
            
            # Weight transfer targets
            long_transfer, lat_transfer, dynamic_dist = self._calculate_weight_transfer(row)
            targets_df.loc[i, 'longitudinal_weight_transfer'] = long_transfer
            targets_df.loc[i, 'lateral_weight_transfer'] = lat_transfer
            targets_df.loc[i, 'dynamic_weight_distribution'] = dynamic_dist
            
            # Tire performance targets
            targets_df.loc[i, 'optimal_grip_window'] = self._estimate_tire_temperatures_optimal_window(row)
            
            slip_angle_eff, slip_ratio_eff = self._calculate_slip_efficiency(row)
            targets_df.loc[i, 'slip_angle_efficiency'] = slip_angle_eff
            targets_df.loc[i, 'slip_ratio_efficiency'] = slip_ratio_eff
        
        return targets_df

    async def train_tire_grip_model(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train tire grip analysis models using provided cleaned telemetry data
        
        Args:
            telemetry_data: Pre-cleaned telemetry data for training
            
        Returns:
            Training results and model performance metrics
        """
        print(f"[INFO] Training tire grip analysis model with provided telemetry data")
        
        if not telemetry_data:
            return {"error": "No telemetry data provided"}
        
        print(f"[INFO] Using provided telemetry data: {len(telemetry_data)} records")
        flattened_telemetry_data = telemetry_data
        
        if len(flattened_telemetry_data) < 100:
            return {"error": "Insufficient telemetry data for training (minimum 100 samples required)"}

        # Convert to DataFrame - data is already cleaned
        df = pd.DataFrame(flattened_telemetry_data)
        print(f"[INFO] Using cleaned telemetry data with {len(df)} records")
        
        # Prepare training data
        X = self._prepare_training_features(df)
        y = self._create_target_variables(df)
        
        if len(X) == 0 or len(y) == 0:
            return {"error": "Failed to prepare training features or targets"}
        
        # Train models for different targets
        models_results = {}
        model_key = "tire_grip_model"  # Generic model key since no track/car specific training
        
        target_variables = [
            'friction_circle_utilization',
            'longitudinal_weight_transfer', 
            'lateral_weight_transfer',
            'dynamic_weight_distribution',
            'optimal_grip_window',
            'slip_angle_efficiency',
            'slip_ratio_efficiency'
        ]
        
        # Store trained models in the class instance
        self.trained_models = {}  # Reset models cache
        
        for target in target_variables:
            if target in y.columns:
                try:
                    model_result = self._train_individual_model(X, y[target], target, model_key)
                    models_results[target] = model_result
                    print(f"[INFO] Trained model for {target}: R² = {model_result.get('r2_score', 'N/A')}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to train model for {target}: {str(e)}")
                    models_results[target] = {"error": str(e)}
        
        # Calculate overall performance summary
        successful_models = [r for r in models_results.values() if 'error' not in r]
        avg_r2 = np.mean([r.get('r2_score', 0) for r in successful_models]) if successful_models else 0
        
        # Store model metadata in the class instance
        self.model_metadata = {
            "model_type": "tire_grip_analysis",
            "track_name": "generic",
            "car_name": "all_cars",
            "models_results": models_results,
            "feature_names": list(X.columns),
            "target_variables": target_variables,
            "training_samples": len(X),
            "average_r2_score": avg_r2,
            "successful_models": len(successful_models),
            "total_models": len(target_variables),
            "training_timestamp": datetime.now().isoformat()
        }
        
        print(f"[INFO] Stored {len(self.trained_models)} trained models in class instance")
        
        # Save models to backend
        try:
            ai_model_dto = {
                "modelType": "tire_grip_analysis",
                "trackName": "generic",
                "carName": "all_cars",
                "modelData": {
                    "models_results": models_results,
                    "feature_names": list(X.columns),
                    "target_variables": target_variables,
                    "training_samples": len(X)
                },
                "metadata": {
                    "average_r2_score": avg_r2,
                    "successful_models": len(successful_models),
                    "total_models": len(target_variables),
                    "training_timestamp": datetime.now().isoformat()
                },
                "isActive": True
            }
            
            await self.backend_service.save_ai_model(ai_model_dto)
            print(f"[INFO] Saved tire grip model to backend")
            
        except Exception as error:
            print(f"[WARNING] Failed to save model to backend: {str(error)}")

        return {
            "success": True,
            "track_name": "generic",
            "car_name": "all_cars",
            "models_trained": len(successful_models),
            "total_targets": len(target_variables),
            "average_r2_score": avg_r2,
            "training_samples": len(X),
            "models_results": models_results,
            "feature_names": list(X.columns),
            "training_timestamp": datetime.now().isoformat()
        }

    def _train_individual_model(self, X: pd.DataFrame, y: pd.Series, target_name: str, model_key: str) -> Dict[str, Any]:
        """
        Train an individual model for a specific target variable
        
        Args:
            X: Feature DataFrame
            y: Target Series
            target_name: Name of the target variable
            model_key: Key for model storage
            
        Returns:
            Model training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Store model in class instance
        model_id = f"{model_key}_{target_name}"
        self.trained_models[model_id] = pipeline
        
        # Store scaler separately for easy access
        self.scalers[model_id] = pipeline.named_steps['scaler']
        
        # Save to disk
        model_path = self.models_directory / f"{model_id}.pkl"
        joblib.dump({
            'model': pipeline,
            'feature_names': list(X.columns),
            'target_name': target_name,
            'model_key': model_key,
            'created_at': datetime.now().isoformat()
        }, model_path)
        
        return {
            "model_id": model_id,
            "target_name": target_name,
            "r2_score": test_r2,
            "train_r2_score": train_r2,
            "mae": test_mae,
            "rmse": test_rmse,
            "feature_count": len(X.columns),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }

    async def extract_tire_grip_features(self, telemetry_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract tire grip features from cleaned telemetry data using saved trained models
        
        Args:
            telemetry_data: List of cleaned telemetry records to predict on
            
        Returns:
            Enhanced telemetry data with tire grip features
        """
        if not telemetry_data:
            return telemetry_data
        
        # Check if we have trained models available
        if not self.trained_models:
            print("[ERROR] No trained models available. Please run train_tire_grip_model first.")
            return telemetry_data
        
        print(f"[INFO] Extracting tire grip features for {len(telemetry_data)} records using saved models")
        print(f"[INFO] Available trained models: {list(self.trained_models.keys())}")
        
        # Convert to DataFrame - data is already cleaned
        df = pd.DataFrame(telemetry_data)
        
        # Prepare features for prediction using the same feature preparation as training
        X = self._prepare_training_features(df)
        
        if len(X) == 0:
            print("[WARNING] No features available for prediction")
            return telemetry_data
        
        # Use the saved models to make predictions
        enhanced_data = []
        model_key = "tire_grip_model"
        
        for i, (original_record, feature_row) in enumerate(zip(telemetry_data, X.to_dict('records'))):
            # Create enhanced record
            enhanced_record = original_record.copy()
            
            # Add basic calculated features
            grip_features = self._calculate_basic_grip_features(df.iloc[i] if i < len(df) else pd.Series())
            enhanced_record.update(grip_features)
            
            # Add ML-based predictions using saved models
            try:
                ml_features = self._predict_with_saved_models(feature_row, model_key)
                enhanced_record.update(ml_features)
            except Exception as e:
                print(f"[WARNING] Failed to get ML predictions for record {i}: {str(e)}")
            
            enhanced_data.append(enhanced_record)
        
        print(f"[INFO] Successfully extracted tire grip features for {len(enhanced_data)} records using saved models")
        return enhanced_data

    def _calculate_basic_grip_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate basic grip features without ML models
        
        Args:
            row: Telemetry data row
            
        Returns:
            Dictionary of basic grip features
        """
        features = {}
        
        try:
            # Basic friction circle utilization
            features['friction_circle_utilization'] = self._calculate_friction_circle_utilization(row)
            
            # Weight transfer analysis
            long_transfer, lat_transfer, dynamic_dist = self._calculate_weight_transfer(row)
            features['longitudinal_weight_transfer'] = long_transfer
            features['lateral_weight_transfer'] = lat_transfer
            features['dynamic_weight_distribution'] = dynamic_dist
            
            # Tire performance metrics
            features['optimal_grip_window'] = self._estimate_tire_temperatures_optimal_window(row)
            
            slip_angle_eff, slip_ratio_eff = self._calculate_slip_efficiency(row)
            features['slip_angle_efficiency'] = slip_angle_eff
            features['slip_ratio_efficiency'] = slip_ratio_eff
            
            # Overall tire grip estimate (simple heuristic)
            temp_factor = features['optimal_grip_window']
            slip_factor = (slip_angle_eff + slip_ratio_eff) / 2
            utilization_penalty = 1 - (features['friction_circle_utilization'] * 0.3)  # Slight penalty for over-utilization
            
            features['overall_tire_grip'] = temp_factor * slip_factor * utilization_penalty
            
            # Tire saturation level (how close to limits)
            features['tire_saturation_level'] = min(features['friction_circle_utilization'] * 1.2, 1.0)
            
        except Exception as e:
            print(f"[WARNING] Error calculating basic grip features: {str(e)}")
            # Provide default values
            default_features = [
                'friction_circle_utilization', 'longitudinal_weight_transfer', 'lateral_weight_transfer',
                'dynamic_weight_distribution', 'optimal_grip_window', 'slip_angle_efficiency',
                'slip_ratio_efficiency', 'overall_tire_grip', 'tire_saturation_level'
            ]
            for feature_name in default_features:
                features[feature_name] = 0.5  # Default moderate value
        
        return features

    def _predict_with_saved_models(self, feature_row: Dict[str, Any], model_key: str) -> Dict[str, float]:
        """
        Predict tire grip features using the saved trained ML models in the class
        
        Args:
            feature_row: Feature data for prediction
            model_key: Key identifying the model set
            
        Returns:
            Dictionary of ML-predicted features
        """
        ml_features = {}
        
        if not self.trained_models:
            print("[WARNING] No trained models available for prediction")
            return ml_features
        
        target_variables = [
            'friction_circle_utilization',
            'longitudinal_weight_transfer', 
            'lateral_weight_transfer',
            'dynamic_weight_distribution',
            'optimal_grip_window',
            'slip_angle_efficiency',
            'slip_ratio_efficiency'
        ]
        
        # Convert feature row to DataFrame for prediction
        feature_df = pd.DataFrame([feature_row])
        
        for target in target_variables:
            model_id = f"{model_key}_{target}"
            
            if model_id in self.trained_models:
                try:
                    # Use the saved model to make prediction
                    prediction = self.trained_models[model_id].predict(feature_df)[0]
                    ml_features[f"ml_{target}"] = float(prediction)
                    
                    # Also add the prediction as the main feature name for compatibility
                    ml_features[target] = float(prediction)
                    
                except Exception as e:
                    print(f"[WARNING] Prediction failed for {target} using saved model: {str(e)}")
            else:
                print(f"[WARNING] Model {model_id} not found in saved models")
        
        return ml_features
    
    def _predict_ml_features(self, feature_row: Dict[str, Any], model_key: str) -> Dict[str, float]:
        """
        Predict tire grip features using trained ML models
        
        Args:
            feature_row: Feature data for prediction
            model_key: Key identifying the model set
            
        Returns:
            Dictionary of ML-predicted features
        """
        ml_features = {}
        
        target_variables = [
            'friction_circle_utilization',
            'longitudinal_weight_transfer', 
            'lateral_weight_transfer',
            'dynamic_weight_distribution',
            'optimal_grip_window',
            'slip_angle_efficiency',
            'slip_ratio_efficiency'
        ]
        
        # Convert feature row to DataFrame for prediction
        feature_df = pd.DataFrame([feature_row])
        
        for target in target_variables:
            model_id = f"{model_key}_{target}"
            
            if model_id in self.trained_models:
                try:
                    prediction = self.trained_models[model_id].predict(feature_df)[0]
                    ml_features[f"ml_{target}"] = float(prediction)
                except Exception as e:
                    print(f"[WARNING] Prediction failed for {target}: {str(e)}")
            else:
                # Try to load from disk
                try:
                    model_path = self.models_directory / f"{model_id}.pkl"
                    if model_path.exists():
                        model_data = joblib.load(model_path)
                        self.trained_models[model_id] = model_data['model']
                        prediction = model_data['model'].predict(feature_df)[0]
                        ml_features[f"ml_{target}"] = float(prediction)
                except Exception as e:
                    print(f"[WARNING] Failed to load and predict {target}: {str(e)}")
        
        return ml_features

    def has_trained_models(self) -> bool:
        """
        Check if the service has trained models available
        
        Returns:
            True if models are available, False otherwise
        """
        return len(self.trained_models) > 0
    
    def get_available_model_targets(self) -> List[str]:
        """
        Get list of available model targets
        
        Returns:
            List of target variable names that have trained models
        """
        if not self.trained_models:
            return []
        
        targets = []
        model_key = "tire_grip_model"
        
        for model_id in self.trained_models.keys():
            if model_id.startswith(f"{model_key}_"):
                target = model_id.replace(f"{model_key}_", "")
                targets.append(target)
        
        return targets

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of trained tire grip models
        
        Returns:
            Model summary information
        """
        model_key = "tire_grip_model"
        
        summary = {
            "model_type": "tire_grip_analysis",
            "models_directory": str(self.models_directory),
            "cached_models": len(self.trained_models),
            "available_models": []
        }
        
        # Check for available model files
        for model_file in self.models_directory.glob(f"{model_key}_*.pkl"):
            try:
                model_data = joblib.load(model_file)
                summary["available_models"].append({
                    "model_id": model_file.stem,
                    "target_name": model_data.get('target_name', 'unknown'),
                    "created_at": model_data.get('created_at', 'unknown'),
                    "feature_count": len(model_data.get('feature_names', []))
                })
            except Exception as e:
                print(f"[WARNING] Failed to load model summary from {model_file}: {str(e)}")
        
        return summary

    def clear_models_cache(self):
        """
        Clear cached tire grip models
        """
        # Clear all cached models
        self.trained_models.clear()
        print("[INFO] Cleared all cached tire grip models")


# Create singleton instance for import
tire_grip_analysis_service = TireGripAnalysisService()

if __name__ == "__main__":
    print("TireGripAnalysisService initialized. Ready for tire grip and friction circle analysis!")
