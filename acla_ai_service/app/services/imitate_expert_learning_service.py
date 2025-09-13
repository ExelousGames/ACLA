"""
Imitation Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service implements imitation learning algorithms to learn from expert driving demonstrations.
It can learn driving behaviors, optimal racing lines, and decision-making patterns from
professional or expert drivers' telemetry data.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Scikit-learn imports for imitation learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import your telemetry models
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor

warnings.filterwarnings('ignore', category=UserWarning)


class BehaviorLearner:
    """Extract driving behavior patterns from telemetry data and train behavior models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.behavior_labels = [
            'aggressive', 'smooth', 'conservative', 'optimal', 'defensive'
        ]
    
    def generate_driving_style_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features that characterize driving style
        
        Args:
            df: Telemetry DataFrame
            
        Returns:
            DataFrame with driving style features
        """
        features = pd.DataFrame()
        
        # Throttle behavior
        if 'Physics_gas' in df.columns:
            features['throttle_mean'] = df['Physics_gas'].rolling(window=10).mean()
            features['throttle_std'] = df['Physics_gas'].rolling(window=10).std()
            features['throttle_max'] = df['Physics_gas'].rolling(window=10).max()
            features['throttle_smoothness'] = df['Physics_gas'].diff().abs().rolling(window=10).mean()
        
        # Brake behavior
        if 'Physics_brake' in df.columns:
            features['brake_mean'] = df['Physics_brake'].rolling(window=10).mean()
            features['brake_std'] = df['Physics_brake'].rolling(window=10).std()
            features['brake_max'] = df['Physics_brake'].rolling(window=10).max()
            features['brake_smoothness'] = df['Physics_brake'].diff().abs().rolling(window=10).mean()
        
        # Steering behavior
        if 'Physics_steer_angle' in df.columns:
            features['steering_mean'] = df['Physics_steer_angle'].rolling(window=10).mean()
            features['steering_std'] = df['Physics_steer_angle'].rolling(window=10).std()
            features['steering_smoothness'] = df['Physics_steer_angle'].diff().abs().rolling(window=10).mean()
            features['steering_frequency'] = df['Physics_steer_angle'].rolling(window=20).apply(
                lambda x: len(np.where(np.diff(np.sign(np.diff(x))))[0])
            )
        
        # Speed behavior
        if 'Physics_speed_kmh' in df.columns:
            features['speed_mean'] = df['Physics_speed_kmh'].rolling(window=10).mean()
            features['speed_std'] = df['Physics_speed_kmh'].rolling(window=10).std()
            features['speed_variance'] = df['Physics_speed_kmh'].rolling(window=10).var()
        
        # G-force behavior (aggressiveness indicators)
        g_force_cols = ['Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z']
        available_g_cols = [col for col in g_force_cols if col in df.columns]
        
        if len(available_g_cols) >= 2:
            g_force_magnitude = np.sqrt(
                sum(df[col]**2 for col in available_g_cols)
            )
            features['g_force_mean'] = g_force_magnitude.rolling(window=10).mean()
            features['g_force_max'] = g_force_magnitude.rolling(window=10).max()
            features['g_force_std'] = g_force_magnitude.rolling(window=10).std()
        
        # Cornering behavior
        if all(col in df.columns for col in ['Physics_steer_angle', 'Physics_speed_kmh']):
            features['cornering_speed'] = df['Physics_speed_kmh'] * np.abs(df['Physics_steer_angle'])
            features['cornering_aggressiveness'] = features['cornering_speed'].rolling(window=10).mean()
        
        # Combined input behavior (trail braking, etc.)
        if all(col in df.columns for col in ['Physics_brake', 'Physics_gas']):
            features['combined_input'] = df['Physics_brake'] + df['Physics_gas']
            features['trail_braking'] = (df['Physics_brake'] > 0) & (df['Physics_gas'] > 0)
            features['input_overlap'] = features['trail_braking'].rolling(window=10).sum()
        
        # Fill missing values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def classify_driving_style(self, features: pd.DataFrame) -> pd.Series:
        """
        Classify driving style based on extracted features
        
        Args:
            features: Driving style features DataFrame
            
        Returns:
            Series with driving style classifications
        """
        # Use clustering to identify driving styles
        feature_cols = features.select_dtypes(include=[np.number]).columns
        X = features[feature_cols].fillna(0)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use K-means clustering to identify driving patterns
        kmeans = KMeans(n_clusters=len(self.behavior_labels), random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Map clusters to behavior labels
        style_mapping = {i: self.behavior_labels[i] for i in range(len(self.behavior_labels))}
        driving_styles = pd.Series([style_mapping.get(cluster, 'unknown') for cluster in clusters])
        
        return driving_styles
    
    def train_behavior_model(self, 
                            features: pd.DataFrame, 
                            styles: pd.Series) -> Dict[str, Any]:
        """
        Train a model to predict driving behavior
        
        Args:
            features: Behavior features DataFrame
            styles: Driving style labels
            
        Returns:
            Trained behavior model information
        """
        # Prepare data
        feature_cols = features.select_dtypes(include=[np.number]).columns
        X = features[feature_cols].fillna(0)
        y = styles
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Check if stratification is possible (each class needs at least 2 samples)
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        min_class_count = np.min(class_counts)
        
        # Split data with or without stratification based on data distribution
        if min_class_count >= 2 and len(unique_classes) > 1:
            # Can use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            print(f"[INFO] Using stratified split with {len(unique_classes)} classes")
        else:
            # Cannot use stratification - some classes have only 1 sample
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            print(f"[WARNING] Using random split (no stratification) - some classes have only {min_class_count} sample(s)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': list(feature_cols),
            'performance_metrics': {
                'accuracy': accuracy,
                'f1_score': f1
            },
            'feature_importance': dict(top_features)
        }


class ExpertTrajectoryLearner:
    """Learn optimal racing lines and trajectories from expert demonstrations"""
    
    def __init__(self):
        self.trajectory_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
    
    def extract_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features relevant to racing line optimization
        
        Args:
            df: Telemetry DataFrame
            
        Returns:
            DataFrame with trajectory features
        """
        features = pd.DataFrame()
        
        # Position and track features
        if 'Graphics_normalized_car_position' in df.columns:
            features['track_position'] = df['Graphics_normalized_car_position']
            features['track_position_rate'] = df['Graphics_normalized_car_position'].diff()
        
        # 3D Player position features for precise trajectory analysis
        if 'Graphics_player_pos_x' in df.columns:
            features['player_pos_x'] = df['Graphics_player_pos_x']
            features['player_pos_x_velocity'] = features['player_pos_x'].diff().rolling(window=3).mean()
            
        if 'Graphics_player_pos_y' in df.columns:
            features['player_pos_y'] = df['Graphics_player_pos_y']
            features['player_pos_y_velocity'] = features['player_pos_y'].diff().rolling(window=3).mean()
            
        if 'Graphics_player_pos_z' in df.columns:
            features['player_pos_z'] = df['Graphics_player_pos_z']
            features['player_pos_z_velocity'] = features['player_pos_z'].diff().rolling(window=3).mean()
            
        # Speed and racing line
        if 'Physics_speed_kmh' in df.columns:
            features['speed'] = df['Physics_speed_kmh']
            features['speed_change'] = df['Physics_speed_kmh'].diff()
            features['acceleration'] = features['speed_change'] / 0.016  # Assuming ~60fps
        
        # Gear information for optimal shifting and trajectory optimization
        if 'Physics_gear' in df.columns:
            features['gear'] = df['Physics_gear']
            features['gear_change'] = df['Physics_gear'].diff()
            # Gear-speed ratio for optimization
            if 'Physics_speed_kmh' in df.columns:
                features['speed_per_gear'] = df['Physics_speed_kmh'] / (df['Physics_gear'] + 1)  # +1 to avoid division by zero
        
        # Steering and line choice
        if 'Physics_steer_angle' in df.columns:
            features['steering_angle'] = df['Physics_steer_angle']
            features['steering_rate'] = df['Physics_steer_angle'].diff()
        
        # Cornering metrics
        if all(col in df.columns for col in ['Physics_steer_angle', 'Physics_speed_kmh']):
            features['cornering_force'] = np.abs(df['Physics_steer_angle']) * df['Physics_speed_kmh']
        
        # Throttle and brake application timing
        if 'Physics_gas' in df.columns:
            features['throttle'] = df['Physics_gas']
            features['throttle_rate'] = df['Physics_gas'].diff()
        
        if 'Physics_brake' in df.columns:
            features['brake'] = df['Physics_brake']
            features['brake_rate'] = df['Physics_brake'].diff()
        
        # Lap time optimization features
        if 'Graphics_current_time' in df.columns:
            features['current_time'] = df['Graphics_current_time']
            features['time_rate'] = df['Graphics_current_time'].diff()
        
        # Fill missing values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def learn_optimal_trajectory(self, 
                               expert_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn optimal trajectory from expert demonstrations
        
        Args:
            expert_df: Expert driver telemetry data
            track_segments: Optional list of track segments to focus on
            
        Returns:
            Dictionary with learned trajectory model and insights
        """
        print(f"[INFO] Learning optimal trajectory from {len(expert_df)} expert data points")
        
        # Extract trajectory features
        trajectory_features = self.extract_trajectory_features(expert_df)
        
        # Define target variables (what we want to optimize)
        targets = {}
        
        # Speed optimization target
        if 'speed' in trajectory_features.columns:
            targets['optimal_speed'] = trajectory_features['speed']
        
        # Steering optimization target
        if 'steering_angle' in trajectory_features.columns:
            targets['optimal_steering'] = trajectory_features['steering_angle']
        
        # Throttle optimization target
        if 'throttle' in trajectory_features.columns:
            targets['optimal_throttle'] = trajectory_features['throttle']
        
        # Brake optimization target
        if 'brake' in trajectory_features.columns:
            targets['optimal_brake'] = trajectory_features['brake']
        
        # Gear optimization target
        if 'gear' in trajectory_features.columns:
            targets['optimal_gear'] = trajectory_features['gear']
        
        # Position optimization targets (3D position for precise trajectory learning)
        if 'player_pos_x' in trajectory_features.columns:
            targets['optimal_player_pos_x'] = trajectory_features['player_pos_x']
        
        if 'player_pos_y' in trajectory_features.columns:
            targets['optimal_player_pos_y'] = trajectory_features['player_pos_y']
            
        if 'player_pos_z' in trajectory_features.columns:
            targets['optimal_player_pos_z'] = trajectory_features['player_pos_z']
        
        # Track position optimization target (fallback if 3D positions not available)
        if 'track_position' in trajectory_features.columns:
            targets['optimal_track_position'] = trajectory_features['track_position']
        
        # Prepare input features (current state)
        input_features = ['track_position', 'speed', 'steering_angle', 'gear', 
                         'player_pos_x', 'player_pos_y', 'player_pos_z',
                         'player_pos_x_velocity', 'player_pos_y_velocity', 'player_pos_z_velocity']
        available_input_features = [f for f in input_features if f in trajectory_features.columns]
        
        if len(available_input_features) < 2:
            raise ValueError("Insufficient features for trajectory learning")
        
        X = trajectory_features[available_input_features].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction if needed
        pca_used = False
        if X_scaled.shape[1] > 10:
            X_scaled = self.pca.fit_transform(X_scaled)
            pca_used = True
        
        # Train models for each target
        models = {}
        performance_metrics = {}
        
        for target_name, target_values in targets.items():
            if target_values.isna().sum() / len(target_values) > 0.5:
                continue  # Skip targets with too many missing values
            
            # Clean target values based on type
            if target_name == 'optimal_gear':
                # For gear, fill missing values with mode (most common gear) and keep as integer
                mode_value = target_values.mode().iloc[0] if not target_values.mode().empty else 1
                y = target_values.fillna(mode_value).astype(int)
            else:
                # For continuous values, use median
                y = target_values.fillna(target_values.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Use different model types based on target variable
            if target_name == 'optimal_gear':
                # Use classifier for discrete gear values, you cant have 3.5 gear   
                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=20, 
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
                # Evaluate classifier
                y_pred = model.predict(X_test)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
            else:
                # Use regressor for continuous values
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=20, 
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
                # Evaluate regressor
                y_pred = model.predict(X_test)
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': model.score(X_test, y_test)
                }
            
            models[target_name] = model
            performance_metrics[target_name] = metrics
            
            # Log metrics based on model type
            if target_name == 'optimal_gear':
                print(f"[INFO] {target_name} model - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
            else:
                print(f"[INFO] {target_name} model - R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
        
        # Store the complete trajectory model
        self.trajectory_model = {
            'models': models,
            'scaler': self.scaler,
            'pca': self.pca if pca_used else None,
            'input_features': available_input_features,
            'performance_metrics': performance_metrics
        }
        
        return {
            'modelData': self.trajectory_model,
            'metadata':{
                'performance_metrics': performance_metrics,
                'input_features': available_input_features,
                'models_trained': list(models.keys())
            }
        }
    
    def predict_optimal_actions(self, current_state: pd.DataFrame) -> Dict[str, float]:
        """
        Predict optimal actions given current state
        
        Args:
            current_state: Current telemetry state
            
        Returns:
            Dictionary with optimal action predictions
        """
        if not self.trajectory_model:
            raise ValueError("No trajectory model trained. Call learn_optimal_trajectory first.")
        
        # Extract features from current state
        trajectory_features = self.extract_trajectory_features(current_state)
        
        # Prepare input
        input_features = self.trajectory_model['input_features']
        
        # Ensure all required features are available
        missing_features = [f for f in input_features if f not in trajectory_features.columns]
        if missing_features:
            raise ValueError(f"Missing required features for prediction: {missing_features}")
            
        X = trajectory_features[input_features].fillna(0)
        
        # Scale features
        X_scaled = self.trajectory_model['scaler'].transform(X)
        
        # Apply PCA if it was used during training and is fitted
        if (self.trajectory_model['pca'] is not None and 
            hasattr(self.trajectory_model['pca'], 'components_')):
            X_scaled = self.trajectory_model['pca'].transform(X_scaled)
        
        # Make predictions
        predictions = {}
        for target_name, model in self.trajectory_model['models'].items():
            try:
                pred = model.predict(X_scaled)
                # Handle gear predictions as integers
                if target_name == 'optimal_gear':
                    predictions[target_name] = int(pred[0] if len(pred) == 1 else int(pred.mean()))
                else:
                    predictions[target_name] = float(pred[0] if len(pred) == 1 else pred.mean())
            except Exception as e:
                print(f"[WARNING] Failed to predict {target_name}: {e}")
                predictions[target_name] = 1 if target_name == 'optimal_gear' else 0.0
        
        return predictions

    def debug_trajectory_model(self) -> Dict[str, Any]:
        """
        Debug method to inspect the current trajectory model state
        
        Returns:
            Dictionary with detailed model debugging information
        """
        if not self.trajectory_model:
            return {
                'status': 'No model trained',
                'has_model': False
            }
        
        debug_info = {
            'status': 'Model available',
            'has_model': True,
            'model_structure': {}
        }
        
        # Check model structure
        for key, value in self.trajectory_model.items():
            if key == 'models':
                debug_info['model_structure']['models'] = {
                    'count': len(value),
                    'model_names': list(value.keys()),
                    'model_types': {name: str(type(model)) for name, model in value.items()}
                }
            elif key == 'scaler':
                debug_info['model_structure']['scaler'] = {
                    'type': str(type(value)),
                    'fitted': hasattr(value, 'scale_'),
                    'n_features': getattr(value, 'n_features_in_', None)
                }
            elif key == 'pca':
                if value is not None:
                    debug_info['model_structure']['pca'] = {
                        'type': str(type(value)),
                        'fitted': hasattr(value, 'components_'),
                        'n_components': getattr(value, 'n_components_', None),
                        'explained_variance_ratio': getattr(value, 'explained_variance_ratio_', None)
                    }
                else:
                    debug_info['model_structure']['pca'] = None
            elif key == 'input_features':
                debug_info['model_structure']['input_features'] = {
                    'count': len(value),
                    'features': value
                }
            elif key == 'performance_metrics':
                debug_info['model_structure']['performance_metrics'] = value
        
        return debug_info
    
    def validate_prediction_input(self, current_state: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data for prediction without actually making predictions
        
        Args:
            current_state: Current telemetry state to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'input_analysis': {},
            'feature_analysis': {}
        }
        
        # Check if model exists
        if not self.trajectory_model:
            validation_results['valid'] = False
            validation_results['errors'].append("No trajectory model trained")
            return validation_results
        
        # Analyze input data
        validation_results['input_analysis'] = {
            'shape': current_state.shape,
            'columns': list(current_state.columns),
            'dtypes': current_state.dtypes.to_dict(),
            'missing_values': current_state.isnull().sum().to_dict(),
            'infinite_values': np.isinf(current_state.select_dtypes(include=[np.number])).sum().to_dict()
        }
        
        # Check for problematic values
        numeric_cols = current_state.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if current_state[col].isnull().sum() > 0:
                validation_results['warnings'].append(f"Column {col} has {current_state[col].isnull().sum()} missing values")
            if np.isinf(current_state[col]).sum() > 0:
                validation_results['warnings'].append(f"Column {col} has {np.isinf(current_state[col]).sum()} infinite values")
        
        current_state
        try:
            # Extract features
            trajectory_features = self.extract_trajectory_features(current_state)
            
            validation_results['feature_analysis'] = {
                'extracted_features_count': trajectory_features.shape[1],
                'extracted_features': list(trajectory_features.columns),
                'required_features': self.trajectory_model['input_features']
            }
            
            # Check for missing features
            required_features = self.trajectory_model['input_features']
            missing_features = [f for f in required_features if f not in trajectory_features.columns]
            
            if missing_features:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Missing required features: {missing_features}")
            
            # Check feature data quality
            feature_subset = trajectory_features[required_features].fillna(0)
            
            for feature in required_features:
                if feature in trajectory_features.columns:
                    feature_data = trajectory_features[feature]
                    if feature_data.isnull().all():
                        validation_results['warnings'].append(f"Feature {feature} is all null values")
                    elif np.isinf(feature_data).any():
                        validation_results['warnings'].append(f"Feature {feature} contains infinite values")
                    elif feature_data.std() == 0:
                        validation_results['warnings'].append(f"Feature {feature} has zero variance")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Feature extraction failed: {str(e)}")
        
        return validation_results


class ImitateExpertLearningService:
    """Main imitation learning service that combines behavior learning and trajectory optimization"""
    
    def __init__(self, models_directory: str = "imitation_models"):
        """
        Initialize the imitation learning service
        
        Args:
            models_directory: Directory to save/load trained imitation models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        self.behavior_learner = BehaviorLearner()
        self.trajectory_learner = ExpertTrajectoryLearner()
        self.trained_models = {}
        
        print(f"[INFO] ImitationLearningService initialized. Models directory: {self.models_directory}")

    def train_ai_model(self, telemetry_data: List[Dict[str, Any]], learning_objectives: List[str] = None) -> Tuple[Dict[str, Any]]:
        """
        Learn from expert driving demonstrations
        
        Args:
            telemetry_data: List of expert telemetry data dictionaries
            learning_objectives: List of what to learn ('behavior', 'trajectory', 'both')
            
        Returns:
            Dictionary with trained models and learning insights, serialized objects and ready for storage
        """
        if learning_objectives is None:
            learning_objectives = ['behavior', 'trajectory']
        
        print(f"[INFO {self.__class__.__name__}] Learning from {len(telemetry_data)} expert demonstrations")
        print(f"[INFO {self.__class__.__name__}] Learning objectives: {learning_objectives}")

        # Convert to DataFrame
        telemetry_df = pd.DataFrame(telemetry_data)
        feature_processor = FeatureProcessor(telemetry_df)
        # Cleaned data
        processed_df = feature_processor.general_cleaning_for_analysis()
        
        results = {}
        
        # Learn driving behavior patterns
        if 'behavior' in learning_objectives:
            print("[INFO] Extracting driving behavior patterns...")
            
            # generate behavior features
            behavior_features = self.behavior_learner.generate_driving_style_features(processed_df)
            
            # Classify driving styles
            driving_styles = self.behavior_learner.classify_driving_style(behavior_features)
            
            # Train behavior prediction model
            behavior_model = self.behavior_learner.train_behavior_model(behavior_features, driving_styles)
            
            results['behavior_learning'] = {
                'modelData': {
                    'model': behavior_model
                },
                'metadata':{
                    'features_shape': behavior_features.shape,
                    'style_distribution': driving_styles.value_counts().to_dict(),
                    'model_performance': behavior_model.get('performance_metrics', {})
                }
            }
        
        # Learn optimal trajectories
        if 'trajectory' in learning_objectives:
            print("[INFO] Learning optimal trajectories...")
            
            trajectory_results = self.trajectory_learner.learn_optimal_trajectory(processed_df)
            
            results['trajectory_learning'] = trajectory_results

        results['learning_summary'] = self._generate_learning_summary(results)
        
        # Store the trained models in the class
        self.trained_models = results
        
        objects_serialized_data = self.serialize_object_inside(results)  
         
        return objects_serialized_data
    
    def predict_expert_actions(self, 
                             processed_df: pd.DataFrame, 
                             model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict what an expert would do in the current situation
        
        Args:
            current_telemetry: Current telemetry state
            model_data: Dictionary containing the trained imitation models
            
        Returns:
            Predicted expert actions and recommendations
        """
        predictions = {}
        
        print(f"[INFO] Predicting expert actions for current telemetry state")
        # Predict driving behavior
        if 'behavior_learning' in model_data:
            behavior_model = model_data['behavior_learning']['modelData']['model']
            
            # Extract behavior features
            behavior_features = self.behavior_learner.generate_driving_style_features(processed_df)
            
            # Prepare features
            feature_cols = behavior_model['feature_names']
            X = behavior_features[feature_cols].fillna(0)
            X_scaled = behavior_model['scaler'].transform(X)
            
            # Predict
            behavior_pred = behavior_model['model'].predict(X_scaled)
            behavior_proba = behavior_model['model'].predict_proba(X_scaled)
            
            # Decode prediction
            predicted_style = behavior_model['label_encoder'].inverse_transform(behavior_pred)[0]
            style_confidence = np.max(behavior_proba)
            
            predictions['driving_behavior'] = {
                'predicted_style': predicted_style,
                'confidence': float(style_confidence),
                'style_probabilities': dict(zip(
                    behavior_model['label_encoder'].classes_,
                    behavior_proba[0]
                ))
            }
        
        # Predict optimal actions
        if 'trajectory_learning' in model_data:
            try:
                # Set the trajectory model
                self.trajectory_learner.trajectory_model = model_data['trajectory_learning']['modelData']
                
                optimal_actions = self.trajectory_learner.predict_optimal_actions(processed_df)
                predictions['optimal_actions'] = optimal_actions
            except Exception as e:
                print(f"[WARNING] Could not predict optimal actions: {e}")
                predictions['optimal_actions'] = {}
        
        return predictions
    
    def _generate_learning_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of learning results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'learning_completed': []
        }
        
        if 'behavior_learning' in results:
            behavior_info = results['behavior_learning']
            summary['learning_completed'].append('behavior')
            summary['behavior_summary'] = {
                'features_extracted': behavior_info['metadata']['features_shape'][1],
                'styles_identified': len(behavior_info['metadata']['style_distribution']),
                'model_accuracy': behavior_info['metadata'].get('model_performance', {}).get('accuracy', 0)
            }
        
        if 'trajectory_learning' in results:
            trajectory_info = results['trajectory_learning']
            summary['learning_completed'].append('trajectory')
            
            # Calculate average performance metric, handling both regression (r2) and classification (accuracy) models
            performance_metrics = trajectory_info['metadata']['performance_metrics']
            
            # Separate regression and classification metrics
            r2_scores = [metrics['r2'] for metrics in performance_metrics.values() if 'r2' in metrics]
            accuracy_scores = [metrics['accuracy'] for metrics in performance_metrics.values() if 'accuracy' in metrics]
            
            # Calculate average scores
            avg_r2 = np.mean(r2_scores) if r2_scores else 0.0
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            summary['trajectory_summary'] = {
                'models_trained': len(trajectory_info['metadata']['models_trained']),
                'input_features': len(trajectory_info['metadata']['input_features']),
                'avg_r2_score': avg_r2,
                'avg_accuracy_score': avg_accuracy,
                'regression_models': len(r2_scores),
                'classification_models': len(accuracy_scores)
            }
        
        return summary
 
    def serialize_object_inside(self, results: any) -> str:
                # Serialize behavior learning model if present
        if 'behavior_learning' in results and 'model' in results['behavior_learning']['modelData']:
            print("[INFO] Serializing behavior learning model...")
            # Only serialize the actual model from the behavior_learning['model'] structure
            behavior_model_to_serialize = results['behavior_learning']['modelData']['model']
            behavior_model_data = self.serialize_data(
                behavior_model_to_serialize
            )
            results['behavior_learning']['modelData']['model'] = behavior_model_data


        # Serialize behavior learning scaler if present
        if 'behavior_learning' in results and 'scaler' in results['behavior_learning']['modelData']:
            print("[INFO] Serializing behavior learning scaler...")
            # Only serialize the actual model from the behavior_learning['model'] structure
            behavior_model_to_serialize = results['behavior_learning']['modelData']['scaler']
            behavior_model_data = self.serialize_data(
                behavior_model_to_serialize
            )
            results['behavior_learning']['modelData']['scaler'] = behavior_model_data
            
        # Serialize trajectory learning models if present
        if 'trajectory_learning' in results and 'models' in results['trajectory_learning']['modelData']:
            print("[INFO] Serializing trajectory learning models...")
            # Only serialize the actual trajectory_model from the trajectory_learning structure
            trajectory_models_to_serialize = results['trajectory_learning']['modelData']['models']
                
            # Serialize each model individually
            serialized_trajectory_models = {}
            for model_name, model in trajectory_models_to_serialize.items():
                print(f"[INFO] Serializing trajectory model: {model_name}")
                serialized_model_data = self.serialize_data(model)
                serialized_trajectory_models[model_name] = serialized_model_data
                
            # Store serialized models back in the trajectory model structure
            results['trajectory_learning']['modelData']['models'] = serialized_trajectory_models
            
            trajectory_scaler_to_serialize = results['trajectory_learning']['modelData']['scaler']
            serialized_scaler_data = self.serialize_data(
                trajectory_scaler_to_serialize
            )
            results['trajectory_learning']['modelData']['scaler'] = serialized_scaler_data
            
            # Serialize trajectory PCA only if it exists and is not None
            trajectory_pca_to_serialize = results['trajectory_learning']['modelData']['pca']
            if trajectory_pca_to_serialize is not None:
                serialized_pca_data = self.serialize_data(trajectory_pca_to_serialize)
                results['trajectory_learning']['modelData']['pca'] = serialized_pca_data
            else:
                results['trajectory_learning']['modelData']['pca'] = None
            
        return results
    
    # Deserialize object inside 
    def deserialize_object_inside(self, serialized_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize the objects that were serialized by serialize_object_inside function
        
        Args:
            serialized_results: Dictionary containing serialized models and metadata
            
        Returns:
            Dictionary with deserialized models and original structure
        """
        results = serialized_results.copy()
        
        # Deserialize behavior learning model if present
        if 'behavior_learning' in results and 'model' in results['behavior_learning']['modelData']:
            print("[INFO] Deserializing behavior learning model...")
            # Deserialize the actual model from the behavior_learning['model'] structure
            behavior_model_serialized = results['behavior_learning']['modelData']['model']
            behavior_model_deserialized = self.deserialize_data(behavior_model_serialized)
            results['behavior_learning']['modelData']['model'] = behavior_model_deserialized

        # Deserialize behavior learning scaler if present
        if 'behavior_learning' in results and 'scaler' in results['behavior_learning']['modelData']:
            print("[INFO] Deserializing behavior learning scaler...")
            # Deserialize the actual scaler from the behavior_learning structure
            behavior_scaler_serialized = results['behavior_learning']['modelData']['scaler']
            behavior_scaler_deserialized = self.deserialize_data(behavior_scaler_serialized)
            results['behavior_learning']['modelData']['scaler'] = behavior_scaler_deserialized
            
        # Deserialize trajectory learning models if present
        if 'trajectory_learning' in results and 'models' in results['trajectory_learning']['modelData']:
            print("[INFO] Deserializing trajectory learning models...")
            # Deserialize each trajectory model individually
            trajectory_models_serialized = results['trajectory_learning']['modelData']['models']
                
            # Deserialize each model individually
            deserialized_trajectory_models = {}
            for model_name, serialized_model in trajectory_models_serialized.items():
                print(f"[INFO] Deserializing trajectory model: {model_name}")
                deserialized_model = self.deserialize_data(serialized_model)
                deserialized_trajectory_models[model_name] = deserialized_model
                
            # Store deserialized models back in the trajectory model structure
            results['trajectory_learning']['modelData']['models'] = deserialized_trajectory_models
            
            # Deserialize trajectory scaler
            trajectory_scaler_serialized = results['trajectory_learning']['modelData']['scaler']
            deserialized_scaler = self.deserialize_data(trajectory_scaler_serialized)
            results['trajectory_learning']['modelData']['scaler'] = deserialized_scaler
            
            # Deserialize trajectory PCA (only if it exists and is not None)
            if 'pca' in results['trajectory_learning']['modelData']:
                trajectory_pca_serialized = results['trajectory_learning']['modelData']['pca']
                if trajectory_pca_serialized is not None:
                    deserialized_pca = self.deserialize_data(trajectory_pca_serialized)
                    results['trajectory_learning']['modelData']['pca'] = deserialized_pca
                else:
                    results['trajectory_learning']['modelData']['pca'] = None
            
        return results
    
    def load_models_from_serialized(self, serialized_model_data: Dict[str, Any]) -> None:
        """
        Load models from serialized data into the class
        
        Args:
            serialized_model_data: Dictionary containing serialized models
        """
        print("[INFO] Loading models from serialized data...")
        self.trained_models = self.deserialize_object_inside(serialized_model_data)
        print("[INFO] Models loaded successfully")
    
    # Serialize models function
    def serialize_data(self, data: any) -> str:
        """
        Serialize trained behavior and trajectory models
        
        Args:
            training_result: Dictionary containing trained models
            
        Returns:
            Serialized model data as base64 encoded string
        """
        # Prepare serialization data
            
        try:
            # Serialize to bytes
            buffer = io.BytesIO()
            pickle.dump(data, buffer)
            buffer.seek(0)
        
            # Encode to base64
            encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
            return encoded_data
                
        except Exception as e:
            print(f"[ERROR] Failed to serialize models: {e}")
            raise e
    
    def deserialize_data(self, model_data: str) -> Dict[str, Any]:
        """
        Deserialize imitation learning models from base64 string
        
        Args:
            model_data: Base64 encoded model data
        
        Returns:
            Dictionary containing deserialized models and metadata
        """
        
        try:
            # Decode from base64
            decoded_data = base64.b64decode(model_data.encode('utf-8'))
            
            # Deserialize using pickle
            buffer = io.BytesIO(decoded_data)
            data_result = pickle.load(buffer)
            return data_result
            
        except Exception as e:
            raise Exception(f"Failed to deserialize imitation learning models: {str(e)}")
    
    def compare_telemetry_with_expert(self, 
                                    incoming_telemetry: List[Dict[str, Any]], 
                                    window_size: int = 5,
                                    min_section_length: int = 10) -> Dict[str, Any]:
        """
        Compare incoming telemetry data with expert actions and identify performance sections
        
        Args:
            incoming_telemetry: List of incoming telemetry data points (already cleaned)
            window_size: Window size for smoothing score trends
            min_section_length: Minimum length for a valid section
            
        Returns:
            Dictionary containing:
            - paired_data: List of incoming telemetry data paired with corresponding expert actions and similarity scores
            - overall statistics and performance sections analysis
            - score statistics for performance evaluation
        """
        print(f"[INFO {self.__class__.__name__}] Comparing {len(incoming_telemetry)} telemetry points with expert model")
        
        # Check if we have trained models
        if not self.trained_models:
            raise ValueError("No trained models available. Please train models first using train_ai_model().")
        
        # Convert to DataFrame (data is already cleaned)
        processed_df = pd.DataFrame(incoming_telemetry)
        
        if processed_df.empty:
            raise ValueError("No telemetry data available")
        
        # Use the stored trained models
        model_data = self.trained_models
        
        # Calculate scores for each row
        scores = []
        expert_predictions = []
        
        print("[INFO] Calculating similarity scores with expert actions...")
        
        for idx in range(len(processed_df)):
            # Get current row as DataFrame (maintaining structure)
            current_row = processed_df.iloc[idx:idx+1].copy()
            
            try:
                # Get expert predictions for this state using stored models
                expert_actions = self.predict_expert_actions(current_row, model_data)
                expert_predictions.append(expert_actions)
                
                # Calculate similarity score between actual and predicted expert actions
                score = self._calculate_similarity_score(current_row, expert_actions)
                scores.append(score)
                
            except Exception as e:
                print(f"[WARNING] Failed to process row {idx}: {e}")
                scores.append(0.0)
                expert_predictions.append({})
        
        # Convert scores to pandas Series for easier manipulation
        scores_series = pd.Series(scores)
        
        # Smooth scores using rolling window
        smoothed_scores = scores_series.rolling(window=min(window_size, len(scores)), center=True).mean()
        smoothed_scores = smoothed_scores.fillna(scores_series)
        
        # Identify sections with different patterns
        sections = self._identify_performance_sections(
            smoothed_scores, 
            window_size, 
            min_section_length
        )
        
        # Create paired data with incoming telemetry and expert actions
        paired_data = []
        for idx in range(len(processed_df)):
            telemetry_point = processed_df.iloc[idx].to_dict()
            expert_actions = expert_predictions[idx]
            similarity_score = scores[idx]
            
            paired_point = {
                'index': idx,
                'telemetry_data': telemetry_point,
                'expert_actions': expert_actions,
                'similarity_score': similarity_score
            }
            paired_data.append(paired_point)
        
        # Identify sections with different patterns for additional analysis
        sections = self._identify_performance_sections(
            smoothed_scores, 
            window_size, 
            min_section_length
        )
        
        # Main result structure with paired data
        result = {
            'total_data_points': len(processed_df),
            'overall_score': float(np.mean(scores)),
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'paired_data': paired_data,
            'performance_sections': []
        }
        
        print(f"[INFO] Identified {len(sections)} performance sections")
        
        # Process performance sections and create training pairs for transformer
        transformer_training_pairs = []
        
        for section in sections:
            start_idx = section['start_index']
            end_idx = section['end_index']
            section_scores = scores[start_idx:end_idx+1]
            
            # Extract telemetry and expert actions for this section
            telemetry_section = []
            expert_actions_section = []
            
            for idx in range(start_idx, end_idx + 1):
                if idx < len(paired_data):
                    telemetry_section.append(paired_data[idx]['telemetry_data'])
                    expert_actions_section.append(paired_data[idx]['expert_actions'])
            
            # Create transformer training pair
            if telemetry_section and expert_actions_section:
                transformer_pair = {
                    'telemetry_section': telemetry_section,
                    'expert_actions': expert_actions_section,
                    'pattern': section['pattern'],
                    'section_stats': {
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'length': end_idx - start_idx + 1,
                        'mean_score': float(np.mean(section_scores)),
                        'trend_slope': section['trend_slope']
                    }
                }
                transformer_training_pairs.append(transformer_pair)
            
            section_result = {
                'pattern': section['pattern'],
                'start_index': start_idx,
                'end_index': end_idx,
                'length': end_idx - start_idx + 1,
                'score_statistics': {
                    'mean': float(np.mean(section_scores)),
                    'std': float(np.std(section_scores)),
                    'min': float(np.min(section_scores)),
                    'max': float(np.max(section_scores)),
                    'trend_slope': section['trend_slope']
                }
            }
            
            result['performance_sections'].append(section_result)
        
        # Add transformer training pairs to the result
        result['transformer_training_pairs'] = transformer_training_pairs
        print(f"[INFO] Created {len(transformer_training_pairs)} transformer training pairs")
        
        return result
    
    def _calculate_similarity_score(self, 
                                  actual_telemetry: pd.DataFrame, 
                                  expert_actions: Dict[str, Any]) -> float:
        """
        Calculate similarity score between actual telemetry and expert predictions
        
        Args:
            actual_telemetry: Single row of actual telemetry data
            expert_actions: Predicted expert actions
            
        Returns:
            Similarity score between 0 and 1 (1 being perfect match)
        """
        if not expert_actions:
            return 0.0
        
        total_score = 0.0
        comparison_count = 0
        
        # Compare optimal actions if available
        if 'optimal_actions' in expert_actions:
            optimal_actions = expert_actions['optimal_actions']
            
            # Compare speed
            if 'optimal_speed' in optimal_actions and 'Physics_speed_kmh' in actual_telemetry.columns:
                actual_speed = actual_telemetry['Physics_speed_kmh'].iloc[0]
                predicted_speed = optimal_actions['optimal_speed']
                if not (np.isnan(actual_speed) or np.isnan(predicted_speed)):
                    speed_diff = abs(actual_speed - predicted_speed)
                    max_speed = max(actual_speed, predicted_speed, 1)  # Avoid division by zero
                    speed_score = max(0, 1 - (speed_diff / max_speed))
                    total_score += speed_score
                    comparison_count += 1
            
            # Compare throttle
            if 'optimal_throttle' in optimal_actions and 'Physics_gas' in actual_telemetry.columns:
                actual_throttle = actual_telemetry['Physics_gas'].iloc[0]
                predicted_throttle = optimal_actions['optimal_throttle']
                if not (np.isnan(actual_throttle) or np.isnan(predicted_throttle)):
                    throttle_diff = abs(actual_throttle - predicted_throttle)
                    throttle_score = max(0, 1 - throttle_diff)  # Assuming throttle is 0-1
                    total_score += throttle_score
                    comparison_count += 1
            
            # Compare brake
            if 'optimal_brake' in optimal_actions and 'Physics_brake' in actual_telemetry.columns:
                actual_brake = actual_telemetry['Physics_brake'].iloc[0]
                predicted_brake = optimal_actions['optimal_brake']
                if not (np.isnan(actual_brake) or np.isnan(predicted_brake)):
                    brake_diff = abs(actual_brake - predicted_brake)
                    brake_score = max(0, 1 - brake_diff)  # Assuming brake is 0-1
                    total_score += brake_score
                    comparison_count += 1
            
            # Compare steering
            if 'optimal_steering' in optimal_actions and 'Physics_steer_angle' in actual_telemetry.columns:
                actual_steering = actual_telemetry['Physics_steer_angle'].iloc[0]
                predicted_steering = optimal_actions['optimal_steering']
                if not (np.isnan(actual_steering) or np.isnan(predicted_steering)):
                    steering_diff = abs(actual_steering - predicted_steering)
                    max_steering = max(abs(actual_steering), abs(predicted_steering), 1)
                    steering_score = max(0, 1 - (steering_diff / max_steering))
                    total_score += steering_score
                    comparison_count += 1
            
            # Compare gear
            if 'optimal_gear' in optimal_actions and 'Physics_gear' in actual_telemetry.columns:
                actual_gear = actual_telemetry['Physics_gear'].iloc[0]
                predicted_gear = optimal_actions['optimal_gear']
                if not (pd.isna(actual_gear) or pd.isna(predicted_gear)):
                    gear_score = 1.0 if int(actual_gear) == int(predicted_gear) else 0.0
                    total_score += gear_score
                    comparison_count += 1
        
        # Compare behavior prediction if available
        if 'driving_behavior' in expert_actions:
            behavior_info = expert_actions['driving_behavior']
            if 'confidence' in behavior_info:
                # Use confidence as a score component
                behavior_score = behavior_info['confidence']
                total_score += behavior_score
                comparison_count += 1
        
        # Return average score
        return total_score / comparison_count if comparison_count > 0 else 0.0
    
    def _identify_performance_sections(self, 
                                     scores: pd.Series, 
                                     window_size: int,
                                     min_section_length: int) -> List[Dict[str, Any]]:
        """
        Identify sections with consistently increasing, high, or decreasing scores
        
        Args:
            scores: Series of similarity scores
            window_size: Window size for trend analysis
            min_section_length: Minimum length for a valid section
            
        Returns:
            List of identified sections with patterns
        """
        sections = []
        
        # Calculate trend using rolling linear regression slope
        trends = []
        for i in range(len(scores)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(scores), i + window_size // 2)
            
            if end_idx - start_idx < 3:  # Need at least 3 points for trend
                trends.append(0.0)
                continue
            
            # Calculate slope using least squares
            x = np.arange(start_idx, end_idx)
            y = scores.iloc[start_idx:end_idx].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(y)
            if np.sum(valid_mask) < 3:
                trends.append(0.0)
                continue
            
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            try:
                slope = np.polyfit(x_clean, y_clean, 1)[0]
                trends.append(slope)
            except:
                trends.append(0.0)
        
        trends = pd.Series(trends)
        
        # Define thresholds for pattern identification
        increasing_threshold = 0.002  # Positive slope threshold
        decreasing_threshold = -0.002  # Negative slope threshold
        high_score_threshold = np.percentile(scores.dropna(), 75)  # Top 25% of scores
        
        # Identify sections
        current_pattern = None
        section_start = 0
        
        for i in range(len(scores)):
            current_score = scores.iloc[i]
            current_trend = trends.iloc[i]
            
            # Determine current pattern
            if current_trend > increasing_threshold:
                pattern = 'increasing'
            elif current_trend < decreasing_threshold:
                pattern = 'decreasing'
            elif current_score > high_score_threshold:
                pattern = 'high_performance'
            else:
                pattern = 'stable'
            
            # Check if pattern changed
            if current_pattern != pattern:
                # Close previous section if it exists and meets minimum length
                if (current_pattern is not None and 
                    i - section_start >= min_section_length and 
                    current_pattern != 'stable'):
                    
                    # Calculate average trend slope for this section
                    section_trends = trends.iloc[section_start:i]
                    avg_slope = section_trends.mean() if len(section_trends) > 0 else 0.0
                    
                    sections.append({
                        'pattern': current_pattern,
                        'start_index': section_start,
                        'end_index': i - 1,
                        'trend_slope': float(avg_slope)
                    })
                
                # Start new section
                current_pattern = pattern
                section_start = i
        
        # Close final section if it meets criteria
        if (current_pattern is not None and 
            len(scores) - section_start >= min_section_length and 
            current_pattern != 'stable'):
            
            section_trends = trends.iloc[section_start:]
            avg_slope = section_trends.mean() if len(section_trends) > 0 else 0.0
            
            sections.append({
                'pattern': current_pattern,
                'start_index': section_start,
                'end_index': len(scores) - 1,
                'trend_slope': float(avg_slope)
            })
        
        return sections
        
    
    
# Example usage and testing
if __name__ == "__main__":
    print("ImitationLearningService initialized. Ready for expert demonstration learning!")
    
    # Example workflow
    service = ImitateExpertLearningService()
    
    # 1. Train the model (this stores models in the class)
    # serialized_results = service.train_ai_model(expert_telemetry_data)
    
    # 2. Or load previously trained models
    # service.load_models_from_serialized(serialized_model_data)
    
    # 3. Compare new telemetry with stored expert models
    # comparison = service.compare_telemetry_with_expert(incoming_telemetry)