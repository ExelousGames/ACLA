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


class DrivingBehaviorExtractor:
    """Extract driving behavior patterns from telemetry data"""
    
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
        
        # Speed and racing line
        if 'Physics_speed_kmh' in df.columns:
            features['speed'] = df['Physics_speed_kmh']
            features['speed_change'] = df['Physics_speed_kmh'].diff()
            features['acceleration'] = features['speed_change'] / 0.016  # Assuming ~60fps
        
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
        
        # Tire and setup features
        tire_temp_cols = [col for col in df.columns if 'tyre_core_temp' in col]
        if tire_temp_cols:
            features['avg_tire_temp'] = df[tire_temp_cols].mean(axis=1)
            features['tire_temp_variance'] = df[tire_temp_cols].var(axis=1)
        
        # Fill missing values
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def learn_optimal_trajectory(self, 
                               expert_df: pd.DataFrame, 
                               track_segments: Optional[List[str]] = None) -> Dict[str, Any]:
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
        
        # Prepare input features (current state)
        input_features = ['track_position', 'speed', 'steering_angle']
        available_input_features = [f for f in input_features if f in trajectory_features.columns]
        
        if len(available_input_features) < 2:
            raise ValueError("Insufficient features for trajectory learning")
        
        X = trajectory_features[available_input_features].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction if needed
        if X_scaled.shape[1] > 10:
            X_scaled = self.pca.fit_transform(X_scaled)
        
        # Train models for each target
        models = {}
        performance_metrics = {}
        
        for target_name, target_values in targets.items():
            if target_values.isna().sum() / len(target_values) > 0.5:
                continue  # Skip targets with too many missing values
            
            # Clean target values
            y = target_values.fillna(target_values.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=20, 
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': model.score(X_test, y_test)
            }
            
            models[target_name] = model
            performance_metrics[target_name] = metrics
            
            
            print(f"[INFO] {target_name} model - R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f}")
        
        # Store the complete trajectory model
        self.trajectory_model = {
            'models': models,
            'scaler': self.scaler,
            'pca': self.pca if hasattr(self, 'pca') else None,
            'input_features': available_input_features,
            'performance_metrics': performance_metrics
        }
        
        return {
            'trajectory_model': self.trajectory_model,
            'performance_metrics': performance_metrics,
            'input_features': available_input_features,
            'models_trained': list(models.keys())
        }
    
    def serialize_model(self, trajectory_model: Dict[str, Any]) -> bytes:
        """
        Serialize the complete trajectory model including all components
        
        Args:
            trajectory_model: Dictionary containing all model components
            
        Returns:
            Serialized model as bytes
        """
        import pickle
        from datetime import datetime
        
        # Create a comprehensive serialization package
        serialization_package = {
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0',
            'trajectory_model': trajectory_model,
            'metadata': {
                'num_models': len(trajectory_model.get('models', {})),
                'input_features_count': len(trajectory_model.get('input_features', [])),
                'has_pca': trajectory_model.get('pca') is not None,
                'has_scaler': trajectory_model.get('scaler') is not None
            }
        }
        
        try:
            # Serialize the complete package
            serialized_data = pickle.dumps(serialization_package)
            print(f"[INFO] Model serialized successfully. Size: {len(serialized_data)} bytes")
            return serialized_data
        except Exception as e:
            print(f"[ERROR] Failed to serialize model: {e}")
            raise
    
    def deserialize_model(self, serialized_data: bytes) -> Dict[str, Any]:
        """
        Deserialize a trajectory model from bytes
        
        Args:
            serialized_data: Serialized model data
            
        Returns:
            Deserialized trajectory model
        """
        import pickle
        
        try:
            serialization_package = pickle.loads(serialized_data)
            
            # Extract the trajectory model
            trajectory_model = serialization_package['trajectory_model']
            
            # Restore the model to the current instance
            self.trajectory_model = trajectory_model
            self.scaler = trajectory_model.get('scaler')
            if trajectory_model.get('pca') is not None:
                self.pca = trajectory_model.get('pca')
            
            print(f"[INFO] Model deserialized successfully from {serialization_package.get('timestamp', 'unknown time')}")
            print(f"[INFO] Model version: {serialization_package.get('model_version', 'unknown')}")

            return trajectory_model
        except Exception as e:
            print(f"[ERROR] Failed to deserialize model: {e}")
            raise
    
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
        X = trajectory_features[input_features].fillna(0)
        
        # Scale features
        X_scaled = self.trajectory_model['scaler'].transform(X)
        
        # Apply PCA if used during training
        if self.trajectory_model['pca'] is not None:
            X_scaled = self.trajectory_model['pca'].transform(X_scaled)
        
        # Make predictions
        predictions = {}
        for target_name, model in self.trajectory_model['models'].items():
            pred = model.predict(X_scaled)
            predictions[target_name] = float(pred[0] if len(pred) == 1 else pred.mean())
        
        return predictions


class ImitationLearningService:
    """Main imitation learning service that combines behavior learning and trajectory optimization"""
    
    def __init__(self, models_directory: str = "imitation_models"):
        """
        Initialize the imitation learning service
        
        Args:
            models_directory: Directory to save/load trained imitation models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        self.behavior_extractor = DrivingBehaviorExtractor()
        self.trajectory_learner = ExpertTrajectoryLearner()
        self.trained_models = {}
        
        print(f"[INFO] ImitationLearningService initialized. Models directory: {self.models_directory}")

    def train_ai_model(self, telemetry_data: List[Dict[str, Any]], learning_objectives: List[str] = None) -> Dict[str, Any]:
        """
        Learn from expert driving demonstrations
        
        Args:
            telemetry_data: List of expert telemetry data dictionaries
            learning_objectives: List of what to learn ('behavior', 'trajectory', 'both')
            
        Returns:
            Dictionary with learning results
        """
        if learning_objectives is None:
            learning_objectives = ['behavior', 'trajectory']
        
        print(f"[INFO] Learning from {len(telemetry_data)} expert demonstrations")
        print(f"[INFO] Learning objectives: {learning_objectives}")
        
        # Convert to DataFrame
        feature_processor = FeatureProcessor(pd.DataFrame(telemetry_data))
        # Cleaned data
        processed_df = feature_processor.general_cleaning_for_analysis()
            
        results = {}
        
        # Learn driving behavior patterns
        if 'behavior' in learning_objectives:
            print("[INFO] Extracting driving behavior patterns...")
            
            # generate behavior features
            behavior_features = self.behavior_extractor.generate_driving_style_features(processed_df)
            
            # Classify driving styles
            driving_styles = self.behavior_extractor.classify_driving_style(behavior_features)
            
            # Train behavior prediction model
            behavior_model = self._train_behavior_model(behavior_features, driving_styles)
            
            results['behavior_learning'] = {
                'model': behavior_model,
                'features_shape': behavior_features.shape,
                'style_distribution': driving_styles.value_counts().to_dict(),
                'model_performance': behavior_model.get('performance_metrics', {})
            }
        
        # Learn optimal trajectories
        if 'trajectory' in learning_objectives:
            print("[INFO] Learning optimal trajectories...")
            
            trajectory_results = self.trajectory_learner.learn_optimal_trajectory(processed_df)
            
            results['trajectory_learning'] = trajectory_results
        
        self._save_imitation_model(results)

        return results, self._generate_learning_summary(results)

    def _train_behavior_model(self, 
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
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
    
    def predict_expert_actions(self, 
                             current_telemetry: Dict[str, Any], 
                             model_id: str) -> Dict[str, Any]:
        """
        Predict what an expert would do in the current situation
        
        Args:
            current_telemetry: Current telemetry state
            model_id: ID of the trained imitation model
            
        Returns:
            Predicted expert actions and recommendations
        """
        # Load model
        model_info = self._load_imitation_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        df = pd.DataFrame([current_telemetry])
        predictions = {}
        
        # Predict driving behavior
        if 'behavior_learning' in model_info:
            behavior_model = model_info['behavior_learning']['model']
            
            # Extract behavior features
            behavior_features = self.behavior_extractor.generate_driving_style_features(df)
            
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
        if 'trajectory_learning' in model_info:
            try:
                # Set the trajectory model
                self.trajectory_learner.trajectory_model = model_info['trajectory_learning']['trajectory_model']
                
                optimal_actions = self.trajectory_learner.predict_optimal_actions(df)
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
                'features_extracted': behavior_info['features_shape'][1],
                'styles_identified': len(behavior_info['style_distribution']),
                'model_accuracy': behavior_info.get('model_performance', {}).get('accuracy', 0)
            }
        
        if 'trajectory_learning' in results:
            trajectory_info = results['trajectory_learning']
            summary['learning_completed'].append('trajectory')
            summary['trajectory_summary'] = {
                'models_trained': len(trajectory_info['models_trained']),
                'input_features': len(trajectory_info['input_features']),
                'avg_r2_score': np.mean([
                    metrics['r2'] for metrics in trajectory_info['performance_metrics'].values()
                ])
            }
        
        return summary

    def prepare_for_analysis(self) -> pd.DataFrame:
        """Prepare the DataFrame for AI analysis by cleaning and preprocessing"""
        
        processed_df = self.df.copy()
        
        # Ensure all column names are strings to prevent AttributeError on .lower()
        if any(not isinstance(col, str) for col in processed_df.columns):
            processed_df.columns = [str(col) for col in processed_df.columns]
        
        # Handle complex nested structures from AC Competizione telemetry
        self._handle_complex_fields(processed_df)
        
        # Handle missing values
        numeric_columns = processed_df.select_dtypes(include=['number']).columns
        processed_df[numeric_columns] = processed_df[numeric_columns].fillna(0)
        
        # Convert string boolean values to actual booleans
        boolean_features = [col for col in processed_df.columns if 
                          isinstance(col, str) and any(keyword in col.lower() for keyword in ['on', 'enabled', 'valid', 'running', 'controlled'])]
        
        for col in boolean_features:
            if col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    processed_df[col] = processed_df[col].map({
                        'True': True, 'False': False, 'true': True, 'false': False,
                        '1': True, '0': False, 1: True, 0: False
                    }).fillna(False)
        
        return processed_df
# Example usage and testing
if __name__ == "__main__":
    print("ImitationLearningService initialized. Ready for expert demonstration learning!")
    
    # Example test
    service = ImitationLearningService()
    
    # You can test with:
    # results = service.learn_from_expert_demonstrations(expert_telemetry_data)
    # predictions = service.predict_expert_actions(current_state, model_id)