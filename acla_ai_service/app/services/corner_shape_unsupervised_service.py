"""
Unsupervised Corner Shape Learning Service for Assetto Corsa Competizione

This service uses unsupervised machine learning techniques to discover and learn
corner shapes from racing telemetry data. It identifies patterns in cornering 
behavior, clusters similar corner types, and creates shape profiles that can be
used for analysis and guidance.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math
import asyncio
from collections import defaultdict
from pathlib import Path
import joblib

# Sklearn imports for unsupervised learning
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture

# Import telemetry models and services
from ..models.telemetry_models import TelemetryFeatures, FeatureProcessor, _safe_float
from .backend_service import backend_service
from .identify_track_cornoring_phases import TrackCorneringAnalyzer

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class CornerShapeFeatureExtractor:
    """Extract features that characterize corner shapes"""
    
    def __init__(self):
        self.telemetry_features = TelemetryFeatures()
        
    def extract_corner_shape_features(self, corner_df: pd.DataFrame, corner_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features that characterize the shape and characteristics of a corner
        
        Args:
            corner_df: DataFrame containing telemetry data for a specific corner
            corner_info: Dictionary with corner metadata (start/end positions, phases, etc.)
            
        Returns:
            Dictionary of corner shape features
        """
        features = {}
        
        # Basic corner geometry features
        features.update(self._extract_geometry_features(corner_df, corner_info))
        
        # Speed profile features
        features.update(self._extract_speed_profile_features(corner_df))
        
        # Steering profile features
        features.update(self._extract_steering_profile_features(corner_df))
        
        # G-force pattern features
        features.update(self._extract_gforce_features(corner_df))
        
        # Braking and throttle patterns
        features.update(self._extract_input_pattern_features(corner_df))
        
        # Trajectory features
        features.update(self._extract_trajectory_features(corner_df))
        
        # Timing and rhythm features
        features.update(self._extract_timing_features(corner_df, corner_info))
        
        return features
    
    def _extract_geometry_features(self, corner_df: pd.DataFrame, corner_info: Dict[str, Any]) -> Dict[str, float]:
        """Extract geometric features of the corner"""
        features = {}
        
        # Corner duration and length
        features['corner_duration'] = len(corner_df)
        
        if 'corner_start_position' in corner_info and 'corner_end_position' in corner_info:
            start_pos = corner_info['corner_start_position']
            end_pos = corner_info['corner_end_position']
            
            # Handle track wraparound
            if end_pos < start_pos:
                corner_length = (1.0 - start_pos) + end_pos
            else:
                corner_length = end_pos - start_pos
                
            features['corner_length'] = corner_length
            features['corner_density'] = len(corner_df) / max(corner_length, 0.001)
        
        # Steering angle characteristics
        if 'Physics_steer_angle' in corner_df.columns:
            steering = corner_df['Physics_steer_angle']
            features['max_steering_angle'] = abs(steering).max()
            features['avg_steering_angle'] = abs(steering).mean()
            features['steering_variability'] = steering.std()
            
            # Steering direction (left/right corner detection)
            features['steering_direction'] = np.sign(steering.mean())  # -1 for left, +1 for right
            
        return features
    
    def _extract_speed_profile_features(self, corner_df: pd.DataFrame) -> Dict[str, float]:
        """Extract speed profile characteristics"""
        features = {}
        
        if 'Physics_speed_kmh' in corner_df.columns:
            speed = corner_df['Physics_speed_kmh']
            
            features['entry_speed'] = speed.iloc[0] if len(speed) > 0 else 0
            features['exit_speed'] = speed.iloc[-1] if len(speed) > 0 else 0
            features['min_speed'] = speed.min()
            features['max_speed'] = speed.max()
            features['avg_speed'] = speed.mean()
            features['speed_drop'] = features['entry_speed'] - features['min_speed']
            features['speed_gain'] = features['exit_speed'] - features['min_speed']
            
            # Speed variation pattern
            features['speed_variability'] = speed.std()
            
            # Find minimum speed position in corner
            min_speed_idx = speed.idxmin()
            min_speed_position = min_speed_idx / len(speed) if len(speed) > 0 else 0.5
            features['min_speed_position'] = min_speed_position  # 0.0 = entry, 1.0 = exit
            
        return features
    
    def _extract_steering_profile_features(self, corner_df: pd.DataFrame) -> Dict[str, float]:
        """Extract steering input pattern features"""
        features = {}
        
        if 'Physics_steer_angle' in corner_df.columns:
            steering = corner_df['Physics_steer_angle']
            abs_steering = abs(steering)
            
            # Steering smoothness
            steering_changes = abs(steering.diff()).fillna(0)
            features['steering_smoothness'] = 1.0 / (1.0 + steering_changes.mean())  # Higher is smoother
            
            # Peak steering position
            max_steering_idx = abs_steering.idxmax()
            max_steering_position = max_steering_idx / len(steering) if len(steering) > 0 else 0.5
            features['max_steering_position'] = max_steering_position
            
            # Steering aggressiveness
            features['steering_aggressiveness'] = steering_changes.max()
            
        return features
    
    def _extract_gforce_features(self, corner_df: pd.DataFrame) -> Dict[str, float]:
        """Extract G-force pattern features"""
        features = {}
        
        # Lateral G-force (cornering)
        if 'Physics_g_force_x' in corner_df.columns:
            g_lat = corner_df['Physics_g_force_x']
            features['max_lateral_g'] = abs(g_lat).max()
            features['avg_lateral_g'] = abs(g_lat).mean()
        
        # Longitudinal G-force (braking/acceleration)
        if 'Physics_g_force_y' in corner_df.columns:
            g_lon = corner_df['Physics_g_force_y']
            features['max_braking_g'] = abs(g_lon[g_lon < 0]).max() if any(g_lon < 0) else 0
            features['max_accel_g'] = g_lon[g_lon > 0].max() if any(g_lon > 0) else 0
        
        # Combined G-force magnitude
        if 'Physics_g_force_x' in corner_df.columns and 'Physics_g_force_y' in corner_df.columns:
            g_combined = np.sqrt(corner_df['Physics_g_force_x']**2 + corner_df['Physics_g_force_y']**2)
            features['max_combined_g'] = g_combined.max()
            features['avg_combined_g'] = g_combined.mean()
        
        return features
    
    def _extract_input_pattern_features(self, corner_df: pd.DataFrame) -> Dict[str, float]:
        """Extract braking and throttle input patterns"""
        features = {}
        
        # Braking patterns
        if 'Physics_brake' in corner_df.columns:
            brake = corner_df['Physics_brake']
            features['max_brake_input'] = brake.max()
            features['avg_brake_input'] = brake.mean()
            
            # Brake trail-off pattern
            brake_points = brake[brake > 0.1]
            if len(brake_points) > 0:
                features['brake_usage_ratio'] = len(brake_points) / len(corner_df)
            else:
                features['brake_usage_ratio'] = 0
        
        # Throttle patterns
        if 'Physics_gas' in corner_df.columns:
            throttle = corner_df['Physics_gas']
            features['max_throttle_input'] = throttle.max()
            features['avg_throttle_input'] = throttle.mean()
            
            # Find throttle application point
            throttle_points = throttle[throttle > 0.1]
            if len(throttle_points) > 0:
                first_throttle_idx = (throttle > 0.1).idxmax()
                throttle_position = first_throttle_idx / len(corner_df) if len(corner_df) > 0 else 0.5
                features['throttle_application_point'] = throttle_position
                features['throttle_usage_ratio'] = len(throttle_points) / len(corner_df)
            else:
                features['throttle_application_point'] = 1.0
                features['throttle_usage_ratio'] = 0
        
        return features
    
    def _extract_trajectory_features(self, corner_df: pd.DataFrame) -> Dict[str, float]:
        """Extract trajectory-related features"""
        features = {}
        
        # Track position variance (racing line consistency)
        if 'Graphics_normalized_car_position' in corner_df.columns:
            positions = corner_df['Graphics_normalized_car_position']
            features['trajectory_consistency'] = 1.0 / (1.0 + positions.std())  # Higher is more consistent
        
        return features
    
    def _extract_timing_features(self, corner_df: pd.DataFrame, corner_info: Dict[str, Any]) -> Dict[str, float]:
        """Extract timing and rhythm features"""
        features = {}
        
        # Phase timing ratios
        if 'phases' in corner_info:
            phases = corner_info['phases']
            total_duration = corner_info.get('total_duration_points', len(corner_df))
            
            for phase_name, phase_data in phases.items():
                duration = phase_data.get('duration_points', 0)
                ratio = duration / max(total_duration, 1)
                features[f'{phase_name}_duration_ratio'] = ratio
        
        return features


class CornerShapeClusterer:
    """Cluster similar corner shapes using unsupervised learning"""
    
    def __init__(self, models_directory: str = "models/corner_shapes"):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.clusterers = {
            'kmeans': None,
            'dbscan': None,
            'gaussian_mixture': None,
            'hierarchical': None
        }
        self.feature_names = []
        self.cluster_centers = {}
        self.cluster_labels = {}
        
    def fit_multiple_clusterers(self, features_df: pd.DataFrame, 
                               clustering_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fit multiple clustering algorithms and select the best one
        
        Args:
            features_df: DataFrame with corner shape features
            clustering_params: Parameters for clustering algorithms
            
        Returns:
            Dictionary with clustering results and best algorithm
        """
        if len(features_df) < 3:
            return {"error": "Need at least 3 corners for clustering"}
        
        # Prepare features
        feature_cols = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[feature_cols].fillna(0)
        self.feature_names = list(feature_cols)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        params = clustering_params or {}
        results = {}
        
        # K-Means Clustering
        try:
            n_clusters = params.get('n_clusters', min(8, max(3, len(X) // 5)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            
            kmeans_score = silhouette_score(X_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0
            
            self.clusterers['kmeans'] = kmeans
            self.cluster_labels['kmeans'] = kmeans_labels
            self.cluster_centers['kmeans'] = kmeans.cluster_centers_
            
            results['kmeans'] = {
                'n_clusters': n_clusters,
                'silhouette_score': kmeans_score,
                'labels': kmeans_labels,
                'algorithm': 'KMeans'
            }
        except Exception as e:
            print(f"[WARNING] K-Means clustering failed: {str(e)}")
        
        # DBSCAN Clustering
        try:
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', max(2, len(X) // 10))
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            
            # Only calculate score if we have valid clusters (not all -1)
            unique_labels = set(dbscan_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                dbscan_score = silhouette_score(X_scaled, dbscan_labels)
            else:
                dbscan_score = -1  # Invalid clustering
            
            self.clusterers['dbscan'] = dbscan
            self.cluster_labels['dbscan'] = dbscan_labels
            
            results['dbscan'] = {
                'n_clusters': len(unique_labels - {-1}),
                'silhouette_score': dbscan_score,
                'labels': dbscan_labels,
                'noise_points': sum(dbscan_labels == -1),
                'algorithm': 'DBSCAN'
            }
        except Exception as e:
            print(f"[WARNING] DBSCAN clustering failed: {str(e)}")
        
        # Gaussian Mixture Model
        try:
            n_components = params.get('n_components', min(6, max(2, len(X) // 6)))
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm_labels = gmm.fit_predict(X_scaled)
            
            gmm_score = silhouette_score(X_scaled, gmm_labels) if len(set(gmm_labels)) > 1 else 0
            
            self.clusterers['gaussian_mixture'] = gmm
            self.cluster_labels['gaussian_mixture'] = gmm_labels
            
            results['gaussian_mixture'] = {
                'n_clusters': n_components,
                'silhouette_score': gmm_score,
                'labels': gmm_labels,
                'algorithm': 'GaussianMixture',
                'bic_score': gmm.bic(X_scaled),
                'aic_score': gmm.aic(X_scaled)
            }
        except Exception as e:
            print(f"[WARNING] Gaussian Mixture clustering failed: {str(e)}")
        
        # Hierarchical Clustering
        try:
            n_clusters = params.get('n_clusters_hierarchical', min(7, max(3, len(X) // 4)))
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            hier_labels = hierarchical.fit_predict(X_scaled)
            
            hier_score = silhouette_score(X_scaled, hier_labels) if len(set(hier_labels)) > 1 else 0
            
            self.clusterers['hierarchical'] = hierarchical
            self.cluster_labels['hierarchical'] = hier_labels
            
            results['hierarchical'] = {
                'n_clusters': n_clusters,
                'silhouette_score': hier_score,
                'labels': hier_labels,
                'algorithm': 'AgglomerativeClustering'
            }
        except Exception as e:
            print(f"[WARNING] Hierarchical clustering failed: {str(e)}")
        
        # Select best clustering algorithm
        best_algorithm = self._select_best_clustering(results)
        
        return {
            'results': results,
            'best_algorithm': best_algorithm,
            'best_clustering': results.get(best_algorithm, {}),
            'feature_names': self.feature_names,
            'n_corners_analyzed': len(X)
        }
    
    def _select_best_clustering(self, results: Dict[str, Any]) -> str:
        """Select the best clustering algorithm based on multiple criteria"""
        if not results:
            return 'kmeans'  # Default fallback
        
        # Score algorithms based on multiple criteria
        algorithm_scores = {}
        
        for algo_name, result in results.items():
            score = 0
            
            # Silhouette score (primary metric)
            silhouette = result.get('silhouette_score', 0)
            if silhouette > 0:
                score += silhouette * 100
            
            # Number of clusters (prefer reasonable number)
            n_clusters = result.get('n_clusters', 0)
            if 3 <= n_clusters <= 8:
                score += 20
            elif n_clusters > 0:
                score += 10
            
            # Algorithm-specific bonuses
            if algo_name == 'dbscan':
                noise_points = result.get('noise_points', 0)
                total_points = len(self.cluster_labels.get(algo_name, []))
                if total_points > 0:
                    noise_ratio = noise_points / total_points
                    if noise_ratio < 0.1:  # Less than 10% noise is good
                        score += 15
                    elif noise_ratio < 0.3:  # Less than 30% noise is acceptable
                        score += 5
            
            algorithm_scores[algo_name] = score
        
        # Return algorithm with highest score
        best_algo = max(algorithm_scores, key=algorithm_scores.get, default='kmeans')
        print(f"[INFO] Selected {best_algo} as best clustering algorithm with score {algorithm_scores[best_algo]:.2f}")
        
        return best_algo
    
    def predict_corner_cluster(self, corner_features: Dict[str, float], algorithm: str = 'best') -> Dict[str, Any]:
        """
        Predict which cluster a corner belongs to
        
        Args:
            corner_features: Dictionary of corner shape features
            algorithm: Which clustering algorithm to use ('best', 'kmeans', etc.)
            
        Returns:
            Dictionary with cluster prediction and confidence
        """
        if algorithm == 'best':
            # Find the algorithm with highest silhouette score
            best_algo = 'kmeans'
            best_score = -1
            for algo_name in self.clusterers:
                if algo_name in self.cluster_labels:
                    labels = self.cluster_labels[algo_name]
                    if len(set(labels)) > 1:
                        try:
                            # This would need the original scaled features, so we'll use a simpler approach
                            best_algo = algo_name
                            break
                        except:
                            continue
            algorithm = best_algo
        
        clusterer = self.clusterers.get(algorithm)
        if clusterer is None:
            return {"error": f"Clusterer '{algorithm}' not available"}
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(corner_features.get(feature_name, 0))
        
        feature_array = np.array([feature_vector])
        feature_scaled = self.scaler.transform(feature_array)
        
        # Predict cluster
        try:
            if algorithm == 'gaussian_mixture':
                cluster_id = clusterer.predict(feature_scaled)[0]
                probabilities = clusterer.predict_proba(feature_scaled)[0]
                confidence = probabilities.max()
            else:
                cluster_id = clusterer.predict(feature_scaled)[0]
                confidence = 0.8  # Default confidence for non-probabilistic methods
            
            return {
                'cluster_id': int(cluster_id),
                'algorithm': algorithm,
                'confidence': float(confidence),
                'features_used': self.feature_names
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def save_clustering_models(self, model_name: str):
        """Save trained clustering models to disk"""
        model_data = {
            'clusterers': self.clusterers,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'cluster_centers': self.cluster_centers,
            'cluster_labels': self.cluster_labels,
            'created_at': datetime.now().isoformat()
        }
        
        model_path = self.models_directory / f"{model_name}_clustering.pkl"
        joblib.dump(model_data, model_path)
        print(f"[INFO] Saved clustering models to {model_path}")
    
    def load_clustering_models(self, model_name: str) -> bool:
        """Load trained clustering models from disk"""
        model_path = self.models_directory / f"{model_name}_clustering.pkl"
        
        if not model_path.exists():
            print(f"[WARNING] Model file {model_path} not found")
            return False
        
        try:
            model_data = joblib.load(model_path)
            
            self.clusterers = model_data['clusterers']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.cluster_centers = model_data['cluster_centers']
            self.cluster_labels = model_data['cluster_labels']
            
            print(f"[INFO] Loaded clustering models from {model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load clustering models: {str(e)}")
            return False


class CornerShapeUnsupervisedService:
    """
    Main service for unsupervised corner shape learning
    
    This service:
    1. Retrieves racing session data from backend
    2. Identifies corners using TrackCorneringAnalyzer
    3. Extracts shape features for each corner
    4. Applies unsupervised learning to discover corner types
    5. Creates shape profiles and provides analysis
    """
    
    def __init__(self, models_directory: str = "models/corner_shapes"):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = CornerShapeFeatureExtractor()
        self.clusterer = CornerShapeClusterer(str(models_directory))
        self.corner_analyzer = TrackCorneringAnalyzer()
        
        self.backend_service = backend_service
        
        # Cache for processed corners
        self.corner_cache = {}
    
    async def learn_corner_shapes(self, trackName: str, 
                                 clustering_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Learn corner shapes for a specific track using unsupervised learning
        
        Args:
            trackName: Name of the track to analyze
            clustering_params: Parameters for clustering algorithms
            
        Returns:
            Dictionary with learning results and corner shape clusters
        """
        print(f"[INFO] Starting unsupervised corner shape learning for track: {trackName}")
        
        # Step 1: Retrieve racing sessions data from backend
        try:
            sessions_data = await self.backend_service.get_all_racing_sessions(trackName)
        except Exception as e:
            return {"error": f"Failed to retrieve racing sessions: {str(e)}"}
        
        if not sessions_data.get("success", False):
            return {"error": f"Backend error: {sessions_data.get('error', 'Unknown error')}"}
        
        sessions = sessions_data.get("sessions", [])
        if not sessions:
            return {"error": f"No racing sessions found for track: {trackName}"}
        
        print(f"[INFO] Retrieved {len(sessions)} sessions for analysis")
        
        # Step 2: Process each session to extract corners
        all_corners_data = []
        all_corners_features = []
        
        for i, session in enumerate(sessions):
            session_telemetry = session.get("data", [])
            if not session_telemetry:
                continue
            
            print(f"[INFO] Processing session {i+1}/{len(sessions)} with {len(session_telemetry)} records")
            
            # Convert to DataFrame
            df = pd.DataFrame(session_telemetry)
            
            # Process telemetry data
            feature_processor = FeatureProcessor(df)
            processed_df = feature_processor.general_cleaning_for_analysis()
            
            # Filter top performance laps (optional, but recommended for learning)
            try:
                filtered_df = self.corner_analyzer.filter_top_performance_laps(processed_df)
            except:
                filtered_df = processed_df
            
            # Identify corners
            try:
                corners_df = self.corner_analyzer.identify_cornering_phases(filtered_df)
                corner_analysis = self.corner_analyzer.get_cornering_analysis_summary(corners_df)
                
                # Extract corner features
                session_corners = self._extract_session_corners(corners_df, corner_analysis, session.get("sessionId", f"session_{i}"))
                all_corners_data.extend(session_corners)
                
                print(f"[INFO] Extracted {len(session_corners)} corners from session {i+1}")
                
            except Exception as e:
                print(f"[WARNING] Failed to process session {i+1}: {str(e)}")
                continue
        
        if not all_corners_data:
            return {"error": "No corners could be extracted from the sessions"}
        
        print(f"[INFO] Total corners extracted: {len(all_corners_data)}")
        
        # Step 3: Extract features for all corners
        for corner_data in all_corners_data:
            corner_df = corner_data['telemetry_data']
            corner_info = corner_data['corner_info']
            
            features = self.feature_extractor.extract_corner_shape_features(corner_df, corner_info)
            features['session_id'] = corner_data['session_id']
            features['corner_id'] = corner_data['corner_id']
            all_corners_features.append(features)
        
        # Convert to DataFrame for clustering
        features_df = pd.DataFrame(all_corners_features)
        
        # Step 4: Apply clustering algorithms
        clustering_results = self.clusterer.fit_multiple_clusterers(features_df, clustering_params)
        
        if 'error' in clustering_results:
            return clustering_results
        
        # Step 5: Analyze cluster characteristics
        cluster_analysis = self._analyze_corner_clusters(features_df, clustering_results)
        
        # Step 6: Save models
        model_name = f"{trackName}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.clusterer.save_clustering_models(model_name)
        
        # Step 7: Generate summary
        learning_summary = {
            'track_name': trackName,
            'model_name': model_name,
            'total_sessions_analyzed': len(sessions),
            'total_corners_found': len(all_corners_data),
            'clustering_results': clustering_results,
            'cluster_analysis': cluster_analysis,
            'feature_names': features_df.select_dtypes(include=[np.number]).columns.tolist(),
            'learning_timestamp': datetime.now().isoformat()
        }
        
        # Cache results
        self.corner_cache[trackName] = learning_summary
        
        print(f"[INFO] Completed unsupervised corner shape learning for {trackName}")
        
        return learning_summary
    
    def _extract_session_corners(self, corners_df: pd.DataFrame, 
                                corner_analysis: Dict[str, Any], 
                                session_id: str) -> List[Dict[str, Any]]:
        """Extract individual corner data from a session"""
        corners_data = []
        
        if 'corner_details' not in corner_analysis:
            return corners_data
        
        corner_details = corner_analysis['corner_details']
        
        for corner_name, corner_info in corner_details.items():
            # Extract corner telemetry data
            corner_id = int(corner_name.split('_')[1]) if '_' in corner_name else 0
            corner_mask = corners_df['corner_id'] == corner_id
            corner_telemetry = corners_df[corner_mask].copy()
            
            if len(corner_telemetry) < 5:  # Skip corners with insufficient data
                continue
            
            corners_data.append({
                'session_id': session_id,
                'corner_id': corner_id,
                'corner_name': corner_name,
                'corner_info': corner_info,
                'telemetry_data': corner_telemetry
            })
        
        return corners_data
    
    def _analyze_corner_clusters(self, features_df: pd.DataFrame, 
                                clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of each corner cluster"""
        best_algorithm = clustering_results['best_algorithm']
        best_results = clustering_results['best_clustering']
        
        if 'labels' not in best_results:
            return {"error": "No cluster labels found"}
        
        labels = best_results['labels']
        unique_clusters = sorted(set(labels))
        
        cluster_characteristics = {}
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_mask = labels == cluster_id
            cluster_data = features_df[cluster_mask]
            
            # Calculate cluster statistics
            cluster_stats = {}
            for feature in numeric_features:
                if feature not in ['session_id', 'corner_id']:
                    cluster_stats[feature] = {
                        'mean': float(cluster_data[feature].mean()),
                        'std': float(cluster_data[feature].std()),
                        'min': float(cluster_data[feature].min()),
                        'max': float(cluster_data[feature].max())
                    }
            
            # Identify dominant characteristics
            dominant_features = self._identify_dominant_features(cluster_stats)
            
            # Generate cluster description
            description = self._generate_cluster_description(cluster_stats, dominant_features)
            
            cluster_characteristics[f"cluster_{cluster_id}"] = {
                'cluster_id': cluster_id,
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(features_df) * 100),
                'statistics': cluster_stats,
                'dominant_features': dominant_features,
                'description': description
            }
        
        return {
            'algorithm_used': best_algorithm,
            'total_clusters': len(unique_clusters),
            'cluster_characteristics': cluster_characteristics,
            'silhouette_score': best_results.get('silhouette_score', 0),
        }
    
    def _identify_dominant_features(self, cluster_stats: Dict[str, Any]) -> List[str]:
        """Identify the most characteristic features of a cluster"""
        # This is a simplified approach - in practice, you might want to compare
        # against the global mean or use more sophisticated feature importance methods
        dominant = []
        
        for feature, stats in cluster_stats.items():
            mean_val = stats['mean']
            std_val = stats['std']
            
            # Identify features with high values or high variability
            if abs(mean_val) > 0.5 or std_val > 0.3:  # Thresholds can be adjusted
                dominant.append(feature)
        
        return dominant[:5]  # Return top 5 dominant features
    
    def _generate_cluster_description(self, cluster_stats: Dict[str, Any], 
                                    dominant_features: List[str]) -> str:
        """Generate a human-readable description of the cluster"""
        descriptions = []
        
        # Check for corner characteristics
        if 'steering_direction' in cluster_stats:
            direction = cluster_stats['steering_direction']['mean']
            if direction < -0.5:
                descriptions.append("left-hand corners")
            elif direction > 0.5:
                descriptions.append("right-hand corners")
        
        if 'max_steering_angle' in cluster_stats:
            angle = cluster_stats['max_steering_angle']['mean']
            if angle > 300:  # Assuming steering angle in degrees * 10
                descriptions.append("tight corners")
            elif angle < 100:
                descriptions.append("gentle corners")
        
        if 'speed_drop' in cluster_stats:
            speed_drop = cluster_stats['speed_drop']['mean']
            if speed_drop > 50:
                descriptions.append("heavy braking zones")
            elif speed_drop < 20:
                descriptions.append("high-speed corners")
        
        if not descriptions:
            descriptions.append("mixed corner characteristics")
        
        return ", ".join(descriptions)
    
    async def predict_corner_shape(self, trackName: str, 
                                 current_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the shape type of a current corner based on learned patterns
        
        Args:
            trackName: Track name for the model
            current_telemetry: Current corner telemetry data
            
        Returns:
            Dictionary with corner shape prediction
        """
        # Check if we have a cached model
        if trackName not in self.corner_cache:
            return {"error": f"No corner shape model found for track: {trackName}"}
        
        # Extract features from current telemetry
        try:
            df = pd.DataFrame([current_telemetry])
            corner_info = {'corner_start_position': 0.0, 'corner_end_position': 0.1}  # Dummy info
            
            features = self.feature_extractor.extract_corner_shape_features(df, corner_info)
            
            # Predict cluster
            prediction = self.clusterer.predict_corner_cluster(features)
            
            if 'error' in prediction:
                return prediction
            
            # Get cluster characteristics
            cached_data = self.corner_cache[trackName]
            cluster_analysis = cached_data.get('cluster_analysis', {})
            cluster_characteristics = cluster_analysis.get('cluster_characteristics', {})
            
            cluster_id = prediction['cluster_id']
            cluster_info = cluster_characteristics.get(f"cluster_{cluster_id}", {})
            
            return {
                'track_name': trackName,
                'predicted_cluster': cluster_id,
                'confidence': prediction['confidence'],
                'cluster_description': cluster_info.get('description', 'Unknown corner type'),
                'cluster_size': cluster_info.get('size', 0),
                'cluster_percentage': cluster_info.get('percentage', 0),
                'features_analyzed': prediction['features_used']
            }
            
        except Exception as e:
            return {"error": f"Failed to predict corner shape: {str(e)}"}
    
    def get_corner_shape_summary(self, trackName: str) -> Dict[str, Any]:
        """
        Get a summary of learned corner shapes for a track
        
        Args:
            trackName: Track name
            
        Returns:
            Dictionary with corner shape learning summary
        """
        if trackName not in self.corner_cache:
            return {"error": f"No corner shape data found for track: {trackName}"}
        
        return self.corner_cache[trackName]
    
    def clear_cache(self, trackName: Optional[str] = None):
        """Clear corner shape cache"""
        if trackName:
            self.corner_cache.pop(trackName, None)
            print(f"[INFO] Cleared corner shape cache for {trackName}")
        else:
            self.corner_cache.clear()
            print("[INFO] Cleared all corner shape cache")


# Create global instance
corner_shape_service = CornerShapeUnsupervisedService()
