"""
Multi-Algorithm Machine Learning Configuration for Racing Telemetry
Defines different algorithms optimized for specific prediction tasks using River for online learning
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np

# River imports for online learning
from river import linear_model, tree, ensemble, neighbors, naive_bayes, neural_net
from river import preprocessing, compose, metrics
from river.base import Regressor, Classifier

class AlgorithmConfiguration:
    """
    Configuration class for different ML algorithms optimized for racing telemetry predictions using River
    River provides online learning algorithms that are perfect for streaming telemetry data
    """
    
    def __init__(self):
        # Define algorithm configurations for different prediction types
        self.model_configs = {
            "lap_time_prediction": {
                "primary": "adaptive_random_forest",
                "alternatives": ["hoeffding_tree", "linear_regression", "sgd_regressor"],
                "description": "Predicts lap times based on telemetry data"
            },
            "sector_time_prediction": {
                "primary": "hoeffding_tree", 
                "alternatives": ["adaptive_random_forest", "linear_regression"],
                "description": "Predicts sector times for track analysis"
            },
            "performance_classification": {
                "primary": "adaptive_random_forest_classifier",
                "alternatives": ["hoeffding_tree_classifier", "naive_bayes_classifier"],
                "description": "Classifies performance levels (fast/medium/slow)"
            },
            "setup_recommendation": {
                "primary": "hoeffding_tree_classifier",
                "alternatives": ["adaptive_random_forest_classifier", "naive_bayes_classifier"],
                "description": "Recommends car setup configurations"
            },
            "tire_strategy": {
                "primary": "hoeffding_tree_classifier",
                "alternatives": ["adaptive_random_forest_classifier", "naive_bayes_classifier"],
                "description": "Optimal tire strategy recommendations"
            },
            "fuel_consumption": {
                "primary": "linear_regression",
                "alternatives": ["sgd_regressor", "hoeffding_tree"],
                "description": "Predicts fuel consumption patterns"
            },
            "brake_performance": {
                "primary": "sgd_regressor",
                "alternatives": ["linear_regression", "hoeffding_tree"],
                "description": "Predicts brake performance and wear"
            },
            "overtaking_opportunity": {
                "primary": "adaptive_random_forest_classifier",
                "alternatives": ["hoeffding_tree_classifier", "naive_bayes_classifier"],
                "description": "Identifies overtaking opportunities"
            },
            "racing_line_optimization": {
                "primary": "sgd_regressor",
                "alternatives": ["linear_regression", "hoeffding_tree"],
                "description": "Optimizes racing line for track sections"
            },
            "weather_adaptation": {
                "primary": "adaptive_random_forest",
                "alternatives": ["hoeffding_tree", "sgd_regressor"],
                "description": "Adapts driving style for weather conditions"
            },
            "consistency_analysis": {
                "primary": "linear_regression",
                "alternatives": ["sgd_regressor", "hoeffding_tree"],
                "description": "Analyzes driving consistency patterns"
            },
            "damage_prediction": {
                "primary": "adaptive_random_forest_classifier",
                "alternatives": ["hoeffding_tree_classifier", "naive_bayes_classifier"],
                "description": "Predicts potential car damage"
            }
        }
        
        # Algorithm implementations using River
        self.algorithms = {
            # Regression algorithms
            "linear_regression": {
                "class": linear_model.LinearRegression,
                "params": {},
                "type": "regression",
                "incremental": True,
                "preprocessing": True
            },
            "sgd_regressor": {
                "class": linear_model.PARegressor,
                "params": {"C": 0.01},
                "type": "regression",
                "incremental": True,
                "preprocessing": True
            },
            "pa_regressor": {
                "class": linear_model.PARegressor,
                "params": {"C": 1.0},
                "type": "regression",
                "incremental": True,
                "preprocessing": True
            },
            "hoeffding_tree": {
                "class": tree.HoeffdingTreeRegressor,
                "params": {"max_depth": 10, "min_samples_leaf": 5},
                "type": "regression",
                "incremental": True,
                "preprocessing": False
            },
            "adaptive_random_forest": {
                "class": ensemble.BaggingRegressor,
                "params": {"n_models": 10},
                "type": "regression",
                "incremental": True,
                "preprocessing": False
            },
            "knn_regressor": {
                "class": neighbors.KNNRegressor,
                "params": {"n_neighbors": 5, "window_size": 1000},
                "type": "regression",
                "incremental": True,
                "preprocessing": True
            },
            
            # Classification algorithms
            "logistic_regression": {
                "class": linear_model.LogisticRegression,
                "params": {},
                "type": "classification",
                "incremental": True,
                "preprocessing": True
            },
            "pa_classifier": {
                "class": linear_model.PAClassifier,
                "params": {"C": 1.0},
                "type": "classification",
                "incremental": True,
                "preprocessing": True
            },
            "naive_bayes_classifier": {
                "class": naive_bayes.GaussianNB,
                "params": {},
                "type": "classification",
                "incremental": True,
                "preprocessing": False
            },
            "hoeffding_tree_classifier": {
                "class": tree.HoeffdingTreeClassifier,
                "params": {"max_depth": 10, "min_samples_leaf": 5},
                "type": "classification",
                "incremental": True,
                "preprocessing": False
            },
            "adaptive_random_forest_classifier": {
                "class": ensemble.BaggingClassifier,
                "params": {"n_models": 10},
                "type": "classification",
                "incremental": True,
                "preprocessing": False
            },
            "knn_classifier": {
                "class": neighbors.KNNClassifier,
                "params": {"n_neighbors": 5, "window_size": 1000},
                "type": "classification",
                "incremental": True,
                "preprocessing": True
            }
        }
    
    def get_algorithm_for_task(self, model_type: str, preferred_algorithm: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the optimal algorithm configuration for a specific prediction task
        
        Args:
            model_type: The type of prediction task
            preferred_algorithm: Override with a specific algorithm
        
        Returns:
            Algorithm configuration dictionary
        """
        if model_type not in self.model_configs:
            # Default to gradient boosting if model type is unknown
            algorithm_name = preferred_algorithm or "gradient_boosting"
        else:
            config = self.model_configs[model_type]
            if preferred_algorithm and preferred_algorithm in config["alternatives"]:
                # Use the preferred algorithm if it's an alternative
                algorithm_name = preferred_algorithm
            else:
                # Use the primary algorithm if no preferred alternative is specified
                algorithm_name = config["primary"]
        
        if algorithm_name not in self.algorithms:
            # Default to gradient boosting if algorithm is unknown
            algorithm_name = "gradient_boosting"  # Fallback
        
        algorithm_config = self.algorithms[algorithm_name].copy()
        algorithm_config["name"] = algorithm_name
        algorithm_config["task_description"] = self.model_configs.get(model_type, {}).get("description", "Unknown task")
        
        return algorithm_config
    
    def create_algorithm_instance(self, algorithm_config: Dict[str, Any]) -> Any:
        """
        Create an instance of the specified algorithm with optional preprocessing pipeline
        
        Args:
            algorithm_config: Algorithm configuration from get_algorithm_for_task
        
        Returns:
            Instantiated algorithm, potentially wrapped in a preprocessing pipeline
        """
        algorithm_class = algorithm_config["class"]
        params = algorithm_config["params"]
        needs_preprocessing = algorithm_config.get("preprocessing", False)
        algorithm_name = algorithm_config.get("name", "")
        
        # Handle special cases for ensemble algorithms that need base models
        if algorithm_name == "adaptive_random_forest":
            # BaggingRegressor needs a base model
            base_model = tree.HoeffdingTreeRegressor()
            algorithm = algorithm_class(model=base_model, **params)
        elif algorithm_name == "adaptive_random_forest_classifier":
            # BaggingClassifier needs a base model  
            base_model = tree.HoeffdingTreeClassifier()
            algorithm = algorithm_class(model=base_model, **params)
        else:
            # Create the base algorithm normally
            algorithm = algorithm_class(**params)
        
        # For River, we can create preprocessing pipelines
        if needs_preprocessing:
            # Create a preprocessing pipeline for algorithms that benefit from it
            pipeline = compose.Pipeline(
                preprocessing.StandardScaler(),
                algorithm
            )
            return pipeline
        
        return algorithm
    
    def create_preprocessing_pipeline(self, algorithm_name: str) -> Optional[Any]:
        """
        Create a preprocessing pipeline for the algorithm if needed
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Preprocessing pipeline or None
        """
        if algorithm_name in ["linear_regression", "sgd_regressor", "logistic_regression", "pa_regressor", "pa_classifier"]:
            return compose.Pipeline(
                preprocessing.StandardScaler()
            )
        return None
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of all supported prediction tasks"""
        return list(self.model_configs.keys())
    
    def get_task_description(self, model_type: str) -> str:
        """Get description of a prediction task"""
        return self.model_configs.get(model_type, {}).get("description", "Unknown task")
    
    def get_algorithm_alternatives(self, model_type: str) -> List[str]:
        """Get alternative algorithms for a task"""
        config = self.model_configs.get(model_type, {})
        alternatives = config.get("alternatives", [])
        primary = config.get("primary", "")
        return [primary] + alternatives if primary else alternatives
    
    def supports_incremental_learning(self, algorithm_name: str) -> bool:
        """Check if algorithm supports incremental learning"""
        return self.algorithms.get(algorithm_name, {}).get("incremental", False)
    
    def get_algorithm_type(self, algorithm_name: str) -> str:
        """Get algorithm type (regression/classification)"""
        return self.algorithms.get(algorithm_name, {}).get("type", "regression")
    
    def optimize_hyperparameters(self, algorithm_name: str, data_size: int, n_features: int) -> Dict[str, Any]:
        """
        Get optimized hyperparameters based on data characteristics for River algorithms
        
        Args:
            algorithm_name: Name of the algorithm
            data_size: Number of samples (for online learning, this is less relevant)
            n_features: Number of features
        
        Returns:
            Optimized parameters dictionary
        """
        base_params = self.algorithms.get(algorithm_name, {}).get("params", {}).copy()
        
        # Optimize based on number of features and expected data characteristics
        if algorithm_name in ["adaptive_random_forest", "adaptive_random_forest_classifier"]:
            # Adjust number of models based on complexity
            if n_features < 10:
                base_params["n_models"] = 5
            elif n_features > 50:
                base_params["n_models"] = 15
            # Adjust depth based on feature count
            base_params["max_depth"] = min(20, max(5, n_features // 2))
                
        elif algorithm_name in ["hoeffding_tree", "hoeffding_tree_classifier"]:
            # Adjust tree depth and minimum samples
            if n_features < 10:
                base_params["max_depth"] = 5
                base_params["min_samples_leaf"] = 10
            elif n_features > 50:
                base_params["max_depth"] = 15
                base_params["min_samples_leaf"] = 2
                
        elif algorithm_name in ["sgd_regressor"]:
            # Adjust learning rate based on feature complexity
            if n_features < 10:
                base_params["learning_rate"] = 0.1
            elif n_features > 50:
                base_params["learning_rate"] = 0.001
                
        elif algorithm_name in ["knn_regressor", "knn_classifier"]:
            # Adjust k and window size
            base_params["n_neighbors"] = min(10, max(3, n_features // 5))
            base_params["window_size"] = max(500, min(5000, data_size // 10)) if data_size > 0 else 1000
                
        return base_params
    
    def get_feature_importance_method(self, algorithm_name: str) -> Optional[str]:
        """
        Get the method to extract feature importance for the algorithm
        Note: River has different approaches to feature importance
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Method name or approach description
        """
        importance_methods = {
            "hoeffding_tree": "tree_feature_importance",
            "hoeffding_tree_classifier": "tree_feature_importance", 
            "adaptive_random_forest": "ensemble_feature_importance",
            "adaptive_random_forest_classifier": "ensemble_feature_importance",
            "linear_regression": "coefficient_based",
            "logistic_regression": "coefficient_based",
            "sgd_regressor": "coefficient_based",
            "pa_regressor": "coefficient_based",
            "pa_classifier": "coefficient_based"
        }
        
        return importance_methods.get(algorithm_name)
    
    def extract_feature_importance(self, model: Any, feature_names: List[str], algorithm_name: str) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from a trained River model
        
        Args:
            model: Trained River model
            feature_names: List of feature names
            algorithm_name: Name of the algorithm
        
        Returns:
            Dictionary of feature importances or None if not supported
        """
        try:
            importance_method = self.get_feature_importance_method(algorithm_name)
            
            if importance_method == "coefficient_based":
                # For linear models, use coefficients as importance
                if hasattr(model, 'weights') and model.weights:
                    importances = {}
                    for feature_name in feature_names:
                        weight = model.weights.get(feature_name, 0.0)
                        importances[feature_name] = abs(weight)
                    return importances
                    
            elif importance_method == "tree_feature_importance":
                # For tree models, we would need to implement custom importance extraction
                # This is more complex in River as it doesn't have built-in feature importance
                return None
                
            elif importance_method == "ensemble_feature_importance":
                # For ensemble models, aggregate importance from individual models
                # This would require custom implementation
                return None
                
            return None
        except Exception:
            return None
    
    def get_prediction_confidence_method(self, algorithm_name: str) -> Optional[str]:
        """
        Get method to calculate prediction confidence for River models
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Method name or None if not supported
        """
        confidence_methods = {
            "logistic_regression": "predict_proba_one",
            "pa_classifier": "predict_proba_one",
            "naive_bayes_classifier": "predict_proba_one",
            "hoeffding_tree_classifier": "predict_proba_one",
            "adaptive_random_forest_classifier": "predict_proba_one",
            "knn_classifier": "predict_proba_one"
        }
        
        return confidence_methods.get(algorithm_name)
    
    def get_recommended_features(self, model_type: str) -> List[str]:
        """
        Get recommended features for specific prediction tasks
        
        Args:
            model_type: Type of prediction task
        
        Returns:
            List of recommended feature categories (now returns actual feature names)
        """
        # Import here to avoid circular imports
        from app.models.telemetry_models import TelemetryFeatures
        
        return TelemetryFeatures.get_features_for_model_type(model_type)
    
    def is_online_learning_algorithm(self, algorithm_name: str) -> bool:
        """
        Check if the algorithm supports online/incremental learning
        All River algorithms support online learning by design
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            True (all River algorithms support online learning)
        """
        return algorithm_name in self.algorithms
    
    def get_learning_method(self, algorithm_name: str) -> str:
        """
        Get the learning method for the algorithm
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Learning method ('learn_one' for River algorithms)
        """
        if algorithm_name in self.algorithms:
            return "learn_one"
        return "unknown"
    
    def get_prediction_method(self, algorithm_name: str) -> str:
        """
        Get the prediction method for the algorithm
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Prediction method name
        """
        algorithm_type = self.get_algorithm_type(algorithm_name)
        
        if algorithm_type == "classification":
            return "predict_proba_one"
        else:
            return "predict_one"
    
    def create_metrics_tracker(self, algorithm_type: str) -> Dict[str, Any]:
        """
        Create appropriate metrics tracker for the algorithm type
        
        Args:
            algorithm_type: Type of algorithm ('regression' or 'classification')
        
        Returns:
            River metrics object
        """
        if algorithm_type == "regression":
            return {"MAE": metrics.MAE(), "RMSE": metrics.RMSE(), "R2": metrics.R2()}
        else:
            return { "Accuracy": metrics.Accuracy() ,"Precision": metrics.Precision() , "Recall": metrics.Recall() , "F1": metrics.F1()}
    
    def get_algorithm_description(self, algorithm_name: str) -> str:
        """
        Get a detailed description of the algorithm
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Algorithm description
        """
        descriptions = {
            "linear_regression": "Online linear regression using normal equations, suitable for linear relationships",
            "sgd_regressor": "Passive-Aggressive regressor for online learning, robust to outliers and suitable for noisy data",
            "pa_regressor": "Passive-Aggressive regressor, robust to outliers and suitable for noisy data",
            "hoeffding_tree": "Online decision tree using Hoeffding bounds, adapts to concept drift",
            "adaptive_random_forest": "Ensemble of online trees that adapts to concept drift, excellent for complex patterns",
            "knn_regressor": "K-Nearest Neighbors with sliding window, good for local patterns",
            "logistic_regression": "Online logistic regression for binary/multi-class classification",
            "pa_classifier": "Passive-Aggressive classifier, robust and suitable for real-time classification",
            "naive_bayes_classifier": "Gaussian Naive Bayes, fast and effective for independent features",
            "hoeffding_tree_classifier": "Online decision tree classifier with concept drift adaptation",
            "adaptive_random_forest_classifier": "Ensemble classifier that adapts to changing data patterns",
            "knn_classifier": "K-Nearest Neighbors classifier with sliding window for pattern recognition"
        }
        
        return descriptions.get(algorithm_name, "Online machine learning algorithm")
    
    def get_algorithm_strengths(self, algorithm_name: str) -> List[str]:
        """
        Get the strengths of a specific algorithm
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            List of algorithm strengths
        """
        strengths = {
            "linear_regression": ["Fast training", "Interpretable", "Low memory usage", "Good for linear relationships"],
            "sgd_regressor": ["Robust to outliers", "Fast convergence", "Good for adversarial settings", "Online learning"],
            "pa_regressor": ["Robust to outliers", "Fast convergence", "Good for adversarial settings"],
            "hoeffding_tree": ["Handles concept drift", "Interpretable", "Fast predictions", "Memory efficient"],
            "adaptive_random_forest": ["Excellent accuracy", "Handles concept drift", "Robust to noise", "Feature importance"],
            "knn_regressor": ["No assumptions about data", "Adapts to local patterns", "Simple to understand"],
            "logistic_regression": ["Probabilistic outputs", "Fast training", "Interpretable", "Good baseline"],
            "pa_classifier": ["Robust classification", "Fast updates", "Good for online settings"],
            "naive_bayes_classifier": ["Very fast", "Handles missing values", "Good with small data"],
            "hoeffding_tree_classifier": ["Interpretable", "Handles concept drift", "Fast predictions"],
            "adaptive_random_forest_classifier": ["High accuracy", "Concept drift adaptation", "Robust ensemble"],
            "knn_classifier": ["Non-parametric", "Local decision boundaries", "Intuitive"]
        }
        
        return strengths.get(algorithm_name, ["Online learning capable"])
