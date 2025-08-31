"""
Multi-Algorithm Machine Learning Configuration for Racing Telemetry
Defines different algorithms optimized for specific prediction tasks
"""

from typing import Dict, Any, List, Optional, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

class AlgorithmConfiguration:
    """
    Configuration class for different ML algorithms optimized for racing telemetry predictions
    Only these scikit-learn algorithms have partial_fit() method for incremental updates:
    """
    
    def __init__(self):
        # Define algorithm configurations for different prediction types
        self.algorithm_configs = {
            "lap_time_prediction": {
                "primary": "gradient_boosting",
                "alternatives": ["random_forest", "neural_network", "svr"],
                "description": "Predicts lap times based on telemetry data"
            },
            "sector_time_prediction": {
                "primary": "random_forest", 
                "alternatives": ["gradient_boosting", "extra_trees"],
                "description": "Predicts sector times for track analysis"
            },
            "performance_classification": {
                "primary": "random_forest_classifier",
                "alternatives": ["gradient_boosting_classifier", "neural_network_classifier"],
                "description": "Classifies performance levels (fast/medium/slow)"
            },
            "setup_recommendation": {
                "primary": "gradient_boosting_classifier",
                "alternatives": ["random_forest_classifier", "svm_classifier"],
                "description": "Recommends car setup configurations"
            },
            "tire_strategy": {
                "primary": "decision_tree_classifier",
                "alternatives": ["random_forest_classifier", "naive_bayes"],
                "description": "Optimal tire strategy recommendations"
            },
            "fuel_consumption": {
                "primary": "linear_regression",
                "alternatives": ["ridge", "gradient_boosting"],
                "description": "Predicts fuel consumption patterns"
            },
            "brake_performance": {
                "primary": "svr",
                "alternatives": ["neural_network", "gradient_boosting"],
                "description": "Predicts brake performance and wear"
            },
            "overtaking_opportunity": {
                "primary": "gradient_boosting_classifier",
                "alternatives": ["random_forest_classifier", "neural_network_classifier"],
                "description": "Identifies overtaking opportunities"
            },
            "racing_line_optimization": {
                "primary": "neural_network",
                "alternatives": ["svr", "gradient_boosting"],
                "description": "Optimizes racing line for track sections"
            },
            "weather_adaptation": {
                "primary": "random_forest",
                "alternatives": ["gradient_boosting", "neural_network"],
                "description": "Adapts driving style for weather conditions"
            },
            "consistency_analysis": {
                "primary": "ridge",
                "alternatives": ["linear_regression", "elastic_net"],
                "description": "Analyzes driving consistency patterns"
            },
            "damage_prediction": {
                "primary": "gradient_boosting_classifier",
                "alternatives": ["random_forest_classifier", "svm_classifier"],
                "description": "Predicts potential car damage"
            }
        }
        
        # Algorithm implementations
        self.algorithms = {
            # Regression algorithms
            "linear_regression": {
                "class": LinearRegression,
                "params": {},
                "type": "regression",
                "incremental": False
            },
            "ridge": {
                "class": Ridge,
                "params": {"alpha": 1.0, "random_state": 42},
                "type": "regression", 
                "incremental": False
            },
            "lasso": {
                "class": Lasso,
                "params": {"alpha": 1.0, "random_state": 42},
                "type": "regression",
                "incremental": False
            },
            "elastic_net": {
                "class": ElasticNet,
                "params": {"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42},
                "type": "regression",
                "incremental": False
            },
            "random_forest": {
                "class": RandomForestRegressor,
                "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                "type": "regression",
                "incremental": False
            },
            "gradient_boosting": {
                "class": GradientBoostingRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                "type": "regression",
                "incremental": False
            },
            "extra_trees": {
                "class": ExtraTreesRegressor,
                "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                "type": "regression",
                "incremental": False
            },
            "svr": {
                "class": SVR,
                "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                "type": "regression",
                "incremental": False
            },
            "neural_network": {
                "class": MLPRegressor,
                "params": {
                    "hidden_layer_sizes": (100, 50),
                    "activation": "relu",
                    "solver": "adam",
                    "random_state": 42,
                    "max_iter": 500
                },
                "type": "regression",
                "incremental": False
            },
            "knn": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 5},
                "type": "regression",
                "incremental": False
            },
            "decision_tree": {
                "class": DecisionTreeRegressor,
                "params": {"random_state": 42},
                "type": "regression",
                "incremental": False
            },
            "ada_boost": {
                "class": AdaBoostRegressor,
                "params": {"n_estimators": 50, "random_state": 42},
                "type": "regression",
                "incremental": False
            },
            "sgd": {
                "class": SGDRegressor,
                "params": {"random_state": 42, "max_iter": 1000},
                "type": "regression",
                "incremental": True
            },
            
            # Classification algorithms
            "random_forest_classifier": {
                "class": RandomForestClassifier,
                "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                "type": "classification",
                "incremental": False
            },
            "gradient_boosting_classifier": {
                "class": GradientBoostingClassifier,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                "type": "classification",
                "incremental": False
            },
            "logistic_regression": {
                "class": LogisticRegression,
                "params": {"random_state": 42, "max_iter": 1000},
                "type": "classification",
                "incremental": False
            },
            "svm_classifier": {
                "class": SVC,
                "params": {"kernel": "rbf", "random_state": 42, "probability": True},
                "type": "classification",
                "incremental": False
            },
            "neural_network_classifier": {
                "class": MLPClassifier,
                "params": {
                    "hidden_layer_sizes": (100, 50),
                    "activation": "relu",
                    "solver": "adam",
                    "random_state": 42,
                    "max_iter": 500
                },
                "type": "classification",
                "incremental": False
            },
            "knn_classifier": {
                "class": KNeighborsClassifier,
                "params": {"n_neighbors": 5},
                "type": "classification",
                "incremental": False
            },
            "naive_bayes": {
                "class": GaussianNB,
                "params": {},
                "type": "classification",
                "incremental": False
            },
            "decision_tree_classifier": {
                "class": DecisionTreeRegressor,
                "params": {"random_state": 42},
                "type": "classification",
                "incremental": False
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
        if model_type not in self.algorithm_configs:
            # Default to gradient boosting for unknown tasks
            algorithm_name = preferred_algorithm or "gradient_boosting"
        else:
            config = self.algorithm_configs[model_type]
            if preferred_algorithm and preferred_algorithm in config["alternatives"]:
                algorithm_name = preferred_algorithm
            else:
                algorithm_name = config["primary"]
        
        if algorithm_name not in self.algorithms:
            algorithm_name = "gradient_boosting"  # Fallback
        
        algorithm_config = self.algorithms[algorithm_name].copy()
        algorithm_config["name"] = algorithm_name
        algorithm_config["task_description"] = self.algorithm_configs.get(model_type, {}).get("description", "Unknown task")
        
        return algorithm_config
    
    def create_algorithm_instance(self, algorithm_config: Dict[str, Any]) -> Any:
        """
        Create an instance of the specified algorithm
        
        Args:
            algorithm_config: Algorithm configuration from get_algorithm_for_task
        
        Returns:
            Instantiated algorithm
        """
        algorithm_class = algorithm_config["class"]
        params = algorithm_config["params"]
        
        return algorithm_class(**params)
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of all supported prediction tasks"""
        return list(self.algorithm_configs.keys())
    
    def get_task_description(self, model_type: str) -> str:
        """Get description of a prediction task"""
        return self.algorithm_configs.get(model_type, {}).get("description", "Unknown task")
    
    def get_algorithm_alternatives(self, model_type: str) -> List[str]:
        """Get alternative algorithms for a task"""
        config = self.algorithm_configs.get(model_type, {})
        alternatives = config.get("alternatives", [])
        primary = config.get("primary", "")
        return [primary] + alternatives if primary else alternatives
    
    def supports_incremental_learning(self, algorithm_name: str) -> bool:
        """Check if algorithm supports incremental learning"""
        return self.algorithms.get(algorithm_name, {}).get("incremental", False)
    
    def get_algorithm_type(self, algorithm_name: str) -> str:
        """Get algorithm type (regression/classification)"""
        return self.algorithms.get(algorithm_name, {}).get("type", "regression")
    
    def optimize_hyperparameters(self, algorithm_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Get optimized hyperparameters based on data characteristics
        
        Args:
            algorithm_name: Name of the algorithm
            X: Feature matrix
            y: Target vector
        
        Returns:
            Optimized parameters dictionary
        """
        base_params = self.algorithms.get(algorithm_name, {}).get("params", {}).copy()
        
        # Simple optimization based on data size
        n_samples, n_features = X.shape
        
        if algorithm_name == "random_forest":
            # Adjust n_estimators based on data size
            if n_samples < 1000:
                base_params["n_estimators"] = 50
            elif n_samples > 10000:
                base_params["n_estimators"] = 200
                
        elif algorithm_name == "gradient_boosting":
            # Adjust learning rate and estimators
            if n_samples < 1000:
                base_params["n_estimators"] = 50
                base_params["learning_rate"] = 0.2
            elif n_samples > 10000:
                base_params["n_estimators"] = 200
                base_params["learning_rate"] = 0.05
                
        elif algorithm_name == "neural_network":
            # Adjust hidden layer size based on features
            if n_features < 10:
                base_params["hidden_layer_sizes"] = (50,)
            elif n_features > 50:
                base_params["hidden_layer_sizes"] = (200, 100, 50)
                
        elif algorithm_name == "svr":
            # Adjust C parameter based on data size
            if n_samples > 5000:
                base_params["C"] = 0.1  # Lower C for larger datasets
                
        return base_params
    
    def get_feature_importance_method(self, algorithm_name: str) -> Optional[str]:
        """
        Get the method to extract feature importance for the algorithm
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Method name or None if not supported
        """
        importance_methods = {
            "random_forest": "feature_importances_",
            "gradient_boosting": "feature_importances_",
            "extra_trees": "feature_importances_",
            "decision_tree": "feature_importances_",
            "ada_boost": "feature_importances_",
            "random_forest_classifier": "feature_importances_",
            "gradient_boosting_classifier": "feature_importances_",
            "decision_tree_classifier": "feature_importances_"
        }
        
        return importance_methods.get(algorithm_name)
    
    def get_prediction_confidence_method(self, algorithm_name: str) -> Optional[str]:
        """
        Get method to calculate prediction confidence
        
        Args:
            algorithm_name: Name of the algorithm
        
        Returns:
            Method name or None if not supported
        """
        confidence_methods = {
            "random_forest": "predict_proba",
            "gradient_boosting_classifier": "predict_proba", 
            "logistic_regression": "predict_proba",
            "svm_classifier": "predict_proba",
            "neural_network_classifier": "predict_proba",
            "knn_classifier": "predict_proba",
            "naive_bayes": "predict_proba"
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
