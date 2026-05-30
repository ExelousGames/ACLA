"""
Scikit-learn Machine Learning Service for Assetto Corsa Competizione Telemetry Analysis

This service provides comprehensive AI model training and prediction capabilities
using your TelemetryFeatures and FeatureProcessor classes.
"""

import json
import base64
import logging
from dataclasses import dataclass
from app.features.tire_grip import TireGripAnalysisService
from app.ml.imitation.service import ExpertImitateLearningService
import numpy as np
import joblib
import warnings
import pickle
import time
from collections import Counter
from typing import Dict, List, Any, Optional, Union, Iterator, AsyncIterator
from pathlib import Path
from app.integrations.backend.schemas import AiModelDto, ActiveModelData
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import your telemetry models
from app.domain.telemetry import (
    TelemetryFeatures,
    _safe_float,
)

# Import backend service
from app.integrations.backend.client import backend_service

# Import model cache service
from app.storage.cache import model_cache_service

# Import hybrid data cache service
from app.storage import get_shared_telemetry_store
from app.infra.config.pipeline import PipelineConfig

# TelemetryLLMOrchestrator imported lazily inside __init__ to break the
# pipelines.chat ↔ pipelines.training circle: pipelines.chat.__init__
# imports Full_dataset_TelemetryMLService, while
# pipelines.chat.orchestrator is a sibling module — loading it requires
# the package's __init__ to finish. (Pre-refactor these lived in
# different packages so the cycle didn't exist.)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class Full_dataset_TelemetryMLService:
    """
    Machine Learning Service for AC Competizione Telemetry Analysis
    """ 
    
    def __init__(self, models_directory: Optional[str] = None, logger: Optional[logging.Logger] = None, pipeline_config: Optional[PipelineConfig] = None):
        """
        Initialize the ML service
        
        Args:
            models_directory: Directory to save/load trained models.
                            If None, defaults to 'models' in the project root.
            logger: Optional logger instance
            pipeline_config: Optional PipelineConfig instance to share cache keys across components.
                           If None, creates a new instance with default (empty) pipeline_id.
        """
        # Resolve to an absolute path so downstream tooling operates on a single location
        if models_directory:
            self.models_directory = Path(models_directory).resolve()
        else:
            # Default to project_root/models.
            # __file__ = app/pipelines/training/full_dataset.py → parents[3]
            # is the project root. (Was parents[2] when this lived at
            # app/services/full_dataset_ml_service.py — fixed in
            # refactor/hexagonal-v2 Step 12 cleanup.)
            self.models_directory = Path(__file__).resolve().parents[3] / "models"

        self.models_directory.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.telemetry_features = TelemetryFeatures()
        self.trained_models = {}
        self.scalers = {}
        self.label_encoders = {}
        # Cache frequently used feature lists to avoid recreating TelemetryFeatures each prediction
        try:
            self._imitate_expert_feature_names = self.telemetry_features.get_features_for_learning_expert()
        except Exception:
            self._imitate_expert_feature_names = []
        
        # Backend service integration
        self.backend_service = backend_service
        
        # Model cache service integration
        self.model_cache_service = model_cache_service
        
        # Telemetry store backed by Lance; see app/storage/__init__.py.
        self.telemetry_store = get_shared_telemetry_store()
        self.logger.info(
            "Using %s telemetry store for large dataset processing (store: %s)",
            type(self.telemetry_store).__name__,
            self.telemetry_store.store_dir,
        )
        
        self.llm_adapter_directory = self.models_directory / "llm_adapters"
        self.llm_adapter_directory.mkdir(parents=True, exist_ok=True)
        self.llm_dataset_directory = self.models_directory / "llm_datasets"
        self.llm_dataset_directory.mkdir(parents=True, exist_ok=True)

        # Deferred to break the chat ↔ training import cycle (see top of file).
        from app.local_llm.orchestrator import TelemetryLLMOrchestrator
        self.llm_orchestrator = TelemetryLLMOrchestrator(
            adapter_directory=self.llm_adapter_directory,
            dataset_directory=self.llm_dataset_directory,
        )

        # Centralize cache key usage for coordinated cleanup
        self.cache_config = pipeline_config if pipeline_config is not None else PipelineConfig()

        # Reusable service instances
        self._expert_service = ExpertImitateLearningService(logger=self.logger)
        self._tire_grip_service = TireGripAnalysisService()

        # Clear entire cache to ensure we start fresh with model instances only
        self.model_cache_service.clear()
        self.logger.info("Cleared entire cache on startup - will cache model instances directly")
    
    def _format_context_window(self, telemetry_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construct and format a context window for inference from a single telemetry record."""

        context_steps = 40  # Default context window size
        repeated_window = [dict(telemetry_record) for _ in range(max(1, context_steps))]
        return repeated_window


if __name__ == "__main__":
    # Example usage
    print("TelemetryMLService with Imitation Learning initialized. Ready for training!")