

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import math
import json
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import contextual feature catalogs for quality weighting
from ..services.tire_grip_analysis_service import TireGripFeatureCatalog
from ..services.imitate_expert_learning_service import ExpertFeatureCatalog
from .telemetry_models import TelemetryFeatures

# Force unbuffered output for real-time print statements
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)


# -----------------------------
# Contextual Feature Ordering
# -----------------------------
def get_canonical_context_feature_order() -> List[str]:
    """
    Get the canonical ordering of contextual features for consistent training/prediction.
    
    This function defines the exact order that contextual features should appear in
    across training and prediction phases. Features are ordered by:
    1. ExpertFeatureCatalog.ContextFeature
    2. TireGripFeatureCatalog.ContextFeature
    
    Returns:
        List[str]: Ordered list of feature names
    """
    context_features = []
    
    # Add expert context features in enum order
    context_features.extend([f.value for f in ExpertFeatureCatalog.ContextFeature])
    
    # Add tire grip context features in enum order
    context_features.extend([f.value for f in TireGripFeatureCatalog.ContextFeature])
    
    return context_features


def extract_context_features_canonical_order(context_data: Dict[str, Any]) -> List[float]:
    """
    Extract contextual features in canonical order, filling missing values with 0.0.
    
    Args:
        context_data: Dictionary containing contextual features
        
    Returns:
        List[float]: Features in canonical order with missing values filled as 0.0
    """
    canonical_order = get_canonical_context_feature_order()
    features = []
    
    for feature_name in canonical_order:
        value = context_data.get(feature_name, 0.0)
        try:
            features.append(float(value))
        except (ValueError, TypeError):
            features.append(0.0)
    
    return features


# -----------------------------
# Per-feature scaling utilities
# -----------------------------
class PerFeatureScaler:
    """Feature-wise scaler that maintains an independent scaler per feature.

    This wrapper allows the training pipeline to apply tailor-made scaling for
    every telemetry/context feature instead of relying on a single scaler that
    treats the feature matrix homogeneously. Each feature receives its own
    ``StandardScaler`` instance by default, but a custom ``scaler_factory`` can
    be provided to build alternative scalers (e.g., ``MinMaxScaler``) on a
    per-feature basis.

    Notes:
        * The scaler enforces explicit feature ordering – callers must supply
          the feature list at fit-time and preserve that ordering for all
          subsequent transforms.
        * Zero-variance features are stabilised by forcing their scale to 1.0
          to avoid division-by-zero and keep their transformed value anchored
          at 0.
    """

    def __init__(self,
                 feature_names: Optional[List[str]] = None,
                 scaler_factory: Optional[Callable[[str], Any]] = None):
        self.feature_names: List[str] = list(feature_names) if feature_names else []
        self.scaler_factory = scaler_factory or self._default_factory
        self._scalers: Dict[str, StandardScaler] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PerFeatureScaler must be fitted before use")

    def _normalise_input(self, data: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, bool]:
        array = np.asarray(data, dtype=np.float32)
        was_one_dimensional = False
        if array.ndim == 1:
            array = array.reshape(1, -1)
            was_one_dimensional = True
        if array.ndim != 2:
            raise ValueError(f"Expected 2D data matrix, got shape {array.shape}")
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Data has {array.shape[1]} features but scaler was fitted with {len(self.feature_names)}"
            )
        return array, was_one_dimensional

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, data: Union[List[List[float]], np.ndarray], feature_names: Optional[List[str]] = None) -> 'PerFeatureScaler':
        if feature_names is not None:
            self.feature_names = list(feature_names)
        if not self.feature_names:
            raise ValueError("Feature names are required to fit PerFeatureScaler")

        array, _ = self._normalise_input(data)

        self._scalers.clear()
        for idx, feature in enumerate(self.feature_names):
            scaler = self.scaler_factory(feature)
            column = array[:, idx].reshape(-1, 1)
            scaler.fit(column)

            # Guard against zero variance to keep transforms stable.
            if hasattr(scaler, 'scale_'):
                scale = float(np.asarray(scaler.scale_).reshape(-1)[0])
                if scale == 0.0:
                    scaler.scale_ = np.array([1.0], dtype=np.float64)
                    if hasattr(scaler, 'var_'):
                        scaler.var_ = np.array([0.0], dtype=np.float64)

            self._scalers[feature] = scaler

        self._fitted = True
        return self

    def transform(self, data: Union[List[List[float]], List[float], np.ndarray]) -> np.ndarray:
        self._require_fitted()
        array, was_one_dimensional = self._normalise_input(data)

        transformed = np.zeros_like(array, dtype=np.float32)
        for idx, feature in enumerate(self.feature_names):
            scaler = self._scalers.get(feature)
            if scaler is None:
                raise KeyError(f"No scaler fitted for feature '{feature}'")
            transformed[:, idx] = scaler.transform(array[:, idx].reshape(-1, 1)).reshape(-1)

        return transformed[0] if was_one_dimensional else transformed

    def inverse_transform(self, data: Union[List[List[float]], List[float], np.ndarray]) -> np.ndarray:
        self._require_fitted()
        array, was_one_dimensional = self._normalise_input(data)

        inversed = np.zeros_like(array, dtype=np.float32)
        for idx, feature in enumerate(self.feature_names):
            scaler = self._scalers.get(feature)
            if scaler is None:
                raise KeyError(f"No scaler fitted for feature '{feature}'")
            inversed[:, idx] = scaler.inverse_transform(array[:, idx].reshape(-1, 1)).reshape(-1)

        return inversed[0] if was_one_dimensional else inversed

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        status = 'fitted' if self._fitted else 'unfitted'
        return f"PerFeatureScaler(features={len(self.feature_names)}, status={status})"

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_factory(_: str) -> StandardScaler:
        return StandardScaler()

    @staticmethod
    def _as_1d_array(value: Any, default: float) -> np.ndarray:
        if isinstance(value, np.ndarray):
            arr = value.astype(np.float64)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float64)
        elif value is None:
            arr = np.asarray([default], dtype=np.float64)
        else:
            arr = np.asarray([value], dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def to_serializable(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of all feature scalers."""
        self._require_fitted()

        serialized_scalers: List[Dict[str, Any]] = []
        for feature in self.feature_names:
            scaler = self._scalers.get(feature)
            if scaler is None:
                raise KeyError(f"Missing scaler for feature '{feature}' during serialization")

            scaler_class = type(scaler).__name__
            if scaler_class != 'StandardScaler':
                raise ValueError(
                    f"PerFeatureScaler serialization currently supports StandardScaler instances only (feature '{feature}')"
                )

            serialized_scalers.append({
                'feature': feature,
                'class': scaler_class,
                'with_mean': getattr(scaler, 'with_mean', True),
                'with_std': getattr(scaler, 'with_std', True),
                'data': {
                    'mean': self._as_1d_array(getattr(scaler, 'mean_', None), 0.0).tolist(),
                    'scale': self._as_1d_array(getattr(scaler, 'scale_', None), 1.0).tolist(),
                    'var': self._as_1d_array(getattr(scaler, 'var_', None), 0.0).tolist(),
                    'n_samples_seen': self._as_1d_array(getattr(scaler, 'n_samples_seen_', None), 1.0).tolist()
                }
            })

        return {
            'version': 1,
            'feature_names': list(self.feature_names),
            'scalers': serialized_scalers
        }

    @classmethod
    def from_serializable(cls, payload: Dict[str, Any], scaler_factory: Optional[Callable[[str], Any]] = None) -> 'PerFeatureScaler':
        """Rehydrate a PerFeatureScaler from its serialized dictionary."""
        if not isinstance(payload, dict):
            raise ValueError("Serialized scaler payload must be a dictionary")

        feature_names = payload.get('feature_names')
        if not feature_names:
            raise ValueError("Serialized scaler payload missing 'feature_names'")

        scaler = cls(feature_names=feature_names, scaler_factory=scaler_factory)
        scaler._scalers.clear()

        scalers_payload = payload.get('scalers', [])
        if len(scalers_payload) != len(feature_names):
            raise ValueError(
                f"Serialized scaler payload contains {len(scalers_payload)} scaler entries for {len(feature_names)} features"
            )

        for entry in scalers_payload:
            feature = entry.get('feature')
            if feature not in scaler.feature_names:
                raise ValueError(f"Serialized scaler entry references unknown feature '{feature}'")

            scaler_class = entry.get('class', 'StandardScaler')
            if scaler_class != 'StandardScaler':
                raise ValueError(
                    f"PerFeatureScaler deserialization currently supports StandardScaler instances only (got '{scaler_class}')"
                )

            scaler_instance = scaler.scaler_factory(feature) if scaler_factory else StandardScaler()
            scaler_instance.with_mean = entry.get('with_mean', True)
            scaler_instance.with_std = entry.get('with_std', True)

            data = entry.get('data', {})
            scaler_instance.mean_ = cls._as_1d_array(data.get('mean'), 0.0)
            scaler_instance.scale_ = cls._as_1d_array(data.get('scale'), 1.0)
            scaler_instance.var_ = cls._as_1d_array(data.get('var'), 0.0)

            n_samples_seen = data.get('n_samples_seen', [1.0])
            if isinstance(n_samples_seen, (int, float)):
                n_samples_seen = [n_samples_seen]
            scaler_instance.n_samples_seen_ = cls._as_1d_array(n_samples_seen, 1.0)
            scaler_instance.n_features_in_ = 1

            scaler._scalers[feature] = scaler_instance

        scaler._fitted = True
        return scaler

    @classmethod
    def from_feature_statistics(cls,
                                feature_names: List[str],
                                means: np.ndarray,
                                variances: np.ndarray,
                                counts: np.ndarray,
                                scaler_factory: Optional[Callable[[str], Any]] = None) -> 'PerFeatureScaler':
        if len(feature_names) == 0:
            raise ValueError("feature_names must not be empty")

        scaler = cls(feature_names=feature_names, scaler_factory=scaler_factory)
        scaler._scalers.clear()

        means = np.asarray(means, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.float64)

        if means.shape[0] != len(feature_names) or variances.shape[0] != len(feature_names) or counts.shape[0] != len(feature_names):
            raise ValueError("Statistic arrays must match length of feature_names")

        for idx, feature in enumerate(feature_names):
            this_count = counts[idx]
            if this_count <= 0:
                this_count = 1.0

            var = max(float(variances[idx]), 0.0)
            scale_value = math.sqrt(var) if var > 0 else 1.0

            scaler_instance = scaler.scaler_factory(feature)
            scaler_instance.mean_ = np.array([float(means[idx])], dtype=np.float64)
            scaler_instance.var_ = np.array([var], dtype=np.float64)
            scaler_instance.scale_ = np.array([scale_value], dtype=np.float64)
            scaler_instance.n_samples_seen_ = np.array([this_count], dtype=np.float64)
            scaler_instance.n_features_in_ = 1

            scaler._scalers[feature] = scaler_instance

        scaler._fitted = True
        return scaler


# -----------------------------
# JSON-safe serialization utils
# -----------------------------
def _safe_number(value: Union[int, float], round_floats: Optional[int] = 6, replace_invalid_with: Any = None) -> Any:
    """Convert a numeric value to a JSON-safe Python primitive.

    - Finite floats are rounded to a configurable precision
    - NaN/Inf/-Inf are replaced with provided fallback (default None)
    - Integers are returned as-is
    """
    try:
        # bool is subclass of int, keep it as-is
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            if math.isfinite(value):
                return round(value, round_floats) if round_floats is not None else value
            return replace_invalid_with
    except Exception:
        return replace_invalid_with
    return value


def make_json_safe(obj: Any, round_floats: Optional[int] = 6, replace_invalid_with: Any = None) -> Any:
    """Recursively convert objects to JSON-safe structures.

    Handles:
    - torch.Tensor -> list of numbers (JSON-safe, with NaN/Inf replaced)
    - numpy scalars/arrays -> Python scalars/lists
    - pandas Series/DataFrame -> list/dict
    - bytes -> base64-encoded string
    - datetime -> ISO string
    - Path -> str
    - dict/list/tuple/set -> recursively processed, sets converted to lists
    - float NaN/Inf -> replaced with `replace_invalid_with` (default None)
    """
    # Short-circuit common primitives
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    # Numbers
    if isinstance(obj, (int, float)):
        return _safe_number(obj, round_floats, replace_invalid_with)

    # Datetime and Paths
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)

    # Bytes -> base64 string
    if isinstance(obj, (bytes, bytearray)):
        import base64 as _b64
        return _b64.b64encode(bytes(obj)).decode('utf-8')

    # Torch tensors
    try:
        import torch as _torch  # local import to avoid issues at import time in some environments
        if isinstance(obj, _torch.Tensor):
            obj = obj.detach().cpu().numpy()
    except Exception:
        pass

    # Numpy scalar/array
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return _safe_number(obj.item(), round_floats, replace_invalid_with)
        if isinstance(obj, _np.ndarray):
            return make_json_safe(obj.tolist(), round_floats, replace_invalid_with)
    except Exception:
        pass

    # Pandas
    try:
        import pandas as _pd
        if isinstance(obj, _pd.DataFrame):
            return make_json_safe(obj.to_dict(orient='records'), round_floats, replace_invalid_with)
        if isinstance(obj, _pd.Series):
            return make_json_safe(obj.tolist(), round_floats, replace_invalid_with)
    except Exception:
        pass

    # Mappings
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v, round_floats, replace_invalid_with) for k, v in obj.items()}

    # Iterables (list/tuple/set)
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v, round_floats, replace_invalid_with) for v in obj]

    # Fallback to string representation
    try:
        return str(obj)
    except Exception:
        return None


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ExpertActionTransformer(nn.Module):
    """
    ExpertActionTransformer - Unified Expert Improvement Learning Model
    
    WHAT IT ACTUALLY DOES:
    This model learns how non-expert drivers progressively move closer to expert-level performance.
    Given the current complete driving state, it predicts the next improved state that moves the
    driver toward expert-like behavior. Instead of separating context and actions, this model
    treats all features as a unified state vector and predicts expert improvement progressions.
    
    UNIFIED FEATURE APPROACH:
    - Input: Complete state vector [context_features + action_features] at timestep t
    - Output: Complete state vector [context_features + action_features] at timestep t+1
    - The attention mechanism learns relationships between ALL features (context and actions)
    
    CORE FUNCTIONALITY:
    1. EXPERT IMPROVEMENT LEARNING: Predicts next improved state in progression toward expert level
    2. UNIFIED FEATURES: Context and actions are treated as single feature vector during improvement
    3. ATTENTION LEARNING: Model learns relationships between context and expert-improving actions
    4. SEQUENTIAL MODELING: Learns temporal dependencies in expert improvement progressions
    
    HOW IT WORKS:
    The model takes a unified feature vector containing:
    - Contextual features: gap analysis, track info, tire grip, environmental data
    - Action features: gas, brake, steer_angle, gear
    
    It predicts the next timestep's complete feature vector during improvement progression:
    - Next contextual features: updated gap analysis showing progress toward expert line
    - Next action features: MORE EXPERT-LIKE gas, brake, steer_angle, gear
    
    TRAINING PROCESS:
    The model is trained on sequences of IMPROVING ACTIONS where non-expert drivers
    progressively move closer to expert-level performance. Each timestep contains
    the complete state during improvement progression. It learns to predict the next
    improved state from the current state, allowing the attention mechanism to discover
    relationships between context and expert-improving actions naturally.
    
    Architecture:
    - Input: Unified state vector [context + actions]
    - Processing: Transformer encoder with self-attention over all features
    - Output: Next unified state vector [next_context + next_actions]
    """
    
    def __init__(self, 
                 total_features_count: int = 46,  # Combined context + action features
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 sequence_length: int = 20,
                 dropout: float = 0.1,
                 time_step_seconds: float = 0.5):
        """
        Initialize the Unified State Transformer
        
        Args:
            total_features_count: Total number of features (context + actions combined)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            sequence_length: Maximum sequence length for predictions
            dropout: Dropout rate
            time_step_seconds: Time duration (in seconds) that each prediction step represents
        """
        super(ExpertActionTransformer, self).__init__()
        
        # Store configuration
        self.total_features_count = total_features_count
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.time_step_seconds = time_step_seconds
        
        # Scaler for normalization during inference (feature-wise scaler)
        self.feature_scaler: Optional[PerFeatureScaler] = None
        
        # Input embedding for unified features
        self.input_embedding = nn.Linear(total_features_count, d_model)

        # Positional encoding : Without positional encoding: The transformer can't distinguish between [brake, throttle, steer] and [steer, brake, throttle], it adds unique positional information
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=sequence_length * 2)
        
        # Transformer encoder for processing unified state sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

        '''
        d_model: The number of expected features in the input (the model dimension).
        nhead: The number of attention heads.
        dim_feedforward: The dimension of the feedforward network model (default is 2048).
        dropout: Dropout value (default is 0.1).
        activation: Activation function of the intermediate layer, either "relu" or "gelu" (default is "relu").
        layer_norm_eps: Epsilon value for layer normalization (default is 1e-5).
        batch_first: If True, then input and output tensors are provided as (batch, seq, feature) (default is False).
        norm_first: If True, layer norm is done before attention and feedforward operations (default is False).
        device: The device on which the module will be allocated.
        dtype: The data type for the module’s parameters.
        '''

        
        # Output projection to unified feature space
        self.output_projection = nn.Linear(d_model, total_features_count)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def set_scalers(self, feature_scaler: Optional[PerFeatureScaler] = None):
        """
        Set the scaler for unified features (context + actions combined).
        
        Args:
            feature_scaler: PerFeatureScaler fitted on unified feature vectors
        """
        self.feature_scaler = feature_scaler
    
    def forward(self, 
                unified_input: torch.Tensor,
                prediction_steps: int = None,
                use_teacher_forcing: bool = None) -> torch.Tensor:
        """
        Forward pass - Teacher Forcing (Training) or Autoregressive (Inference)
        
        This method supports two modes:
        1. TEACHER FORCING (Training): Direct prediction from input to target without feedback
        2. AUTOREGRESSIVE (Inference): Sequential prediction where each output feeds into next input
        
        TEACHER FORCING FLOW (use_teacher_forcing=True or training mode):
        1. Process input sequence through transformer
        2. Predict all target states simultaneously 
        3. Much faster and more stable for training
        
        AUTOREGRESSIVE FLOW (use_teacher_forcing=False or inference mode):
        1. Start with input sequence (real states)
        2. For each prediction step:
           - Process current sequence through transformer
           - Predict next state from the sequence
           - Append prediction to sequence for next iteration
        3. Return all generated predictions
        
        Args:
            unified_input: Initial unified states [batch_size, input_length, total_features]
                          Each timestep contains: [context_features + action_features]  
            prediction_steps: Number of future steps to predict (if None, predicts 1 step)
            use_teacher_forcing: If True, use teacher forcing. If None, auto-detect based on training mode
            
        Returns:
            Generated predictions [batch_size, prediction_steps, total_features]
        """
        if prediction_steps is None:
            prediction_steps = 1
        
        # Auto-detect teacher forcing mode if not specified
        if use_teacher_forcing is None:
            use_teacher_forcing = self.training  # Use teacher forcing during training
            
        batch_size = unified_input.shape[0]
        device = unified_input.device
        
        if use_teacher_forcing:
            # Validate input sequence dimensions
            expected_features = self.total_features_count
            actual_features = unified_input.shape[-1]
            assert actual_features == expected_features, f"Expected {expected_features} features, got {actual_features}"
            
            # Embed input sequence
            embedded_input = self.input_embedding(unified_input)
            
            # Apply positional encoding
            encoded_input = self.pos_encoding(embedded_input)
            
            # Process through transformer
            transformer_output = self.transformer(encoded_input)
            
            # Project to unified feature space
            sequence_predictions = self.output_projection(transformer_output)
            
            # For teacher forcing: input[t] -> predict[t+1]
            # Data preparation gives us:
            # - input_sequence: scaled_sequence[:-1] (timesteps 0 to N-2)
            # - target_sequence: scaled_sequence[1:] (timesteps 1 to N-1)
            # So we need to return predictions that align with targets
            
            # sequence_predictions has same length as input, so it's already correctly aligned
            # Each prediction at position t predicts the next state (t+1)
            return sequence_predictions
        
        else:
            # Start with input sequence
            current_sequence = unified_input
            predictions = []
            
            # Generate predictions autoregressively
            for step in range(prediction_steps):
                # Validate current sequence dimensions
                expected_features = self.total_features_count
                actual_features = current_sequence.shape[-1]
                assert actual_features == expected_features, f"Expected {expected_features} features, got {actual_features}"
                
                # Embed current sequence
                embedded_input = self.input_embedding(current_sequence)
                
                # Apply positional encoding
                encoded_input = self.pos_encoding(embedded_input)
                
                # Process through transformer
                transformer_output = self.transformer(encoded_input)
                
                # Project to unified feature space
                sequence_predictions = self.output_projection(transformer_output)
                
                # Take the last prediction as next state
                next_state = sequence_predictions[:, -1:, :]  # [B, 1, features]
                
                # Store prediction
                predictions.append(next_state)
                
                # Append prediction to sequence for next iteration (autoregressive)
                current_sequence = torch.cat([current_sequence, next_state], dim=1)
                
                # Limit sequence length to prevent memory issues
                max_context = 100
                if current_sequence.shape[1] > max_context:
                    current_sequence = current_sequence[:, -max_context:, :]
            
            # Return all predictions
            result = torch.cat(predictions, dim=1) if len(predictions) > 1 else predictions[0]
            return result

    def unified_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor) -> torch.Tensor:
        """
        MSE loss for unified expert improvement learning with NaN protection
        
        Since training segments are filtered to contain only improving action sequences
        (non-expert → expert-like progressions), this loss learns how to predict the
        next improved state in the progression toward expert-level performance.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, total_features]
            targets: Target improved unified states [batch_size, seq_len, total_features]
                    These targets represent expert-improving states in filtered segments
            
        Returns:
            MSE loss tensor (scalar) - measures accuracy of expert improvement predictions
        """
        # Ensure loss computation in full precision to avoid dtype issues
        predictions = predictions.float()
        targets = targets.float()
        
        # Check for NaN/Inf in inputs BEFORE computing loss
        pred_has_nan = torch.isnan(predictions).any()
        pred_has_inf = torch.isinf(predictions).any()
        target_has_nan = torch.isnan(targets).any()
        target_has_inf = torch.isinf(targets).any()
        
        if pred_has_nan or pred_has_inf or target_has_nan or target_has_inf:
            print(f"[ERROR] NaN/Inf detected in loss inputs:")
            print(f"  Predictions - NaN: {pred_has_nan}, Inf: {pred_has_inf}")
            print(f"  Targets - NaN: {target_has_nan}, Inf: {target_has_inf}")
            print(f"  Pred range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"  Target range: [{targets.min():.6f}, {targets.max():.6f}]")
            
            # Replace NaN/Inf with safe values to prevent training crash
            if pred_has_nan:
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            if pred_has_inf:
                predictions = torch.clamp(predictions, min=-1e6, max=1e6)
            if target_has_nan:
                targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
            if target_has_inf:
                targets = torch.clamp(targets, min=-1e6, max=1e6)
        
        # Compute MSE loss
        loss = F.mse_loss(predictions, targets, reduction='mean')
        
        # Final NaN check on computed loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] NaN/Inf loss computed: {loss}")
            print(f"  Input shapes - Pred: {predictions.shape}, Target: {targets.shape}")
            print(f"  Input stats - Pred mean: {predictions.mean():.6f}, Target mean: {targets.mean():.6f}")
            # Return a small positive loss instead of NaN to prevent training crash
            loss = torch.tensor(1e-6, device=loss.device, dtype=loss.dtype)
        
        return loss
    

    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def predict_segment_progression(self, 
                                  combined_input: torch.Tensor,
                                  prediction_length: Optional[int] = None) -> torch.Tensor:
        """
        Predict unified state progression for a segment
        
        This method processes input unified states and predicts the next sequence of
        unified states (context + actions) for the specified prediction length.
        
        Args:
            combined_input: Unified state features [batch_size, input_len, total_features]
                           Contains combined context + action features for autoregressive prediction
            prediction_length: Number of future steps to predict. If None, uses model's sequence_length
            
        Returns:
            Predicted unified state progression [batch_size, prediction_length, total_features]
            Shows next sequence of unified states (context + improved actions)
        """
        self.eval()
        
        # Store original sequence length and temporarily update if needed
        original_sequence_length = self.sequence_length
        if prediction_length is not None:
            self.sequence_length = prediction_length
            
        try:
            with torch.no_grad():
                return self.forward(
                    unified_input=combined_input, 
                    prediction_steps=prediction_length
                )
        finally:
            # Restore original sequence length
            self.sequence_length = original_sequence_length
    
    
    def _apply_unified_inverse_scaling(self, scaled_predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse scaling to unified model predictions to convert from normalized to original scale.
        
        Args:
            scaled_predictions: Scaled unified predictions [batch_size, seq_len, total_features]
            
        Returns:
            Unscaled unified predictions [batch_size, seq_len, total_features]
        """
        if self.feature_scaler is None:
            return scaled_predictions
            
        # Convert to numpy for sklearn scaler
        device = scaled_predictions.device
        original_shape = scaled_predictions.shape
        
        # Reshape to 2D: (batch_size * seq_len, total_features)
        predictions_2d = scaled_predictions.view(-1, original_shape[-1]).cpu().numpy()
        
        # Apply inverse transform
        unscaled_predictions = self.feature_scaler.inverse_transform(predictions_2d)
        
        # Convert back to tensor and reshape
        unscaled_tensor = torch.from_numpy(unscaled_predictions).float().to(device)
        return unscaled_tensor.view(original_shape)
    
    def get_contextual_features_summary(self, context_feature_names: List[str]) -> Dict[str, Any]:
        """
        Get summary of contextual features available for loss weighting
        
        Args:
            context_feature_names: List of context feature names
            
        Returns:
            Dictionary with summary of available contextual features
        """
        # Get all possible context features from catalogs
        expert_features = [f.value for f in ExpertFeatureCatalog.ContextFeature]
        tire_grip_features = [f.value for f in TireGripFeatureCatalog.ContextFeature]
        
        # Filter to only those present in the dataset
        available_expert = [f for f in expert_features if f in context_feature_names]
        available_tire_grip = [f for f in tire_grip_features if f in context_feature_names]
        
        return {
            'available_for_weighting': available_expert + available_tire_grip,
            'expert_features': available_expert,
            'tire_grip_features': available_tire_grip,
            'total_context_features': len(context_feature_names)
        }
    
    def predict_human_readable(self, 
                              current_telemetry: Dict[str, Any],
                              context_data: Optional[Dict[str, Any]] = None,
                              sequence_length: int = 10) -> Dict[str, Any]:
        """
        Generate human-readable driving improvement predictions from current telemetry data.
        
        This function serves as the main interface for real-time racing coaching, converting
        raw telemetry data into actionable driving advice that shows how a non-expert driver
        should improve their actions to move toward expert-level performance.
        
        INFERENCE APPROACH:
        Unlike training which uses fixed-length sequences, inference uses a single timestep
        of current telemetry data to predict a sequence of future improved actions through
        autoregressive generation. This mirrors real-world racing where you make decisions
        based on current state to plan future actions.
        
        Process Flow:
        1. Validate and preprocess input telemetry data
        2. Convert telemetry to model input format (normalization, feature extraction)
        3. Create single-timestep input (not repeated sequence)
        4. Generate improved action sequence predictions using autoregressive generation
            (conditioned on expert-to-non-expert gap context features)
        5. Convert raw numerical predictions to human-readable advice
        6. Format everything into structured JSON response
        
        Args:
            current_telemetry: Dictionary containing current driver telemetry data
                              Expected keys: speed, position, forces, steering, throttle, brake, etc.
            context_data: Optional dictionary with gap and environmental context information
                         Should include gap features from ExpertFeatureCatalog.ContextFeature:
                         - expert_velocity_alignment: How aligned current velocity is with expert
                         - speed_difference: Speed difference from expert optimal
                         - distance_to_expert_line: Distance from expert racing line
                         Can include: tire grip levels, weather conditions
            sequence_length: Number of future action steps to predict (default: 10)
            
        Returns:
            Structured JSON dictionary with complete target predictions:
            {
                "status": "success" | "error",
                "timestamp": ISO timestamp,
                "sequence_predictions": [
                    {
                        "step": 1,
                        "time_ahead": "0.1s",
                        "all_targets": {
                            "Physics_gas": 0.2,
                            "Physics_brake": 0.6,
                            "Physics_steer_angle": -0.15,
                            "Physics_gear": 3,
                            "... all other features": "... values"
                        }
                    }
                ]
            }
        """
        try:
            # Get device from model parameters first
            device = next(self.parameters()).device
            
            # Prepare combined input data for model input
            combined_features = self._extract_combined_features_for_prediction(current_telemetry, context_data)
            
            # For inference, create a single timestep input (not repeated sequence)
            # The model will use autoregressive generation to create the sequence
            combined_tensor = torch.tensor([[combined_features]], dtype=torch.float32).to(device)  # [1, 1, features]
            
            # Generate predictions
            try:
                self.eval()
                with torch.no_grad():
                    predictions = self.predict_segment_progression(
                        combined_input=combined_tensor,
                        prediction_length=sequence_length
                    )
                    # During inference, apply inverse scaling to get original unified feature values
                    predictions = self._apply_unified_inverse_scaling(predictions)
            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {str(e)}")
            
            # Convert predictions to numpy for processing
            predictions_np = predictions.cpu().numpy()[0]  # Remove batch dimension
            
            # Create sequence predictions
            sequence_predictions = self._create_sequence_predictions(predictions_np, sequence_length)

            # Build response
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "sequence_predictions": sequence_predictions,
            }
            
            return make_json_safe(response)
            
        except Exception as e:
            error_payload = make_json_safe({
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            raise RuntimeError(json.dumps(error_payload))
    
    def _extract_combined_features_for_prediction(self, telemetry: Dict[str, Any], context_data: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Extract combined telemetry and gap features for model input.
        
        Args:
            telemetry: Dictionary containing non-expert telemetry data
            context_data: Optional dictionary containing gap features from ExpertFeatureCatalog.ContextFeature:
                         - expert_velocity_alignment: Alignment with expert velocity direction
                         - speed_difference: Speed difference from expert optimal
                         - distance_to_expert_line: Distance from expert racing line
            
        Returns:
            List[float]: Combined features for model input (telemetry + gap features + environmental context)
        """
        # Extract telemetry features
        try:
            from ..services.imitate_expert_learning_service import get_features_for_imitate_expert
            telemetry_feature_names = get_features_for_imitate_expert()
        except Exception:
            try:
                telemetry_feature_names = TelemetryFeatures.get_features_for_imitate_expert()
            except Exception:
                raise ValueError("Could not get telemetry feature names")
        
        telemetry_features = []
        for feature_name in telemetry_feature_names:
            try:
                value = float(telemetry.get(feature_name, 0.0))
                telemetry_features.append(value)
            except (ValueError, TypeError):
                telemetry_features.append(0.0)
        
        # Extract context features if available
        context_features = []
        if context_data:
            context_features = extract_context_features_canonical_order(context_data)
        else:
            raise ValueError("Context data is required for gap features")
        
        # Combine telemetry and context features
        combined_features = telemetry_features + context_features
        
        # Ensure the feature vector matches the model's expected input size
        expected_len = self.total_features_count
        if len(combined_features) < expected_len:
            combined_features.extend([0.0] * (expected_len - len(combined_features)))
        elif len(combined_features) > expected_len:
            combined_features = combined_features[:expected_len]

        # Apply unified feature scaler if available
        if self.feature_scaler is not None:
            import numpy as np
            features_array = np.array(combined_features).reshape(1, -1)
            scaled_features = self.feature_scaler.transform(features_array)
            combined_features = scaled_features.flatten().tolist()

        return combined_features
    
    def _create_sequence_predictions(self, predictions: np.ndarray, sequence_length: int) -> List[Dict[str, Any]]:
        """Create sequence of future predictions from unified feature vectors - includes ALL targets"""
        sequence = []
        
        # Get all feature names for complete output
        from ..models.telemetry_models import TelemetryFeatures
        feature_names = TelemetryFeatures.get_features_for_imitate_expert()
        
        for i in range(min(sequence_length, len(predictions))):
            pred = predictions[i]
            
            # Create prediction step with ALL target features
            step_data = {
                "step": i + 1,
                "time_ahead": f"{(i + 1) * self.time_step_seconds:.1f}s",
                "all_targets": {}
            }
            
            # Include all features in the prediction output
            for j, feature_name in enumerate(feature_names):
                if j < len(pred):
                    step_data["all_targets"][feature_name] = round(float(pred[j]), 4)
            
            sequence.append(step_data)
        
        return sequence
    
    def serialize_model(self) -> Dict[str, Any]:
        """
        Serialize the model to a JSON-serializable dictionary
        
        Returns:
            Dictionary containing model state and configuration for JSON serialization
        """
        import base64
        import io
        import torch as _torch
        
        # Save model state to bytes
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        state_dict_bytes = buffer.getvalue()
        
        # Serialize unified feature scaler
        feature_scaler_data = None
        
        if self.feature_scaler is not None:
            feature_scaler_data = self.feature_scaler.to_serializable()
        
        model_data = {
            'model_type': 'ExpertActionTransformer',
            'state_dict': base64.b64encode(state_dict_bytes).decode('utf-8'),
            'feature_scaler': feature_scaler_data,
            'config': {
                'total_features_count': self.total_features_count,
                'd_model': self.d_model,
                'sequence_length': self.sequence_length,
                'time_step_seconds': self.time_step_seconds,  # Include time step configuration
                'nhead': getattr(self.transformer.layers[0].self_attn, 'num_heads', 8),
                'num_layers': len(self.transformer.layers),
                'dim_feedforward': getattr(self.transformer.layers[0].linear1, 'out_features', 1024),
                'dropout': 0.1  # Default, could extract from layers if needed
            },
            'serialization_timestamp': datetime.now().isoformat(),
            'pytorch_version': getattr(_torch, '__version__', 'unknown')
        }
        # Ensure the entire payload is JSON-safe (no NaN/Inf, tensors, numpy, etc.)
        return make_json_safe(model_data)
    

    @classmethod
    def deserialize_transformer_model(cls, serialized_data: Dict[str, Any]) -> 'ExpertActionTransformer':
        """
        Create and return a new ExpertActionTransformer instance from serialized data.
        
        This class method creates a new trained ExpertActionTransformer model from serialized data
        created by serialize_model(). It instantiates a new model with the correct architecture
        and loads the trained weights, making the model ready for inference.
        
        Args:
            serialized_data: Dictionary containing serialized model data with keys:
                           - 'model_type': Should be 'ExpertActionTransformer'
                           - 'state_dict': Base64-encoded model weights and biases
                           - 'config': Model architecture configuration
                           - 'serialization_timestamp': When model was serialized
        
        Returns:
            ExpertActionTransformer: New instance with restored state
        
        Raises:
            ValueError: If serialized data format is invalid or incompatible
            RuntimeError: If model state restoration fails
        """
        import base64
        import io
        
        try:
            # Validate serialized data format
            required_keys = ['model_type', 'state_dict', 'config']
            for key in required_keys:
                if key not in serialized_data:
                    raise ValueError(f"Missing required key '{key}' in serialized data")
            
            # Verify this is the correct model type
            if serialized_data['model_type'] != 'ExpertActionTransformer':
                raise ValueError(f"Invalid model type: {serialized_data['model_type']}. Expected 'ExpertActionTransformer'")
            
            # Extract configuration
            config = serialized_data['config']

            # Gather architecture parameters from config with sensible fallbacks
            cfg_total_features = config.get('total_features_count', config.get('input_features_count', 42))  # Backward compatibility
            cfg_d_model = config.get('d_model', 256)
            cfg_seq_len = config.get('sequence_length', 20)
            cfg_time_step = config.get('time_step_seconds', 0.5)
            cfg_nhead = config.get('nhead', 8)
            cfg_num_layers = config.get('num_layers', 6)
            cfg_dim_ff = config.get('dim_feedforward', 1024)
            cfg_dropout = config.get('dropout', 0.1)

            print(f"[INFO] Creating ExpertActionTransformer from serialized config:")
            print(f"[INFO] - Features: {cfg_total_features} total unified features")
            print(f"[INFO] - Architecture: d_model={cfg_d_model}, nhead={cfg_nhead}, layers={cfg_num_layers}")
            print(f"[INFO] - Sequence: length={cfg_seq_len}, time_step={cfg_time_step}s")

            # Create new model instance with the serialized configuration
            model = cls(
                total_features_count=cfg_total_features,
                d_model=cfg_d_model,
                nhead=cfg_nhead,
                num_layers=cfg_num_layers,
                dim_feedforward=cfg_dim_ff,
                sequence_length=cfg_seq_len,
                dropout=cfg_dropout,
                time_step_seconds=cfg_time_step,
            )
            
            # Decode and restore model state
            state_dict_base64 = serialized_data['state_dict']
            state_dict_bytes = base64.b64decode(state_dict_base64)
            
            # Load state dict from bytes
            buffer = io.BytesIO(state_dict_bytes)
            state_dict = torch.load(buffer, map_location='cpu')  # Load to CPU first
            
            # Load the state dict into the new model
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as load_err:
                print(f"[WARNING] Strict state_dict load failed: {load_err}. Trying non-strict load...")
                incompatible = model.load_state_dict(state_dict, strict=False)
                # Handle both tuple return (older PyTorch) and IncompatibleKeys object (newer)
                missing = getattr(incompatible, 'missing_keys', None) or (incompatible[0] if isinstance(incompatible, (list, tuple)) and len(incompatible) > 0 else [])
                unexpected = getattr(incompatible, 'unexpected_keys', None) or (incompatible[1] if isinstance(incompatible, (list, tuple)) and len(incompatible) > 1 else [])
                if missing:
                    print(f"[WARNING] Missing keys during load: {missing}")
                if unexpected:
                    print(f"[WARNING] Unexpected keys during load: {unexpected}")
            
            # Restore unified feature scaler if available (with backward compatibility)
            scaler_payload = serialized_data.get('feature_scaler')
            if scaler_payload is None:
                model.feature_scaler = None
            elif not isinstance(scaler_payload, dict):
                raise ValueError("Serialized model 'feature_scaler' must be a dictionary payload")
            else:
                model.feature_scaler = PerFeatureScaler.from_serializable(scaler_payload)
                print("[INFO] - Restored unified feature scaler from serialized payload")
            
            # Set model to evaluation mode (ready for inference)
            model.eval()
            
            # Log successful restoration
            serialization_time = serialized_data.get('serialization_timestamp', 'unknown')
            scaler_status = " (unified feature scaler)" if model.feature_scaler is not None else " (no scaler)"
            
            print(f"[INFO] Successfully created ExpertActionTransformer model from serialized data")
            print(f"[INFO] - Model features: {model.total_features_count} total unified features")
            print(f"[INFO] - Architecture: d_model={model.d_model}, seq_len={model.sequence_length}")
            print(f"[INFO] - Originally serialized: {serialization_time}")
            print(f"[INFO] - Model ready for inference{scaler_status}")
            
            return model
            
        except Exception as e:
            error_msg = f"Failed to deserialize ExpertActionTransformer model: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def check_dataset_quality(self, dataset: 'TelemetryActionDataset', dataset_name: str = "Training Dataset") -> Dict[str, Any]:
        """
        Comprehensive dataset quality check for training data validation.
        
        This function performs extensive quality validation on the dataset to ensure
        it meets the requirements for effective transformer training. It checks for:
        - Data integrity and completeness
        - Statistical distribution health
        - Feature correlation analysis
        - Temporal consistency within segments
        - Outlier detection and analysis
        - Missing value patterns
        - Scale and normalization verification
        
        Args:
            dataset: TelemetryActionDataset to validate
            dataset_name: Name for reporting (e.g., "Training Dataset", "Validation Dataset")
            
        Returns:
            Dictionary containing comprehensive quality metrics and recommendations
        """
        print(f"\n{'='*80}")
        print(f"🔍 DATASET QUALITY ANALYSIS: {dataset_name}")
        print(f"{'='*80}")
        
        quality_report = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'overall_quality': 'UNKNOWN',
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            # Get basic dataset information
            print(f"[DEBUG] Getting segment info...")
            segment_info = dataset.get_segment_info()
            print(f"[DEBUG] Feature names obtained successfully")
            context_features = dataset.get_context_feature_names()
            
            print(f"📊 Basic Dataset Information:")
            print(f"   • Total segments: {segment_info['num_segments']:,}")
            print(f"   • Segment length: {segment_info['segment_length']}")
            print(f"   • Total samples: {segment_info['total_samples']:,}")
            input_features = dataset.get_input_feature_names()
            # In unified approach, target feature names mirror input feature names
            action_features = input_features
            print(f"   • Context features: {len(context_features)}")
            
            quality_report['metrics']['basic_info'] = {
                'num_segments': segment_info['num_segments'],
                'segment_length': segment_info['segment_length'],
                'total_samples': segment_info['total_samples'],
                'total_features_count': len(input_features),
                'context_features_count': len(context_features)
            }
            
            # Check minimum dataset size requirements
            min_segments_required = 100
            min_samples_required = 10000
            
            if segment_info['num_segments'] < min_segments_required:
                quality_report['critical_issues'].append(
                    f"Insufficient segments: {segment_info['num_segments']} < {min_segments_required} (minimum required)"
                )
            
            if segment_info['total_samples'] < min_samples_required:
                quality_report['critical_issues'].append(
                    f"Insufficient total samples: {segment_info['total_samples']} < {min_samples_required} (minimum required)"
                )
            
            # Sample-based quality analysis for streaming dataset
            print(f"\n🔬 Sample-Based Quality Analysis:")
            sample_size = min(50, len(dataset))  # Sample up to 50 segments
            import random
            sample_indices = random.sample(range(len(dataset)), sample_size)
            
            print(f"   • Analyzing {sample_size} sample segments for quality...")
            
            # Collect sample data using chunk-based approach
            sample_inputs = []
            sample_targets = []
            nan_count = 0
            inf_count = 0
            
            # Sample from the first few chunks instead of using dataset[idx]
            sample_chunks = min(3, len(sample_indices))  # Use first 3 chunks for quality check
            batches_sampled = 0
            
            for chunk_idx in range(sample_chunks):
                try:
                    batch_count = 0
                    # Get a few batches from each chunk for quality assessment
                    for input_batch, target_batch in dataset.get_chunk_batches(chunk_idx):
                        sample_inputs.append(input_batch)
                        sample_targets.append(target_batch)
                        
                        # Check for NaN/Inf in this batch
                        nan_count += torch.isnan(input_batch).sum().item() + torch.isnan(target_batch).sum().item()
                        inf_count += torch.isinf(input_batch).sum().item() + torch.isinf(target_batch).sum().item()
                        
                        batch_count += 1
                        batches_sampled += 1
                        
                        # Only sample a few batches per chunk for efficiency
                        if batch_count >= 2:
                            break
                    
                except Exception as e:
                    quality_report['critical_issues'].append(f"Failed to load chunk {chunk_idx}: {str(e)}")
            
            if not sample_inputs:
                quality_report['critical_issues'].append("Failed to load any sample segments")
                quality_report['overall_quality'] = 'ERROR'
                return quality_report
            
            # Stack samples for analysis
            sample_input_tensor = torch.stack(sample_inputs)
            sample_target_tensor = torch.stack(sample_targets)
            
            print(f"   • Sample input shape: {sample_input_tensor.shape}")
            print(f"   • Sample target shape: {sample_target_tensor.shape}")
            print(f"   • NaN values in samples: {nan_count}")
            print(f"   • Inf values in samples: {inf_count}")
            
            # Store sample metrics
            quality_report['metrics'].update({
                'sample_size': sample_size,
                'sample_nan_count': nan_count,
                'sample_inf_count': inf_count
            })
            
            # Critical issues for NaN/Inf values
            if nan_count > 0:
                quality_report['critical_issues'].append(f"NaN values detected in samples: {nan_count}")
            if inf_count > 0:
                quality_report['critical_issues'].append(f"Infinite values detected in samples: {inf_count}")
            
            # Statistical distribution analysis on samples
            print(f"\n� Sample Statistical Analysis:")
            input_mean = torch.mean(sample_input_tensor)
            input_std = torch.std(sample_input_tensor)
            input_min = torch.min(sample_input_tensor)
            input_max = torch.max(sample_input_tensor)
            
            target_mean = torch.mean(sample_target_tensor)
            target_std = torch.std(sample_target_tensor)
            target_min = torch.min(sample_target_tensor)
            target_max = torch.max(sample_target_tensor)
            
            # Check for zero variance features
            zero_var_input = (input_std < 1e-6).sum().item()
            zero_var_target = (target_std < 1e-6).sum().item()
            
            print(f"   • Input features with zero variance: {zero_var_input}")
            print(f"   • Target features with zero variance: {zero_var_target}")
            print(f"   • Input mean range: [{input_mean.min():.4f}, {input_mean.max():.4f}]")
            print(f"   • Input std range: [{input_std.min():.4f}, {input_std.max():.4f}]")
            print(f"   • Target mean range: [{target_mean.min():.4f}, {target_mean.max():.4f}]")
            print(f"   • Target std range: [{target_std.min():.4f}, {target_std.max():.4f}]")
            
            if zero_var_input > 0:
                quality_report['warnings'].append(f"{zero_var_input} input features have zero variance")
            
            if zero_var_target > 0:
                quality_report['critical_issues'].append(f"{zero_var_target} target features have zero variance")
            
            quality_report['metrics']['statistics'] = {
                'input_stats': {
                    'mean_range': [float(input_mean.min()), float(input_mean.max())],
                    'std_range': [float(input_std.min()), float(input_std.max())],
                    'value_range': [float(input_min.min()), float(input_max.max())],
                    'zero_variance_count': zero_var_input
                },
                'target_stats': {
                    'mean_range': [float(target_mean.min()), float(target_mean.max())],
                    'std_range': [float(target_std.min()), float(target_std.max())],
                    'value_range': [float(target_min.min()), float(target_max.max())],
                    'zero_variance_count': zero_var_target
                }
            }
            
            # Outlier detection
            print(f"\n🎯 Outlier Detection Analysis:")
            
            # Calculate z-scores and identify outliers (>3 standard deviations)
            input_z_scores = torch.abs((sample_input_tensor - input_mean) / (input_std + 1e-8))
            target_z_scores = torch.abs((sample_target_tensor - target_mean) / (target_std + 1e-8))
            
            input_outliers = (input_z_scores > 3.0).sum().item()
            target_outliers = (target_z_scores > 3.0).sum().item()
            
            input_outlier_percentage = (input_outliers / sample_input_tensor.numel()) * 100
            target_outlier_percentage = (target_outliers / sample_target_tensor.numel()) * 100
            
            print(f"   • Input outliers (>3σ): {input_outliers:,} ({input_outlier_percentage:.2f}%)")
            print(f"   • Target outliers (>3σ): {target_outliers:,} ({target_outlier_percentage:.2f}%)")
            
            if input_outlier_percentage > 5.0:
                quality_report['warnings'].append(f"High input outlier percentage: {input_outlier_percentage:.2f}%")
            
            if target_outlier_percentage > 5.0:
                quality_report['warnings'].append(f"High target outlier percentage: {target_outlier_percentage:.2f}%")
            
            quality_report['metrics']['outliers'] = {
                'input_outliers': input_outliers,
                'input_outlier_percentage': float(input_outlier_percentage),
                'target_outliers': target_outliers,
                'target_outlier_percentage': float(target_outlier_percentage)
            }
            
            # Temporal consistency check within segments
            print(f"\n⏱️ Temporal Consistency Analysis:")
            
            # Calculate temporal derivatives (changes between consecutive timesteps)
            input_derivatives = torch.diff(sample_input_tensor, dim=1)  # [segments, seq_len-1, features]
            target_derivatives = torch.diff(sample_target_tensor, dim=1)
            
            # Calculate mean absolute temporal changes
            input_temporal_change = torch.mean(torch.abs(input_derivatives))
            target_temporal_change = torch.mean(torch.abs(target_derivatives))
            
            # Check for sudden jumps (large temporal derivatives)
            input_large_jumps = (torch.abs(input_derivatives) > 5.0).sum().item()
            target_large_jumps = (torch.abs(target_derivatives) > 2.0).sum().item()
            
            print(f"   • Mean input temporal change: {input_temporal_change:.4f}")
            print(f"   • Mean target temporal change: {target_temporal_change:.4f}")
            print(f"   • Large input jumps: {input_large_jumps:,}")
            print(f"   • Large target jumps: {target_large_jumps:,}")
            
            if input_large_jumps > sample_input_tensor.numel() * 0.01:  # >1% of values
                quality_report['warnings'].append(f"Many large temporal jumps in input data: {input_large_jumps}")
            
            if target_large_jumps > sample_target_tensor.numel() * 0.01:
                quality_report['warnings'].append(f"Many large temporal jumps in target data: {target_large_jumps}")
            
            quality_report['metrics']['temporal_consistency'] = {
                'input_temporal_change': float(input_temporal_change),
                'target_temporal_change': float(target_temporal_change),
                'input_large_jumps': input_large_jumps,
                'target_large_jumps': target_large_jumps
            }
            
            # Feature correlation analysis
            print(f"\n🔗 Feature Correlation Analysis:")
            
            # Flatten tensors for correlation analysis
            input_flat = sample_input_tensor.view(-1, sample_input_tensor.size(-1)).numpy()
            target_flat = sample_target_tensor.view(-1, sample_target_tensor.size(-1)).numpy()
            
            # Calculate correlation matrices with proper handling of zero variance features
            high_corr_input_pairs = []
            high_corr_target_pairs = []
            
            try:
                # Check for sufficient variance before correlation calculation
                input_std_flat = np.std(input_flat, axis=0)
                valid_input_features = input_std_flat > 1e-10  # Only features with sufficient variance
                
                if np.sum(valid_input_features) > 1:  # Need at least 2 features for correlation
                    valid_input_data = input_flat[:, valid_input_features]
                    valid_input_feature_names = [input_features[i] for i in range(len(input_features)) if valid_input_features[i]]
                    
                    # Suppress numpy warnings for correlation calculation
                    with np.errstate(divide='ignore', invalid='ignore'):
                        input_corr_matrix = np.corrcoef(valid_input_data.T)
                    
                    # Replace NaN values with 0 (no correlation)
                    input_corr_matrix = np.nan_to_num(input_corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Find highly correlated feature pairs
                    for i in range(len(valid_input_feature_names)):
                        for j in range(i+1, len(valid_input_feature_names)):
                            if abs(input_corr_matrix[i, j]) > 0.95:
                                high_corr_input_pairs.append((valid_input_feature_names[i], valid_input_feature_names[j], input_corr_matrix[i, j]))
                else:
                    print(f"   • Skipping input correlation analysis - insufficient valid features ({np.sum(valid_input_features)})")
                    
            except Exception as e:
                print(f"   • Input correlation analysis failed: {str(e)}")
                quality_report['warnings'].append("Input correlation analysis failed due to data issues")
            
            try:
                # Same for action features
                target_std_flat = np.std(target_flat, axis=0)
                valid_target_features = target_std_flat > 1e-10  # Only features with sufficient variance
                
                if np.sum(valid_target_features) > 1:  # Need at least 2 features for correlation
                    valid_target_data = target_flat[:, valid_target_features]
                    valid_target_feature_names = [action_features[i] for i in range(len(action_features)) if valid_target_features[i]]
                    
                    # Suppress numpy warnings for correlation calculation
                    with np.errstate(divide='ignore', invalid='ignore'):
                        target_corr_matrix = np.corrcoef(valid_target_data.T)
                    
                    # Replace NaN values with 0 (no correlation)
                    target_corr_matrix = np.nan_to_num(target_corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Find highly correlated feature pairs
                    for i in range(len(valid_target_feature_names)):
                        for j in range(i+1, len(valid_target_feature_names)):
                            if abs(target_corr_matrix[i, j]) > 0.95:
                                high_corr_target_pairs.append((valid_target_feature_names[i], valid_target_feature_names[j], target_corr_matrix[i, j]))
                else:
                    print(f"   • Skipping target correlation analysis - insufficient valid features ({np.sum(valid_target_features)})")
                    
            except Exception as e:
                print(f"   • Target correlation analysis failed: {str(e)}")
                quality_report['warnings'].append("Target correlation analysis failed due to data issues")
            
            print(f"   • Highly correlated input pairs (>0.95): {len(high_corr_input_pairs)}")
            print(f"   • Highly correlated target pairs (>0.95): {len(high_corr_target_pairs)}")
            
            if len(high_corr_input_pairs) > 0:
                quality_report['warnings'].append(f"{len(high_corr_input_pairs)} highly correlated input feature pairs detected")
                for pair in high_corr_input_pairs[:3]:  # Show first 3
                    print(f"     - {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
            
            if len(high_corr_target_pairs) > 0:
                quality_report['warnings'].append(f"{len(high_corr_target_pairs)} highly correlated target feature pairs detected")
                for pair in high_corr_target_pairs[:3]:  # Show first 3
                    print(f"     - {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
            
            # Normalization verification
            print(f"\n🎛️ Normalization Verification:")
            
            feature_scaler = dataset.get_scalers()
            
            scaler_quality = {
                'feature_scaler_available': feature_scaler is not None
            }
            
            if feature_scaler:
                # Check if data is approximately normalized (mean~0, std~1)
                normalized_mean = torch.abs(input_mean).max().item()
                normalized_std_deviation = torch.abs(input_std - 1.0).max().item()
                
                print(f"   • Input normalization quality:")
                print(f"     - Max absolute mean: {normalized_mean:.4f} (should be ~0)")
                print(f"     - Max std deviation from 1: {normalized_std_deviation:.4f} (should be ~0)")
                
                scaler_quality['input_normalization'] = {
                    'max_abs_mean': float(normalized_mean),
                    'max_std_deviation': float(normalized_std_deviation)
                }
                
                if normalized_mean > 0.1:
                    quality_report['warnings'].append(f"Input normalization: high mean {normalized_mean:.4f}")
                if normalized_std_deviation > 0.2:
                    quality_report['warnings'].append(f"Input normalization: std deviation {normalized_std_deviation:.4f}")
            
            # In unified approach, target uses same normalization as input
            target_normalized_mean = torch.abs(target_mean).max().item()
            target_normalized_std_dev = torch.abs(target_std - 1.0).max().item()
            
            print(f"   • Target normalization quality:")
            print(f"     - Max absolute mean: {target_normalized_mean:.4f} (should be ~0)")
            print(f"     - Max std deviation from 1: {target_normalized_std_dev:.4f} (should be ~0)")
            
            scaler_quality['target_normalization'] = {
                'max_abs_mean': float(target_normalized_mean),
                'max_std_deviation': float(target_normalized_std_dev)
            }
            
            quality_report['metrics']['normalization'] = scaler_quality
            
            # Context feature validation
            print(f"\n🎯 Context Feature Validation:")
            
            if len(context_features) > 0:
                # Count context features by type (no need to access actual data)
                expert_context_count = 0
                tire_context_count = 0
                
                for feature_name in context_features:
                    if 'expert' in feature_name.lower():
                        expert_context_count += 1
                    elif 'tire' in feature_name.lower() or 'grip' in feature_name.lower():
                        tire_context_count += 1
                
                print(f"   • Expert context features: {expert_context_count}")
                print(f"   • Tire grip context features: {tire_context_count}")
                
                quality_report['metrics']['context_features'] = {
                    'expert_features': expert_context_count,
                    'tire_features': tire_context_count
                }
                
                if expert_context_count == 0:
                    quality_report['warnings'].append("No expert context features detected")
            else:
                quality_report['warnings'].append("No context features available")
            
            # Generate overall quality assessment
            print(f"\n🎖️ QUALITY ASSESSMENT SUMMARY:")
            
            critical_count = len(quality_report['critical_issues'])
            warning_count = len(quality_report['warnings'])
            
            if critical_count == 0 and warning_count == 0:
                overall_quality = "EXCELLENT"
                quality_color = "🟢"
            elif critical_count == 0 and warning_count <= 2:
                overall_quality = "GOOD"
                quality_color = "🟡"
            elif critical_count <= 1 and warning_count <= 5:
                overall_quality = "ACCEPTABLE"
                quality_color = "🟠"
            else:
                overall_quality = "POOR"
                quality_color = "🔴"
            
            quality_report['overall_quality'] = overall_quality
            
            print(f"   {quality_color} Overall Quality: {overall_quality}")
            print(f"   • Critical Issues: {critical_count}")
            print(f"   • Warnings: {warning_count}")
            
            # Generate recommendations
            recommendations = []
            
            if segment_info['num_segments'] < min_segments_required:
                recommendations.append("Collect more training segments for better model generalization")
            
            if zero_var_input > 0:
                recommendations.append("Remove or fix zero-variance input features")
            
            if zero_var_target > 0:
                recommendations.append("Investigate zero-variance target features - may indicate data collection issues")
            
            if input_outlier_percentage > 5.0 or target_outlier_percentage > 5.0:
                recommendations.append("Consider outlier removal or robust scaling methods")
            
            if len(high_corr_input_pairs) > 5:
                recommendations.append("Consider dimensionality reduction for highly correlated input features")
            
            if not feature_scaler:
                recommendations.append("Ensure proper data normalization with StandardScaler")
            
            if len(context_features) == 0:
                recommendations.append("Add contextual features for better learning guidance")
            
            quality_report['recommendations'] = recommendations
            
            # Print detailed summary
            print(f"\n📋 DETAILED SUMMARY:")
            print(f"{'─'*60}")
            
            if quality_report['critical_issues']:
                print(f"🚨 Critical Issues ({len(quality_report['critical_issues'])}):")
                for i, issue in enumerate(quality_report['critical_issues'], 1):
                    print(f"   {i}. {issue}")
                print()
            
            if quality_report['warnings']:
                print(f"⚠️ Warnings ({len(quality_report['warnings'])}):")
                for i, warning in enumerate(quality_report['warnings'], 1):
                    print(f"   {i}. {warning}")
                print()
            
            if quality_report['recommendations']:
                print(f"💡 Recommendations ({len(quality_report['recommendations'])}):")
                for i, rec in enumerate(quality_report['recommendations'], 1):
                    print(f"   {i}. {rec}")
                print()
            
            print(f"✅ Dataset quality check completed for {dataset_name}")
            print(f"📊 Ready for training: {'Yes' if overall_quality in ['EXCELLENT', 'GOOD'] else 'With caution' if overall_quality == 'ACCEPTABLE' else 'Not recommended'}")
            
        except Exception as e:
            error_msg = f"Error during dataset quality analysis: {str(e)}"
            print(f"❌ {error_msg}")
            quality_report['critical_issues'].append(error_msg)
            quality_report['overall_quality'] = 'ERROR'
        
        print(f"{'='*80}\n")
        
        return make_json_safe(quality_report)




class _RunningFeatureStats:
    """Streaming aggregator for per-feature statistics across all dataset chunks."""

    def __init__(self, num_features: int):
        self.counts = np.zeros(num_features, dtype=np.float64)
        self.sums = np.zeros(num_features, dtype=np.float64)
        self.sums_sq = np.zeros(num_features, dtype=np.float64)

    def update(self, matrix: np.ndarray) -> None:
        if matrix.size == 0:
            return
        # Ensure float64 for numerical stability
        data = matrix.astype(np.float64, copy=False)
        self.counts += data.shape[0]
        self.sums += np.sum(data, axis=0, dtype=np.float64)
        self.sums_sq += np.sum(np.square(data, dtype=np.float64), axis=0, dtype=np.float64)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = np.divide(self.sums, self.counts, out=np.zeros_like(self.sums), where=self.counts > 0)
        averaged_sq = np.divide(self.sums_sq, self.counts, out=np.zeros_like(self.sums_sq), where=self.counts > 0)
        variances = averaged_sq - np.square(means)
        variances = np.clip(variances, 0.0, None)
        counts = np.where(self.counts > 0, self.counts, 1.0)
        return counts, means, variances


class TelemetryActionDataset(Dataset):
    """
    Simplified Large Chunk Dataset for GPU Batch Training
    
    ASSUMPTIONS:
    - Each chunk contains 10k+ segments (large chunks)
    - Load one chunk at a time during training  
    - Batch segments within chunk for efficient GPU training
    - No complex fallback mechanisms needed
    
    APPROACH:
    - __len__ returns number of chunks available
    - __getitem__ loads one chunk and returns GPU-sized batches from it
    - Simple and efficient for large-scale training
    """
    
    def __init__(self,
                 data_cache,
                 segments_cache_key: str,
                 fixed_segment_length: int,
                 batch_size: int = 32):
        """
        Initialize simplified dataset for large chunks
        
        Args:
            data_cache: Training cache instance to load chunks from
            segments_cache_key: Cache key where chunks are stored
            fixed_segment_length: Required length for all segments
            batch_size: GPU batch size for processing segments within chunks
        """
        self.data_cache = data_cache
        self.segments_cache_key = segments_cache_key
        self.fixed_segment_length = fixed_segment_length
        self.batch_size = batch_size
        
        # Get basic chunk information
        self.chunk_count = self._count_chunks()
        self.unified_features = self._get_feature_names()

        # Initialize feature preprocessing
        self.feature_scaler = PerFeatureScaler(self.unified_features)
        self._features_fitted = False

        print(f"[INFO] ✓ Simplified dataset initialized: {self.chunk_count} large chunks")
        print(f"[INFO] ✓ GPU batch size: {batch_size}")
        print(f"[INFO] ✓ Features: {len(self.unified_features)}")
        print(f"[INFO] ✓ Assuming 10k+ segments per chunk")
    
    def _count_chunks(self) -> int:
        """Count available chunks without loading them all into memory"""
        try:
            chunks_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)
            chunk_count = 0
            for _ in chunks_iterator:
                chunk_count += 1
            print(f"[INFO] Found {chunk_count} chunks available")
            return chunk_count
        except Exception as e:
            raise ValueError(f"Failed to count chunks: {str(e)}")
    
    def _get_feature_names(self) -> List[str]:
        """Extract feature names from first chunk without caching everything"""
        try:
            chunks_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)
            first_chunk = next(chunks_iterator)
            
            # Get first record and first timestep to extract feature names
            first_record = first_chunk.iloc[0].to_dict()
            first_timestep_data = first_record[0]  # Column 0 contains first timestep
            
            if isinstance(first_timestep_data, dict):
                feature_names = list(first_timestep_data.keys())
                print(f"[INFO] Extracted {len(feature_names)} feature names")
                return feature_names
            else:
                raise ValueError(f"Unexpected data structure: expected dict, got {type(first_timestep_data)}")
                
        except Exception as e:
            raise ValueError(f"Failed to extract feature names: {str(e)}")
    
    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        """Load a specific chunk by index"""
        chunks_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)
        
        # Skip to the desired chunk
        for i, chunk_df in enumerate(chunks_iterator):
            if i == chunk_idx:
                chunk_records = chunk_df.to_dict('records')
                print(f"[INFO] Loaded chunk {chunk_idx} with {len(chunk_records)} segments")
                return chunk_records
        
        raise IndexError(f"Chunk {chunk_idx} not found")
    
    
    def _ensure_features_fitted(self):
        """Ensure feature scaling is fitted using sample from first chunk"""
        if self._features_fitted:
            return
        
        print(f"[INFO] Fitting feature scaling using all available chunks...")

        stats = _RunningFeatureStats(len(self.unified_features))
        total_rows = 0

        chunk_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)

        for chunk_idx, chunk_df in enumerate(chunk_iterator):
            chunk_records = chunk_df.to_dict('records')
            chunk_rows: List[List[float]] = []

            for record in chunk_records:
                for timestep in range(self.fixed_segment_length):
                    timestep_data = record.get(timestep)
                    if not isinstance(timestep_data, dict):
                        continue

                    row: List[float] = []
                    for feature in self.unified_features:
                        value = timestep_data.get(feature, 0.0)
                        try:
                            row.append(float(value))
                        except (ValueError, TypeError):
                            row.append(0.0)
                    chunk_rows.append(row)

            if chunk_rows:
                chunk_matrix = np.array(chunk_rows, dtype=np.float32)
                if np.isnan(chunk_matrix).any() or np.isinf(chunk_matrix).any():
                    print(f"[WARNING] NaN/Inf detected in chunk {chunk_idx}; applying cleaning before stats update")
                    chunk_matrix = np.nan_to_num(chunk_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

                stats.update(chunk_matrix)
                total_rows += chunk_matrix.shape[0]

            print(f"[DEBUG] Processed chunk {chunk_idx}: rows accumulated={total_rows}")

        if total_rows == 0:
            raise ValueError("No data available across chunks to fit feature scaler")

        counts, means, variances = stats.finalize()
        self.feature_scaler = PerFeatureScaler.from_feature_statistics(
            self.unified_features,
            means,
            variances,
            counts,
            scaler_factory=self.feature_scaler.scaler_factory if self.feature_scaler else None
        )
        self._features_fitted = True

        print(f"[INFO] ✓ Feature scaling fitted across {total_rows} timesteps from all chunks")
    
    def _build_matrix(self, data_list: List[Dict[str, Any]], feature_names: List[str]) -> np.ndarray:
        """Extract features and build a matrix from list of dictionaries"""
        matrix = []
        for record in data_list:
            row = []
            for feature in feature_names:
                value = record.get(feature, 0.0)
                # Convert to float, handle various data types
                try:
                    if isinstance(value, (int, float)):
                        row.append(float(value))
                    elif isinstance(value, str):
                        row.append(float(value) if value.replace('.', '').replace('-', '').isdigit() else 0.0)
                    else:
                        row.append(0.0)
                except (ValueError, TypeError):
                    row.append(0.0)
            matrix.append(row)
        
        return np.array(matrix, dtype=np.float32)
    
    def __len__(self) -> int:
        """Return number of chunks available"""
        return self.chunk_count
    
    def _process_segment_record(self, segment_record: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert one cached segment record into input/target sequences (numpy arrays).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (input_seq, target_seq) each with shape [seq_len-1, features]
            or None if the record cannot be processed.
        """
        try:
            # Reconstruct the time series from the flattened segment data
            sequence_data: List[List[float]] = []

            for timestep in range(self.fixed_segment_length):
                timestep_data = segment_record.get(timestep, None)
                if not isinstance(timestep_data, dict):
                    # Skip malformed timesteps
                    return None

                feature_array: List[float] = []
                for feature_name in self.unified_features:
                    try:
                        value = float(timestep_data.get(feature_name, 0.0))
                    except (ValueError, TypeError):
                        value = 0.0
                    feature_array.append(value)
                sequence_data.append(feature_array)

            if len(sequence_data) != self.fixed_segment_length:
                return None

            # [timesteps, features]
            sequence_matrix = np.array(sequence_data, dtype=np.float32)
            
            # Check for NaN/Inf in raw data before scaling
            if np.isnan(sequence_matrix).any() or np.isinf(sequence_matrix).any():
                print(f"[WARNING] NaN/Inf detected in raw segment data - skipping segment")
                return None

            # Scale per-timestep using fitted scaler
            scaled_sequence = self.feature_scaler.transform(sequence_matrix)
            
            # Check for NaN/Inf after scaling
            if np.isnan(scaled_sequence).any() or np.isinf(scaled_sequence).any():
                print(f"[WARNING] NaN/Inf detected after scaling - skipping segment")
                return None

            # Teacher-forcing style next-step target
            input_sequence = scaled_sequence[:-1]   # [seq_len-1, features]
            target_sequence = scaled_sequence[1:]   # [seq_len-1, features]
            return input_sequence, target_sequence
        except Exception:
            return None



    def get_chunk_batches(self, chunk_idx: int):
        """
        Generator that yields GPU-sized batches from a large chunk
        
        Args:
            chunk_idx: Index of the chunk to process
            
        Yields:
            Tuples of (batch_inputs, batch_targets) for GPU training
        """
        if chunk_idx >= self.chunk_count:
            raise IndexError(f"Chunk index {chunk_idx} out of range")
            
        # Ensure features are fitted
        self._ensure_features_fitted()
        
        # Load the chunk
        chunk_records = self._load_chunk(chunk_idx)
        
        batch_inputs = []
        batch_targets = []
        
        for segment_record in chunk_records:
            processed = self._process_segment_record(segment_record)
            if processed is None:
                continue
                
            input_sequence, target_sequence = processed
            batch_inputs.append(input_sequence)
            batch_targets.append(target_sequence)
            
            # Yield batch when we reach desired size
            if len(batch_inputs) >= self.batch_size:
                batch_input_tensor = torch.FloatTensor(np.stack(batch_inputs))
                batch_target_tensor = torch.FloatTensor(np.stack(batch_targets))
                yield batch_input_tensor, batch_target_tensor
                
                # Reset for next batch
                batch_inputs = []
                batch_targets = []
        
        # Yield remaining segments as final batch
        if batch_inputs:
            batch_input_tensor = torch.FloatTensor(np.stack(batch_inputs))
            batch_target_tensor = torch.FloatTensor(np.stack(batch_targets))
            yield batch_input_tensor, batch_target_tensor

    def __getitem__(self, chunk_idx: int):
        """Return chunk index for simple iteration - actual batching done by get_chunk_batches"""
        return chunk_idx
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """
        Get feature names for unified state prediction model.
        
        In the unified approach, both input and target use the same feature set,
        as we're predicting the next state which has the same structure as input state.
        
        Returns:
            Tuple of (input_features, target_features) where both are identical unified_features
        """
        return self.unified_features, self.unified_features
    
    def get_scalers(self) -> PerFeatureScaler:
        return self.feature_scaler
    
    def get_segment_info(self) -> Dict[str, Any]:
        return {
            "num_chunks": self.chunk_count,
            "segment_length": self.fixed_segment_length,
            "sequence_length": self.fixed_segment_length - 1,  # Input/target sequences are segment_length - 1
            "total_features": len(self.unified_features),
            "feature_names": self.unified_features,
            "tensor_shapes": {
                "input": [self.fixed_segment_length - 1, len(self.unified_features)],
                "target": [self.fixed_segment_length - 1, len(self.unified_features)]
            }
        }
    
    def get_context_feature_names(self) -> List[str]:
        """Get context feature names from unified features"""
        # Filter for context features (exclude basic telemetry features)
        context_features = []
        for feature in self.unified_features:
            if any(context_prefix in feature.lower() for context_prefix in [
                'expert_', 'tire_', 'grip_', 'distance_', 
                'velocity_alignment', 'speed_difference'
            ]):
                context_features.append(feature)
        return context_features

    # Compatibility helper used by quality checks
    def get_input_feature_names(self) -> List[str]:
        """Return the unified input feature names (for quality reports)."""
        return list(self.unified_features)
    
    @staticmethod
    def validate_segments(unified_segments: List[List[Dict[str, Any]]], expected_length: int) -> Dict[str, Any]:
        """
        Validate that all segments have the expected fixed length
        
        Args:
            unified_segments: List of segments to validate
            expected_length: Expected length for each segment
            
        Returns:
            Dict with validation results
        """
        errors = []
        valid_segments = 0
        invalid_segments = 0
        
        for i, segment in enumerate(unified_segments):
            if len(segment) == expected_length:
                valid_segments += 1
            else:
                invalid_segments += 1
                errors.append(f"Segment {i}: length {len(segment)} != expected {expected_length}")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'num_segments': len(unified_segments),
            'statistics': {
                'valid_segments': valid_segments,
                'invalid_segments': invalid_segments,
                'expected_length': expected_length
            }
        }
    


class ExpertActionTrainer:
    """
    Unified State Trainer class for the Expert Action Transformer.
    
    This trainer handles unified feature sequences for next-state prediction training.
    Each training sample consists of input state and target next state, where states
    contain unified features (context + actions combined).
    
    DATASET QUALITY VALIDATION USAGE:
    
    # Example 1: Basic quality check before training
    model = ExpertActionTransformer(total_features_count=46)
    trainer = ExpertActionTrainer(model)
    
    # Check dataset quality and get detailed report
    quality_report = model.check_dataset_quality(train_dataset, "My Training Dataset")
    print(f"Dataset quality: {quality_report['overall_quality']}")
    
    # Example 2: Comprehensive pre-training validation (recommended)
    try:
        validation_results = trainer.validate_training_data_quality(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            require_good_quality=True  # Will raise exception if quality is poor
        )
        
        # Proceed with training if validation passes
        training_results = trainer.train(train_dataset, val_dataset, epochs=50)
        
    except ValueError as e:
        print(f"Training blocked due to data quality issues: {e}")
        # Fix data quality issues before proceeding
    
    # Example 3: Quality check with custom handling
    validation_results = trainer.validate_training_data_quality(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        require_good_quality=False  # Don't raise exception, just report
    )
    
    if validation_results['overall_recommendation'] in ['EXCELLENT', 'GOOD']:
        print("✅ Starting training with high-quality data")
        training_results = trainer.train(train_dataset, val_dataset, epochs=100)
    elif validation_results['overall_recommendation'] == 'ACCEPTABLE':
        print("⚠️ Starting training with acceptable data, monitoring closely")
        training_results = trainer.train(train_dataset, val_dataset, epochs=30, patience=5)
    else:
        print("🔴 Skipping training due to poor data quality")
        print("Critical issues:", validation_results['aggregated_critical_issues'])
        print("Recommendations:", validation_results['aggregated_recommendations'])
    """
    
    def __init__(self,
                 model: ExpertActionTransformer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """
        Initialize the simplified trainer for large chunks
        
        Args:
            model: The transformer model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        # Device and precision setup
        self.device = device
        self.model = model.to(device)
        self._cuda = device.startswith('cuda') and torch.cuda.is_available()

        # Enable cuDNN benchmark for faster kernels on GPU
        try:
            if self._cuda:
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Mixed precision (AMP) configuration
        # Use float16 for broader compatibility; bfloat16 can have tensor dtype mismatches
        self.amp_dtype = torch.float16 if self._cuda else None
        
        # GradScaler for float16 AMP 
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._cuda and self.amp_dtype == torch.float16)

        # Favor higher matmul precision/tensor cores where applicable (PyTorch 2.0+)
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss function - will be contextual weighted loss if context available
        self.criterion = nn.MSELoss()  # Fallback for when no context is available
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def set_scalers_from_dataset(self, dataset: TelemetryActionDataset):
        """
        Extract and set scaler from the unified dataset on the model.
        
        Args:
            dataset: Training dataset containing fitted scaler
        """
        feature_scaler = dataset.get_scalers()
        
        # Set scaler on the model
        self.model.set_scalers(feature_scaler)
        
        print(f"[INFO] Set unified feature scaler on model: {'✓' if feature_scaler else '✗'}")
    
    def train_epoch(self, dataset: TelemetryActionDataset) -> float:
        """
        Simplified training: one chunk at a time with GPU batching
        
        Args:
            dataset: Dataset with large chunks (10k+ segments each)
            
        Returns:
            Average loss across all batches
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"[INFO] Starting training epoch with {len(dataset)} large chunks")
        
        # Process chunks in random order
        chunk_indices = np.random.permutation(len(dataset))
        
        for chunk_idx in chunk_indices:
            print(f"[INFO] Processing chunk {chunk_idx+1}/{len(dataset)}")
            
            try:
                # Get GPU batches from this chunk
                for batch_inputs, batch_targets in dataset.get_chunk_batches(chunk_idx):
                    # Move to device
                    batch_inputs = batch_inputs.to(self.device, non_blocking=self._cuda)
                    batch_targets = batch_targets.to(self.device, non_blocking=self._cuda)
                    
                    # Forward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                        target_seq_len = batch_targets.shape[1]
                        predictions = self.model(
                            unified_input=batch_inputs,
                            prediction_steps=target_seq_len,
                            use_teacher_forcing=True,
                        )
                    
                    # Verify shapes match before computing loss
                    if predictions.shape != batch_targets.shape:
                        print(f"[WARNING] Shape mismatch: predictions {predictions.shape} vs targets {batch_targets.shape}")
                        # Try to align shapes if possible
                        if predictions.shape[1] != batch_targets.shape[1]:
                            min_seq_len = min(predictions.shape[1], batch_targets.shape[1])
                            predictions = predictions[:, :min_seq_len, :]
                            batch_targets = batch_targets[:, :min_seq_len, :]
                            print(f"[INFO] Aligned to shape: {predictions.shape}")
                    
                    loss = self.model.unified_loss(predictions=predictions, targets=batch_targets)
                    
                    # Skip batch if loss is NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[WARNING] Skipping batch due to NaN/Inf loss: {loss}")
                        continue
                    
                    # Backward pass
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clean up
                    del batch_inputs, batch_targets, predictions, loss
                    
                    if num_batches % 100 == 0:
                        print(f"[INFO] Processed {num_batches} batches, current avg loss: {total_loss/num_batches:.6f}")
                        
            except Exception as e:
                print(f"[WARNING] Failed to process chunk {chunk_idx}: {str(e)}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[INFO] ✓ Epoch complete: {num_batches} batches processed, avg loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def validate_epoch(self, dataset: TelemetryActionDataset) -> float:
        """
        Simplified validation: one chunk at a time with GPU batching
        
        Args:
            dataset: Dataset with large chunks
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        print(f"[INFO] Starting validation with {len(dataset)} large chunks")
        
        with torch.no_grad():
            for chunk_idx in range(len(dataset)):
                try:
                    # Get GPU batches from this chunk
                    for batch_inputs, batch_targets in dataset.get_chunk_batches(chunk_idx):
                        batch_inputs = batch_inputs.to(self.device, non_blocking=self._cuda)
                        batch_targets = batch_targets.to(self.device, non_blocking=self._cuda)
                        
                        target_seq_len = batch_targets.shape[1]
                        predictions = self.model(
                            unified_input=batch_inputs,
                            prediction_steps=target_seq_len,
                            use_teacher_forcing=True,
                        )
                        
                        loss = self.model.unified_loss(predictions=predictions, targets=batch_targets)
                        total_loss += loss.item()
                        num_batches += 1
                        
                        del batch_inputs, batch_targets, predictions, loss
                        
                except Exception as e:
                    print(f"[WARNING] Failed to process validation chunk {chunk_idx}: {str(e)}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[INFO] ✓ Validation complete: {num_batches} batches processed, avg loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def train(self, 
              train_dataset: TelemetryActionDataset,
              val_dataset: Optional[TelemetryActionDataset] = None,
              epochs: int = 50,
              patience: int = 15,
              save_best: bool = True) -> Dict[str, Any]:
        """
        Main training loop for the Expert Action Transformer model.
        
        This function implements a complete training pipeline with the following key components:
        
        1. TRAINING LOOP ARCHITECTURE:
           - Iterates through specified number of epochs
           - Each epoch processes entire training dataset via train_epoch()
           - Optionally validates on validation set via validate_epoch()
           - Tracks losses and training progress over time
        
        2. ADAPTIVE LEARNING RATE:
           - Uses ReduceLROnPlateau scheduler to automatically reduce learning rate
           - Monitors validation loss; reduces LR when loss plateaus
           - Helps model converge to better local minima during training
        
        3. EARLY STOPPING MECHANISM:
           - Prevents overfitting by stopping training when validation loss stops improving
           - Tracks consecutive epochs without validation loss improvement
           - Stops training if no improvement for 'patience' epochs
           - Balances training time vs model generalization
        
        4. BEST MODEL CHECKPOINTING:
           - Automatically saves model state when validation loss reaches new minimum
           - Stores complete model weights, epoch number, and corresponding loss
           - Loads best performing model at end of training (not final epoch)
           - Ensures returned model represents peak performance, not final iteration
        
        5. PROGRESS MONITORING:
           - Prints comprehensive training metrics each epoch
           - Tracks both training and validation losses over time  
           - Displays current learning rate for debugging purposes
           - Maintains history for post-training analysis
        
        Training Process Flow:
        - Initialize tracking variables (best_val_loss, epochs_without_improvement)
        - For each epoch:
          a) Train model on training data using train_epoch()
          b) Evaluate model on validation data using validate_epoch() 
          c) Update learning rate scheduler based on validation performance
          d) Check if current model is best seen so far (lowest val loss)
          e) Save model checkpoint if it's the best performing
          f) Check early stopping criteria
          g) Print epoch statistics
        - After training completion, load the best saved model
        - Return comprehensive training statistics and metrics
        
        Args:
            train_dataloader: DataLoader with training sequences (telemetry -> expert actions)
            val_dataloader: Optional DataLoader for validation during training
            epochs: Maximum number of training epochs to run
            patience: Number of epochs to wait for val loss improvement before early stopping  
            save_best: Whether to checkpoint and restore best performing model
            
        Returns:
            Dictionary containing complete training history:
            - train_losses: List of training losses per epoch
            - val_losses: List of validation losses per epoch  
            - best_val_loss: Lowest validation loss achieved
            - epochs_trained: Actual number of epochs completed
            - final_lr: Final learning rate after training
        """
        print(f"[INFO] Starting segmented training for {epochs} epochs on {self.device}")
        print(f"[INFO] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Display dataset and contextual feature information
        segment_info = train_dataset.get_segment_info()
        print(f"[INFO] Training dataset: {segment_info['num_chunks']} chunks, "
              f"each segment with length {segment_info['segment_length']}")
        
        context_feature_names = train_dataset.get_context_feature_names()
        if context_feature_names:
            context_summary = self.model.get_contextual_features_summary(context_feature_names)
            print(f"[INFO] Contextual weighted loss enabled with {len(context_summary['available_for_weighting'])} weighting features:")
            if context_summary['tire_grip_features']:
                print(f"[INFO]   - Tire grip features: {context_summary['tire_grip_features']}")
            if context_summary['expert_features']:
                print(f"[INFO]   - Expert alignment features: {context_summary['expert_features']}")
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Train on segments
            train_loss = self.train_epoch(train_dataset)
            self.train_losses.append(train_loss)
            
            # Validate on segments
            if val_dataset is not None:
                val_loss = self.validate_epoch(val_dataset)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping and best model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if save_best:
                        self.best_model_state = {
                            'state_dict': self.model.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss
                        }
                else:
                    epochs_without_improvement += 1
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"[INFO] Early stopping after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}")
        
        # Load best model if available
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state['state_dict'])
            print(f"[INFO] Loaded best model from epoch {self.best_model_state['epoch']+1}")
        
        # Set scalers on the model from training dataset for inference
        self.set_scalers_from_dataset(train_dataset)
        
        return make_json_safe({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'final_lr': self.optimizer.param_groups[0]['lr']
        })
    
    def validate_training_data_quality(self, 
                                     train_dataset: TelemetryActionDataset,
                                     val_dataset: Optional[TelemetryActionDataset] = None,
                                     require_good_quality: bool = True) -> Dict[str, Any]:
        """
        Validate dataset quality before training and provide comprehensive quality reports.
        
        This method performs thorough quality checks on training and validation datasets
        to ensure they meet the requirements for effective transformer training. It can
        optionally enforce quality standards by raising exceptions for poor quality data.
        
        Args:
            train_dataset: Training dataset to validate
            val_dataset: Optional validation dataset to validate
            require_good_quality: If True, raises exception for poor quality datasets
            
        Returns:
            Dictionary containing quality reports for all datasets
            
        Raises:
            ValueError: If require_good_quality=True and dataset quality is poor
        """
        print(f"\n🔍 COMPREHENSIVE DATASET QUALITY VALIDATION")
        print(f"{'='*80}")
        
        quality_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'datasets_checked': [],
            'overall_recommendation': 'UNKNOWN'
        }
        
        # Check training dataset quality
        try:
            train_quality = self.model.check_dataset_quality(train_dataset, "Training Dataset")
        except Exception as e:
            print(f"[ERROR] check_dataset_quality failed: {str(e)}")
            raise e
        quality_results['training_quality'] = train_quality
        quality_results['datasets_checked'].append('training')
        
        # Check validation dataset quality if provided
        val_quality = None
        if val_dataset is not None:
            print(f"Validating validation dataset...")
            val_quality = self.model.check_dataset_quality(val_dataset, "Validation Dataset")
            quality_results['validation_quality'] = val_quality
            quality_results['datasets_checked'].append('validation')
        
        # Determine overall recommendation
        train_quality_level = train_quality.get('overall_quality', 'ERROR')
        val_quality_level = val_quality.get('overall_quality', 'EXCELLENT') if val_quality else 'N/A'
        
        # Overall recommendation based on worst dataset quality
        quality_levels = ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'ERROR']
        
        worst_quality = train_quality_level
        if val_quality_level != 'N/A':
            train_idx = quality_levels.index(train_quality_level) if train_quality_level in quality_levels else -1
            val_idx = quality_levels.index(val_quality_level) if val_quality_level in quality_levels else -1
            worst_idx = max(train_idx, val_idx)
            worst_quality = quality_levels[worst_idx] if worst_idx >= 0 else 'ERROR'
        
        quality_results['overall_recommendation'] = worst_quality
        
        # Print overall summary
        print(f"\n🎯 OVERALL TRAINING READINESS ASSESSMENT:")
        print(f"{'─'*60}")
        print(f"Training Dataset Quality: {train_quality_level}")
        if val_quality:
            print(f"Validation Dataset Quality: {val_quality_level}")
        print(f"Overall Recommendation: {worst_quality}")
        
        # Training readiness recommendations
        if worst_quality == 'EXCELLENT':
            recommendation = "✅ READY FOR TRAINING - Excellent data quality detected"
            print(f"{recommendation}")
        elif worst_quality == 'GOOD':
            recommendation = "✅ READY FOR TRAINING - Good data quality with minor issues"
            print(f"{recommendation}")
        elif worst_quality == 'ACCEPTABLE':
            recommendation = "⚠️ TRAINING WITH CAUTION - Acceptable quality but monitor training closely"
            print(f"{recommendation}")
        elif worst_quality == 'POOR':
            recommendation = "🔴 NOT RECOMMENDED FOR TRAINING - Poor data quality detected"
            print(f"{recommendation}")
        else:
            recommendation = "❌ TRAINING BLOCKED - Critical data quality issues"
            print(f"{recommendation}")
        
        quality_results['training_recommendation'] = recommendation
        
        # Aggregate all critical issues and recommendations
        all_critical_issues = train_quality.get('critical_issues', [])
        all_recommendations = train_quality.get('recommendations', [])
        
        if val_quality:
            all_critical_issues.extend(val_quality.get('critical_issues', []))
            all_recommendations.extend(val_quality.get('recommendations', []))
        
        quality_results['aggregated_critical_issues'] = list(set(all_critical_issues))
        quality_results['aggregated_recommendations'] = list(set(all_recommendations))
        
        if all_critical_issues:
            print(f"\n🚨 ALL CRITICAL ISSUES TO ADDRESS:")
            for i, issue in enumerate(all_critical_issues, 1):
                print(f"   {i}. {issue}")
        
        if all_recommendations:
            print(f"\n💡 AGGREGATED RECOMMENDATIONS:")
            for i, rec in enumerate(all_recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Enforce quality requirements if requested
        if require_good_quality and worst_quality in ['POOR', 'ERROR']:
            error_msg = (f"Dataset quality check failed: {worst_quality} quality detected. "
                        f"Critical issues: {len(all_critical_issues)}. "
                        f"Set require_good_quality=False to proceed anyway.")
            print(f"\n❌ {error_msg}")
            raise ValueError(error_msg)
        
        print(f"{'='*80}\n")
        
        return make_json_safe(quality_results)
    
    def evaluate(self, dataset: TelemetryActionDataset) -> Dict[str, Any]:
        """
        Run a targeted single-sequence evaluation comparing teacher-forced and
        autoregressive predictions against the ground-truth target sequence.
        
        Args:
            dataset: Dataset providing cached unified telemetry segments.
            
        Returns:
            Dictionary describing prediction quality for both inference modes.
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty – cannot run evaluation")

        self.model.eval()

        print("[INFO] Evaluating model on a single representative segment...")

        # Ensure feature scaling is ready and gather one valid segment
        dataset._ensure_features_fitted()

        selected_input = None
        selected_target = None
        segment_metadata: Dict[str, int] = {}

        for chunk_idx in range(len(dataset)):
            chunk_records = dataset._load_chunk(chunk_idx)
            for segment_idx, segment_record in enumerate(chunk_records):
                processed = dataset._process_segment_record(segment_record)
                if processed is None:
                    continue

                input_seq_np, target_seq_np = processed
                selected_input = input_seq_np
                selected_target = target_seq_np
                segment_metadata = {
                    'chunk_index': chunk_idx,
                    'segment_index': segment_idx
                }
                break

            if selected_input is not None:
                break

        if selected_input is None or selected_target is None:
            raise ValueError("Unable to locate a valid segment for evaluation")

        print(
            f"[INFO] Using chunk {segment_metadata['chunk_index']} "
            f"segment {segment_metadata['segment_index']} for evaluation"
        )

        input_sequence_serialized = selected_input.tolist()

        # Prepare tensors
        input_tensor = torch.from_numpy(np.expand_dims(selected_input, axis=0)).to(self.device)
        target_tensor = torch.from_numpy(np.expand_dims(selected_target, axis=0)).to(self.device)

        target_seq_len = target_tensor.shape[1]
        feature_count = target_tensor.shape[-1]

        print(f"[INFO] Sequence length: {target_seq_len} | Features: {feature_count}")

        def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
            diff = predictions - targets
            mse = torch.mean(diff ** 2).item()
            mae = torch.mean(torch.abs(diff)).item()
            rmse = float(np.sqrt(max(mse, 0.0)))
            max_abs = torch.max(torch.abs(diff)).item()
            final_step_mae = torch.mean(torch.abs(diff[:, -1, :])).item()
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'max_abs_error': max_abs,
                'final_step_mae': final_step_mae
            }

        with torch.no_grad():
            # Teacher-forced prediction (mirrors training behaviour)
            with torch.autocast(
                device_type='cuda',
                dtype=self.amp_dtype,
                enabled=self._cuda and self.amp_dtype is not None
            ):
                teacher_predictions = self.model(
                    unified_input=input_tensor,
                    prediction_steps=target_seq_len,
                    use_teacher_forcing=True
                )

            # Autoregressive rollout using model predictions as future inputs
            with torch.autocast(
                device_type='cuda',
                dtype=self.amp_dtype,
                enabled=self._cuda and self.amp_dtype is not None
            ):
                autoregressive_predictions = self.model(
                    unified_input=input_tensor,
                    prediction_steps=target_seq_len,
                    use_teacher_forcing=False
                )

        # Align shapes if autoregressive path returns squeezed tensor for single step
        if autoregressive_predictions.dim() == 3:
            autoreg_tensor = autoregressive_predictions
        else:
            autoreg_tensor = autoregressive_predictions.unsqueeze(0)

        if teacher_predictions.shape[1] != target_seq_len:
            raise ValueError(
                f"Teacher-forced prediction length {teacher_predictions.shape[1]} "
                f"does not match target length {target_seq_len}"
            )

        if autoreg_tensor.shape[1] != target_seq_len:
            raise ValueError(
                f"Autoregressive prediction length {autoreg_tensor.shape[1]} "
                f"does not match target length {target_seq_len}"
            )

        teacher_tensor = teacher_predictions.float()
        autoreg_tensor = autoreg_tensor.float()
        target_eval_tensor = target_tensor.float()

        teacher_metrics = _compute_metrics(teacher_tensor, target_eval_tensor)
        autoreg_metrics = _compute_metrics(autoreg_tensor, target_eval_tensor)

        # Collect comparison snapshots (detach to CPU for readability)
        teacher_np = teacher_tensor.detach().cpu().numpy()
        autoreg_np = autoreg_tensor.detach().cpu().numpy()
        target_np = target_eval_tensor.detach().cpu().numpy()
        input_np = input_tensor.detach().cpu().numpy()

        # Build per-step breakdowns
        teacher_per_step: List[Dict[str, Any]] = []
        autoreg_per_step: List[Dict[str, Any]] = []

        teacher_seq = teacher_np[0]
        autoreg_seq = autoreg_np[0]
        input_seq = input_np[0]
        target_seq = target_np[0]

        if input_seq.shape[0] != target_seq_len:
            raise ValueError(
                f"Input sequence length {input_seq.shape[0]} does not match target length {target_seq_len}"
            )

        if teacher_seq.shape[0] != target_seq_len:
            raise ValueError(
                f"Teacher-forced sequence length {teacher_seq.shape[0]} does not match target length {target_seq_len}"
            )

        if autoreg_seq.shape[0] != target_seq_len:
            raise ValueError(
                f"Autoregressive sequence length {autoreg_seq.shape[0]} does not match target length {target_seq_len}"
            )

        for step_idx in range(target_seq_len):
            teacher_per_step.append({
                'step': step_idx,
                'input_state': input_seq[step_idx].tolist(),
                'prediction': teacher_seq[step_idx].tolist(),
                'target_state': target_seq[step_idx].tolist()
            })

            autoreg_per_step.append({
                'step': step_idx,
                'generated_state': autoreg_seq[step_idx].tolist(),
                'target_state': target_seq[step_idx].tolist()
            })

        print(
            "[INFO] Teacher-forced MSE: {:.6f} | MAE: {:.6f}".format(
                teacher_metrics['mse'], teacher_metrics['mae']
            )
        )
        print(
            "[INFO] Autoregressive MSE: {:.6f} | MAE: {:.6f}".format(
                autoreg_metrics['mse'], autoreg_metrics['mae']
            )
        )

        return make_json_safe({
            'segment': segment_metadata,
            'sequence_length': int(target_seq_len),
            'feature_count': int(feature_count),
            'input_sequence': input_sequence_serialized,
            'teacher_forcing': {
                'metrics': teacher_metrics,
                'predictions': teacher_np.tolist(),
                'per_step': teacher_per_step
            },
            'autoregressive': {
                'metrics': autoreg_metrics,
                'predictions': autoreg_np.tolist(),
                'per_step': autoreg_per_step
            },
            'target_sequence': target_np.tolist()
        })
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and training state"""
        return make_json_safe({
            'model_config': {
                'total_features': self.model.total_features_count,
                'd_model': self.model.d_model,
                'sequence_length': self.model.sequence_length
            },
            'training_config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'device': self.device
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'epochs_trained': len(self.train_losses)
            },
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        })