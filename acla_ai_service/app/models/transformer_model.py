

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
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterable
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import contextual feature catalogs for quality weighting
from ..services.tire_grip_analysis_service import TireGripFeatureCatalog
from ..services.imitate_expert_learning_service import (
    ExpertFeatureCatalog,
    ExpertImitateLearningService,
)
from .telemetry_models import TelemetryFeatures

# Force unbuffered output for real-time print statements
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)


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

        # Positional encoding injects timestep order so the transformer knows which state comes next;
        # without it, self-attention would treat the sequence as orderless and lose the progression signal.
        # If max_len were set to exactly sequence_length, 
        # the moment autoregressive inference asked for a timestep beyond that budget you’d hit an index error (positional cache too short). 
        # Multiplying by two gives a safety margin so the model can consume the original context and a comparably long stretch of generated future steps without rebuilding the positional encoding table.
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

    def validate_input_features(self, input_payload: Union[Dict[str, Any], Iterable[str]]) -> None:
        """Ensure that the provided payload contains every required unified feature.

        Args:
            input_payload: Mapping from feature name to value or iterable of feature names.

        Raises:
            RuntimeError: If the model does not have feature metadata available.
            TypeError: If the payload type is unsupported.
            KeyError: If one or more required features are missing.
        """
        required = self.feature_scaler.get_feature_names()
        if not required:
            raise RuntimeError(
                "Model has no registered feature names; cannot validate input payload."
            )

        if isinstance(input_payload, dict):
            provided = {str(key) for key in input_payload.keys()}
        elif isinstance(input_payload, (list, tuple, set)):
            provided = {str(key) for key in input_payload}
        else:
            try:
                provided = {str(key) for key in input_payload}  # type: ignore[arg-type]
            except TypeError as type_err:
                raise TypeError(
                    "Input payload must be a mapping or iterable of feature names"
                ) from type_err

        missing = [name for name in required if name not in provided]
        if missing:
            preview = ', '.join(missing[:5])
            suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ''
            raise KeyError(f"Missing required feature(s) for transformer inference: {preview}{suffix}")
    
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

            if self.feature_scaler is None or not self.feature_scaler.is_fitted():
                raise RuntimeError(
                    "ExpertActionTransformer requires a fitted PerFeatureScaler for inference"
                )

            required_features = self.feature_scaler.get_feature_names()

            if isinstance(current_telemetry, dict):
                self.validate_input_features(current_telemetry)
                ordered_values = [current_telemetry[feature] for feature in required_features]
            else:
                raise TypeError(
                    "current_telemetry must be a mapping of feature values or an ordered iterable of feature values"
                )

            try:
                numeric_vector = [float(value) for value in ordered_values]
            except (TypeError, ValueError) as cast_err:
                raise ValueError("Telemetry payload must contain numeric values for all required features") from cast_err
            
            # Scale telemetry using the fitted feature scaler before inference
            numeric_array = np.asarray(numeric_vector, dtype=np.float32).reshape(1, -1)
            scaled_array = self.feature_scaler.transform(numeric_array).astype(np.float32, copy=False)

            # For inference, create a single timestep input (not repeated sequence)
            # The model will use autoregressive generation to create the sequence
            combined_tensor = torch.from_numpy(scaled_array).to(device).unsqueeze(1)
            
            # Generate predictions
            try:
                self.eval()
                with torch.no_grad():
                    predictions = self.predict_segment_progression(
                        combined_input=combined_tensor,
                        prediction_length=sequence_length
                    )
            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {str(e)}")
            
            # Convert predictions back to original scale using the feature scaler
            predictions_cpu = predictions.detach().cpu()
            original_shape = predictions_cpu.shape
            predictions_2d = predictions_cpu.reshape(-1, original_shape[-1]).numpy()
            unscaled_predictions = self.feature_scaler.inverse_transform(predictions_2d)
            predictions_np = unscaled_predictions.reshape(original_shape)[0]
            
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
    
    def _create_sequence_predictions(self, predictions: np.ndarray, sequence_length: int) -> List[Dict[str, Any]]:
        """Create sequence of future predictions from unified feature vectors - includes ALL targets"""
        sequence = []
        
        # Prefer the scaler's feature metadata to keep ordering identical to training tensors.
        if self.feature_scaler is not None and self.feature_scaler.is_fitted():
            feature_names = self.feature_scaler.get_feature_names()
        else:
            feature_names = TelemetryFeatures.get_features_for_imitate_expert()

        feature_count = len(feature_names)
        
        for i in range(min(sequence_length, len(predictions))):
            pred = predictions[i]
            
            # Create prediction step with ALL target features
            step_data = {
                "step": i + 1,
                "time_ahead": f"{(i + 1) * self.time_step_seconds:.1f}s",
                "all_targets": {}
            }
            
            # Include all features in the prediction output
            for feature_idx, value in enumerate(pred):
                if feature_idx < feature_count:
                    feature_name = feature_names[feature_idx]
                else:
                    feature_name = f"feature_{feature_idx}"
                step_data["all_targets"][feature_name] = round(float(value), 4)
            
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
        
        if self.feature_scaler is None:
            raise RuntimeError(
                "Cannot serialize transformer without a fitted PerFeatureScaler"
            )

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
                raise ValueError(
                    "Serialized model is missing 'feature_scaler' data; this is required to restore feature ordering"
                )
            if not isinstance(scaler_payload, dict):
                raise ValueError("Serialized model 'feature_scaler' must be a dictionary payload")

            restored_scaler = PerFeatureScaler.from_serializable(scaler_payload)
            print("[INFO] - Restored unified feature scaler from serialized payload")

            model.set_scalers(restored_scaler)
            
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

        evaluation_segments: List[Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray], Dict[str, int]]] = []
        max_segments_to_collect = 10

        for chunk_idx in range(len(dataset)):
            chunk_records = dataset._load_chunk(chunk_idx)
            for segment_idx, segment_record in enumerate(chunk_records):
                processed = dataset._process_segment_record(segment_record)
                if processed is None:
                    continue

                metadata = {
                    'chunk_index': chunk_idx,
                    'segment_index': segment_idx
                }
                evaluation_segments.append((segment_record, processed, metadata))

                if len(evaluation_segments) >= max_segments_to_collect:
                    break

            if len(evaluation_segments) >= max_segments_to_collect:
                break

        if not evaluation_segments:
            raise ValueError("Unable to locate a valid segment for evaluation")

        primary_segment_raw, (selected_input, selected_target), segment_metadata = evaluation_segments[0]

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

        # Restore original scale for easier interpretation if scaler is available
        feature_scaler = dataset.get_scalers()

        def _inverse_scale(array: np.ndarray) -> np.ndarray:
            flat = array.reshape(-1, feature_count)
            unscaled = feature_scaler.inverse_transform(flat)
            return np.asarray(unscaled, dtype=np.float32).reshape(array.shape)

        teacher_np_unscaled = _inverse_scale(teacher_np)
        autoreg_np_unscaled = _inverse_scale(autoreg_np)
        target_np_unscaled = _inverse_scale(target_np)
        input_np_unscaled = _inverse_scale(input_np)

        # Build per-step breakdowns
        teacher_per_step: List[Dict[str, Any]] = []
        autoreg_per_step: List[Dict[str, Any]] = []

        teacher_seq = teacher_np[0]
        teacher_seq_unscaled = teacher_np_unscaled[0]
        autoreg_seq = autoreg_np[0]
        autoreg_seq_unscaled = autoreg_np_unscaled[0]
        input_seq = input_np[0]
        input_seq_unscaled = input_np_unscaled[0]
        target_seq = target_np[0]
        target_seq_unscaled = target_np_unscaled[0]
        feature_names, _ = dataset.get_feature_names()
        if len(feature_names) != feature_count:
            feature_names = [f"feature_{idx}" for idx in range(feature_count)]

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

        def _named(values: np.ndarray) -> Dict[str, float]:
            return {name: float(val) for name, val in zip(feature_names, values)}

        teacher_predictions_named: List[Dict[str, float]] = []
        teacher_predictions_unscaled_named: List[Dict[str, float]] = []
        autoreg_predictions_named: List[Dict[str, float]] = []
        autoreg_predictions_unscaled_named: List[Dict[str, float]] = []
        target_named: List[Dict[str, float]] = []
        target_unscaled_named: List[Dict[str, float]] = []
        input_named: List[Dict[str, float]] = []
        input_unscaled_named: List[Dict[str, float]] = []

        for step_idx in range(target_seq_len):
            teacher_named = _named(teacher_seq[step_idx])
            teacher_unscaled_named = _named(teacher_seq_unscaled[step_idx])
            autoreg_named = _named(autoreg_seq[step_idx])
            autoreg_unscaled_named = _named(autoreg_seq_unscaled[step_idx])
            target_named_step = _named(target_seq[step_idx])
            target_unscaled_named_step = _named(target_seq_unscaled[step_idx])
            input_named_step = _named(input_seq[step_idx])
            input_unscaled_named_step = _named(input_seq_unscaled[step_idx])

            teacher_per_step.append({
                'step': step_idx,
                'input_state': input_named_step,
                'input_state_unscaled': input_unscaled_named_step,
                'prediction': teacher_named,
                'prediction_unscaled': teacher_unscaled_named,
                'target_state': target_named_step,
                'target_state_unscaled': target_unscaled_named_step
            })

            autoreg_per_step.append({
                'step': step_idx,
                'generated_state': autoreg_named,
                'generated_state_unscaled': autoreg_unscaled_named,
                'target_state': target_named_step,
                'target_state_unscaled': target_unscaled_named_step
            })

            teacher_predictions_named.append(teacher_named)
            teacher_predictions_unscaled_named.append(teacher_unscaled_named)
            autoreg_predictions_named.append(autoreg_named)
            autoreg_predictions_unscaled_named.append(autoreg_unscaled_named)
            target_named.append(target_named_step)
            target_unscaled_named.append(target_unscaled_named_step)
            input_named.append(input_named_step)
            input_unscaled_named.append(input_unscaled_named_step)

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
        evaluation_payload = {
            'segment': segment_metadata,
            'sequence_length': int(target_seq_len),
            'feature_count': int(feature_count),
            'feature_names': feature_names,
            'input_sequence': input_sequence_serialized,
             'input_sequence_unscaled': input_np_unscaled[0].tolist(),
            'input_sequence_named': input_named,
            'input_sequence_unscaled_named': input_unscaled_named,
            'teacher_forcing': {
                'metrics': teacher_metrics,
                'predictions': teacher_np.tolist(),
                'predictions_named': teacher_predictions_named,
                'predictions_unscaled': teacher_np_unscaled.tolist(),
                'predictions_unscaled_named': teacher_predictions_unscaled_named,
                'per_step': teacher_per_step
            },
            'autoregressive': {
                'metrics': autoreg_metrics,
                'predictions': autoreg_np.tolist(),
                'predictions_named': autoreg_predictions_named,
                'predictions_unscaled': autoreg_np_unscaled.tolist(),
                'predictions_unscaled_named': autoreg_predictions_unscaled_named,
                'per_step': autoreg_per_step
            },
            'target_sequence': target_np.tolist(),
            'target_sequence_unscaled': target_np_unscaled.tolist(),
            'target_sequence_named': target_named,
            'target_sequence_unscaled_named': target_unscaled_named,
            'segments_sampled': [meta for _, _, meta in evaluation_segments]
        }

        def _segment_to_timesteps(segment_record: Dict[str, Any]) -> List[Dict[str, Any]]:
            timesteps: List[Dict[str, Any]] = []
            for timestep in range(dataset.fixed_segment_length):
                timestep_data = segment_record.get(timestep)
                if isinstance(timestep_data, dict):
                    timesteps.append(timestep_data)
            return timesteps

        visualization_segments_rendered = 0
        visualization_metadata: List[Dict[str, int]] = []

        segments_to_visualize: List[List[Dict[str, Any]]] = []
        for raw_segment, _, meta in evaluation_segments[:max_segments_to_collect]:
            timesteps = _segment_to_timesteps(raw_segment)
            if len(timesteps) < 2:
                continue
            segments_to_visualize.append(timesteps)
            visualization_metadata.append(meta)

        if segments_to_visualize:
            try:
                viz_service = ExpertImitateLearningService()
                viz_service.visualize_optimal_segments(
                    segments_to_visualize,
                    max_segments=len(segments_to_visualize),
                    show=False,
                    output_dir=None,
                    return_base64=False,
                )
                visualization_segments_rendered = len(segments_to_visualize)
                for meta in visualization_metadata:
                    print(
                        f"[INFO] Rendered evaluation visualization for chunk {meta['chunk_index']} "
                        f"segment {meta['segment_index']}"
                    )
            except Exception as service_error:
                print(f"[WARN] Evaluation visualization failed: {service_error}")

        evaluation_payload['visualizations_rendered'] = visualization_segments_rendered
        evaluation_payload['visualization_metadata'] = visualization_metadata

        evaluation_payload_safe = make_json_safe(evaluation_payload)

        try:
            output_dir = Path(__file__).resolve().parent.parent / 'scripts' / 'debug_output' / 'transformer_eval'
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = (
                f"eval_segment_{segment_metadata['chunk_index']}_"
                f"{segment_metadata['segment_index']}_{timestamp}.json"
            )
            output_path = output_dir / filename
            with output_path.open('w', encoding='utf-8') as outfile:
                json.dump(evaluation_payload_safe, outfile, indent=2)
            print(f"[INFO] Raw prediction comparisons saved to {output_path}")
        except Exception as file_error:
            print(f"[WARN] Failed to persist raw prediction outputs: {file_error}")

        return evaluation_payload_safe
    
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