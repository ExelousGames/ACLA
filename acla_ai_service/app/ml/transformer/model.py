
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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
from app.domain.tire_grip_features import TireGripFeatureCatalog
from app.domain.expert_features import ExpertFeatureCatalog
from app.domain.telemetry import TelemetryFeatures, _safe_float  # Force unbuffered output for real-time print statements
import os

# Extracted in refactor/hexagonal-v4 — pulled out of this 2740-line module.
# Re-imported so the trainer and entry function (still in this file)
# see the same symbols.
from app.storage.datasets.transformer_scaler import PerFeatureScaler, _RunningFeatureStats
from app.storage.datasets.telemetry_dataset import TelemetryActionDataset

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)


# --- class PerFeatureScaler: removed; see app/ml/transformer/scaler.py or app/storage/datasets/telemetry_dataset.py ---
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
    ExpertActionTransformer - Universal Expert Trajectory Prediction Model
    
    WHAT IT ACTUALLY DOES:
    This model learns to predict the optimal expert trajectory from ANY initial state.
    Whether the driver is on the expert line, off the line, or in the pits, the model
    predicts the sequence of states that leads towards optimal expert behavior.
    Instead of separating context and actions, this model treats all features as a 
    unified state vector and predicts the progression towards the expert ideal.
    
    UNIFIED FEATURE APPROACH:
    - Input: Complete state vector [context_features + action_features] at timestep t
    - Output: Complete state vector [context_features + action_features] at timestep t+1
    - The attention mechanism learns relationships between ALL features (context and actions)
    
    CORE FUNCTIONALITY:
    1. UNIVERSAL TRAJECTORY PREDICTION: Predicts optimal path from any starting condition
       (e.g., staying on expert line, recovering to line, merging from pits)
    2. UNIFIED FEATURES: Context and actions are treated as single feature vector
    3. ATTENTION LEARNING: Model learns relationships between context and optimal actions
    4. SEQUENTIAL MODELING: Learns temporal dependencies in trajectory progressions
    
    HOW IT WORKS:
    The model takes a unified feature vector containing:
    - Contextual features: gap analysis, track info, tire grip, environmental data
    - Action features: gas, brake, steer_angle, gear
    
    It predicts the next timestep's complete feature vector:
    - Next contextual features: updated gap analysis showing progress toward expert line
    - Next action features: OPTIMAL gas, brake, steer_angle, gear for the situation
    
    TRAINING PROCESS:
    The model is trained on a diverse set of driving scenarios including:
    - Expert line maintenance (staying on line)
    - Recovery trajectories (moving from off-line to on-line)
    - Pit exit and merging behaviors
    Each timestep contains the complete state. It learns to predict the next
    optimal state from the current state, allowing the attention mechanism to discover
    relationships between context and optimal actions naturally.
    
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

    # Optional per-feature loss weighting (configured post-scaler)
        self._loss_weight_vector = None
        self._loss_weight_names = []

    # Cache causal masks per device/length to avoid regenerating every batch
        self._mask_cache = {}
    
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
        # Reset cached loss weights – they depend on feature ordering
        if feature_scaler and feature_scaler.is_fitted():
            self._loss_weight_names = feature_scaler.get_feature_names()
            self._loss_weight_vector = torch.ones(len(self._loss_weight_names), dtype=torch.float32)
        else:
            self._loss_weight_names = []
            self._loss_weight_vector = None

    def configure_loss_weights(
        self,
        overrides: Optional[Dict[str, float]] = None,
        default_weight: float = 1.0,
        brake_weight: float = 2.0,
        steering_weight: float = 2.0,
        throttle_weight: float = 0.75,
        minimum_weight: float = 0.1,
        maximum_weight: float = 5.0,
    ) -> None:
        """Configure per-feature loss weights based on feature names.

        Args:
            overrides: Explicit feature -> weight mapping to apply last.
            default_weight: Baseline weight for unspecified features.
            brake_weight: Weight for features containing "brake".
            steering_weight: Weight for features containing steering keywords.
            throttle_weight: Weight for throttle/gas features.
            minimum_weight: Lower clamp for any assigned weight.
            maximum_weight: Upper clamp for any assigned weight.
        """
        if not self.feature_scaler or not self.feature_scaler.is_fitted():
            raise RuntimeError("Loss weights require a fitted feature scaler")

        feature_names = self.feature_scaler.get_feature_names()
        weights = torch.full((len(feature_names),), float(default_weight), dtype=torch.float32)

        for idx, name in enumerate(feature_names):
            lname = name.lower()
            weight = default_weight

            if "brake" in lname or "decel" in lname:
                weight = brake_weight
            elif any(token in lname for token in ("steer", "yaw", "turn")):
                weight = steering_weight
            elif any(token in lname for token in ("gas", "throttle")):
                weight = throttle_weight

            if overrides and name in overrides:
                weight = float(overrides[name])

            weight = max(minimum_weight, min(maximum_weight, weight))
            weights[idx] = weight

        self._loss_weight_names = feature_names
        self._loss_weight_vector = weights

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
                use_teacher_forcing: bool = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            
        Args:
            unified_input: Batched input sequences.
            prediction_steps: Number of steps to predict (defaults to 1 for inference).
            use_teacher_forcing: Force teacher forcing path when True.
            padding_mask: Optional boolean mask identifying padded timesteps (True -> pad).

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

            # Prevent attention from peeking at future targets during teacher forcing
            seq_len = encoded_input.size(1)
            causal_mask = self._get_causal_mask(seq_len, encoded_input.device)
            key_padding_mask = None
            if padding_mask is not None:
                key_padding_mask = padding_mask.bool()
            
            # Process through transformer
            transformer_output = self.transformer(
                encoded_input,
                mask=causal_mask,
                src_key_padding_mask=key_padding_mask
            )
            
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
                
                # Create or reuse causal mask for current rollout length
                seq_len = encoded_input.size(1)
                causal_mask = self._get_causal_mask(seq_len, encoded_input.device)

                # Process through transformer
                transformer_output = self.transformer(encoded_input, mask=causal_mask)
                
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

    def unified_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        timestep_weights: Optional[torch.Tensor] = None,
        feature_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MSE loss for unified expert trajectory learning with NaN protection
        
        This loss learns how to predict the next optimal state in the progression 
        toward expert-level performance from any initial condition (on-line, off-line, 
        recovery, etc.).
        
        Args:
            predictions: Model predictions [batch_size, seq_len, total_features]
            targets: Target optimal unified states [batch_size, seq_len, total_features]
                    These targets represent the optimal path/actions
            
        Returns:
            MSE loss tensor (scalar) - measures accuracy of trajectory predictions
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
        
        # Element-wise squared error
        squared_error = F.mse_loss(predictions, targets, reduction='none')

        # Apply per-feature weighting if available
        weight_vector = feature_weights
        if weight_vector is None and self._loss_weight_vector is not None:
            weight_vector = self._loss_weight_vector

        if weight_vector is not None:
            weight_vector = weight_vector.to(predictions.device)
            squared_error = squared_error * weight_vector.view(1, 1, -1)

        # Apply timestep weights (e.g., braking/turning emphasis)
        if timestep_weights is not None:
            timestep_weights = timestep_weights.to(predictions.device)
            if timestep_weights.dim() == 2:
                timestep_weights = timestep_weights.unsqueeze(-1)
            squared_error = squared_error * timestep_weights

        # Compute mean loss after weighting
        loss = squared_error.mean()
        
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

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a cached float32 causal mask tailored for the given device and sequence length."""
        key = (device.type, device.index, seq_len)
        cached = self._mask_cache.get(key)
        if cached is None:
            mask = self._generate_square_subsequent_mask(seq_len).to(device=device, dtype=torch.float32)
            self._mask_cache[key] = mask
            return mask
        if cached.device != device:
            cached = cached.to(device=device, dtype=torch.float32)
            self._mask_cache[key] = cached
        return cached
    
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
        Generate human-readable driving advice/predictions from current telemetry data.
        
        This function serves as the main interface for real-time racing coaching, converting
        raw telemetry data into actionable driving advice. It predicts the optimal future
        trajectory from the current state, whether that involves maintaining the expert line,
        recovering to it, or merging from pits.
        
        INFERENCE APPROACH:
        Unlike training which uses fixed-length sequences, inference uses a single timestep
        of current telemetry data to predict a sequence of future optimal actions through
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
            feature_names = TelemetryFeatures.get_features_for_learning_expert()

        feature_count = len(feature_names)
        
        for i in range(min(sequence_length, len(predictions))):
            pred = predictions[i]
            
            # Create prediction step with ALL target features
            step_data = {
                "step": i + 1,
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


# --- class _RunningFeatureStats: removed; see app/ml/transformer/scaler.py or app/storage/datasets/telemetry_dataset.py ---
# --- class TelemetryActionDataset(Dataset): removed; see app/ml/transformer/scaler.py or app/storage/datasets/telemetry_dataset.py ---
# ExpertActionTrainer + prepare_and_train_coach_transformer_model moved to
# app/pipelines/training/transformer_trainer.py in refactor/hexagonal-v5.
# Callers should import from there directly.
