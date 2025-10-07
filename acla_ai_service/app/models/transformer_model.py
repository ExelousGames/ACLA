

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
from typing import Dict, List, Any, Optional, Tuple, Union
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
        
        # Scaler for normalization during inference (single scaler for unified features)
        self.feature_scaler: Optional[StandardScaler] = None
        
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
    
    def set_scalers(self, feature_scaler: Optional[StandardScaler] = None):
        """
        Set the scaler for unified features (context + actions combined).
        
        Args:
            feature_scaler: StandardScaler fitted on unified feature vectors
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
            
            # For teacher forcing, we predict the next states directly
            # Take all predictions except the first one (since input[0] -> target[1], etc.)
            if sequence_predictions.shape[1] >= prediction_steps:
                result = sequence_predictions[:, -prediction_steps:, :]
            else:
                # If we need more predictions than available, pad with last prediction
                last_pred = sequence_predictions[:, -1:, :].repeat(1, prediction_steps, 1)
                result = last_pred
                
            return result
        
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
        MSE loss for unified expert improvement learning
        
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
        
        # Base MSE loss over all unified features
        loss = F.mse_loss(predictions, targets, reduction='mean')
        
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
            scaler_buffer = io.BytesIO()
            import pickle
            pickle.dump(self.feature_scaler, scaler_buffer)
            feature_scaler_data = base64.b64encode(scaler_buffer.getvalue()).decode('utf-8')
        
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
            import pickle
            
            # Try new unified scaler first
            if 'feature_scaler' in serialized_data and serialized_data['feature_scaler'] is not None:
                try:
                    scaler_bytes = base64.b64decode(serialized_data['feature_scaler'])
                    scaler_buffer = io.BytesIO(scaler_bytes)
                    model.feature_scaler = pickle.load(scaler_buffer)
                    print("[INFO] - Restored unified feature scaler")
                except Exception as e:
                    print(f"[WARNING] Failed to restore unified feature scaler: {e}")
                    model.feature_scaler = None
            # Backward compatibility: try to load old input_scaler as feature_scaler
            elif 'input_scaler' in serialized_data and serialized_data['input_scaler'] is not None:
                try:
                    scaler_bytes = base64.b64decode(serialized_data['input_scaler'])
                    scaler_buffer = io.BytesIO(scaler_bytes)
                    model.feature_scaler = pickle.load(scaler_buffer)
                    print("[INFO] - Restored feature scaler from legacy input_scaler (backward compatibility)")
                except Exception as e:
                    print(f"[WARNING] Failed to restore legacy input scaler: {e}")
                    model.feature_scaler = None
            else:
                model.feature_scaler = None
            
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
            
            # Collect sample data
            sample_inputs = []
            sample_targets = []
            nan_count = 0
            inf_count = 0
            
            for idx in sample_indices:
                try:
                    input_seq, target_seq = dataset[idx]
                    sample_inputs.append(input_seq)
                    sample_targets.append(target_seq)
                    
                    # Check for NaN/Inf in this sample
                    nan_count += torch.isnan(input_seq).sum().item() + torch.isnan(target_seq).sum().item()
                    inf_count += torch.isinf(input_seq).sum().item() + torch.isinf(target_seq).sum().item()
                    
                except Exception as e:
                    quality_report['critical_issues'].append(f"Failed to load segment {idx}: {str(e)}")
            
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




class TelemetryActionDataset(Dataset):
    """
    Unified State Dataset for sequence prediction - Streaming Implementation
    
    This dataset efficiently handles large datasets by streaming segments from cache
    without loading all segments into memory at once. PyTorch DataLoader handles
    the efficient loading during training.
    
    Each segment represents a temporal sequence where:
    - Input: state[t] = [context_features + action_features] at timestep t
    - Target: state[t+1] = [context_features + action_features] at timestep t+1
    
    The attention mechanism learns relationships between all features naturally.
    """
    
    def __init__(self,
                 data_cache,
                 segments_cache_key: str,
                 fixed_segment_length: int):
        """
        Initialize the streaming unified state dataset
        
        Args:
            data_cache: Training cache instance to load segments from
            segments_cache_key: Cache key where segments are stored
            fixed_segment_length: Required length for all segments
        """
        self.data_cache = data_cache
        self.segments_cache_key = segments_cache_key
        self.fixed_segment_length = fixed_segment_length
        
        # Cache the manifest path and session files list for efficiency
        self._manifest_path = None
        self._session_files = []
        
        # Load segment index from cache without loading actual segment data
        self._build_segment_index()
        
        # Initialize feature preprocessing components
        self.feature_scaler = StandardScaler()
        self._features_fitted = False
        
        print(f"[INFO] ✓ Streaming dataset initialized with {self.total_segments} segments")
        print(f"[INFO] ✓ Fixed segment length: {fixed_segment_length}")
        print(f"[INFO] ✓ Memory efficient streaming from cache key: {segments_cache_key}")
        print(f"[INFO] ✓ Direct parquet file access for optimal performance")
    
    def _build_segment_index(self):
        """Build index of segments from cached chunk files with new chunked structure"""
        try:
            # The segments_cache_key is already a complete cache key, use it directly
            cached_info = self.data_cache._get_cache_metadata(self.segments_cache_key, max_age_hours=24)
            
            if not cached_info:
                raise ValueError(f"No cached segments found for key: {self.segments_cache_key}")
            
            # Get manifest file path directly
            manifest_file = Path(cached_info["file_path"])
            if not manifest_file.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
            
            # Read chunk files from manifest
            with open(manifest_file, 'r') as f:
                chunk_files = [line.strip() for line in f.readlines() if line.strip()]
            
            if not chunk_files:
                raise ValueError("No chunk files found in manifest")
            
            # Build segment index by extracting segments from cached chunks
            self.segment_index = []  # [(chunk_file_path, segment_idx_in_chunk)]
            self.total_segments = 0
            
            print(f"[INFO] Loading segments from {len(chunk_files)} chunk files...")
            
            for chunk_filename in chunk_files:
                chunk_path = manifest_file.parent / chunk_filename
                if not chunk_path.exists():
                    continue
                
                try:
                    # Load chunk data
                    chunk_df = pd.read_parquet(chunk_path)
                    
                    # Extract chunk data - cache stored it as single record
                    for _, chunk_row in chunk_df.iterrows():
                        chunk_dict = chunk_row.to_dict()
                        
                        # Extract segments from the chunk
                        segments = chunk_dict.get('segments', [])
                        if not segments:
                            print(f"[WARNING] No segments found in chunk: {chunk_filename}")
                            continue
                        
                        # Add each segment to index
                        for segment_idx, segment_data in enumerate(segments):
                            if len(segment_data) == self.fixed_segment_length:
                                self.segment_index.append((chunk_path, f"{chunk_dict.get('batch_number', 0)}_{segment_idx}"))
                                self.total_segments += 1
                            else:
                                print(f"[WARNING] Segment {segment_idx} in chunk {chunk_filename} has wrong length {len(segment_data)}, expected {self.fixed_segment_length}")
                            
                except Exception as e:
                    print(f"[WARNING] Failed to process chunk file {chunk_filename}: {str(e)}")
                    continue
            
            if self.total_segments == 0:
                raise ValueError("No valid segments found in chunk files")
            
            # Load feature names from first segment
            first_chunk_path, first_segment_id = self.segment_index[0]
            first_segment_data = self._load_segment_on_demand(first_chunk_path, first_segment_id)
            
            if len(first_segment_data) != self.fixed_segment_length:
                raise ValueError(f"Segment length mismatch: {len(first_segment_data)} != {self.fixed_segment_length}")
            
            self.unified_features = list(first_segment_data[0].keys())
            
            print(f"[INFO] ✓ Segment index built: {self.total_segments} segments across {len(chunk_files)} chunk files")
            print(f"[INFO] ✓ Found {len(self.unified_features)} unified features")
            
        except Exception as e:
            raise ValueError(f"Failed to build segment index: {str(e)}")
    
    def _load_segment_on_demand(self, chunk_path: Path, segment_id: str) -> List[Dict[str, Any]]:
        """Load a specific segment from chunk file by segment_id"""
        try:
            # Load chunk data from parquet file
            chunk_df = pd.read_parquet(chunk_path)
            
            # Extract the chunk data (cache stored it as single record)
            chunk_row = chunk_df.iloc[0]  # First (and only) row contains the chunk
            chunk_dict = chunk_row.to_dict()
            
            # Extract segments from chunk
            segments = chunk_dict.get('segments', [])
            if not segments:
                raise ValueError(f"No segments found in chunk: {chunk_path}")
            
            # Parse segment_id to get batch_number and segment_index
            batch_number, segment_idx = segment_id.split('_')
            segment_idx = int(segment_idx)
            
            # Get the specific segment
            if segment_idx >= len(segments):
                raise ValueError(f"Segment index {segment_idx} out of range for chunk with {len(segments)} segments")
            
            segment_data = segments[segment_idx]
            
            # Validate segment length
            if len(segment_data) != self.fixed_segment_length:
                raise ValueError(f"Segment length mismatch: {len(segment_data)} != {self.fixed_segment_length}")
            
            return segment_data
            
        except Exception as e:
            raise ValueError(f"Failed to load segment {segment_id} from {chunk_path}: {str(e)}")
    
    def _ensure_features_fitted(self):
        """Ensure feature scaling is fitted using sampling approach"""
        if self._features_fitted:
            return
        
        print(f"[INFO] Fitting feature scaling using sample segments...")
        
        # Sample a few segments to fit the scaler (memory efficient)
        sample_size = min(100, self.total_segments)  # Sample max 100 segments
        import random
        sample_indices = random.sample(range(self.total_segments), sample_size)
        
        all_sample_data = []
        for idx in sample_indices:
            chunk_path, segment_id = self.segment_index[idx]
            segment_data = self._load_segment_on_demand(chunk_path, segment_id)
            all_sample_data.extend(segment_data)
        
        # Build feature matrix from sample
        feature_matrix = self._build_matrix(all_sample_data, self.unified_features)
        
        # Fit scaler
        self.feature_scaler.fit(feature_matrix)
        self._features_fitted = True
        
        print(f"[INFO] ✓ Feature scaling fitted using {len(all_sample_data)} sample records")
        
        # Clear sample data immediately
        del all_sample_data, feature_matrix
    
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
        return self.total_segments
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample (segment) - loads on-demand for memory efficiency
        
        Returns:
            Tuple of (input_sequence, target_sequence) tensors
        """
        # Ensure features are fitted
        self._ensure_features_fitted()
        
        # Get segment location from index
        chunk_path, segment_id = self.segment_index[idx]
        
        # Load segment from chunk file on-demand
        segment = self._load_segment_on_demand(chunk_path, segment_id)
        
        # Build feature matrix for this segment
        feature_matrix = self._build_matrix(segment, self.unified_features)
        
        # Scale features
        scaled_features = self.feature_scaler.transform(feature_matrix)
        
        # Create input sequence (all timesteps except last)
        input_sequence = scaled_features[:-1]  # [seq_len-1, features]
        
        # Create target sequence (all timesteps except first)
        target_sequence = scaled_features[1:]  # [seq_len-1, features]
        
        return torch.FloatTensor(input_sequence), torch.FloatTensor(target_sequence)
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """
        Get feature names for unified state prediction model.
        
        In the unified approach, both input and target use the same feature set,
        as we're predicting the next state which has the same structure as input state.
        
        Returns:
            Tuple of (input_features, target_features) where both are identical unified_features
        """
        return self.unified_features, self.unified_features
    
    def get_scalers(self) -> StandardScaler:
        return self.feature_scaler
    
    def get_segment_info(self) -> Dict[str, Any]:
        return {
            "num_segments": self.total_segments,
            "segment_length": self.fixed_segment_length,
            "sequence_length": self.fixed_segment_length - 1,  # Input/target sequences are segment_length - 1
            "total_features": len(self.unified_features),
            "total_samples": self.total_segments * (self.fixed_segment_length - 1),  # Total training samples
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
    
    @staticmethod
    def validate_segments(unified_segments: List[List[Dict[str, Any]]],
                         expected_length: int) -> Dict[str, Any]:
        """
        Validate unified segment data before creating dataset
        
        Args:
            unified_segments: List of unified segments to validate
            expected_length: Expected length for all segments
            
        Returns:
            Dict with validation results and statistics
        """
        validation_result = {
            'is_valid': True,
            'num_segments': len(unified_segments),
            'expected_length': expected_length,
            'errors': [],
            'warnings': [],
            'statistics': {
                'segment_lengths': [],
                'valid_segments': 0,
                'invalid_segments': 0
            }
        }
        
        if len(unified_segments) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("No segments provided")
            return validation_result
        
        # Validate each segment
        for i, segment in enumerate(unified_segments):
            seg_len = len(segment)
            validation_result['statistics']['segment_lengths'].append(seg_len)
            
            segment_valid = True
            
            if seg_len != expected_length:
                validation_result['errors'].append(f"Segment {i}: length {seg_len} != expected {expected_length}")
                segment_valid = False
            
            if segment_valid:
                validation_result['statistics']['valid_segments'] += 1
            else:
                validation_result['statistics']['invalid_segments'] += 1
                validation_result['is_valid'] = False
        
        # Add summary statistics
        if validation_result['statistics']['segment_lengths']:
            lengths = validation_result['statistics']['segment_lengths']
            
            validation_result['statistics'].update({
                'length_range': (min(lengths), max(lengths)),
                'all_segments_same_length': len(set(lengths)) == 1,
            })
        
        return validation_result

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
        Initialize the segmented trainer
        
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
        Train for one epoch using unified state prediction
        
        Args:
            dataset: Unified feature dataset
            
        Returns:
            Average loss across all batches
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create DataLoader optimized for streaming large datasets
        # NOTE: shuffle=True for better training (segments are independent sequences)
        # NOTE: num_workers=0 to avoid multiprocessing with parquet files (can cause locks)
        # NOTE: Larger batch size for better GPU utilization with streaming
        dataloader = DataLoader(
            dataset, 
            batch_size=64,  # Larger batch for better GPU utilization
            shuffle=True,   # Shuffle segments for better training
            num_workers=0,  # Avoid multiprocessing with file I/O
            pin_memory=True if self._cuda else False,  # Pin memory for faster GPU transfer
            persistent_workers=False  # Don't persist workers to save memory
        )

        print(f"[INFO] Training on {len(dataset)} unified sequences in batches...")
        
        for batch_data in dataloader:
            # Debug: Check what the dataloader is actually returning
            if len(batch_data) != 2:
                print(f"[ERROR] DataLoader returning {len(batch_data)} items instead of 2")
                print(f"[ERROR] Items: {[type(item) for item in batch_data]}")
                raise ValueError(f"DataLoader returning {len(batch_data)} items, expected 2")
            
            batch_input_states, batch_target_states = batch_data
            # Move to device
            batch_input_states = batch_input_states.to(self.device, non_blocking=self._cuda)
            batch_target_states = batch_target_states.to(self.device, non_blocking=self._cuda)
            
            self.optimizer.zero_grad(set_to_none=True)

            # Autocast for mixed precision on GPU
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                # Forward pass: use teacher forcing for fast training
                target_seq_len = batch_target_states.shape[1]
                predictions = self.model(
                    unified_input=batch_input_states, 
                    prediction_steps=target_seq_len, 
                    use_teacher_forcing=True  # Explicit teacher forcing for training
                )
            
            # Loss computation: unified state prediction loss
            loss = self.model.unified_loss(
                predictions=predictions, 
                targets=batch_target_states
            )

            # Backward + optimizer step (with AMP support)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                # Unscale before clipping
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
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, dataset: TelemetryActionDataset) -> float:
        """
        Validate for one epoch using batch processing on fixed-size segments
        
        Args:
            dataset: Validation dataset with fixed-size segmented data
            
        Returns:
            Average validation loss across all batches
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Create DataLoader optimized for streaming validation
        dataloader = DataLoader(
            dataset, 
            batch_size=64,  # Larger batch for better GPU utilization 
            shuffle=False,  # No shuffle for validation
            num_workers=0,  # Avoid multiprocessing with file I/O
            pin_memory=True if self._cuda else False,
            persistent_workers=False
        )
        
        # Get context feature names from the dataset
        context_feature_names = dataset.get_context_feature_names()
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Debug: Check what the dataloader is actually returning
                if len(batch_data) != 2:
                    print(f"[ERROR] DataLoader returning {len(batch_data)} items instead of 2")
                    print(f"[ERROR] Items: {[type(item) for item in batch_data]}")
                    raise ValueError(f"DataLoader returning {len(batch_data)} items, expected 2")
                
                batch_input_states, batch_target_states = batch_data
                # Move to device
                batch_input_states = batch_input_states.to(self.device, non_blocking=self._cuda).float()
                batch_target_states = batch_target_states.to(self.device, non_blocking=self._cuda).float()
                
                # Forward pass - no mixed precision for validation to avoid dtype issues
                target_seq_len = batch_target_states.shape[1]
                predictions = self.model(
                    unified_input=batch_input_states, 
                    prediction_steps=target_seq_len,
                    use_teacher_forcing=True  # Use teacher forcing for validation too
                )
                
                # Loss computation: unified state prediction loss
                loss = self.model.unified_loss(
                    predictions=predictions, 
                    targets=batch_target_states
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
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
        print(f"[INFO] Training dataset: {segment_info['num_segments']} segments, "
              f"each with length {segment_info['segment_length']}")
        
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
    
    def evaluate(self, dataset: TelemetryActionDataset) -> Dict[str, float]:
        """
        Evaluate the model on segmented test data using efficient batch processing
        
        Args:
            dataset: Test dataset with segmented data
            
        Returns:
            Evaluation metrics across all segments
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        context_feature_names = dataset.get_context_feature_names()
        num_segments = len(dataset)
        
        print(f"[INFO] Evaluating on {num_segments} segments using batch processing...")
        
        # Create DataLoader for efficient batch processing (same as training)
        # Use larger batch size for evaluation since we don't need gradients
        batch_size = 64  # Larger batch size for faster evaluation
        # NOTE: num_workers=0 to avoid multiprocessing issues in Docker/Windows
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        num_batches = len(dataloader)
        print(f"[INFO] Processing {num_batches} batches with batch size {batch_size}")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # Debug: Check what the dataloader is actually returning
                if len(batch_data) != 2:
                    print(f"[ERROR] DataLoader returning {len(batch_data)} items instead of 2")
                    print(f"[ERROR] Items: {[type(item) for item in batch_data]}")
                    raise ValueError(f"DataLoader returning {len(batch_data)} items, expected 2")
                
                batch_input_states, batch_target_states = batch_data
                
                # Show progress updates
                if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
                    progress_pct = (batch_idx + 1) / num_batches * 100
                    segments_processed = (batch_idx + 1) * batch_size
                    if segments_processed > num_segments:
                        segments_processed = num_segments
                    print(f"[INFO] Evaluation progress: batch {batch_idx + 1}/{num_batches} ({progress_pct:.1f}%) - {segments_processed}/{num_segments} segments")
                
                # Move to device
                batch_input_states = batch_input_states.to(self.device, non_blocking=self._cuda)
                batch_target_states = batch_target_states.to(self.device, non_blocking=self._cuda)
                
                batch_size_actual = batch_input_states.shape[0]
                segment_length = batch_input_states.shape[1]
                
                # Use teacher forcing for evaluation consistency with training
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                    target_seq_len = batch_target_states.shape[1]
                    predictions = self.model(
                        unified_input=batch_input_states, 
                        prediction_steps=target_seq_len,
                        use_teacher_forcing=True  # Use teacher forcing for consistent evaluation
                    )
                
                # Loss computation outside autocast: unified state prediction loss
                loss = self.model.unified_loss(
                    predictions=predictions, 
                    targets=batch_target_states
                )
                
                batch_samples = batch_size_actual * segment_length
                total_loss += loss.item() * batch_samples
                total_samples += batch_samples
                
                # Show running average loss periodically
                if batch_idx > 0 and (batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1):
                    running_avg_loss = total_loss / total_samples
                    print(f"[INFO] Running average loss: {running_avg_loss:.6f}")
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_target_states.cpu().numpy())
        
        # Compute additional metrics - concatenate all batch data efficiently
        print(f"[INFO] Computing final evaluation metrics...")
        
        # Concatenate along the batch dimension first, then reshape
        predictions_array = np.concatenate(all_predictions, axis=0)  # Shape: [total_segments, seq_len, features]
        targets_array = np.concatenate(all_targets, axis=0)
        
        print(f"[INFO] Evaluation data shape - Predictions: {predictions_array.shape}, Targets: {targets_array.shape}")
        
        # Flatten for overall metrics (more efficient than reshaping twice)
        pred_flat = predictions_array.flatten()
        target_flat = targets_array.flatten()
        
        print(f"[INFO] Computing metrics on {len(pred_flat):,} prediction values...")
        
        # Compute metrics efficiently
        mse = np.mean((target_flat - pred_flat) ** 2)
        mae = np.mean(np.abs(target_flat - pred_flat))
        
        # R² score (vectorized computation)
        target_mean = np.mean(target_flat)
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - target_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        final_test_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Display final evaluation results
        print(f"\n[INFO] ===== EVALUATION COMPLETE =====")
        print(f"[INFO] Segments processed: {num_segments}")
        print(f"[INFO] Total samples: {total_samples}")
        print(f"[INFO] Test Loss: {final_test_loss:.6f}")
        print(f"[INFO] Mean Squared Error (MSE): {mse:.6f}")
        print(f"[INFO] Mean Absolute Error (MAE): {mae:.6f}")
        print(f"[INFO] R² Score: {r2:.4f}")
        print(f"[INFO] ================================\n")
        
        return make_json_safe({
            'test_loss': final_test_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'num_samples': total_samples,
            'num_segments': num_segments
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