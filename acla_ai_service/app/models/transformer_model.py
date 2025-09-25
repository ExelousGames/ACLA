

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
from ..services.corner_identification_unsupervised_service import CornerFeatureCatalog
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
    2. CornerFeatureCatalog.ContextFeature  
    3. TireGripFeatureCatalog.ContextFeature
    
    Returns:
        List[str]: Ordered list of feature names
    """
    context_features = []
    
    # Add expert context features in enum order
    context_features.extend([f.value for f in ExpertFeatureCatalog.ContextFeature])
    
    # Add corner context features in enum order
    context_features.extend([f.value for f in CornerFeatureCatalog.ContextFeature])
    
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
    ExpertActionTransformer - AI Model for Non-Expert Driver Progression Learning
    
    WHAT IT ACTUALLY DOES:
    This model learns how a NON-EXPERT DRIVER progressively improves their driving actions 
    over time to reach expert-level performance. It does NOT directly predict expert actions.
    Instead, it models the learning trajectory of a non-expert driver as they receive 
    guidance and improve toward expert-level performance.
    
    Think of it as modeling the "learning curve" of a student driver who is getting coaching
    from an expert instructor. The model learns: "Given where I am now (non-expert state) 
    and what the expert target looks like (contextual guidance), what should my next 
    improved action be?"
    
    CORE FUNCTIONALITY:
    1. PROGRESSION MODELING: Models how non-expert actions evolve toward expert performance
    2. GAP-AWARE LEARNING: Uses delta-to-expert features to understand improvement direction
    3. CONTEXTUAL GUIDANCE: Uses expert targets as contextual guidance, not direct output
    4. SEQUENTIAL IMPROVEMENT: Learns step-by-step improvement sequences over time
    
    PRACTICAL APPLICATIONS:
    - Driver Training Systems: Models how students progress during training sessions
    - Adaptive Racing Coaching: Provides personalized improvement trajectories
    - Performance Analysis: Understands learning patterns and improvement rates
    - Skill Development: Models how drivers acquire expert-level techniques
    
    HOW IT WORKS:
    The model takes non-expert telemetry and enriched contextual data that includes:
    - Expert optimal targets (what the expert would do)
    - Delta-to-expert gap features (how far off the non-expert currently is)
    - Track/corner contextual information
    
    It then predicts what the non-expert driver's NEXT IMPROVED ACTIONS should be,
    not what the expert would do. This models the gradual improvement process.
    
    INPUT DATA:
    - Non-expert telemetry: Current non-expert driver's state and actions
    - Enriched context: Expert targets + gap features + track/environmental context
    
    OUTPUT PREDICTIONS:
    - Non-expert's next improved actions: The driver's improved gas, brake, steering, etc.
    - These represent the driver's evolving actions as they learn, NOT expert actions
    
    TRAINING PROCESS:
    The model is trained on sequences of non-expert telemetry data where drivers are 
    progressively improving over time. Expert targets are provided as contextual guidance
    through the enriched_contextual_data, helping the model understand the improvement direction.
    
    Architecture:
    - Input: Non-expert telemetry + enriched context (expert targets, gaps, track info)
    - Output: Non-expert's improved action sequences (learning trajectory)
    - Uses attention mechanism to focus on relevant improvement patterns
    """
    
    def __init__(self, 
                 telemetry_features_count: int = 42,  # Telemetry features from get_features_for_imitate_expert()
                 context_features_count: Optional[int] = None,  # Auto-determined from canonical feature catalogs
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 sequence_length: int = 20,
                 dropout: float = 0.1,
                 time_step_seconds: float = 0.5):
        """
        Initialize the Expert Action Transformer
        
        Args:
            telemetry_features_count: Number of telemetry input features 
            context_features_count: Number of enriched contextual features. If None, auto-determined from canonical feature catalogs.
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            sequence_length: Maximum sequence length for predictions
            dropout: Dropout rate
            time_step_seconds: Time duration (in seconds) that each prediction step represents (default: 0.5s)
        """
        super(ExpertActionTransformer, self).__init__()
        
        # Auto-determine context features count if not provided
        if context_features_count is None:
            canonical_features = get_canonical_context_feature_order()
            context_features_count = len(canonical_features)
            print(f"[INFO] Auto-determined context_features_count: {context_features_count} from canonical feature catalogs")
        
        # Store configuration
        self.input_features_count = telemetry_features_count
        self.context_features_count = context_features_count 
        self.output_features_count = 4  # Fixed output size: gas, brake, steer_angle, gear
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.time_step_seconds = time_step_seconds  # Control how much real time each step represents
        
        # Scalers for normalization during inference
        self.telemetry_scaler: Optional[StandardScaler] = None
        self.context_scaler: Optional[StandardScaler] = None
        self.action_scaler: Optional[StandardScaler] = None
        
        # Input embeddings
        self.telemetry_embedding = nn.Linear(telemetry_features_count, d_model)
        self.context_embedding = nn.Linear(context_features_count, d_model) if context_features_count > 0 else None

        # Positional encoding : Without positional encoding: The transformer can't distinguish between [brake, throttle, steer] and [steer, brake, throttle], it adds unique positional information
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=sequence_length * 2)
        
        # Transformer encoder for processing current state
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer decoder for generating action sequences  
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Action sequence embedding (for decoder inputs during training)
        self.action_embedding = nn.Linear(self.output_features_count, d_model)
        
        # Output projection to action space
        self.action_projection = nn.Linear(d_model, self.output_features_count)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def set_scalers(self, telemetry_scaler: Optional[StandardScaler] = None, context_scaler: Optional[StandardScaler] = None, action_scaler: Optional[StandardScaler] = None):
        """
        Set the scalers for telemetry, context, and action features.
        
        Args:
            telemetry_scaler: StandardScaler fitted on telemetry features
            context_scaler: StandardScaler fitted on context features
            action_scaler: StandardScaler fitted on action features
        """
        self.telemetry_scaler = telemetry_scaler
        self.context_scaler = context_scaler
        self.action_scaler = action_scaler
    
    def forward(self, 
                telemetry: torch.Tensor,
                context: Optional[torch.Tensor] = None, 
                target_actions: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None,
                segment_length: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass - Segmented Non-Expert Driver Progression Learning Pipeline
        
        This method processes variable-length segments of telemetry data where each segment
        represents a coherent effort to improve toward expert performance. The model
        handles segments of different lengths and learns progression patterns within
        each improvement attempt.
        
        SEGMENTED PROCESSING APPROACH:
        Unlike traditional fixed-length sequence processing, this model handles:
        - Variable-length segments (10-200+ timesteps per segment)
        - Each segment represents a complete improvement effort
        - Segments are processed independently during training
        - Dynamic sequence handling with proper masking
        
        ARCHITECTURAL FLOW FOR SEGMENTS:
        
        Step 1: DYNAMIC INPUT HANDLING
        - Accept variable-length telemetry and context tensors
        - Handle single segments during training (batch_size=1, variable seq_len)
        - Maintain consistent feature dimensions across segments
        
        Step 2: EMBEDDING LAYER
        - Non-expert telemetry features → high-dimensional space (d_model)
        - Enriched context features → same space (d_model)
        - Variable sequence lengths handled dynamically
        
        Step 3: FEATURE FUSION
        - Combines telemetry + context for each timestep in segment
        - Creates unified representation for the entire improvement effort
        - Maintains temporal relationships within the segment
        
        Step 4: POSITIONAL ENCODING
        - Applies positional encoding up to actual segment length
        - Handles variable lengths without padding artifacts
        - Preserves temporal order within each improvement attempt
        
        Step 5: TRANSFORMER PROCESSING
        - Encoder processes the complete segment context
        - Decoder generates action sequences for the segment
        - Uses segment-specific sequence length for generation
        
        Args:
            telemetry: Non-expert telemetry features [batch_size, seq_len, input_features]
                      (typically batch_size=1 for single segment processing)
            context: Enriched contextual features [batch_size, seq_len, context_features]
            target_actions: Target action sequences [batch_size, seq_len, action_features]
            target_mask: Optional mask for variable-length sequences
            segment_length: Length of the current segment being processed
            
        Returns:
            Predicted action sequence [batch_size, seq_len, action_features]
            Where seq_len matches the input segment length
        """
        batch_size = telemetry.shape[0]
        seq_len = telemetry.shape[1]
        
        # Use provided segment_length or infer from input
        actual_seq_len = segment_length if segment_length is not None else seq_len
        
        # Embed telemetry
        telemetry_embedded = self.telemetry_embedding(telemetry)  # [B, L, d_model]
        
        # Combine with context if available
        if context is not None and self.context_embedding is not None:
            context_embedded = self.context_embedding(context)  # [B, L, d_model]
            # Combine telemetry and context
            encoder_input = telemetry_embedded + context_embedded
        else:
            encoder_input = telemetry_embedded
        
        # Apply positional encoding only up to actual sequence length
        encoder_input = self.pos_encoding(encoder_input)
        
        # Encode current state
        memory = self.transformer_encoder(encoder_input)  # [B, L, d_model]
        
        # Choose generation strategy based on training mode and target availability
        if self.training and target_actions is not None:
            # TRAINING MODE: Use teacher forcing for fast parallel training
            decoder_output = self._generate_actions_teacher_forcing(memory, target_actions)
            # During training, return scaled predictions for loss calculation
            return decoder_output
        else:
            # INFERENCE MODE: Use autoregressive generation for realistic prediction
            decoder_output = self._generate_actions_autoregressive(memory, actual_seq_len)
            # During inference, apply inverse scaling to get original action values
            unscaled_output = self._apply_action_inverse_scaling(decoder_output)
            return unscaled_output
    
    def contextual_weighted_loss(self, 
                                predictions: torch.Tensor, 
                                target_actions: torch.Tensor, 
                                context: Optional[torch.Tensor] = None,
                                context_feature_names: Optional[List[str]] = None) -> torch.Tensor:
        """
        Contextual weighted loss that emphasizes learning from high-quality examples
        
        This loss function weights the standard MSE loss based on contextual quality indicators:
        - Higher weight for samples with good grip utilization (close to optimal)
        - Higher weight for samples with good expert alignment
        - Lower weight for samples showing poor physics utilization
        
        Args:
            predictions: Model predictions [batch_size, seq_len, action_features]
            target_actions: Target actions [batch_size, seq_len, action_features]  
            context: Contextual features [batch_size, seq_len, context_features]
            context_feature_names: Names of context features for indexing
            
        Returns:
            Weighted loss tensor (scalar)
        """
        # Ensure loss computation in full precision to avoid dtype issues
        predictions = predictions.float()
        target_actions = target_actions.float()
        if context is not None:
            context = context.float()
        
        # Base MSE loss (element-wise, not reduced)
        base_loss = F.mse_loss(predictions, target_actions, reduction='none')  # [B, L, A]
        
        if context is None or context_feature_names is None:
            # Fallback to standard MSE if no context available
            return base_loss.mean()
        
        # Initialize quality weight as ones (neutral weighting)
        quality_weight = torch.ones(context.shape[0], context.shape[1], device=context.device)  # [B, L]
        
        try:
            # Extract contextual quality indicators by feature name
            context_indices = {}
            
            # Tire grip features
            grip_features = TireGripFeatureCatalog.ContextFeature
            if grip_features.TURNING_GRIP_UTILIZATION.value in context_feature_names:
                context_indices['grip_util'] = context_feature_names.index(grip_features.TURNING_GRIP_UTILIZATION.value)
            if grip_features.OPTIMAL_GRIP_WINDOW.value in context_feature_names:
                context_indices['grip_window'] = context_feature_names.index(grip_features.OPTIMAL_GRIP_WINDOW.value)
            
            # Expert alignment features  
            expert_features = ExpertFeatureCatalog.ContextFeature
            if expert_features.EXPERT_VELOCITY_ALIGNMENT.value in context_feature_names:
                context_indices['expert_alignment'] = context_feature_names.index(expert_features.EXPERT_VELOCITY_ALIGNMENT.value)
            
            # Corner context features (for additional weighting)
            corner_features = CornerFeatureCatalog.ContextFeature
            if corner_features.CORNER_CONFIDENCE.value in context_feature_names:
                context_indices['corner_confidence'] = context_feature_names.index(corner_features.CORNER_CONFIDENCE.value)
            
            weight_components = []
            
            # 1. Grip utilization quality (peak at ~1.0, lower at extremes)
            if 'grip_util' in context_indices:
                grip_util = context[:, :, context_indices['grip_util']].float()  # [B, L]
                # Quality peaks at 1.0 (optimal grip), decreases as we move away
                grip_quality = 1.0 - torch.abs(grip_util - 1.0).clamp(0, 1)
                weight_components.append(grip_quality)
            
            # 2. Grip window quality (higher is better)  
            if 'grip_window' in context_indices:
                grip_window = context[:, :, context_indices['grip_window']].float()  # [B, L]
                weight_components.append(grip_window)
            
            # 3. Expert alignment quality (higher is better)
            if 'expert_alignment' in context_indices:
                expert_alignment = context[:, :, context_indices['expert_alignment']].float()  # [B, L]
                weight_components.append(expert_alignment)
            
            # 4. Corner confidence (higher is better)
            if 'corner_confidence' in context_indices:
                corner_confidence = context[:, :, context_indices['corner_confidence']].float()  # [B, L]
                weight_components.append(corner_confidence)
            
            # Combine weight components if any were found
            if weight_components:
                # Average all quality components
                quality_weight = torch.stack(weight_components, dim=-1).mean(dim=-1)  # [B, L]
                
                # Add small epsilon to prevent zero weights
                quality_weight = quality_weight + 0.1
                
                # Normalize to prevent extreme weighting
                quality_weight = quality_weight.clamp(0.1, 2.0)
            
        except Exception as e:
            print(f"[WARNING] Error in contextual weighting: {e}. Using standard loss.")
            quality_weight = torch.ones_like(quality_weight)
        
        # Apply quality weights to loss (expand dims to match action dimensions)
        weighted_loss = base_loss * quality_weight.unsqueeze(-1)  # [B, L, A]
        
        # Return mean weighted loss
        return weighted_loss.mean()
    
    def get_contextual_features_summary(self, context_feature_names: List[str]) -> Dict[str, Any]:
        """
        Get a summary of which contextual features are available for quality weighting.
        Also validates that the provided features match canonical ordering.
        
        Args:
            context_feature_names: List of available context feature names
            
        Returns:
            Dictionary summarizing available contextual guidance features and ordering validation
        """
        canonical_order = get_canonical_context_feature_order()
        
        summary = {
            'total_context_features': len(context_feature_names) if context_feature_names else 0,
            'available_for_weighting': [],
            'tire_grip_features': [],
            'expert_features': [],
            'corner_features': [],
            'canonical_order_matches': context_feature_names == canonical_order if context_feature_names else True
        }
        
        if not context_feature_names:
            return summary
        
        # Validate feature ordering consistency
        if context_feature_names != canonical_order:
            print(f"[WARNING] Context features do not match canonical order!")
            print(f"[WARNING] Expected count: {len(canonical_order)}, Received count: {len(context_feature_names)}")
            if len(context_feature_names) > 0 and len(canonical_order) > 0:
                print(f"[WARNING] First few expected: {canonical_order[:5]}")
                print(f"[WARNING] First few received: {context_feature_names[:5]}")
        
        # Check tire grip features
        grip_features = TireGripFeatureCatalog.ContextFeature
        for feature in grip_features:
            if feature.value in context_feature_names:
                summary['tire_grip_features'].append(feature.value)
                summary['available_for_weighting'].append(feature.value)
        
        # Check expert alignment features
        expert_features = ExpertFeatureCatalog.ContextFeature
        for feature in expert_features:
            if feature.value in context_feature_names:
                summary['expert_features'].append(feature.value)
                summary['available_for_weighting'].append(feature.value)
        
        # Check corner features
        corner_features = CornerFeatureCatalog.ContextFeature
        for feature in corner_features:
            if feature.value in context_feature_names:
                summary['corner_features'].append(feature.value)
                summary['available_for_weighting'].append(feature.value)
        
        summary['weighting_enabled'] = len(summary['available_for_weighting']) > 0
        
        return summary
    
    def validate_context_features(self, context_features: List[str]) -> Dict[str, Any]:
        """
        Validate that context features match the canonical ordering expected by the model.
        
        Args:
            context_features: List of context feature names to validate
            
        Returns:
            Dict containing validation results and recommendations
        """
        canonical_order = get_canonical_context_feature_order()
        
        validation_result = {
            'is_valid': True,
            'expected_count': len(canonical_order),
            'received_count': len(context_features),
            'order_matches': context_features == canonical_order,
            'missing_features': [],
            'extra_features': [],
            'recommendations': []
        }
        
        # Check count mismatch
        if len(context_features) != len(canonical_order):
            validation_result['is_valid'] = False
            validation_result['recommendations'].append(
                f"Expected {len(canonical_order)} context features, got {len(context_features)}"
            )
        
        # Check for missing and extra features
        canonical_set = set(canonical_order)
        received_set = set(context_features)
        
        validation_result['missing_features'] = list(canonical_set - received_set)
        validation_result['extra_features'] = list(received_set - canonical_set)
        
        if validation_result['missing_features']:
            validation_result['is_valid'] = False
            validation_result['recommendations'].append(
                f"Missing features: {validation_result['missing_features'][:5]}..."
            )
        
        if validation_result['extra_features']:
            validation_result['is_valid'] = False
            validation_result['recommendations'].append(
                f"Extra features: {validation_result['extra_features'][:5]}..."
            )
        
        # Check ordering
        if not validation_result['order_matches'] and len(context_features) == len(canonical_order):
            validation_result['is_valid'] = False
            validation_result['recommendations'].append(
                "Features are present but not in canonical order. Use get_canonical_context_feature_order()."
            )
        
        return validation_result
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _generate_actions_autoregressive(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Generate expert racing actions autoregressively during real-time inference.
        
        WHAT IS AUTOREGRESSIVE GENERATION?
        Autoregressive generation is a sequential prediction approach where each new prediction
        depends on all previously generated predictions. In racing terms, this means:
        - Step 1: Predict immediate action based on current telemetry
        - Step 2: Predict next action based on current telemetry + predicted action from Step 1
        - Step 3: Predict next action based on current telemetry + actions from Steps 1-2
        - Continue until full racing sequence is generated
        
        This mirrors how human racing drivers think: each driving decision influences the next,
        creating a chain of interdependent actions that form an optimal racing strategy.
        
        WHY AUTOREGRESSIVE FOR RACING?
        Racing actions are highly sequential and interdependent:
        1. PHYSICS CAUSALITY: Current throttle affects next corner speed, which affects next braking
        2. STRATEGIC PLANNING: Early braking enables later acceleration, optimizing overall lap time
        3. REAL-TIME CONSTRAINTS: Driver must make decisions without knowing future exact conditions
        4. TEMPORAL DEPENDENCIES: Racing line decisions now affect racing line options later
        
        IMPLEMENTATION ARCHITECTURE:
        
        Phase 1: INITIALIZATION
        - Creates "start token" (zero tensor) representing beginning of action sequence
        - This acts like a driver sitting in car before taking any actions
        
        Phase 2: ITERATIVE GENERATION LOOP (for each future time step)
        Step A: POSITIONAL ENCODING
          - Injects temporal position information into current sequence
          - Tells model "this is action at time T+1, T+2, etc."
          - Critical for understanding action timing and sequence order
        
        Step B: CAUSAL MASKING
          - Creates attention mask preventing "looking ahead" at future actions
          - Simulates real-time racing: driver can't see future decisions
          - Ensures each prediction uses only current telemetry + past actions
        
        Step C: TRANSFORMER DECODING
          - Feeds current sequence + encoded telemetry through decoder
          - Decoder attends to relevant patterns from training (expert demonstrations)
          - Produces high-dimensional representation of optimal next action
        
        Step D: ACTION PROJECTION
          - Converts high-dimensional decoder output to concrete racing actions
          - Maps internal representation → [gas, brake, steer_angle, gear]
          - These are the actual control inputs driver/car should execute
        
        Step E: SEQUENCE EXTENSION
          - Embeds predicted action back into model's internal representation
          - Appends to growing sequence of predicted actions
          - This prediction becomes input for next time step's decision
        
        Phase 3: SEQUENCE COMPLETION
        - Concatenates all individual predictions into full action sequence
        - Returns complete racing strategy: immediate through future actions
        
        RACING-SPECIFIC EXAMPLES:
        
        Corner Approach Sequence:
        T=0: Model sees "approaching corner at 180 km/h"
        → Predicts: "Start braking, 60% brake pressure" 
        T=1: Model sees "approaching corner + predicted braking"
        → Predicts: "Continue braking, 80% brake pressure, slight left turn"
        T=2: Model sees "corner entry + previous braking/turning"
        → Predicts: "Release brake, increase steering, prepare for apex"
        T=3: Model sees "at apex + full turning sequence"
        → Predicts: "Begin throttle application, reduce steering"
        
        This creates coherent racing strategy where each action logically follows from
        previous actions, just like expert human drivers plan ahead.
        
        TECHNICAL ADVANTAGES:
        1. COHERENT SEQUENCES: Each action considers full context of previous decisions
        2. ADAPTIVE PLANNING: Can adjust strategy based on predicted outcomes
        3. TEMPORAL CONSISTENCY: Maintains logical action flow over time
        4. EXPERT MIMICKING: Replicates how expert drivers think sequentially
        
        PERFORMANCE CHARACTERISTICS:
        - Computational: O(seq_len²) due to growing attention sequence
        - Memory: O(seq_len * d_model) for maintaining decoder state
        - Quality: High coherence but potential error accumulation over long sequences
        - Real-time: Suitable for real-time racing applications (millisecond latency)
        
        Args:
            memory: Encoded current telemetry state [batch_size, input_seq_len, d_model]
                   Contains transformer encoder's understanding of current racing situation
            seq_len: Number of future action steps to predict (typically 10-20 for racing)
                    Represents prediction horizon: how far ahead to plan
        
        Returns:
            Complete action sequence [batch_size, seq_len, action_features]
            Sequential racing actions from immediate next step through prediction horizon
            Format: [gas%, brake%, steer_angle, gear] per time step
        """
        batch_size = memory.shape[0]
        device = memory.device
        
        # Initialize with zeros or learned start token
        decoder_input = torch.zeros(batch_size, 1, self.d_model, device=device)
        outputs = []
        
        for i in range(seq_len):
            # Add positional encoding
            decoder_input_pos = self.pos_encoding(decoder_input)
            
            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(decoder_input_pos.shape[1])
            tgt_mask = tgt_mask.to(device)
            
            # Decode
            decoder_output = self.transformer_decoder(
                tgt=decoder_input_pos,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [B, i+1, d_model]
            
            # Get the last output and project to action space
            last_output = decoder_output[:, -1:, :]  # [B, 1, d_model] 
            action_output = self.action_projection(last_output)  # [B, 1, action_features]
            outputs.append(action_output)
            
            # Prepare next decoder input (embed the predicted action)
            next_embedded = self.action_embedding(action_output)  # [B, 1, d_model]
            decoder_input = torch.cat([decoder_input, next_embedded], dim=1)
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)  # [B, seq_len, action_features]
    
    def _generate_actions_teacher_forcing(self, memory: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """
        Generate actions using teacher forcing for fast parallel training.
        
        Teacher forcing uses the ground truth target actions as decoder input, enabling
        parallel processing of the entire sequence rather than sequential generation.
        This dramatically speeds up training while maintaining learning quality.
        
        How Teacher Forcing Works:
        1. Take target actions and shift them right by one position (add start token)
        2. Feed the shifted sequence as decoder input all at once
        3. Apply causal masking so each position can only attend to previous positions
        4. Decoder processes the entire sequence in parallel
        5. Output predictions for all positions simultaneously
        
        Benefits:
        - Much faster training (parallel vs sequential processing)
        - More stable gradients (entire sequence contributes to loss)
        - Better GPU utilization (batched operations instead of loops)
        - Maintains causal structure through attention masking
        
        Args:
            memory: Encoded telemetry state [batch_size, input_seq_len, d_model]
            target_actions: Ground truth actions [batch_size, seq_len, action_features]
                           These are the correct actions the model should learn to predict
        
        Returns:
            Predicted actions [batch_size, seq_len, action_features]
        """
        batch_size, seq_len, _ = target_actions.shape
        device = memory.device
        
        # Create decoder input by shifting target actions right and adding start token
        # Start token is zeros (representing "no action" at beginning)
        start_tokens = torch.zeros(batch_size, 1, self.output_features_count, device=device)
        
        # Shift target actions right: [action1, action2, action3] -> [start, action1, action2]
        # This ensures decoder input at position i is target at position i-1
        decoder_input_actions = torch.cat([start_tokens, target_actions[:, :-1, :]], dim=1)  # [B, L, action_features]
        
        # Embed the action sequence for decoder processing
        decoder_input_embedded = self.action_embedding(decoder_input_actions)  # [B, L, d_model]
        
        # Add positional encoding to decoder input
        decoder_input_embedded = self.pos_encoding(decoder_input_embedded)
        
        # Create causal mask to prevent future information leakage
        # Each position can only attend to current and previous positions
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
        
        # Apply transformer decoder with teacher forcing
        # memory contains encoded current state, decoder_input contains shifted target actions
        decoder_output = self.transformer_decoder(
            tgt=decoder_input_embedded,          # Shifted target actions as input
            memory=memory,                       # Encoded current state  
            tgt_mask=causal_mask,               # Prevent future information leakage
            memory_mask=None                     # No masking on encoder output
        )  # [B, L, d_model]
        
        # Project decoder output to action space
        predictions = self.action_projection(decoder_output)  # [B, L, action_features]
        
        return predictions
    
    def predict_segment_progression(self, 
                                  telemetry: torch.Tensor,
                                  context: Optional[torch.Tensor] = None,
                                  temperature: float = 1.0,
                                  deterministic: bool = False) -> torch.Tensor:
        """
        Predict progression actions for a complete segment
        
        This method processes a variable-length segment and predicts the non-expert
        driver's improved actions throughout that segment. Each segment represents
        a coherent improvement effort (e.g., a corner approach, lap section).
        
        Args:
            telemetry: Non-expert telemetry features [batch_size, segment_len, input_features]
            context: Enriched contextual features [batch_size, segment_len, context_features]
                     Contains expert targets and gap features to guide improvement
            temperature: Temperature for sampling (higher = more random) - currently unused
            deterministic: If True, use greedy decoding instead of sampling - currently unused
            
        Returns:
            Predicted segment progression [batch_size, segment_len, action_features]
            Shows improved non-expert actions throughout the segment
        """
        self.eval()
        segment_length = telemetry.shape[1]
            
        with torch.no_grad():
            return self.forward(
                telemetry=telemetry, 
                context=context, 
                target_actions=None,
                segment_length=segment_length
            )
                
            # Apply activation functions for different action types
            actions = self._apply_action_constraints(decoder_output)
            
            return actions
    
    def _apply_action_constraints(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """
        Apply physical constraints to predicted actions
        
        Args:
            raw_actions: Raw action predictions [batch_size, seq_len, action_features]
            
        Returns:
            Constrained actions [batch_size, seq_len, action_features]
        """
        # Assume action order: [gas, brake, steer_angle, gear]
        constrained = raw_actions.clone()
        
        # Gas and brake: [0, 1]
        constrained[..., 0] = torch.sigmoid(raw_actions[..., 0])  # gas
        constrained[..., 1] = torch.sigmoid(raw_actions[..., 1])  # brake
        
        # Steering: [-1, 1]  
        constrained[..., 2] = torch.tanh(raw_actions[..., 2])     # steer_angle
        
        # Gear: typically [1, 6], use clamp for discrete values
        if raw_actions.shape[-1] > 3:
            constrained[..., 3] = torch.clamp(raw_actions[..., 3], 1, 6)  # gear
        
        return constrained
    
    def _apply_action_inverse_scaling(self, scaled_actions: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse scaling to model predictions to convert from normalized to original scale.
        
        Args:
            scaled_actions: Scaled action predictions [batch_size, seq_len, action_features]
            
        Returns:
            Unscaled actions [batch_size, seq_len, action_features]
        """
        if self.action_scaler is None:
            return scaled_actions
            
        # Convert to numpy for sklearn scaler
        device = scaled_actions.device
        original_shape = scaled_actions.shape
        
        # Reshape to 2D: (batch_size * seq_len, action_features)
        actions_2d = scaled_actions.view(-1, original_shape[-1]).cpu().numpy()
        
        # Apply inverse transform
        unscaled_actions = self.action_scaler.inverse_transform(actions_2d)
        
        # Convert back to tensor and reshape
        unscaled_tensor = torch.from_numpy(unscaled_actions).float().to(device)
        return unscaled_tensor.view(original_shape)
    
    def predict_human_readable(self, 
                              current_telemetry: Dict[str, Any],
                              context_data: Optional[Dict[str, Any]] = None,
                              sequence_length: int = 10) -> Dict[str, Any]:
        """
        Generate human-readable expert driving predictions from current telemetry data.
        
        This function serves as the main interface for real-time racing guidance, converting
        raw telemetry data into actionable driving advice that can be easily understood
        by human drivers or displayed in user interfaces.
        
        Process Flow:
        1. Validate and preprocess input telemetry data
        2. Convert telemetry to model input format (normalization, feature extraction)
        3. Generate expert action sequence predictions using the trained model
            (optionally conditioned on delta-to-expert gap context)
        4. Convert raw numerical predictions to human-readable advice
        5. Format everything into structured JSON response
        
        Args:
            current_telemetry: Dictionary containing current driver telemetry data
                              Expected keys: speed, position, forces, steering, throttle, brake, etc.
            context_data: Optional dictionary with track/tire context information
                         Can include: corner info, tire grip levels, weather conditions
            sequence_length: Number of future action steps to predict (default: 10)
            
        Returns:
            Structured JSON dictionary with human-readable predictions:
            {
                "status": "success" | "error",
                "timestamp": ISO timestamp,
                "current_situation": {
                    "speed": "120 km/h",
                    "track_position": "mid-corner",
                    "racing_line": "optimal" | "suboptimal",
                    "tire_grip": "good" | "losing grip"
                },
                "sequence_predictions": [
                    {
                        "step": 1,
                        "time_ahead": "0.1s",
                        "action": "Begin braking",
                        "throttle": 0.2,
                        "brake": 0.6,
                        "steering": -0.15
                    }
                ],
                "contextual_info": {
                    "track_sector": "Sector 2, Turn 5",
                    "weather_impact": "Dry conditions, full grip",
                    "optimal_speed_estimate": "95 km/h for current section"
                }
            }
        """
        try:
            # Get device from model parameters first
            device = next(self.parameters()).device
            
            # Validate context features if provided
            if context_data and self.context_embedding is not None:
                context_features = self._extract_context_features(context_data)
                validation_result = self.validate_context_features(list(context_data.keys()))
                
                if not validation_result['is_valid']:
                    print(f"[WARNING] Context feature validation failed during prediction:")
                    for rec in validation_result['recommendations']:
                        print(f"[WARNING] - {rec}")
                    
                context_tensor = torch.tensor([context_features], dtype=torch.float32).unsqueeze(0).to(device)
            else:
                context_tensor = None
            
            # Prepare telemetry data for model input
            telemetry_features = self._extract_telemetry_features(current_telemetry)
            
            # Convert to tensor format
            telemetry_tensor = torch.tensor([telemetry_features], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Generate predictions
            self.eval()
            with torch.no_grad():
                predictions = self.predict_segment_progression(
                    telemetry=telemetry_tensor,
                    context=context_tensor,
                    sequence_length=sequence_length,
                    deterministic=True
                )
            
            # Convert predictions to numpy for processing
            predictions_np = predictions.cpu().numpy()[0]  # Remove batch dimension
            
            # Analyze current situation
            current_situation = self._analyze_current_situation(current_telemetry, context_data)
            
            # Create sequence predictions
            sequence_predictions = self._create_sequence_predictions(predictions_np, sequence_length)
            
            # Contextual information
            contextual_info = self._extract_prediction_contextual_info(current_telemetry, context_data)
            
            # Build response
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "current_situation": current_situation,
                "sequence_predictions": sequence_predictions,
                "contextual_info": contextual_info
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
    
    def _extract_telemetry_features(self, telemetry: Dict[str, Any]) -> List[float]:
        """Extract and normalize telemetry features for model input"""
        # Use the canonical feature list for imitation expert models
        try:
            feature_names = TelemetryFeatures.get_features_for_imitate_expert()
        except Exception:
            raise ValueError("TelemetryFeatures class is not properly defined.")
        
        features = []
        for feature in feature_names:
            value = telemetry.get(feature, 0.0)
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(0.0)
        # Ensure the feature vector matches the model's expected input size
        expected_len = getattr(self, 'input_features_count', len(features))
        if len(features) < expected_len:
            features.extend([0.0] * (expected_len - len(features)))
        elif len(features) > expected_len:
            features = features[:expected_len]

        # Apply telemetry scaler if available
        if self.telemetry_scaler is not None:
            import numpy as np
            features_array = np.array(features).reshape(1, -1)  # Shape: (1, n_features)
            scaled_features = self.telemetry_scaler.transform(features_array)
            features = scaled_features.flatten().tolist()

        return features
    
    def _extract_context_features(self, context_data: Dict[str, Any]) -> List[float]:
        """
        Extract contextual features for model input using canonical ordering.
        
        This method ensures the same feature order is used during prediction
        as was used during training. Features are extracted in canonical order
        as defined by the feature catalogs, with missing values filled as 0.0.
        
        Args:
            context_data: Dictionary containing contextual features
            
        Returns:
            List[float]: Features in canonical order
        """
        if context_data is None:
            # Return zeros for all canonical features if no context provided
            canonical_order = get_canonical_context_feature_order()
            features = [0.0] * len(canonical_order)
        else:
            # Use canonical extraction function for consistency
            features = extract_context_features_canonical_order(context_data)
        
        # Apply context scaler if available
        if self.context_scaler is not None:
            import numpy as np
            features_array = np.array(features).reshape(1, -1)  # Shape: (1, n_features)
            scaled_features = self.context_scaler.transform(features_array)
            features = scaled_features.flatten().tolist()
            
        return features
    
    def _analyze_current_situation(self, telemetry: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze current driving situation"""
        speed = float(telemetry.get('Physics_speed_kmh', 0))
        steer_angle = float(telemetry.get('Physics_steer_angle', 0))
        throttle = float(telemetry.get('Physics_gas', 0))
        brake = float(telemetry.get('Physics_brake', 0))
        
        # Determine track position
        if abs(steer_angle) > 0.1:
            track_position = "in-corner"
        elif throttle > 0.8:
            track_position = "straight-line"
        else:
            track_position = "corner-approach"
        
        # Determine racing line quality
        if abs(steer_angle) < 0.05 and speed > 100:
            racing_line = "optimal"
        else:
            racing_line = "suboptimal"
        
        # Tire grip assessment (simplified)
        g_lateral = abs(float(telemetry.get('Physics_g_force_x', 0)))
        if g_lateral < 1.0:
            tire_grip = "good grip"
        elif g_lateral < 1.5:
            tire_grip = "moderate grip"
        else:
            tire_grip = "losing grip"
        
        return {
            "speed": f"{speed:.0f} km/h",
            "track_position": track_position,
            "racing_line": racing_line,
            "tire_grip": tire_grip
        }
    
    def _create_sequence_predictions(self, predictions: np.ndarray, sequence_length: int) -> List[Dict[str, Any]]:
        """Create sequence of future predictions"""
        sequence = []
        
        for i in range(min(sequence_length, len(predictions))):
            pred = predictions[i]
            
            # Determine main action for this step - now only 4 actions: [gas, brake, steer_angle, gear]
            gas, brake, steering, gear = pred[0], pred[1], pred[2], int(pred[3])
            
            if brake > 0.3:
                action = "Apply brakes"
            elif gas > 0.7:
                action = "Accelerate"
            elif abs(steering) > 0.1:
                direction = "right" if steering > 0 else "left"
                action = f"Turn {direction}"
            else:
                action = "Maintain course"
            
            sequence.append({
                "step": i + 1,
                "time_ahead": f"{(i + 1) * self.time_step_seconds:.1f}s",
                "action": action,
                "throttle": round(float(gas), 2),
                "brake": round(float(brake), 2), 
                "steering": round(float(steering), 2),
                "gear": int(gear)
            })
        
        return sequence
    
    def _extract_prediction_contextual_info(self, telemetry: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Extract contextual information for response"""
        info = {}
        
        # Track information
        position = float(telemetry.get('Graphics_normalized_car_position', 0))
        if position < 0.33:
            info["track_sector"] = "Sector 1"
        elif position < 0.66:
            info["track_sector"] = "Sector 2" 
        else:
            info["track_sector"] = "Sector 3"
        
        # Weather (simplified)
        info["weather_impact"] = "Dry conditions, full grip available"
        
        # Optimal speed for current section (estimated)
        current_speed = float(telemetry.get('Physics_speed_kmh', 0))
        steer_angle = abs(float(telemetry.get('Physics_steer_angle', 0)))
        
        if steer_angle > 0.2:
            optimal_speed = current_speed * 0.9  # Corner
        else:
            optimal_speed = current_speed * 1.1  # Straight
        
        info["optimal_speed_estimate"] = f"{optimal_speed:.0f} km/h for current section"
        
        return info
    
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
        
        # Serialize scalers
        telemetry_scaler_data = None
        context_scaler_data = None
        action_scaler_data = None
        
        if self.telemetry_scaler is not None:
            scaler_buffer = io.BytesIO()
            import pickle
            pickle.dump(self.telemetry_scaler, scaler_buffer)
            telemetry_scaler_data = base64.b64encode(scaler_buffer.getvalue()).decode('utf-8')
            
        if self.context_scaler is not None:
            scaler_buffer = io.BytesIO()
            import pickle
            pickle.dump(self.context_scaler, scaler_buffer)
            context_scaler_data = base64.b64encode(scaler_buffer.getvalue()).decode('utf-8')
            
        if self.action_scaler is not None:
            scaler_buffer = io.BytesIO()
            import pickle
            pickle.dump(self.action_scaler, scaler_buffer)
            action_scaler_data = base64.b64encode(scaler_buffer.getvalue()).decode('utf-8')
        
        model_data = {
            'model_type': 'ExpertActionTransformer',
            'state_dict': base64.b64encode(state_dict_bytes).decode('utf-8'),
            'telemetry_scaler': telemetry_scaler_data,
            'context_scaler': context_scaler_data,
            'action_scaler': action_scaler_data,
            'config': {
                'input_features': self.input_features_count,
                'context_features': self.context_features_count,
                'action_features': self.output_features_count,
                'd_model': self.d_model,
                'sequence_length': self.sequence_length,
                'time_step_seconds': self.time_step_seconds,  # Include time step configuration
                'nhead': getattr(self.transformer_encoder.layers[0].self_attn, 'num_heads', 8),
                'num_layers': len(self.transformer_encoder.layers),
                'dim_feedforward': getattr(self.transformer_encoder.layers[0].linear1, 'out_features', 1024),
                'dropout': 0.1  # Default, could extract from layers if needed
            },
            'serialization_timestamp': datetime.now().isoformat(),
            'pytorch_version': getattr(_torch, '__version__', 'unknown')
        }
        # Ensure the entire payload is JSON-safe (no NaN/Inf, tensors, numpy, etc.)
        return make_json_safe(model_data)
    

    def deserialize_transformer_model(self, serialized_data: Dict[str, Any]) -> 'ExpertActionTransformer':
        """
        Deserialize model from JSON-serializable dictionary and restore state to current instance
        
        This method restores a trained ExpertActionTransformer model from serialized data
        created by serialize_model(). It updates the current instance's configuration and
        loads the trained weights, making the model ready for inference.
        
        Args:
            serialized_data: Dictionary containing serialized model data with keys:
                           - 'model_type': Should be 'ExpertActionTransformer'
                           - 'state_dict': Base64-encoded model weights and biases
                           - 'config': Model architecture configuration
                           - 'serialization_timestamp': When model was serialized
        
        Returns:
            Self (ExpertActionTransformer): The current instance with restored state
        
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
            cfg_input_features = config.get('input_features', self.input_features_count)
            cfg_context_features = config.get('context_features', self.context_features_count)
            cfg_action_features = config.get('action_features', self.output_features_count)
            cfg_d_model = config.get('d_model', self.d_model)
            cfg_seq_len = config.get('sequence_length', self.sequence_length)
            cfg_time_step = config.get('time_step_seconds', getattr(self, 'time_step_seconds', 0.1))
            cfg_nhead = config.get('nhead', getattr(self.transformer_encoder.layers[0].self_attn, 'num_heads', 8))
            cfg_num_layers = config.get('num_layers', len(self.transformer_encoder.layers) if hasattr(self, 'transformer_encoder') else 6)
            cfg_dim_ff = config.get('dim_feedforward', getattr(self.transformer_encoder.layers[0].linear1, 'out_features', 1024) if hasattr(self, 'transformer_encoder') else 1024)
            cfg_dropout = config.get('dropout', 0.1)

            # Determine if architecture rebuild is required
            needs_rebuild = False
            try:
                current_nhead = getattr(self.transformer_encoder.layers[0].self_attn, 'num_heads', cfg_nhead)
                current_num_layers = len(self.transformer_encoder.layers)
                current_dim_ff = getattr(self.transformer_encoder.layers[0].linear1, 'out_features', cfg_dim_ff)
            except Exception:
                current_nhead = cfg_nhead
                current_num_layers = cfg_num_layers
                current_dim_ff = cfg_dim_ff

            if (
                cfg_input_features != self.input_features_count or
                cfg_context_features != self.context_features_count or
                cfg_action_features != self.output_features_count or
                cfg_d_model != self.d_model or
                cfg_seq_len != self.sequence_length or
                cfg_nhead != current_nhead or
                cfg_num_layers != current_num_layers or
                cfg_dim_ff != current_dim_ff or
                abs(cfg_dropout - 0.1) > 1e-9  # dropout used in module construction
            ):
                needs_rebuild = True

            # Log any mismatches for visibility
            if needs_rebuild:
                print("[INFO] Rebuilding model architecture from serialized config to match checkpoint...")
                print(f"[INFO] Serialized config: in={cfg_input_features}, ctx={cfg_context_features}, act={cfg_action_features}, d_model={cfg_d_model}, nhead={cfg_nhead}, layers={cfg_num_layers}, dim_ff={cfg_dim_ff}, seq_len={cfg_seq_len}, dropout={cfg_dropout}")

            # Preserve current device
            try:
                device = next(self.parameters()).device
            except Exception:
                device = torch.device('cpu')

            # Rebuild architecture if needed so state_dict keys match
            if needs_rebuild:
                # Re-run __init__ with the serialized configuration to rebuild modules
                self.__init__(
                    telemetry_features_count=cfg_input_features,
                    context_features_count=cfg_context_features,
                    d_model=cfg_d_model,
                    nhead=cfg_nhead,
                    num_layers=cfg_num_layers,
                    dim_feedforward=cfg_dim_ff,
                    sequence_length=cfg_seq_len,
                    dropout=cfg_dropout,
                    time_step_seconds=cfg_time_step,
                )
                # Ensure the model is on the original device
                self.to(device)
            else:
                # Even if not rebuilding, update simple config fields
                self.input_features_count = cfg_input_features
                self.context_features_count = cfg_context_features
                self.output_features_count = cfg_action_features
                self.d_model = cfg_d_model
                self.sequence_length = cfg_seq_len
                self.time_step_seconds = cfg_time_step
            
            # Decode and restore model state
            state_dict_base64 = serialized_data['state_dict']
            state_dict_bytes = base64.b64decode(state_dict_base64)
            
            # Load state dict from bytes
            buffer = io.BytesIO(state_dict_bytes)
            state_dict = torch.load(buffer, map_location='cpu')  # Load to CPU first
            
            # Load the state dict into current model
            try:
                self.load_state_dict(state_dict, strict=True)
            except Exception as load_err:
                print(f"[WARNING] Strict state_dict load failed: {load_err}. Trying non-strict load...")
                incompatible = self.load_state_dict(state_dict, strict=False)
                # Handle both tuple return (older PyTorch) and IncompatibleKeys object (newer)
                missing = getattr(incompatible, 'missing_keys', None) or (incompatible[0] if isinstance(incompatible, (list, tuple)) and len(incompatible) > 0 else [])
                unexpected = getattr(incompatible, 'unexpected_keys', None) or (incompatible[1] if isinstance(incompatible, (list, tuple)) and len(incompatible) > 1 else [])
                if missing:
                    print(f"[WARNING] Missing keys during load: {missing}")
                if unexpected:
                    print(f"[WARNING] Unexpected keys during load: {unexpected}")
            
            # Restore scalers if available
            import pickle
            if 'telemetry_scaler' in serialized_data and serialized_data['telemetry_scaler'] is not None:
                try:
                    scaler_bytes = base64.b64decode(serialized_data['telemetry_scaler'])
                    scaler_buffer = io.BytesIO(scaler_bytes)
                    self.telemetry_scaler = pickle.load(scaler_buffer)
                    print("[INFO] - Restored telemetry scaler")
                except Exception as e:
                    print(f"[WARNING] Failed to restore telemetry scaler: {e}")
                    self.telemetry_scaler = None
            else:
                self.telemetry_scaler = None
                
            if 'context_scaler' in serialized_data and serialized_data['context_scaler'] is not None:
                try:
                    scaler_bytes = base64.b64decode(serialized_data['context_scaler'])
                    scaler_buffer = io.BytesIO(scaler_bytes)
                    self.context_scaler = pickle.load(scaler_buffer)
                    print("[INFO] - Restored context scaler")
                except Exception as e:
                    print(f"[WARNING] Failed to restore context scaler: {e}")
                    self.context_scaler = None
            else:
                self.context_scaler = None
                
            if 'action_scaler' in serialized_data and serialized_data['action_scaler'] is not None:
                try:
                    scaler_bytes = base64.b64decode(serialized_data['action_scaler'])
                    scaler_buffer = io.BytesIO(scaler_bytes)
                    self.action_scaler = pickle.load(scaler_buffer)
                    print("[INFO] - Restored action scaler")
                except Exception as e:
                    print(f"[WARNING] Failed to restore action scaler: {e}")
                    self.action_scaler = None
            else:
                self.action_scaler = None
            
            # Set model to evaluation mode (ready for inference)
            self.eval()
            
            # Log successful restoration
            serialization_time = serialized_data.get('serialization_timestamp', 'unknown')
            scaler_info = []
            if self.telemetry_scaler is not None:
                scaler_info.append("telemetry")
            if self.context_scaler is not None:
                scaler_info.append("context")
            if self.action_scaler is not None:
                scaler_info.append("action")
            scaler_status = f" (scalers: {', '.join(scaler_info)})" if scaler_info else " (no scalers)"
            
            print(f"[INFO] Successfully restored ExpertActionTransformer model")
            print(f"[INFO] - Model features: {self.input_features_count} telemetry, "
                  f"{self.context_features_count} context, {self.output_features_count} actions")
            print(f"[INFO] - Architecture: d_model={self.d_model}, seq_len={self.sequence_length}")
            print(f"[INFO] - Originally serialized: {serialization_time}")
            print(f"[INFO] - Model ready for inference{scaler_status}")
            
            return self
            
        except Exception as e:
            error_msg = f"Failed to deserialize ExpertActionTransformer model: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg) from e


class TelemetryActionDataset(Dataset):
    """
    Segmented Dataset class for learning non-expert driver progression toward expert performance

    This dataset handles lists of telemetry segments where each segment represents an effort
    to improve toward expert performance. Each segment can have variable length and represents
    a coherent improvement attempt (e.g., a corner approach, a lap section, or a training session).
    
    Key insight: The model processes one segment at a time, where each segment shows
    a non-expert driver's progression toward expert performance within that specific context.
    """
    
    def __init__(self,
                 telemetry_segments: List[List[Dict[str, Any]]],
                 enriched_contextual_segments: List[List[Dict[str, Any]]]):
        """
        Initialize the segmented dataset
        
        Args:
            telemetry_segments: List of telemetry segments, where each segment is a list of 
                               non-expert telemetry records showing progression
            enriched_contextual_segments: List of contextual segments, where each segment 
                                        contains expert targets, gap features, and environmental context
        """
        assert len(telemetry_segments) == len(enriched_contextual_segments), \
            "Number of telemetry segments must match contextual segments"
        
        # Validate that each segment pair has the same length
        for i, (tel_seg, ctx_seg) in enumerate(zip(telemetry_segments, enriched_contextual_segments)):
            assert len(tel_seg) == len(ctx_seg), \
                f"Segment {i}: telemetry length ({len(tel_seg)}) != contextual length ({len(ctx_seg)})"
        
        self.telemetry_segments = telemetry_segments
        self.enriched_contextual_segments = enriched_contextual_segments
        self.num_segments = len(telemetry_segments)
        
        # Extract feature names from first segment
        if telemetry_segments and telemetry_segments[0]:
            self.action_features = self._get_default_action_features()
            self.telemetry_features = [f for f in telemetry_segments[0][0].keys() 
                                     if f not in self.action_features]
        else:
            raise ValueError("Empty telemetry segments provided")
        
        # Get canonical context feature order
        self.context_features = get_canonical_context_feature_order()
        
        print(f"[INFO] Initialized segmented dataset with {self.num_segments} segments")
        print(f"[INFO] Segment lengths: {[len(seg) for seg in telemetry_segments[:5]]}..." +
              (f" (showing first 5 of {self.num_segments})" if self.num_segments > 5 else ""))
        print(f"[INFO] Features - Telemetry: {len(self.telemetry_features)}, " +
              f"Actions: {len(self.action_features)}, Context: {len(self.context_features)}")
        
        # Preprocess all segments
        self._preprocess_segments()
    
    def _get_default_action_features(self) -> List[str]:
        """Get default action features to predict (non-expert driver's actual actions)""" 
        return [
            "Physics_gas", "Physics_brake", "Physics_steer_angle", "Physics_gear"
        ]
    
    def _preprocess_segments(self):
        """
        Preprocess and normalize segmented telemetry data.
        
        This method processes each segment individually while maintaining consistent
        normalization across all segments. Each segment represents a coherent improvement
        effort with variable length.
        
        Processing steps:
        1. Collect all data points from all segments for global normalization
        2. Build separate matrices for telemetry, actions, and context features
        3. Fit scalers on the complete dataset for consistent normalization
        4. Store processed segments with normalized features
        """
        print("[INFO] Preprocessing segmented data...")
        
        # Collect all data points from all segments for global normalization
        all_telemetry_data = []
        all_action_data = []
        all_context_data = []
        
        for tel_segment, ctx_segment in zip(self.telemetry_segments, self.enriched_contextual_segments):
            all_telemetry_data.extend(tel_segment)
            all_action_data.extend(tel_segment)  # Actions are in telemetry data
            all_context_data.extend(ctx_segment)
        
        # Build global feature matrices for fitting scalers
        global_telemetry_matrix = self._build_matrix(all_telemetry_data, self.telemetry_features)
        global_action_matrix = self._build_matrix(all_action_data, self.action_features)
        global_context_matrix = self._build_context_matrix_canonical(all_context_data, self.context_features)
        
        # Fit scalers on global data for consistent normalization
        self.telemetry_scaler = StandardScaler()
        self.action_scaler = StandardScaler()  
        self.context_scaler = StandardScaler()
        
        self.telemetry_scaler.fit(global_telemetry_matrix)
        self.action_scaler.fit(global_action_matrix)
        self.context_scaler.fit(global_context_matrix)
        
        # Process each segment individually with fitted scalers
        self.processed_segments = []
        for i, (tel_segment, ctx_segment) in enumerate(zip(self.telemetry_segments, self.enriched_contextual_segments)):
            # Build matrices for this segment
            seg_telemetry_matrix = self._build_matrix(tel_segment, self.telemetry_features)
            seg_action_matrix = self._build_matrix(tel_segment, self.action_features)
            seg_context_matrix = self._build_context_matrix_canonical(ctx_segment, self.context_features)
            
            # Apply normalization
            seg_telemetry_normalized = self.telemetry_scaler.transform(seg_telemetry_matrix)
            seg_action_normalized = self.action_scaler.transform(seg_action_matrix)
            seg_context_normalized = self.context_scaler.transform(seg_context_matrix)
            
            # Store processed segment
            processed_segment = {
                'telemetry': torch.tensor(seg_telemetry_normalized, dtype=torch.float32),
                'actions': torch.tensor(seg_action_normalized, dtype=torch.float32),
                'context': torch.tensor(seg_context_normalized, dtype=torch.float32),
                'length': len(tel_segment)
            }
            self.processed_segments.append(processed_segment)
        
        print(f"[INFO] Preprocessed {len(self.processed_segments)} segments")
        print(f"[INFO] Global normalization: {global_telemetry_matrix.shape[0]} total samples")
        print(f"[INFO] Feature dimensions - Telemetry: {global_telemetry_matrix.shape[1]}, " +
              f"Actions: {global_action_matrix.shape[1]}, Context: {global_context_matrix.shape[1]}")
    
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
    
    def _build_context_matrix_canonical(self, data_list: List[Dict[str, Any]], feature_names: List[str]) -> np.ndarray:
        """
        Extract contextual features and build matrix using canonical feature ordering.
        
        This method ensures contextual features are extracted in the exact same order
        during training and prediction, using the canonical feature order defined by
        the feature catalogs. Missing features are filled with 0.0.
        
        Args:
            data_list: List of contextual data dictionaries
            feature_names: List of feature names in canonical order
            
        Returns:
            np.ndarray: Feature matrix with consistent ordering
        """
        matrix = []
        for record in data_list:
            # Use the canonical extraction function to ensure consistency
            row = extract_context_features_canonical_order(record)
            matrix.append(row)
        
        return np.array(matrix, dtype=np.float32)
    
    def __len__(self) -> int:
        """Return number of segments"""
        return self.num_segments
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training segment
        
        Args:
            idx: Segment index
            
        Returns:
            Tuple of (telemetry_seq, context_seq, action_seq) as tensors
            Each tensor has shape [segment_length, features] where segment_length
            varies per segment
        """
        if idx >= len(self.processed_segments):
            raise IndexError(f"Segment index {idx} out of range (0-{len(self.processed_segments)-1})")
        
        segment = self.processed_segments[idx]
        return segment['telemetry'], segment['context'], segment['actions']
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Get telemetry and action feature names"""
        return self.telemetry_features, self.action_features
    
    def get_context_feature_names(self) -> List[str]:
        """Get context feature names in canonical order"""
        return self.context_features
    
    def get_scalers(self) -> Dict[str, StandardScaler]:
        """Get the fitted scalers for denormalization"""
        return {
            'telemetry': self.telemetry_scaler,
            'actions': self.action_scaler,
            'context': self.context_scaler
        }
    
    def get_segment_info(self) -> Dict[str, Any]:
        """Get information about the segments in the dataset"""
        segment_lengths = [seg['length'] for seg in self.processed_segments]
        return {
            'num_segments': self.num_segments,
            'segment_lengths': segment_lengths,
            'min_length': min(segment_lengths) if segment_lengths else 0,
            'max_length': max(segment_lengths) if segment_lengths else 0,
            'avg_length': sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0,
            'total_samples': sum(segment_lengths)
        }

class ExpertActionTrainer:
    """
    Segmented Trainer class for the Expert Action Transformer.
    
    This trainer handles segmented telemetry data where each training example
    is a complete segment of variable length representing an improvement effort.
    Training processes one segment at a time rather than using traditional batching.
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
        Extract and set scalers from the segmented dataset on the model.
        
        Args:
            dataset: Training dataset containing fitted scalers
        """
        scalers = dataset.get_scalers()
        telemetry_scaler = scalers.get('telemetry')
        context_scaler = scalers.get('context')
        action_scaler = scalers.get('actions')  
        
        # Set scalers on the model
        self.model.set_scalers(telemetry_scaler, context_scaler, action_scaler)
        
        print(f"[INFO] Set scalers on model - Telemetry: {'✓' if telemetry_scaler else '✗'}, Context: {'✓' if context_scaler else '✗'}, Action: {'✓' if action_scaler else '✗'}")
    
    def train_epoch(self, dataset: TelemetryActionDataset) -> float:
        """
        Train for one epoch processing segments individually
        
        Unlike traditional batch training, this method processes each segment
        individually since segments have variable lengths. Each segment represents
        a complete improvement effort that should be learned as a coherent unit.
        
        Args:
            dataset: Segmented telemetry action dataset
            
        Returns:
            Average loss across all segments
        """
        self.model.train()
        total_loss = 0.0
        num_segments = len(dataset)
        
        # Get context feature names from the dataset
        context_feature_names = dataset.get_context_feature_names()
        
        print(f"[INFO] Training on {num_segments} segments...")
        
        for segment_idx in range(num_segments):
            # Get single segment (no batching due to variable lengths)
            telemetry, context, target_actions = dataset[segment_idx]
            
            # Add batch dimension (batch_size=1 for single segment)
            telemetry = telemetry.unsqueeze(0).to(self.device, non_blocking=self._cuda)
            context = context.unsqueeze(0).to(self.device, non_blocking=self._cuda)
            target_actions = target_actions.unsqueeze(0).to(self.device, non_blocking=self._cuda)
            
            segment_length = telemetry.shape[1]  # Actual length of this segment
            
            self.optimizer.zero_grad(set_to_none=True)

            # Autocast for mixed precision on GPU
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                # Forward pass with teacher forcing during training
                predictions = self.model(
                    telemetry=telemetry,
                    context=context,
                    target_actions=target_actions,  # Enable teacher forcing in training mode
                    segment_length=segment_length
                )
            
            # Loss computation OUTSIDE autocast to ensure proper gradient scaling
            loss = self.model.contextual_weighted_loss(
                predictions=predictions, 
                target_actions=target_actions,
                context=context,
                context_feature_names=context_feature_names
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
            
            # Progress reporting for long training
            if (segment_idx + 1) % max(1, num_segments // 10) == 0:
                avg_loss_so_far = total_loss / (segment_idx + 1)
                print(f"  Processed {segment_idx + 1}/{num_segments} segments, "
                      f"avg loss: {avg_loss_so_far:.6f}, "
                      f"segment length: {segment_length}")
        
        return total_loss / num_segments if num_segments > 0 else 0.0
    
    def validate_epoch(self, dataset: TelemetryActionDataset) -> float:
        """
        Validate for one epoch processing segments individually
        
        Args:
            dataset: Validation dataset with segmented data
            
        Returns:
            Average validation loss across all segments
        """
        self.model.eval()
        total_loss = 0.0
        num_segments = len(dataset)
        
        # Get context feature names from the dataset
        context_feature_names = dataset.get_context_feature_names()
        
        with torch.no_grad():
            for segment_idx in range(num_segments):
                # Get single segment
                telemetry, context, target_actions = dataset[segment_idx]
                
                # Add batch dimension
                telemetry = telemetry.unsqueeze(0).to(self.device, non_blocking=self._cuda).float()
                context = context.unsqueeze(0).to(self.device, non_blocking=self._cuda).float()
                target_actions = target_actions.unsqueeze(0).to(self.device, non_blocking=self._cuda).float()
                
                segment_length = telemetry.shape[1]
                
                # Forward pass - no mixed precision for validation to avoid dtype issues
                predictions = self.model(
                    telemetry=telemetry,
                    context=context,
                    target_actions=None,  # No teacher forcing in validation
                    segment_length=segment_length
                )
                
                # Loss computation
                loss = self.model.contextual_weighted_loss(
                    predictions=predictions, 
                    target_actions=target_actions,
                    context=context,
                    context_feature_names=context_feature_names
                )
                
                total_loss += loss.item()
        
        return total_loss / num_segments if num_segments > 0 else 0.0
    
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
              f"lengths {segment_info['min_length']}-{segment_info['max_length']} "
              f"(avg: {segment_info['avg_length']:.1f})")
        
        context_feature_names = train_dataset.get_context_feature_names()
        if context_feature_names:
            context_summary = self.model.get_contextual_features_summary(context_feature_names)
            print(f"[INFO] Contextual weighted loss enabled with {len(context_summary['available_for_weighting'])} weighting features:")
            if context_summary['tire_grip_features']:
                print(f"[INFO]   - Tire grip features: {context_summary['tire_grip_features']}")
            if context_summary['expert_features']:
                print(f"[INFO]   - Expert alignment features: {context_summary['expert_features']}")
            if context_summary['corner_features']:
                print(f"[INFO]   - Corner context features: {context_summary['corner_features']}")
        
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
    
    def evaluate(self, dataset: TelemetryActionDataset) -> Dict[str, float]:
        """
        Evaluate the model on segmented test data
        
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
        
        print(f"[INFO] Evaluating on {num_segments} segments...")
        
        with torch.no_grad():
            for segment_idx in range(num_segments):
                telemetry, context, target_actions = dataset[segment_idx]
                
                # Add batch dimension
                telemetry = telemetry.unsqueeze(0).to(self.device, non_blocking=self._cuda)
                context = context.unsqueeze(0).to(self.device, non_blocking=self._cuda)
                target_actions = target_actions.unsqueeze(0).to(self.device, non_blocking=self._cuda)
                
                segment_length = telemetry.shape[1]
                
                # Use standard forward (autoregressive) under AMP for GPU (evaluation mode)
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                    predictions = self.model(
                        telemetry=telemetry,
                        context=context,
                        target_actions=None,  # No teacher forcing during evaluation
                        segment_length=segment_length
                    )
                
                # Loss computation outside autocast
                loss = self.model.contextual_weighted_loss(
                    predictions=predictions, 
                    target_actions=target_actions,
                    context=context,
                    context_feature_names=context_feature_names
                )
                
                total_loss += loss.item() * segment_length
                total_samples += segment_length
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_actions.cpu().numpy())
        
        # Compute additional metrics - concatenate all segment data
        predictions_array = np.concatenate(all_predictions, axis=1)  # Concatenate along sequence dimension
        targets_array = np.concatenate(all_targets, axis=1)
        
        # Flatten for overall metrics
        pred_flat = predictions_array.reshape(-1)
        target_flat = targets_array.reshape(-1)
        
        mse = mean_squared_error(target_flat, pred_flat)
        mae = mean_absolute_error(target_flat, pred_flat)
        
        # R² score
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return make_json_safe({
            'test_loss': total_loss / total_samples if total_samples > 0 else 0.0,
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
                'input_features': self.model.input_features_count,
                'context_features': self.model.context_features_count,
                'action_features': self.model.output_features_count,
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


# Utility functions for converting to segmented format
def create_segments_from_continuous_data(telemetry_data: List[Dict[str, Any]], 
                                        enriched_contextual_data: List[Dict[str, Any]], 
                                        segment_length: int = 50,
                                        overlap: int = 0) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Convert continuous telemetry data into segments for the new segmented model.
    
    This utility function helps migrate from the old continuous data format to the new
    segmented format required by the updated transformer model.
    
    Args:
        telemetry_data: Continuous list of telemetry records
        enriched_contextual_data: Continuous list of contextual records
        segment_length: Length of each segment (default: 50)
        overlap: Number of overlapping samples between segments (default: 0)
        
    Returns:
        Tuple of (telemetry_segments, contextual_segments) where each is a list of segments
    """
    assert len(telemetry_data) == len(enriched_contextual_data), \
        "Telemetry and contextual data must have same length"
    
    telemetry_segments = []
    contextual_segments = []
    
    total_samples = len(telemetry_data)
    step_size = segment_length - overlap
    
    for start_idx in range(0, total_samples - segment_length + 1, step_size):
        end_idx = start_idx + segment_length
        
        tel_segment = telemetry_data[start_idx:end_idx]
        ctx_segment = enriched_contextual_data[start_idx:end_idx]
        
        telemetry_segments.append(tel_segment)
        contextual_segments.append(ctx_segment)
    
    print(f"[INFO] Created {len(telemetry_segments)} segments from {total_samples} continuous samples")
    print(f"[INFO] Segment parameters: length={segment_length}, overlap={overlap}, step_size={step_size}")
    
    return telemetry_segments, contextual_segments


def create_custom_segments(telemetry_data: List[Dict[str, Any]], 
                          enriched_contextual_data: List[Dict[str, Any]], 
                          segment_boundaries: List[Tuple[int, int]]) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Create custom segments based on specific boundary indices.
    
    This function allows creating segments based on semantic boundaries like:
    - Corner approaches and exits
    - Lap sections
    - Training session parts
    - Any other meaningful racing segments
    
    Args:
        telemetry_data: Continuous list of telemetry records
        enriched_contextual_data: Continuous list of contextual records
        segment_boundaries: List of (start_idx, end_idx) tuples defining segments
        
    Returns:
        Tuple of (telemetry_segments, contextual_segments)
    """
    assert len(telemetry_data) == len(enriched_contextual_data), \
        "Telemetry and contextual data must have same length"
    
    telemetry_segments = []
    contextual_segments = []
    
    for start_idx, end_idx in segment_boundaries:
        if start_idx < 0 or end_idx > len(telemetry_data) or start_idx >= end_idx:
            print(f"[WARNING] Invalid segment boundary ({start_idx}, {end_idx}), skipping")
            continue
            
        tel_segment = telemetry_data[start_idx:end_idx]
        ctx_segment = enriched_contextual_data[start_idx:end_idx]
        
        telemetry_segments.append(tel_segment)
        contextual_segments.append(ctx_segment)
    
    segment_lengths = [len(seg) for seg in telemetry_segments]
    print(f"[INFO] Created {len(telemetry_segments)} custom segments")
    print(f"[INFO] Segment lengths: min={min(segment_lengths)}, max={max(segment_lengths)}, avg={sum(segment_lengths)/len(segment_lengths):.1f}")
    
    return telemetry_segments, contextual_segments


def segment_by_improvement_attempts(telemetry_data: List[Dict[str, Any]], 
                                   enriched_contextual_data: List[Dict[str, Any]], 
                                   improvement_indicator_key: str = 'expert_gap_total',
                                   min_segment_length: int = 10,
                                   max_segment_length: int = 200) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Automatically segment data based on improvement attempts.
    
    This function analyzes the improvement indicator (e.g., gap to expert) and creates
    segments where the driver shows consistent improvement effort. Segments are created
    when the improvement trajectory changes significantly.
    
    Args:
        telemetry_data: Continuous list of telemetry records
        enriched_contextual_data: Continuous list of contextual records  
        improvement_indicator_key: Key in contextual data indicating improvement (e.g., 'expert_gap_total')
        min_segment_length: Minimum length for a segment
        max_segment_length: Maximum length for a segment
        
    Returns:
        Tuple of (telemetry_segments, contextual_segments)
    """
    assert len(telemetry_data) == len(enriched_contextual_data), \
        "Telemetry and contextual data must have same length"
    
    # Extract improvement indicator values
    improvement_values = []
    for ctx_record in enriched_contextual_data:
        value = ctx_record.get(improvement_indicator_key, 0.0)
        try:
            improvement_values.append(float(value))
        except (ValueError, TypeError):
            improvement_values.append(0.0)
    
    if not improvement_values:
        print("[WARNING] No improvement indicator values found, using fixed segments")
        return create_segments_from_continuous_data(telemetry_data, enriched_contextual_data, 50, 0)
    
    # Find segment boundaries based on improvement trend changes
    segment_boundaries = []
    current_start = 0
    
    for i in range(min_segment_length, len(improvement_values)):
        # Check if we should end current segment
        should_segment = False
        
        # End if we've reached maximum segment length
        if i - current_start >= max_segment_length:
            should_segment = True
        
        # End if improvement trend changes significantly
        elif i >= min_segment_length:
            # Calculate improvement trend over recent window
            window_size = min(10, i - current_start)
            if window_size >= 3:
                recent_trend = improvement_values[i-window_size:i]
                overall_trend = improvement_values[current_start:i]
                
                # Simple trend change detection (can be made more sophisticated)
                recent_avg = sum(recent_trend) / len(recent_trend)
                overall_avg = sum(overall_trend) / len(overall_trend)
                
                if abs(recent_avg - overall_avg) > 0.1 * abs(overall_avg) and i - current_start >= min_segment_length:
                    should_segment = True
        
        if should_segment:
            segment_boundaries.append((current_start, i))
            current_start = i
    
    # Add final segment
    if current_start < len(improvement_values) - min_segment_length:
        segment_boundaries.append((current_start, len(improvement_values)))
    
    return create_custom_segments(telemetry_data, enriched_contextual_data, segment_boundaries)