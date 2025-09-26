

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
                 input_features_count: int = 42,  # Combined telemetry and context features
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
            input_features_count: Number of combined input features (telemetry + context)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            sequence_length: Maximum sequence length for predictions
            dropout: Dropout rate
            time_step_seconds: Time duration (in seconds) that each prediction step represents (default: 0.5s)
        """
        super(ExpertActionTransformer, self).__init__()
        
        # Store configuration
        self.input_features_count = input_features_count
        self.output_features_count = 4  # Fixed output size: gas, brake, steer_angle, gear
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.time_step_seconds = time_step_seconds  # Control how much real time each step represents
        
        # Scaler for normalization during inference
        self.input_scaler: Optional[StandardScaler] = None
        self.action_scaler: Optional[StandardScaler] = None
        
        # Input embedding (single embedding for combined features)
        self.input_embedding = nn.Linear(input_features_count, d_model)

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
    
    def set_scalers(self, input_scaler: Optional[StandardScaler] = None, action_scaler: Optional[StandardScaler] = None):
        """
        Set the scalers for combined input and action features.
        
        Args:
            input_scaler: StandardScaler fitted on combined input features (telemetry + context)
            action_scaler: StandardScaler fitted on action features
        """
        self.input_scaler = input_scaler
        self.action_scaler = action_scaler
    
    def forward(self, 
                combined_input: torch.Tensor,
                target_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - Fixed-Size Segment Non-Expert Driver Progression Learning Pipeline
        
        This method processes fixed-length segments of telemetry data where each segment
        represents a coherent effort to improve toward expert performance. All segments
        are preprocessed to have the same length (self.sequence_length).
        
        FIXED SEGMENTED PROCESSING APPROACH:
        - Fixed-length segments (self.sequence_length timesteps per segment)
        - Each segment represents a complete improvement effort
        - Consistent batch processing with fixed dimensions
        - No masking required due to fixed lengths
        
        ARCHITECTURAL FLOW:
        
        Step 1: INPUT VALIDATION
        - Ensure input dimensions match expected fixed size
        - Validate batch and sequence dimensions
        
        Step 2: EMBEDDING LAYER
        - Combined telemetry and context features → high-dimensional space (d_model)
        
        Step 3: POSITIONAL ENCODING
        - Applies positional encoding to fixed sequence length
        - Preserves temporal order within each improvement attempt
        
        Step 4: TRANSFORMER PROCESSING
        - Encoder processes the complete segment context
        - Decoder generates action sequences for the segment
        
        Args:
            combined_input: Combined telemetry and context features [batch_size, sequence_length, input_features]
            target_actions: Target action sequences [batch_size, sequence_length, action_features]
            
        Returns:
            Predicted action sequence [batch_size, sequence_length, action_features]
        """
        batch_size = combined_input.shape[0]
        seq_len = combined_input.shape[1]
        
        # Validate fixed sequence length
        assert seq_len == self.sequence_length, f"Expected sequence length {self.sequence_length}, got {seq_len}"
        
        # Embed combined input
        embedded_input = self.input_embedding(combined_input)  # [B, L, d_model]
        
        # Apply positional encoding
        encoder_input = self.pos_encoding(embedded_input)
        
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
            decoder_output = self._generate_actions_autoregressive(memory, self.sequence_length)
            # During inference, apply inverse scaling to get original action values
            unscaled_output = self._apply_action_inverse_scaling(decoder_output)
            return unscaled_output
    
    def standard_loss(self, 
                     predictions: torch.Tensor, 
                     target_actions: torch.Tensor) -> torch.Tensor:
        """
        Standard MSE loss for action prediction
        
        Args:
            predictions: Model predictions [batch_size, seq_len, action_features]
            target_actions: Target actions [batch_size, seq_len, action_features]  
            
        Returns:
            MSE loss tensor (scalar)
        """
        # Ensure loss computation in full precision to avoid dtype issues
        predictions = predictions.float()
        target_actions = target_actions.float()
        
        # Base MSE loss
        loss = F.mse_loss(predictions, target_actions, reduction='mean')
        
        return loss
    

    
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
                                  combined_input: torch.Tensor,
                                  temperature: float = 1.0,
                                  deterministic: bool = False) -> torch.Tensor:
        """
        Predict progression actions for a complete segment
        
        This method processes a variable-length segment and predicts the non-expert
        driver's improved actions throughout that segment. Each segment represents
        a coherent improvement effort (e.g., a corner approach, lap section).
        
        Args:
            combined_input: Combined telemetry and context features [batch_size, segment_len, input_features]
                           Contains expert targets and gap features to guide improvement
            temperature: Temperature for sampling (higher = more random) - currently unused
            deterministic: If True, use greedy decoding instead of sampling - currently unused
            
        Returns:
            Predicted segment progression [batch_size, segment_len, action_features]
            Shows improved non-expert actions throughout the segment
        """
        self.eval()
        segment_length = combined_input.shape[1]
            
        with torch.no_grad():
            return self.forward(
                combined_input=combined_input, 
                target_actions=None
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
        corner_features = [f.value for f in CornerFeatureCatalog.ContextFeature]
        tire_grip_features = [f.value for f in TireGripFeatureCatalog.ContextFeature]
        
        # Filter to only those present in the dataset
        available_expert = [f for f in expert_features if f in context_feature_names]
        available_corner = [f for f in corner_features if f in context_feature_names]
        available_tire_grip = [f for f in tire_grip_features if f in context_feature_names]
        
        return {
            'available_for_weighting': available_expert + available_corner + available_tire_grip,
            'expert_features': available_expert,
            'corner_features': available_corner,
            'tire_grip_features': available_tire_grip,
            'total_context_features': len(context_feature_names)
        }
    
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
            
            # Prepare combined input data for model input
            combined_features = self._extract_combined_features(current_telemetry, context_data)
            
            # Convert to tensor format
            combined_tensor = torch.tensor([combined_features], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Generate predictions
            self.eval()
            with torch.no_grad():
                predictions = self.predict_segment_progression(
                    combined_input=combined_tensor,
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
    
    def _extract_combined_features(self, telemetry: Dict[str, Any], context_data: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Extract combined telemetry and context features for model input.
        
        Args:
            telemetry: Dictionary containing telemetry data
            context_data: Optional dictionary containing context data
            
        Returns:
            List[float]: Combined features for model input
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
            # If no context provided, use zeros for all canonical context features
            canonical_order = get_canonical_context_feature_order()
            context_features = [0.0] * len(canonical_order)
        
        # Combine telemetry and context features
        combined_features = telemetry_features + context_features
        
        # Ensure the feature vector matches the model's expected input size
        expected_len = getattr(self, 'input_features_count', len(combined_features))
        if len(combined_features) < expected_len:
            combined_features.extend([0.0] * (expected_len - len(combined_features)))
        elif len(combined_features) > expected_len:
            combined_features = combined_features[:expected_len]

        # Apply input scaler if available
        if self.input_scaler is not None:
            import numpy as np
            features_array = np.array(combined_features).reshape(1, -1)
            scaled_features = self.input_scaler.transform(features_array)
            combined_features = scaled_features.flatten().tolist()

        return combined_features
    
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
        input_scaler_data = None
        action_scaler_data = None
        
        if self.input_scaler is not None:
            scaler_buffer = io.BytesIO()
            import pickle
            pickle.dump(self.input_scaler, scaler_buffer)
            input_scaler_data = base64.b64encode(scaler_buffer.getvalue()).decode('utf-8')
            
        if self.action_scaler is not None:
            scaler_buffer = io.BytesIO()
            import pickle
            pickle.dump(self.action_scaler, scaler_buffer)
            action_scaler_data = base64.b64encode(scaler_buffer.getvalue()).decode('utf-8')
        
        model_data = {
            'model_type': 'ExpertActionTransformer',
            'state_dict': base64.b64encode(state_dict_bytes).decode('utf-8'),
            'input_scaler': input_scaler_data,
            'action_scaler': action_scaler_data,
            'config': {
                'input_features_count': self.input_features_count,
                'output_features_count': self.output_features_count,
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
            cfg_input_features = config.get('input_features_count', self.input_features_count)
            cfg_action_features = config.get('output_features_count', self.output_features_count)
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
                print(f"[INFO] Serialized config: in={cfg_input_features}, act={cfg_action_features}, d_model={cfg_d_model}, nhead={cfg_nhead}, layers={cfg_num_layers}, dim_ff={cfg_dim_ff}, seq_len={cfg_seq_len}, dropout={cfg_dropout}")

            # Preserve current device
            try:
                device = next(self.parameters()).device
            except Exception:
                device = torch.device('cpu')

            # Rebuild architecture if needed so state_dict keys match
            if needs_rebuild:
                # Re-run __init__ with the serialized configuration to rebuild modules
                self.__init__(
                    input_features_count=cfg_input_features,
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
            if 'input_scaler' in serialized_data and serialized_data['input_scaler'] is not None:
                try:
                    scaler_bytes = base64.b64decode(serialized_data['input_scaler'])
                    scaler_buffer = io.BytesIO(scaler_bytes)
                    self.input_scaler = pickle.load(scaler_buffer)
                    print("[INFO] - Restored input scaler")
                except Exception as e:
                    print(f"[WARNING] Failed to restore input scaler: {e}")
                    self.input_scaler = None
            else:
                self.input_scaler = None
                
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
            if self.input_scaler is not None:
                scaler_info.append("input")
            if self.action_scaler is not None:
                scaler_info.append("action")
            scaler_status = f" (scalers: {', '.join(scaler_info)})" if scaler_info else " (no scalers)"
            
            print(f"[INFO] Successfully restored ExpertActionTransformer model")
            print(f"[INFO] - Model features: {self.input_features_count} input, {self.output_features_count} actions")
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
    Fixed-Size Segmented Dataset class for learning non-expert driver progression toward expert performance

    This dataset handles lists of telemetry segments where each segment represents an effort
    to improve toward expert performance. All segments must have the same fixed length,
    representing a coherent improvement attempt (e.g., a corner approach, a lap section, or a training session).
    
    Key insight: The model processes fixed-size segments in batches, where each segment shows
    a non-expert driver's progression toward expert performance within that specific context.
    """
    
    def __init__(self,
                 combined_segments: List[List[Dict[str, Any]]],
                 fixed_segment_length: int):
        """
        Initialize the fixed-size segmented dataset with combined telemetry and context data
        
        Args:
            combined_segments: List of pre-created combined segments, where each segment is a list of 
                              dictionaries containing both telemetry and context features together.
                              All segments must have exactly fixed_segment_length elements.
            fixed_segment_length: Required length for all segments
        """
        # Basic validation
        assert len(combined_segments) > 0, "At least one segment must be provided"
        assert fixed_segment_length > 0, f"Fixed segment length must be positive, got {fixed_segment_length}"
        
        # Validate that all segments have exactly the fixed length
        invalid_segments = []
        for i, segment in enumerate(combined_segments):
            if len(segment) != fixed_segment_length:
                invalid_segments.append(f"Combined segment {i}: length {len(segment)} != {fixed_segment_length}")
        
        if invalid_segments:
            error_msg = f"Fixed-size validation failed for {len(invalid_segments)} segment(s):\n" + "\n".join(invalid_segments[:10])
            if len(invalid_segments) > 10:
                error_msg += f"\n... and {len(invalid_segments) - 10} more segments"
            raise ValueError(error_msg)
        
        # Store validated segments
        self.combined_segments = combined_segments
        self.num_segments = len(combined_segments)
        self.fixed_segment_length = fixed_segment_length
        
        # Extract feature names from first segment
        if combined_segments and combined_segments[0]:
            self.action_features = self._get_default_action_features()
            # All features except actions are input features (telemetry + context combined)
            self.input_features = [f for f in combined_segments[0][0].keys() 
                                  if f not in self.action_features]
        else:
            raise ValueError("Empty combined segments provided")
        
        print(f"[INFO] ✓ Validated fixed-size segmented dataset with {self.num_segments} segments")
        print(f"[INFO] ✓ All segments have fixed length: {fixed_segment_length}")
        print(f"[INFO] ✓ Features - Input: {len(self.input_features)}, Actions: {len(self.action_features)}")
        
        # Preprocess all validated segments
        self._preprocess_segments()
    
    def _get_default_action_features(self) -> List[str]:
        """Get default action features to predict (non-expert driver's actual actions)""" 
        return [
            "Physics_gas", "Physics_brake", "Physics_steer_angle", "Physics_gear"
        ]
    
    def _preprocess_segments(self):
        """
        Preprocess and normalize fixed-size segmented telemetry data.
        
        This method processes each fixed-size segment while maintaining consistent
        normalization across all segments. All segments have the same fixed length.
        
        Processing steps:
        1. Collect all data points from all segments for global normalization
        2. Build separate matrices for telemetry, actions, and context features
        3. Fit scalers on the complete dataset for consistent normalization
        4. Store processed segments as tensors ready for batch processing
        """
        print("[INFO] Preprocessing fixed-size segmented data...")
        
        # Collect all data points from all segments for global normalization
        all_telemetry_data = []
        all_action_data = []
        all_context_data = []
        
        for segment in self.combined_segments:
            all_telemetry_data.extend(segment)
            all_action_data.extend(segment)  # Actions are in combined data
        
        # Build global feature matrices for fitting scalers
        global_input_matrix = self._build_matrix(all_telemetry_data, self.input_features)
        global_action_matrix = self._build_matrix(all_action_data, self.action_features)
        
        # Fit scalers on global data for consistent normalization
        self.input_scaler = StandardScaler()
        self.action_scaler = StandardScaler()  
        
        self.input_scaler.fit(global_input_matrix)
        self.action_scaler.fit(global_action_matrix)
        
        # Process segments as matrices for efficient batch processing
        # IMPORTANT: Preserve temporal order within each segment for coherent driving sequences
        input_matrices = []
        action_matrices = []
        
        for segment in self.combined_segments:
            # Build matrices for this segment (preserving temporal order)
            seg_input_matrix = self._build_matrix(segment, self.input_features)
            seg_action_matrix = self._build_matrix(segment, self.action_features)
            
            # Apply normalization
            seg_input_normalized = self.input_scaler.transform(seg_input_matrix)
            seg_action_normalized = self.action_scaler.transform(seg_action_matrix)
            
            input_matrices.append(seg_input_normalized)
            action_matrices.append(seg_action_normalized)
        
        # Convert to tensors - shape: [num_segments, fixed_segment_length, features]
        # Temporal order within each segment is preserved for coherent driving sequences
        self.input_tensor = torch.tensor(np.array(input_matrices), dtype=torch.float32)
        self.action_tensor = torch.tensor(np.array(action_matrices), dtype=torch.float32)
        
        print(f"[INFO] Preprocessed {len(input_matrices)} fixed-size segments")
        print(f"[INFO] ✓ Temporal order preserved within each segment")
        print(f"[INFO] Tensor shapes - Input: {self.input_tensor.shape}, Actions: {self.action_tensor.shape}")
        print(f"[INFO] Global normalization: {global_input_matrix.shape[0]} total samples")
    
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training segment
        
        Args:
            idx: Segment index
            
        Returns:
            Tuple of (combined_input_seq, action_seq) as tensors
            Each tensor has shape [fixed_segment_length, features]
        """
        if idx >= self.num_segments:
            raise IndexError(f"Segment index {idx} out of range (0-{self.num_segments-1})")
        
        return self.input_tensor[idx], self.action_tensor[idx]
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Get input and action feature names"""
        return self.input_features, self.action_features
    
    def get_scalers(self) -> Dict[str, StandardScaler]:
        """Get the fitted scalers for denormalization"""
        return {
            'input': self.input_scaler,
            'actions': self.action_scaler
        }
    
    def get_context_feature_names(self) -> List[str]:
        """Get the context feature names from the canonical ordering"""
        return get_canonical_context_feature_order()
    
    def get_segment_info(self) -> Dict[str, Any]:
        """Get information about the fixed-size segments in the dataset"""
        return {
            'num_segments': self.num_segments,
            'segment_length': self.fixed_segment_length,
            'total_samples': self.num_segments * self.fixed_segment_length,
            'tensor_shapes': {
                'input': list(self.input_tensor.shape),
                'actions': list(self.action_tensor.shape)
            }
        }
    
    @staticmethod
    def validate_segments(combined_segments: List[List[Dict[str, Any]]],
                         expected_length: int) -> Dict[str, Any]:
        """
        Validate combined segment data before creating dataset
        
        Args:
            combined_segments: List of combined segments to validate
            expected_length: Expected length for all segments
            
        Returns:
            Dict with validation results and statistics
        """
        validation_result = {
            'is_valid': True,
            'num_segments': len(combined_segments),
            'expected_length': expected_length,
            'errors': [],
            'warnings': [],
            'statistics': {
                'segment_lengths': [],
                'valid_segments': 0,
                'invalid_segments': 0
            }
        }
        
        if len(combined_segments) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("No segments provided")
            return validation_result
        
        # Validate each segment
        for i, segment in enumerate(combined_segments):
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
    Fixed-Size Segmented Trainer class for the Expert Action Transformer.
    
    This trainer handles fixed-size segmented telemetry data with efficient batch processing.
    All segments have the same fixed length, enabling traditional batching for faster training.
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
        input_scaler = scalers.get('input')
        action_scaler = scalers.get('actions')  
        
        # Set scalers on the model
        self.model.set_scalers(input_scaler, action_scaler)
        
        print(f"[INFO] Set scalers on model - Input: {'✓' if input_scaler else '✗'}, Action: {'✓' if action_scaler else '✗'}")
    
    def train_epoch(self, dataset: TelemetryActionDataset) -> float:
        """
        Train for one epoch using batch processing on fixed-size segments
        
        Args:
            dataset: Fixed-size segmented telemetry action dataset
            
        Returns:
            Average loss across all batches
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create DataLoader for efficient batch processing
        # NOTE: shuffle=False to preserve temporal order within segments
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"[INFO] Training on {len(dataset)} segments in batches...")
        print(f"[INFO] Preserving temporal order within each segment (no shuffling)")
        
        for batch_combined_input, batch_target_actions in dataloader:
            # Move to device
            batch_combined_input = batch_combined_input.to(self.device, non_blocking=self._cuda)
            batch_target_actions = batch_target_actions.to(self.device, non_blocking=self._cuda)
            
            self.optimizer.zero_grad(set_to_none=True)

            # Autocast for mixed precision on GPU
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                # Forward pass with teacher forcing during training
                predictions = self.model(
                    combined_input=batch_combined_input,
                    target_actions=batch_target_actions
                )
            
            # Loss computation OUTSIDE autocast to ensure proper gradient scaling
            loss = self.model.standard_loss(
                predictions=predictions, 
                target_actions=batch_target_actions
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
        
        # Create DataLoader for efficient batch processing
        # NOTE: shuffle=False to preserve temporal order within segments
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # Get context feature names from the dataset
        context_feature_names = dataset.get_context_feature_names()
        
        with torch.no_grad():
            for batch_combined_input, batch_target_actions in dataloader:
                # Move to device
                batch_combined_input = batch_combined_input.to(self.device, non_blocking=self._cuda).float()
                batch_target_actions = batch_target_actions.to(self.device, non_blocking=self._cuda).float()
                
                # Forward pass - no mixed precision for validation to avoid dtype issues
                predictions = self.model(
                    combined_input=batch_combined_input,
                    target_actions=None  # No teacher forcing in validation
                )
                
                # Loss computation
                loss = self.model.standard_loss(
                    predictions=predictions, 
                    target_actions=batch_target_actions
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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        num_batches = len(dataloader)
        print(f"[INFO] Processing {num_batches} batches with batch size {batch_size}")
        
        with torch.no_grad():
            for batch_idx, (batch_combined_input, batch_target_actions) in enumerate(dataloader):
                # Show progress updates
                if batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
                    progress_pct = (batch_idx + 1) / num_batches * 100
                    segments_processed = (batch_idx + 1) * batch_size
                    if segments_processed > num_segments:
                        segments_processed = num_segments
                    print(f"[INFO] Evaluation progress: batch {batch_idx + 1}/{num_batches} ({progress_pct:.1f}%) - {segments_processed}/{num_segments} segments")
                
                # Move to device
                batch_combined_input = batch_combined_input.to(self.device, non_blocking=self._cuda)
                batch_target_actions = batch_target_actions.to(self.device, non_blocking=self._cuda)
                
                batch_size_actual = batch_combined_input.shape[0]
                segment_length = batch_combined_input.shape[1]
                
                # Use standard forward (autoregressive) under AMP for GPU (evaluation mode)
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                    predictions = self.model(
                        combined_input=batch_combined_input,
                        target_actions=None  # No teacher forcing during evaluation
                    )
                
                # Loss computation outside autocast
                loss = self.model.standard_loss(
                    predictions=predictions, 
                    target_actions=batch_target_actions
                )
                
                batch_samples = batch_size_actual * segment_length
                total_loss += loss.item() * batch_samples
                total_samples += batch_samples
                
                # Show running average loss periodically
                if batch_idx > 0 and (batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1):
                    running_avg_loss = total_loss / total_samples
                    print(f"[INFO] Running average loss: {running_avg_loss:.6f}")
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_target_actions.cpu().numpy())
        
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
                'input_features': self.model.input_features_count,
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