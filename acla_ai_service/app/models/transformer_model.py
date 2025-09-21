

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

# Force unbuffered output for real-time print statements
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)


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
    - Non-expert's next improved actions: The driver's improved throttle, brake, steering, etc.
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
                 context_features_count: int = 31,  # Enriched context (e.g., corners + grip + expert diff )
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 sequence_length: int = 20,
                 dropout: float = 0.1,
                 time_step_seconds: float = 0.1):
        """
        Initialize the Expert Action Transformer
        
        Args:
            telemetry_features_count: Number of telemetry input features 
            context_features_count: Number of enriched contextual features (corners, expert targets, delta-to-expert gaps)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            sequence_length: Maximum sequence length for predictions
            dropout: Dropout rate
            time_step_seconds: Time duration (in seconds) that each prediction step represents (default: 0.1s)
        """
        super(ExpertActionTransformer, self).__init__()
        
        # Store configuration
        self.input_features_count = telemetry_features_count
        self.context_features_count = context_features_count 
        self.output_features_count = 5  # Fixed output size: throttle, brake, steering, gear, speed
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.time_step_seconds = time_step_seconds  # Control how much real time each step represents
        
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
    
    def forward(self, 
                telemetry: torch.Tensor,
                context: Optional[torch.Tensor] = None, 
                target_actions: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - Non-Expert Driver Progression Learning Pipeline
        
        This method implements the complete forward pass through the transformer model,
        transforming non-expert telemetry and enriched context into predicted non-expert 
        action sequences that show progression toward expert performance over time.
        
        CRITICAL UNDERSTANDING:
        The model does NOT predict what an expert would do. Instead, it predicts what 
        the non-expert driver should do next as they progressively improve toward expert 
        performance. Expert targets are provided as contextual guidance within the 
        enriched context data.
        
        MODIFIED TRAINING APPROACH - NO TEACHER FORCING:
        Unlike traditional sequence-to-sequence models, this implementation uses autoregressive 
        generation for both training and inference. This means:
        
        - TRAINING: Model generates non-expert's next improved actions step-by-step
        - INFERENCE: Same autoregressive generation process
        - More realistic training that matches inference behavior
        - Models the actual learning progression of a non-expert driver
        
        ARCHITECTURAL FLOW:
        
        Step 1: EMBEDDING LAYER
        - Non-expert telemetry features (42-dim) → high-dimensional space (256-dim)
        - Enriched context features (31-dim) → same space (256-dim)
        - Creates rich feature representations for transformer processing
        
        Step 2: FEATURE FUSION
        - Combines non-expert telemetry + enriched context (expert targets, gap features, track info)
        - Allows model to correlate current non-expert state with improvement targets
        - Gap-aware signals help the model understand how to improve over time
        - Creates unified input representation for encoder
        
        Step 3: POSITIONAL ENCODING
        - Injects sequence position information into embeddings
        - Critical for understanding temporal order in racing telemetry
        - Enables model to distinguish "brake before corner" vs "accelerate after corner"
        
        Step 4: TRANSFORMER ENCODER
        - Processes current driver state through multi-head attention
        - Each attention head focuses on different aspects (speed, position, forces)
        - Creates contextualized representation of current racing situation
        - Output "memory" contains encoded understanding of current state
        
        Step 5: AUTOREGRESSIVE DECODER
        - Always uses autoregressive generation (no teacher forcing)
        - Starts with zero/start token, generates actions sequentially
        - Each predicted action becomes input for next prediction
        - Simulates real-time expert decision making process
        - Same behavior during training and inference
        
        Step 6: ACTION PROJECTION
        - Maps high-dimensional decoder output back to action space
        - 256-dim → 5-dim: [throttle, brake, steering, gear, speed]
        
        Args:
            telemetry: Non-expert telemetry features [batch_size, seq_len, input_features]
                      Contains non-expert driver's current state: speed, position, forces, actions, etc.
            context: Enriched contextual features [batch_size, seq_len, context_features] 
                    Contains expert targets, gap features, track info, environmental conditions
            target_actions: Non-expert's target action sequences for training [batch_size, seq_len, action_features]
                          The non-expert's actual improved actions for supervised learning
            target_mask: Mask for target sequence [batch_size, seq_len]
                        Currently unused, reserved for variable-length sequences
            
        Returns:
            Predicted action sequence [batch_size, seq_len, action_features]
            Non-expert's predicted improved actions as they progress toward expert performance
        """
        batch_size = telemetry.shape[0]
        seq_len = telemetry.shape[1]
        
        # Embed telemetry
        telemetry_embedded = self.telemetry_embedding(telemetry)  # [B, L, d_model]
        
        # Combine with context if available
        if context is not None and self.context_embedding is not None:
            context_embedded = self.context_embedding(context)  # [B, L, d_model]
            # Combine telemetry and context
            encoder_input = telemetry_embedded + context_embedded
        else:
            encoder_input = telemetry_embedded
        
        # Add positional encoding
        encoder_input = self.pos_encoding(encoder_input)
        
        # Encode current state
        memory = self.transformer_encoder(encoder_input)  # [B, L, d_model]
        
        # Always use autoregressive generation (no teacher forcing)
        decoder_output = self._generate_actions_autoregressive(memory, seq_len)

        # decoder_output is already in action space [B, L, action_features]
        return decoder_output
    
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
                grip_util = context[:, :, context_indices['grip_util']]  # [B, L]
                # Quality peaks at 1.0 (optimal grip), decreases as we move away
                grip_quality = 1.0 - torch.abs(grip_util - 1.0).clamp(0, 1)
                weight_components.append(grip_quality)
            
            # 2. Grip window quality (higher is better)  
            if 'grip_window' in context_indices:
                grip_window = context[:, :, context_indices['grip_window']]  # [B, L]
                weight_components.append(grip_window)
            
            # 3. Expert alignment quality (higher is better)
            if 'expert_alignment' in context_indices:
                expert_alignment = context[:, :, context_indices['expert_alignment']]  # [B, L]
                weight_components.append(expert_alignment)
            
            # 4. Corner confidence (higher is better)
            if 'corner_confidence' in context_indices:
                corner_confidence = context[:, :, context_indices['corner_confidence']]  # [B, L]
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
        Get a summary of which contextual features are available for quality weighting
        
        Args:
            context_feature_names: List of available context feature names
            
        Returns:
            Dictionary summarizing available contextual guidance features
        """
        summary = {
            'total_context_features': len(context_feature_names) if context_feature_names else 0,
            'available_for_weighting': [],
            'tire_grip_features': [],
            'expert_features': [],
            'corner_features': []
        }
        
        if not context_feature_names:
            return summary
        
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
          - Maps internal representation → [throttle, brake, steering, gear, speed]
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
            Format: [throttle%, brake%, steering_angle, gear, target_speed] per time step
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
    
    def predict_non_expert_progression_sequence(self, 
                               telemetry: torch.Tensor,
                               context: Optional[torch.Tensor] = None,
                               sequence_length: Optional[int] = None,
                               temperature: float = 1.0,
                               deterministic: bool = False) -> torch.Tensor:
        """
        Predict a sequence of non-expert driver actions showing progression toward expert performance
        
        IMPORTANT: This method predicts what the NON-EXPERT driver should do next as they 
        progressively improve, NOT what an expert would do. Expert targets are provided 
        in the context to guide the improvement direction.
        
        Args:
            telemetry: Non-expert telemetry features [batch_size, input_seq_len, input_features]
            context: Enriched contextual features [batch_size, input_seq_len, context_features]
                     Must include expert targets and delta-to-expert features to guide improvement.
            sequence_length: Length of action sequence to predict (default: self.sequence_length)
            temperature: Temperature for sampling (higher = more random)
            deterministic: If True, use greedy decoding instead of sampling
            
        Returns:
            Predicted non-expert progression sequence [batch_size, sequence_length, action_features]
            Shows how the non-expert driver should improve their actions over time
        """
        self.eval()
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        with torch.no_grad():
            batch_size = telemetry.shape[0]
            device = telemetry.device
            
            # Embed telemetry
            telemetry_embedded = self.telemetry_embedding(telemetry)
            
            # Combine with context if available
            if context is not None and self.context_embedding is not None:
                context_embedded = self.context_embedding(context)
                encoder_input = telemetry_embedded + context_embedded
            else:
                encoder_input = telemetry_embedded
            
            # Add positional encoding and encode
            encoder_input = self.pos_encoding(encoder_input)
            memory = self.transformer_encoder(encoder_input)
            
            # Generate action sequence autoregressively
            decoder_output = self._generate_actions_autoregressive(memory, sequence_length)
            
            # Apply temperature and sampling if not deterministic
            if not deterministic and temperature != 1.0:
                decoder_output = decoder_output / temperature
                
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
        # Assume action order: [throttle, brake, steering, gear, speed]
        constrained = raw_actions.clone()
        
        # Throttle and brake: [0, 1]
        constrained[..., 0] = torch.sigmoid(raw_actions[..., 0])  # throttle
        constrained[..., 1] = torch.sigmoid(raw_actions[..., 1])  # brake
        
        # Steering: [-1, 1]  
        constrained[..., 2] = torch.tanh(raw_actions[..., 2])     # steering
        
        # Gear: typically [1, 6], use softmax for discrete selection
        if raw_actions.shape[-1] > 3:
            constrained[..., 3] = torch.clamp(raw_actions[..., 3], 1, 6)  # gear
        
        # Speed: [0, inf) but practically [0, 350] km/h, use ReLU + clamp
        if raw_actions.shape[-1] > 4:
            constrained[..., 4] = torch.clamp(F.relu(raw_actions[..., 4]), 0, 350)  # speed
        
        return constrained
    
    def predict_human_readable(self, 
                              current_telemetry: Dict[str, Any],
                              context_data: Optional[Dict[str, Any]] = None,
                              sequence_length: int = 10,
                              include_confidence: bool = True) -> Dict[str, Any]:
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
        5. Calculate confidence scores and contextual information
        6. Format everything into structured JSON response
        
        Args:
            current_telemetry: Dictionary containing current driver telemetry data
                              Expected keys: speed, position, forces, steering, throttle, brake, etc.
            context_data: Optional dictionary with track/tire context information
                         Can include: corner info, tire grip levels, weather conditions
            sequence_length: Number of future action steps to predict (default: 10)
            include_confidence: Whether to include prediction confidence metrics
            
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
                "expert_advice": {
                    "immediate_action": "Brake moderately and turn in earlier",
                    "throttle_guidance": "Maintain current throttle (65%)",
                    "braking_guidance": "Apply 40% brake pressure now",
                    "steering_guidance": "Turn steering wheel 15° left",
                    "gear_guidance": "Downshift to gear 3"
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
                "performance_analysis": {
                    "vs_expert_gap": "+0.8s per lap",
                    "main_improvement": "Earlier braking points",
                    "confidence_score": 0.85
                },
                "contextual_info": {
                    "track_sector": "Sector 2, Turn 5",
                    "weather_impact": "Dry conditions, full grip",
                    "optimal_speed": "95 km/h for this corner"
                }
            }
        """
        try:
            # Prepare telemetry data for model input
            telemetry_features = self._extract_telemetry_features(current_telemetry)
            
            # Convert to tensor format
            device = next(self.parameters()).device
            telemetry_tensor = torch.tensor([telemetry_features], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Process context data if provided
            context_tensor = None
            if context_data and self.context_embedding is not None:
                context_features = self._extract_context_features(context_data)
                context_tensor = torch.tensor([context_features], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Generate predictions
            self.eval()
            with torch.no_grad():
                predictions = self.predict_non_expert_progression_sequence(
                    telemetry=telemetry_tensor,
                    context=context_tensor,
                    sequence_length=sequence_length,
                    deterministic=True
                )
            
            # Convert predictions to numpy for processing
            predictions_np = predictions.cpu().numpy()[0]  # Remove batch dimension
            
            # Analyze current situation
            current_situation = self._analyze_current_situation(current_telemetry, context_data)
            
            # Generate expert advice
            expert_advice = self._generate_expert_advice(predictions_np[0], current_telemetry)  # First prediction
            
            # Create sequence predictions
            sequence_predictions = self._create_sequence_predictions(predictions_np, sequence_length)
            
            # Performance analysis
            performance_analysis = self._analyze_performance_gap(current_telemetry, predictions_np[0])
            
            # Add confidence scoring if requested
            if include_confidence:
                performance_analysis["confidence_score"] = self._calculate_prediction_confidence(predictions_np)
            
            # Contextual information
            contextual_info = self._extract_contextual_info(current_telemetry, context_data)
            
            # Build response
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "current_situation": current_situation,
                "expert_advice": expert_advice,
                "sequence_predictions": sequence_predictions,
                "performance_analysis": performance_analysis,
                "contextual_info": contextual_info
            }
            
            return make_json_safe(response)
            
        except Exception as e:
            return make_json_safe({
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "error_type": type(e).__name__
            })
    
    def _extract_telemetry_features(self, telemetry: Dict[str, Any]) -> List[float]:
        """Extract and normalize telemetry features for model input"""
        # Default telemetry features expected by the model
        feature_names = [
            "Graphics_normalized_car_position", "Graphics_player_pos_x", "Graphics_player_pos_y", 
            "Graphics_player_pos_z", "Graphics_current_time", "Physics_speed_kmh", "Physics_gas",
            "Physics_brake", "Physics_steer_angle", "Physics_gear", "Physics_rpm", "Physics_g_force_x",
            "Physics_g_force_y", "Physics_g_force_z", "Physics_slip_angle_front_left", 
            "Physics_slip_angle_front_right", "Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right", 
            "Physics_velocity_x", "Physics_velocity_y", "Physics_velocity_z"
        ]
        
        features = []
        for feature in feature_names:
            value = telemetry.get(feature, 0.0)
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(0.0)
        
        return features
    
    def _extract_context_features(self, context_data: Dict[str, Any]) -> List[float]:
        """Extract contextual features for model input"""
        # This should match the context features used during training
        # Placeholder implementation - adjust based on your actual context structure
        features = []
        
        # Corner information (16 features)
        corner_features = context_data.get('corner_info', {})
        corner_keys = ['radius', 'entry_speed', 'exit_speed', 'banking', 'elevation_change'] 
        for key in corner_keys:
            features.append(float(corner_features.get(key, 0.0)))
        
        # Add more features to reach expected context size
        while len(features) < 31:  # Expected context_features size
            features.append(0.0)
        
        return features[:31]  # Ensure exact size
    
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
    
    def _generate_expert_advice(self, prediction: np.ndarray, current_telemetry: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable expert advice from predictions"""
        # Prediction format: [throttle, brake, steering, gear, speed]
        pred_throttle = float(prediction[0])
        pred_brake = float(prediction[1])
        pred_steering = float(prediction[2])
        pred_gear = int(prediction[3])
        pred_speed = float(prediction[4])
        
        # Current values
        curr_throttle = float(current_telemetry.get('Physics_gas', 0))
        curr_brake = float(current_telemetry.get('Physics_brake', 0))
        curr_steering = float(current_telemetry.get('Physics_steer_angle', 0))
        curr_gear = int(current_telemetry.get('Physics_gear', 1))
        curr_speed = float(current_telemetry.get('Physics_speed_kmh', 0))
        
        # Generate advice
        advice = {}
        
        # Throttle guidance
        throttle_diff = pred_throttle - curr_throttle
        if abs(throttle_diff) < 0.1:
            advice["throttle_guidance"] = f"Maintain current throttle ({curr_throttle*100:.0f}%)"
        elif throttle_diff > 0.1:
            advice["throttle_guidance"] = f"Increase throttle to {pred_throttle*100:.0f}% (currently {curr_throttle*100:.0f}%)"
        else:
            advice["throttle_guidance"] = f"Reduce throttle to {pred_throttle*100:.0f}% (currently {curr_throttle*100:.0f}%)"
        
        # Braking guidance  
        brake_diff = pred_brake - curr_brake
        if pred_brake > 0.1:
            advice["braking_guidance"] = f"Apply {pred_brake*100:.0f}% brake pressure"
        elif curr_brake > 0.1 and pred_brake < 0.1:
            advice["braking_guidance"] = "Release brakes"
        else:
            advice["braking_guidance"] = "No braking needed"
        
        # Steering guidance
        steering_diff = pred_steering - curr_steering
        if abs(steering_diff) > 0.05:
            direction = "right" if steering_diff > 0 else "left"
            advice["steering_guidance"] = f"Turn steering wheel {abs(steering_diff)*100:.0f}% more to the {direction}"
        else:
            advice["steering_guidance"] = "Maintain current steering"
        
        # Gear guidance
        if pred_gear != curr_gear:
            if pred_gear > curr_gear:
                advice["gear_guidance"] = f"Upshift to gear {pred_gear}"
            else:
                advice["gear_guidance"] = f"Downshift to gear {pred_gear}"
        else:
            advice["gear_guidance"] = f"Stay in gear {curr_gear}"
        
        # Overall immediate action
        if pred_brake > 0.3:
            advice["immediate_action"] = "Brake harder and prepare for corner"
        elif pred_throttle > curr_throttle + 0.2:
            advice["immediate_action"] = "Accelerate out of corner"
        elif abs(steering_diff) > 0.1:
            direction = "right" if steering_diff > 0 else "left"
            advice["immediate_action"] = f"Turn more to the {direction}"
        else:
            advice["immediate_action"] = "Maintain current driving line"
        
        return advice
    
    def _create_sequence_predictions(self, predictions: np.ndarray, sequence_length: int) -> List[Dict[str, Any]]:
        """Create sequence of future predictions"""
        sequence = []
        
        for i in range(min(sequence_length, len(predictions))):
            pred = predictions[i]
            
            # Determine main action for this step
            throttle, brake, steering, gear, speed = pred[0], pred[1], pred[2], int(pred[3]), pred[4]
            
            if brake > 0.3:
                action = "Apply brakes"
            elif throttle > 0.7:
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
                "throttle": round(float(throttle), 2),
                "brake": round(float(brake), 2), 
                "steering": round(float(steering), 2),
                "gear": int(gear),
                "target_speed": round(float(speed), 1)
            })
        
        return sequence
    
    def _analyze_performance_gap(self, current_telemetry: Dict[str, Any], expert_prediction: np.ndarray) -> Dict[str, Any]:
        """Analyze performance gap vs expert"""
        curr_speed = float(current_telemetry.get('Physics_speed_kmh', 0))
        expert_speed = float(expert_prediction[4])  # Predicted optimal speed
        
        speed_diff = expert_speed - curr_speed
        
        # Estimate lap time impact (simplified)
        if abs(speed_diff) < 2:
            gap_estimate = "On pace with expert"
            improvement = "Minor adjustments needed"
        elif speed_diff > 5:
            gap_estimate = f"+{speed_diff*0.1:.1f}s per sector"
            improvement = "Carry more speed through corners"
        elif speed_diff < -5:
            gap_estimate = f"+{abs(speed_diff)*0.05:.1f}s per sector"
            improvement = "Focus on earlier braking points"
        else:
            gap_estimate = f"+{abs(speed_diff)*0.08:.1f}s per sector"
            improvement = "Optimize racing line"
        
        return {
            "vs_expert_gap": gap_estimate,
            "main_improvement": improvement,
            "speed_delta": f"{speed_diff:+.1f} km/h"
        }
    
    def _calculate_prediction_confidence(self, predictions: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        # Simple confidence based on prediction stability
        if len(predictions) < 2:
            return 0.5
        
        # Calculate variance in predictions as inverse confidence measure
        variances = np.var(predictions, axis=0)
        avg_variance = np.mean(variances)
        
        # Convert variance to confidence (0-1 scale)
        confidence = max(0.1, min(0.95, 1.0 / (1.0 + avg_variance * 10)))
        
        return round(confidence, 2)
    
    def _extract_contextual_info(self, telemetry: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, str]:
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
        
        # Save model state to bytes
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        state_dict_bytes = buffer.getvalue()
        
        model_data = {
            'model_type': 'ExpertActionTransformer',
            'state_dict': base64.b64encode(state_dict_bytes).decode('utf-8'),
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
            'serialization_timestamp': datetime.now().isoformat()
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
            
            # Validate current model configuration matches serialized model
            config_checks = [
                ('input_features', self.input_features_count),
                ('context_features', self.context_features_count),
                ('action_features', self.output_features_count),
                ('d_model', self.d_model),
                ('sequence_length', self.sequence_length)
            ]
            
            for config_key, current_value in config_checks:
                if config_key in config and config[config_key] != current_value:
                    print(f"[WARNING] Configuration mismatch for {config_key}: "
                          f"serialized={config[config_key]}, current={current_value}")
            
            # Update model configuration from serialized data
            self.input_features_count = config.get('input_features', self.input_features_count)
            self.context_features_count = config.get('context_features', self.context_features_count) 
            self.output_features_count = config.get('action_features', self.output_features_count)
            self.d_model = config.get('d_model', self.d_model)
            self.sequence_length = config.get('sequence_length', self.sequence_length)
            self.time_step_seconds = config.get('time_step_seconds', self.time_step_seconds)
            
            # Decode and restore model state
            state_dict_base64 = serialized_data['state_dict']
            state_dict_bytes = base64.b64decode(state_dict_base64)
            
            # Load state dict from bytes
            buffer = io.BytesIO(state_dict_bytes)
            state_dict = torch.load(buffer, map_location='cpu')  # Load to CPU first
            
            # Load the state dict into current model
            self.load_state_dict(state_dict)
            
            # Set model to evaluation mode (ready for inference)
            self.eval()
            
            # Log successful restoration
            serialization_time = serialized_data.get('serialization_timestamp', 'unknown')
            print(f"[INFO] Successfully restored ExpertActionTransformer model")
            print(f"[INFO] - Model features: {self.input_features_count} telemetry, "
                  f"{self.context_features_count} context, {self.output_features_count} actions")
            print(f"[INFO] - Architecture: d_model={self.d_model}, seq_len={self.sequence_length}")
            print(f"[INFO] - Originally serialized: {serialization_time}")
            print(f"[INFO] - Model ready for inference")
            
            return self
            
        except Exception as e:
            error_msg = f"Failed to deserialize ExpertActionTransformer model: {str(e)}"
            print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg) from e


class TelemetryActionDataset(Dataset):
    """
    Dataset class for learning non-expert driver progression toward expert performance

    This dataset uses non-expert telemetry and actions, with enriched contextual data that includes
    expert targets and delta-to-expert gap features. The model learns how a non-expert driver
    should adjust their actions over time to progressively reach expert performance levels.
    
    Key insight: The model output represents the non-expert driver's actual actions as they
    improve over time, NOT the expert's actions. Expert actions are provided as contextual
    guidance within enriched_contextual_data.
    """
    
    def __init__(self,
                 telemetry_data: List[Dict[str, Any]],
                 enriched_contextual_data: List[Dict[str, Any]],
                 sequence_length: int = 20):
        """
        Initialize the dataset
        
        Args:
            telemetry_data: List of non-expert telemetry records (includes both telemetry and actions)
            enriched_contextual_data: List of enriched contextual features including:
                                    - Expert optimal targets (expert_optimal_*)
                                    - Delta-to-expert gap features (expert_gap_*, expert_velocity_alignment)  
                                    - Corner/track/tire contextual information
                                    - Target actions for training (now included in enriched contextual data)
            sequence_length: Length of sequences to generate
            telemetry_features: List of telemetry feature names to extract (position, speed, forces, etc.)
        """
        assert len(telemetry_data) == len(enriched_contextual_data), "Telemetry and contextual data must have same length"
        
        self.telemetry_data = telemetry_data
        self.enriched_contextual_data = enriched_contextual_data
        self.sequence_length = sequence_length
        
        # Default feature lists
        self.telemetry_features = list(telemetry_data[0].keys())
        
        # Preprocessing
        self._preprocess_data()
        
        # Generate sequence indices
        self._generate_sequences()
    
    def _get_default_action_features(self) -> List[str]:
        """Get default action features to predict (non-expert driver's actual actions)""" 
        return [
            "Physics_gas", "Physics_brake", "Physics_steer_angle", "Physics_gear", "Physics_speed_kmh"
        ]
    
    def _preprocess_data(self):
        """
        Preprocess and normalize the data for transformer training.
        
        This function performs the following steps:
        1. Converts raw telemetry dictionaries into numerical feature matrices
        2. Extracts non-expert action targets from the same telemetry data 
        3. Processes enriched contextual data (corner info, expert targets, delta-to-expert features)
        4. Applies standardization (zero mean, unit variance) to all feature matrices
        5. Stores fitted scalers for later denormalization during inference
        
        The preprocessing ensures all input features are on similar scales, which is
        critical for stable transformer training and attention mechanism performance.
        """
        # Extract feature matrices from raw dictionary data
        # Convert list of telemetry dictionaries -> numpy matrix [samples, features]
        self.telemetry_matrix = self._build_matrix(self.telemetry_data, self.telemetry_features)
        
        # Extract contextual features if available (includes expert targets and gap features)
        if self.enriched_contextual_data:
            self.context_features = list(self.enriched_contextual_data[0].keys()) if self.enriched_contextual_data else []
            self.context_matrix = self._build_matrix(self.enriched_contextual_data, self.context_features)
        else:
            self.context_features = []
            self.context_matrix = None
        
        # Normalize features, because features have vastly different scales
        self.telemetry_scaler = StandardScaler()
        self.context_scaler = StandardScaler() if self.context_matrix is not None else None
        self.telemetry_matrix = self.telemetry_scaler.fit_transform(self.telemetry_matrix)
        if self.context_matrix is not None:
            self.context_matrix = self.context_scaler.fit_transform(self.context_matrix)
        
        print(f"[INFO] Preprocessed dataset: {self.telemetry_matrix.shape[0]} samples, "
              f"{self.telemetry_matrix.shape[1]} telemetry features")
        
        if self.context_matrix is not None:
            print(f"[INFO] Context matrix: {self.context_matrix.shape[1]} contextual features")
    
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
    
    def _generate_sequences(self):
        """
        Generate valid sequence start indices for transformer training.
        
        Purpose:
        - Creates fixed-length training sequences from continuous telemetry data
        - Determines how the dataset will be chunked for batch processing
        - Ensures sequences fit within available data boundaries
        
        How it works:
        1. Iterates through telemetry data in non-overlapping windows
        2. Each window starts at index i and spans sequence_length samples
        3. Only creates sequences where there's enough data (i + sequence_length <= total_samples)
        4. Stores valid start indices in self.sequence_indices list
        
        Strategy:
        - Non-overlapping sequences prevent data leakage between training samples
        - Step size equals sequence_length to maximize data efficiency
        - Alternative strategies could use overlapping windows or sliding windows
        
        Example:
        - Data length: 1000 samples, sequence_length: 20
        - Generated indices: [0, 20, 40, 60, ..., 980] 
        - Result: 49 non-overlapping sequences of 20 samples each
        """
        self.sequence_indices = []
        
        # Generate non-overlapping sequences to prevent data leakage
        # Step by sequence_length to avoid overlap between training samples
        for i in range(0, len(self.telemetry_data) - self.sequence_length + 1, self.sequence_length):
            self.sequence_indices.append(i)
        
        print(f"[INFO] Generated {len(self.sequence_indices)} sequences of length {self.sequence_length}")
    
    def __len__(self) -> int:
        """Return number of sequences"""
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a training sample
        
        Returns:
            Tuple of (telemetry_seq, context_seq, action_seq) as tensors
        """
        start_idx = self.sequence_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract sequences
        telemetry_seq = torch.tensor(self.telemetry_matrix[start_idx:end_idx], dtype=torch.float32)
        
        # Extract action features from telemetry data (actions are already included in telemetry)
        # Get action feature indices from telemetry features
        action_feature_names = self._get_default_action_features()
        action_indices = []
        for action_feature in action_feature_names:
            if action_feature in self.telemetry_features:
                action_indices.append(self.telemetry_features.index(action_feature))
        
        # Extract action sequence from telemetry matrix using the identified indices
        if action_indices:
            action_seq = torch.tensor(self.telemetry_matrix[start_idx:end_idx, action_indices], dtype=torch.float32)
        else:
            # Fallback: create zeros if no action features found
            action_seq = torch.zeros(self.sequence_length, len(action_feature_names), dtype=torch.float32)
        
        if self.context_matrix is not None:
            context_seq = torch.tensor(self.context_matrix[start_idx:end_idx], dtype=torch.float32)
            return telemetry_seq, context_seq, action_seq
        else:
            return telemetry_seq, action_seq
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Get telemetry and action feature names"""
        action_features = self._get_default_action_features()
        return self.telemetry_features, action_features
    
    def get_context_feature_names(self) -> Optional[List[str]]:
        """Get context feature names if available"""
        if not hasattr(self, 'context_features') or not self.context_features:
            # Try to reconstruct from enriched_contextual_data if available
            if self.enriched_contextual_data and len(self.enriched_contextual_data) > 0:
                self.context_features = list(self.enriched_contextual_data[0].keys())
            else:
                return None
        return self.context_features if hasattr(self, 'context_features') else None
    
    def get_scalers(self) -> Dict[str, StandardScaler]:
        """Get the fitted scalers for denormalization"""
        scalers = {
            'telemetry': self.telemetry_scaler
        }
        if self.context_scaler is not None:
            scalers['context'] = self.context_scaler
        return scalers

class ExpertActionTrainer:
    """
    Trainer class for the Expert Action Transformer.
    """
    
    def __init__(self,
                 model: ExpertActionTransformer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """
        Initialize the trainer
        
        Args:
            model: The transformer model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
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
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch using contextual weighted loss"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get context feature names from the dataset if available
        context_feature_names = None
        if hasattr(dataloader.dataset, 'get_context_feature_names'):
            context_feature_names = dataloader.dataset.get_context_feature_names()
        
        for batch in dataloader:
            if len(batch) == 3:  # telemetry, context, actions
                telemetry, context, target_actions = batch
                telemetry = telemetry.to(self.device)
                context = context.to(self.device)
                target_actions = target_actions.to(self.device)
            else:  # telemetry, actions (no context)
                telemetry, target_actions = batch
                telemetry = telemetry.to(self.device)
                target_actions = target_actions.to(self.device)
                context = None
            
            self.optimizer.zero_grad()
            
            # Forward pass with autoregressive generation (no teacher forcing)
            # Model will generate actions step by step using its own predictions
            predictions = self.model(
                telemetry=telemetry,
                context=context
            )
            
            # Use contextual weighted loss if context is available
            if context is not None and context_feature_names is not None:
                loss = self.model.contextual_weighted_loss(
                    predictions=predictions, 
                    target_actions=target_actions,
                    context=context,
                    context_feature_names=context_feature_names
                )
            else:
                # Fallback to standard MSE loss
                loss = self.criterion(predictions, target_actions)
                if num_batches == 0:
                    print(f"[INFO] Using standard MSE loss (no context features available)")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch using contextual weighted loss"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Get context feature names from the dataset if available
        context_feature_names = None
        if hasattr(dataloader.dataset, 'get_context_feature_names'):
            context_feature_names = dataloader.dataset.get_context_feature_names()
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:  # telemetry, context, actions
                    telemetry, context, target_actions = batch
                    telemetry = telemetry.to(self.device)
                    context = context.to(self.device)
                    target_actions = target_actions.to(self.device)
                else:  # telemetry, actions (no context)
                    telemetry, target_actions = batch
                    telemetry = telemetry.to(self.device)
                    target_actions = target_actions.to(self.device)
                    context = None
                
                # Forward pass with autoregressive generation (no teacher forcing)
                predictions = self.model(
                    telemetry=telemetry,
                    context=context
                )
                
                # Use contextual weighted loss if context is available
                if context is not None and context_feature_names is not None:
                    loss = self.model.contextual_weighted_loss(
                        predictions=predictions, 
                        target_actions=target_actions,
                        context=context,
                        context_feature_names=context_feature_names
                    )
                else:
                    # Fallback to standard MSE loss
                    loss = self.criterion(predictions, target_actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
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
        print(f"[INFO] Starting training for {epochs} epochs on {self.device}")
        print(f"[INFO] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Display contextual feature information
        if hasattr(train_dataloader.dataset, 'get_context_feature_names'):
            context_feature_names = train_dataloader.dataset.get_context_feature_names()
            if context_feature_names:
                context_summary = self.model.get_contextual_features_summary(context_feature_names)
                print(f"[INFO] Contextual weighted loss enabled with {len(context_summary['available_for_weighting'])} weighting features:")
                if context_summary['tire_grip_features']:
                    print(f"[INFO]   - Tire grip features: {context_summary['tire_grip_features']}")
                if context_summary['expert_features']:
                    print(f"[INFO]   - Expert alignment features: {context_summary['expert_features']}")
                if context_summary['corner_features']:
                    print(f"[INFO]   - Corner context features: {context_summary['corner_features']}")
            else:
                print(f"[INFO] No contextual features available - using standard MSE loss")
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate
            if val_dataloader is not None:
                val_loss = self.validate_epoch(val_dataloader)
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
        
        return make_json_safe({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'final_lr': self.optimizer.param_groups[0]['lr']
        })
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data using contextual weighted loss
        
        Args:
            dataloader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        # Get context feature names from the dataset if available
        context_feature_names = None
        if hasattr(dataloader.dataset, 'get_context_feature_names'):
            context_feature_names = dataloader.dataset.get_context_feature_names()
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    telemetry, context, target_actions = batch
                    telemetry = telemetry.to(self.device)
                    context = context.to(self.device)
                    target_actions = target_actions.to(self.device)
                else:
                    telemetry, target_actions = batch
                    telemetry = telemetry.to(self.device)
                    target_actions = target_actions.to(self.device)
                    context = None
                
                # Use standard forward method (autoregressive generation)
                predictions = self.model(
                    telemetry=telemetry,
                    context=context
                )
                
                # Use contextual weighted loss if context is available
                if context is not None and context_feature_names is not None:
                    loss = self.model.contextual_weighted_loss(
                        predictions=predictions, 
                        target_actions=target_actions,
                        context=context,
                        context_feature_names=context_feature_names
                    )
                else:
                    # Fallback to standard MSE loss
                    loss = self.criterion(predictions, target_actions)
                
                total_loss += loss.item() * target_actions.shape[0]
                total_samples += target_actions.shape[0]
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_actions.cpu().numpy())
        
        # Compute additional metrics
        predictions_array = np.concatenate(all_predictions, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)
        
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
            'num_samples': total_samples
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