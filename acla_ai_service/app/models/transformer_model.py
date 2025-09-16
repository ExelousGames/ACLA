"""
Transformer Model for Expert Action Prediction in Racing Telemetry

This model predicts the sequence of actions needed to reach expert optimal performance
over a period of time. It analyzes current driver behavior and suggests step-by-step
improvements to match expert driving patterns.
"""

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

# Force unbuffered output for real-time print statements
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

def print_progress(message: str, force_flush: bool = True):
    """Print with automatic flushing for real-time output during training"""
    print(message, flush=force_flush)
    # Also force flush to system stdout 
    sys.stdout.flush()

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model to handle sequence position information
    """
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ExpertActionTransformer(nn.Module):
    """
    Transformer model for predicting expert action sequences from current telemetry
    
    Architecture:
    - Input: Current telemetry features + contextual data (corner info, tire grip, etc.)
    - Output: Sequence of expert actions (steering, throttle, brake) for next N timesteps
    - Uses attention mechanism to focus on relevant past patterns
    """
    
    def __init__(self,
                 input_features: int,
                 action_features: int = 3,  # steering, throttle, brake
                 reasoning_features: int = 30,  # contextual reasoning features (targets for auxiliary reasoning head)
                 context_features: int = 0,   # number of exogenous enriched context features provided as separate encoder stream
                 context_feature_names: Optional[List[str]] = None,  # ordering for context features (stored for prediction)
                 context_fusion: str = "add",  # one of: 'add', 'gate', 'concat' (concat not yet implemented)
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_sequence_length: int = 256):
        """
        Initialize the Expert Action Transformer
        
        Args:
            input_features: Number of input telemetry features (basic telemetry only)
            action_features: Number of action outputs (steering, throttle, brake)
            reasoning_features: Number of contextual reasoning features (corners, grip, etc.)
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_sequence_length: Maximum sequence length
        """
        super(ExpertActionTransformer, self).__init__()
        
    # Steering value conventions used throughout this file:
    # ---------------------------------------------------
    # 1. Raw model OUTPUT / decoder tokens for steering are assumed NORMALIZED in [0,1]
    #    with 0.5 meaning "wheel centered". (See generate_expert_action_instructions(normalized_steering_0_1=True)).
    # 2. For human reasoning / reporting we frequently CONVERT that 0..1 value to a signed
    #    range [-1,1] via (value - 0.5) * 2.0. Anywhere a variable name contains 'steering_signed_-1_1'
    #    or logic references "decode_normalized_steering" it is working in the signed domain.
    # 3. Legacy datasets may still provide 'Physics_steer_angle' in physical degrees (e.g. -500..500).
    #    We detect this by magnitude (>2 treated as degrees) when constructing basic reasoning
    #    features and then approximate-normalize. Those legacy degree paths are clearly commented.
    # 4. Action extraction (_extract_action_features) does NOT rescale steering; downstream code
    #    must interpret whether it is already 0..1 or in degrees. New pipelines should ensure
    #    the stored value is already normalized 0..1 (center ~0.5) for consistency.
    # 5. There is NO use of a [-1,0] range; if you saw that elsewhere it likely meant [-1,1].

        self.input_features = input_features
        self.action_features = action_features
        self.reasoning_features = reasoning_features
        self.context_features = context_features
        self.context_feature_names = context_feature_names or []
        self.context_fusion = context_fusion.lower() if context_features > 0 else "none"
        if self.context_fusion not in {"add", "gate", "concat", "none"}:
            raise ValueError("context_fusion must be one of {'add','gate','concat','none'}")
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        
        # Input embedding layers (telemetry + optional context fused by addition after separate projection)
        self.input_embedding = nn.Linear(input_features, d_model)
        if context_features and context_features > 0:
            if self.context_fusion == "concat":
                # For concat we increase dimension before projection
                self.context_embedding = nn.Linear(context_features, d_model)
                self.fusion_linear = nn.Linear(d_model * 2, d_model)
            else:
                self.context_embedding = nn.Linear(context_features, d_model)
                if self.context_fusion == "gate":
                    # gating network takes concatenated telemetry+context embeddings
                    self.gate_linear = nn.Linear(d_model * 2, d_model)
        else:
            self.context_embedding = None
        self.action_embedding = nn.Linear(action_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_sequence_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        # Multi-head output projections
        self.action_projection = nn.Linear(d_model, action_features)  # Actions (steering, throttle, brake)
        self.reasoning_projection = nn.Linear(d_model, reasoning_features)  # Contextual reasoning features
        self.performance_head = nn.Linear(d_model, 1)  # Performance score prediction
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Create padding mask for variable length sequences"""
        batch_size, max_len = seq.shape[:2]
        mask = torch.arange(max_len).expand(batch_size, max_len) >= seq_len.unsqueeze(1)
        return mask
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder (prevent looking at future tokens)"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self,
                src_telemetry: torch.Tensor,
                tgt_actions: torch.Tensor,
                src_context: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            src_telemetry: Source telemetry data [seq_len, batch_size, input_features]
            tgt_actions: Target action sequence [seq_len, batch_size, action_features]
            src_padding_mask: Padding mask for source
            tgt_padding_mask: Padding mask for target
            
        Returns:
            Tuple of (predicted_actions, predicted_reasoning, performance_scores)
        """
        seq_len = tgt_actions.shape[0]
        
        # Embed inputs (telemetry + optional context)
        src_embedded = self.input_embedding(src_telemetry)
        if self.context_embedding is not None and src_context is not None:
            if src_context.shape[0] != src_telemetry.shape[0]:
                raise ValueError("src_context must have same seq_len as src_telemetry")
            ctx_emb = self.context_embedding(src_context)
            if self.context_fusion == "add":
                src_embedded = src_embedded + ctx_emb
            elif self.context_fusion == "gate":
                gate = torch.sigmoid(self.gate_linear(torch.cat([src_embedded, ctx_emb], dim=-1)))
                src_embedded = src_embedded + gate * ctx_emb
            elif self.context_fusion == "concat":
                combined = torch.cat([src_embedded, ctx_emb], dim=-1)
                src_embedded = self.fusion_linear(combined)
        src_embedded = src_embedded * math.sqrt(self.d_model)
        tgt_embedded = self.action_embedding(tgt_actions) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src_embedded = self.pos_encoder(src_embedded)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # Create causal mask for target
        tgt_mask = self.create_causal_mask(seq_len).to(tgt_actions.device)
        
        # Transformer forward pass
        transformer_output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to multiple output spaces
        predicted_actions = self.action_projection(transformer_output)
        predicted_reasoning = self.reasoning_projection(transformer_output)
        performance_scores = self.performance_head(transformer_output)
        
        return predicted_actions, predicted_reasoning, performance_scores
    
    def predict_expert_sequence(self,
                               src_telemetry: torch.Tensor,
                               sequence_length: int = 20,
                               temperature: float = 1.0,
                               initial_action: Optional[torch.Tensor] = None,
                               early_stop_tolerance: float = 0.0,
                               early_stop_min_steps: int = 5,
                               src_padding_mask: Optional[torch.Tensor] = None,
                               src_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressively predict an expert action sequence with decoder-only steps.

        Maintains the original public behavior (returns full fixed-length sequences) while
        internally avoiding redundant encoder passes and adding optional early stopping.

        Args:
            src_telemetry: [src_seq_len, batch, input_features]
            sequence_length: number of future steps to predict
            temperature: deterministic smoothing factor applied to incremental deltas:
                1.0 leaves prediction unchanged; <1 dampens change; >1 amplifies.
            initial_action: optional seed action [batch, action_features] or [1,batch,action_features]
            early_stop_tolerance: absolute per-dimension threshold below which changes
                are considered negligible for early stopping (0 disables)
            early_stop_min_steps: minimum steps before early stopping can trigger
            src_padding_mask: optional boolean mask [batch, src_seq_len] for source padding

        Returns:
            actions: [T, batch, action_features]
            reasoning: [T, batch, reasoning_features]
            performance: [T, batch, 1]
        """
        self.eval()
        if src_telemetry.dim() != 3:
            raise ValueError("src_telemetry must be [seq_len, batch, features]")

        device = src_telemetry.device
        batch_size = src_telemetry.shape[1]

        with torch.no_grad():
            # Encode source once
            src_emb = self.input_embedding(src_telemetry)
            if self.context_embedding is not None and src_context is not None:
                if src_context.shape[0] != src_telemetry.shape[0]:
                    raise ValueError("src_context must have same seq_len as src_telemetry")
                ctx_emb = self.context_embedding(src_context)
                if self.context_fusion == "add":
                    src_emb = src_emb + ctx_emb
                elif self.context_fusion == "gate":
                    gate = torch.sigmoid(self.gate_linear(torch.cat([src_emb, ctx_emb], dim=-1)))
                    src_emb = src_emb + gate * ctx_emb
                elif self.context_fusion == "concat":
                    combined = torch.cat([src_emb, ctx_emb], dim=-1)
                    src_emb = self.fusion_linear(combined)
            src_emb = src_emb * math.sqrt(self.d_model)
            src_emb = self.pos_encoder(src_emb)
            memory = self.transformer.encoder(src_emb, mask=None, src_key_padding_mask=src_padding_mask)

            # Prepare initial decoder token
            if initial_action is None:
                dec_tokens = torch.zeros(1, batch_size, self.action_features, device=device)
            else:
                if initial_action.dim() == 2:
                    dec_tokens = initial_action.unsqueeze(0).to(device)
                elif initial_action.dim() == 3 and initial_action.size(0) == 1:
                    dec_tokens = initial_action.to(device)
                else:
                    raise ValueError("initial_action must be shape [batch,F] or [1,batch,F]")

            actions_out: List[torch.Tensor] = []
            reasoning_out: List[torch.Tensor] = []
            perf_out: List[torch.Tensor] = []
            prev_action: Optional[torch.Tensor] = None

            for step in range(sequence_length):
                tgt_emb = self.action_embedding(dec_tokens) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_mask = self.create_causal_mask(tgt_emb.size(0)).to(device)
                dec_out = self.transformer.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                last_hidden = dec_out[-1:, :, :]
                last_action = self.action_projection(last_hidden)
                last_reasoning = self.reasoning_projection(last_hidden)
                last_perf = self.performance_head(last_hidden)

                if temperature != 1.0 and prev_action is not None:
                    last_action = prev_action + (last_action - prev_action) * temperature

                actions_out.append(last_action)
                reasoning_out.append(last_reasoning)
                perf_out.append(last_perf)

                if early_stop_tolerance > 0 and prev_action is not None and step + 1 >= early_stop_min_steps:
                    if torch.all(torch.abs(last_action - prev_action) <= early_stop_tolerance):
                        remaining = sequence_length - (step + 1)
                        if remaining > 0:
                            actions_out.extend([last_action.clone()] * remaining)
                            reasoning_out.extend([last_reasoning.clone()] * remaining)
                            perf_out.extend([last_perf.clone()] * remaining)
                        break

                dec_tokens = torch.cat([dec_tokens, last_action], dim=0)
                prev_action = last_action.detach()

            actions_tensor = torch.cat(actions_out, dim=0)
            reasoning_tensor = torch.cat(reasoning_out, dim=0)
            perf_tensor = torch.cat(perf_out, dim=0)

        return actions_tensor, reasoning_tensor, perf_tensor
    
    def serialize_model(self) -> Dict[str, Any]:
        """
        Serialize the model to a JSON-serializable dictionary
        
        Returns:
            Dictionary containing model state and configuration for JSON serialization
        """
        import base64
        import io
        
        # Get model state dict
        state_dict = self.state_dict()
        
        # Convert tensors to base64 encoded strings for JSON serialization
        serialized_state_dict = {}
        for key, tensor in state_dict.items():
            # Convert tensor to bytes
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            buffer.seek(0)
            # Encode to base64 string
            serialized_state_dict[key] = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create serializable model data
        model_data = {
            'model_type': 'ExpertActionTransformer',
            'model_config': {
                'input_features': self.input_features,
                'action_features': self.action_features,
                'reasoning_features': self.reasoning_features,
                'context_features': self.context_features,
                'context_feature_names': self.context_feature_names,
                'context_fusion': self.context_fusion,
                'd_model': self.d_model,
                'max_sequence_length': self.max_sequence_length,
                'targets_are_deltas': getattr(self, 'targets_are_deltas', False)
            },
            'state_dict': serialized_state_dict,
            'model_architecture': {
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'num_decoder_layers': self.num_decoder_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout_rate
            }
        }
        
        return model_data
    
    @classmethod
    def deserialize_model(cls, serialized_data: Dict[str, Any]) -> 'ExpertActionTransformer':
        """
        Deserialize a model from JSON-serializable data and create a new instance
        
        Args:
            serialized_data: Dictionary containing serialized model data
            
        Returns:
            New ExpertActionTransformer instance with loaded weights
        """
        import base64
        import io
        
        # Extract configuration
        config = serialized_data['model_config']
        architecture = serialized_data['model_architecture']
        
        # Create new model instance with the same configuration
        model = cls(
            input_features=config['input_features'],
            action_features=config['action_features'],
            reasoning_features=config.get('reasoning_features', 30),
            context_features=config.get('context_features', 0),
            context_feature_names=config.get('context_feature_names', []),
            context_fusion=config.get('context_fusion', 'none'),
            d_model=config['d_model'],
            nhead=architecture['nhead'],
            num_encoder_layers=architecture['num_encoder_layers'],
            num_decoder_layers=architecture['num_decoder_layers'],
            dim_feedforward=architecture['dim_feedforward'],
            dropout=architecture['dropout'],
            max_sequence_length=config['max_sequence_length']
        )
        setattr(model, 'targets_are_deltas', bool(config.get('targets_are_deltas', False)))
        
        # Deserialize state dict
        state_dict = {}
        for key, encoded_tensor in serialized_data['state_dict'].items():
            # Decode base64 string to bytes
            tensor_bytes = base64.b64decode(encoded_tensor)
            # Create buffer from bytes
            buffer = io.BytesIO(tensor_bytes)
            # Load tensor from buffer
            tensor = torch.load(buffer, map_location='cpu')
            state_dict[key] = tensor
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        
        return model
    
    def load_serialized_weights(self, serialized_data: Dict[str, Any]):
        """
        Load weights from serialized data into the current model instance
        
        Args:
            serialized_data: Dictionary containing serialized model data
        """
        import base64
        import io
        
        # Deserialize state dict
        state_dict = {}
        for key, encoded_tensor in serialized_data['state_dict'].items():
            # Decode base64 string to bytes
            tensor_bytes = base64.b64decode(encoded_tensor)
            # Create buffer from bytes
            buffer = io.BytesIO(tensor_bytes)
            # Load tensor from buffer
            tensor = torch.load(buffer, map_location='cpu')
            state_dict[key] = tensor
        
        # Load state dict into current model
        self.load_state_dict(state_dict)
        
        # Update model attributes if needed
        if 'model_config' in serialized_data:
            config = serialized_data['model_config']
            self.input_features = config['input_features']
            self.action_features = config['action_features']
            self.reasoning_features = config.get('reasoning_features', 30)  # Default for backward compatibility
            self.context_features = config.get('context_features', 0)
            self.context_feature_names = config.get('context_feature_names', [])
            self.context_fusion = config.get('context_fusion', 'none')
            self.d_model = config['d_model']
            self.max_sequence_length = config['max_sequence_length']
            setattr(self, 'targets_are_deltas', bool(config.get('targets_are_deltas', False)))
        
        if 'model_architecture' in serialized_data:
            architecture = serialized_data['model_architecture']
            self.nhead = architecture['nhead']
            self.num_encoder_layers = architecture['num_encoder_layers']
            self.num_decoder_layers = architecture['num_decoder_layers']
            self.dim_feedforward = architecture['dim_feedforward']
            self.dropout_rate = architecture['dropout']

    # -------------------------------------------------------------
    # Human-interpretable expert action plan generation
    # -------------------------------------------------------------
    def generate_expert_action_instructions(self,
                                            src_telemetry: torch.Tensor,
                                            sequence_length: int = 20,
                                            temperature: float = 1.0,
                                            thresholds: Optional[Dict[str, float]] = None,
                                            steering_range: Tuple[float, float] = (0.0, 1.0),
                                            steering_left_negative: bool = True,
                                            normalized_steering_0_1: bool = True,
                                            split_large_changes: bool = True,
                                            max_substeps: int = 3,
                                            device: Optional[str] = None,
                                            seconds_per_step: float = 0.1,
                                            src_context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Predict sequence and produce baseline-relative action steps.

        Returns a dict with predicted per-step actions, recommended absolute actions,
        and human-readable steps computed relative to the current driver inputs.
        """
        device = device or next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            if src_telemetry.dim() != 3:
                raise ValueError("src_telemetry must have shape [seq_len, batch, features]")
            if src_telemetry.shape[1] != 1:
                src_telemetry = src_telemetry[:, :1, :]
            src_telemetry = src_telemetry.to(device)

            # Extract position/speed for metadata
            start_norm_pos = 0.0
            start_speed_kmh: Optional[float] = None
            feature_index: Optional[int] = None
            try:
                from ..models.telemetry_models import TelemetryFeatures  # type: ignore
                tf = TelemetryFeatures()
                feat_list = tf.get_features_for_imitate_expert()
                if 'Graphics_normalized_car_position' in feat_list:
                    feature_index = feat_list.index('Graphics_normalized_car_position')
                    start_norm_pos = float(max(0.0, min(1.0, src_telemetry[-1, 0, feature_index].item())))
                if 'Physics_speed_kmh' in feat_list:
                    sp = feat_list.index('Physics_speed_kmh')
                    start_speed_kmh = float(max(0.0, src_telemetry[-1, 0, sp].item()))
            except Exception:
                pass

            # Predict action tokens
            actions_tok, _, _ = self.predict_expert_sequence(
                src_telemetry,
                sequence_length=sequence_length,
                temperature=temperature,
                src_context=src_context
            )
            preds = actions_tok.squeeze(1).cpu().numpy()  # [T,3]

            # Baseline (current driver inputs at last encoder step)
            predictions_are_deltas = bool(getattr(self, 'targets_are_deltas', False))
            baseline = np.zeros(3, dtype=np.float32)
            try:
                from ..models.telemetry_models import TelemetryFeatures  # type: ignore
                tf2 = TelemetryFeatures()
                names = tf2.get_features_for_imitate_expert()
                def idx(n: str) -> Optional[int]:
                    return names.index(n) if n in names else None
                si, gi, bi = idx('Physics_steer_angle'), idx('Physics_gas'), idx('Physics_brake')
                if si is not None:
                    baseline[0] = float(src_telemetry[-1, 0, si].item())
                if gi is not None:
                    baseline[1] = float(src_telemetry[-1, 0, gi].item())
                if bi is not None:
                    baseline[2] = float(src_telemetry[-1, 0, bi].item())
            except Exception:
                pass

            # Thresholds and helpers
            thr = {
                'steer_deg': 0.1,
                'steer_pct': 0.05,
                'throttle': 0.15,
                'brake': 0.10,
                'min_combined_change_score': 0.15
            }
            if thresholds:
                thr.update(thresholds)

            steer_min, steer_max = steering_range
            steer_abs_max = max(abs(steer_min), abs(steer_max)) or 1.0

            def decode_normalized_steering(val: float) -> float:
                v = max(0.0, min(1.0, val))
                return (v - 0.5) * 2.0

            def steering_direction(angle_signed: float) -> str:
                if abs(angle_signed) < 1e-3:
                    return 'straight'
                if steering_left_negative:
                    return 'left' if angle_signed < 0 else 'right'
                return 'left' if angle_signed > 0 else 'right'

            def steering_percent(angle_signed: float) -> float:
                return min(1.0, max(0.0, abs(angle_signed))) if normalized_steering_0_1 else abs(angle_signed) / steer_abs_max

            def format_percent(frac: float, digits: int = 0) -> str:
                return f"{max(0.0, min(1.0, frac)) * 100:.{digits}f}%"

            # Optional context-derived notes
            ctx_vector: Optional[np.ndarray] = None
            ctx_name_list: List[str] = self.context_feature_names if hasattr(self, 'context_feature_names') else []
            if src_context is not None and src_context.numel() > 0 and len(ctx_name_list) == int(src_context.shape[-1]):
                try:
                    ctx_vector = src_context.squeeze().detach().cpu().numpy()
                except Exception:
                    ctx_vector = None

            def context_notes_for_step(step_idx: int) -> List[str]:
                notes: List[str] = []
                if ctx_vector is None or not ctx_name_list:
                    return notes
                ctx_map = {name: float(ctx_vector[i]) for i, name in enumerate(ctx_name_list)}
                corner_keys = [k for k in ctx_map.keys() if any(x in k.lower() for x in ['corner', 'curvature', 'apex', 'entry', 'exit'])]
                if corner_keys:
                    curv_vals = [ctx_map[k] for k in corner_keys if 'curv' in k.lower()]
                    if curv_vals and max(curv_vals) > 0.6:
                        notes.append('tight corner ahead')
                    dir_keys = [k for k in corner_keys if 'direction' in k.lower()]
                    if dir_keys:
                        val = ctx_map[dir_keys[0]]
                        if val > 0.5:
                            notes.append('right-hand turn upcoming')
                        elif val < -0.5:
                            notes.append('left-hand turn upcoming')
                grip_like = {k: v for k, v in ctx_map.items() if any(x in k.lower() for x in ['grip', 'friction', 'saturation'])}
                if grip_like:
                    overall_keys = [k for k in grip_like if 'overall' in k.lower() or 'grip' in k.lower()]
                    metric = grip_like[overall_keys[0]] if overall_keys else (max(grip_like.values()) if grip_like else None)
                    if metric is not None:
                        if metric < 0.3:
                            notes.append('very low tire grip; be cautious')
                        elif metric < 0.6:
                            notes.append('moderate grip; avoid aggressive inputs')
                lat_keys = [k for k in ctx_map if 'lateral_weight_transfer' in k]
                if lat_keys:
                    lat = abs(ctx_map[lat_keys[0]])
                    if lat > 0.6:
                        notes.append('high lateral load; maintain stability')
                return notes

            predicted_actions_list: List[Dict[str, Any]] = []
            recommended_actions_abs: List[List[float]] = []
            for t, (steer_val, throttle, brake) in enumerate(preds):
                if sequence_length <= 1:
                    norm_pos = start_norm_pos
                else:
                    progression = (t / (sequence_length - 1))
                    norm_pos = start_norm_pos + progression * (1.0 - start_norm_pos)
                norm_pos = max(0.0, min(1.0, norm_pos))
                future_time_s = max(0.0, t * seconds_per_step)

                # Reconstruct absolute recommendation if model outputs deltas
                s_abs = float(baseline[0] + steer_val) if predictions_are_deltas else float(steer_val)
                g_abs = float(baseline[1] + throttle) if predictions_are_deltas else float(throttle)
                b_abs = float(baseline[2] + brake) if predictions_are_deltas else float(brake)
                recommended_actions_abs.append([s_abs, g_abs, b_abs])

                if normalized_steering_0_1:
                    steer_signed = decode_normalized_steering(float(s_abs))
                    steer_pct = steering_percent(steer_signed)
                    steer_dir = steering_direction(steer_signed)
                    steering_raw_norm = float(max(0.0, min(1.0, s_abs)))
                    steering_angle_report = steer_signed
                else:
                    steering_raw_norm = float(s_abs)
                    steer_signed = float(s_abs)
                    steer_pct = steering_percent(steer_signed)
                    steer_dir = steering_direction(steer_signed)
                    steering_angle_report = float(s_abs)

                predicted_actions_list.append({
                    't': t,
                    'normalized_position': norm_pos,
                    'time_s': future_time_s,
                    'steering_angle_deg': steering_angle_report,
                    'steering_direction': steer_dir,
                    'steering_percent': steer_pct,
                    'steering_raw_norm_0_1': steering_raw_norm,
                    'steering_signed_-1_1': steer_signed,
                    'throttle': float(g_abs),
                    'brake': float(b_abs),
                    'context_notes': context_notes_for_step(t)
                })

            steps: List[Dict[str, Any]] = []
            text_instructions: List[str] = []

            def create_step(t_idx: int, _prev_action: Dict[str, Any], curr_action: Dict[str, Any],
                            steer_delta: float, throttle_delta: float, brake_delta: float,
                            substep_index: Optional[int] = None, total_substeps: Optional[int] = None):
                norm_pos = curr_action['normalized_position']
                steer_pct_curr = curr_action['steering_percent']
                steer_dir_curr = curr_action['steering_direction']
                parts: List[str] = []
                if abs(steer_delta) >= max(thr['steer_deg'], thr['steer_pct'] * steer_abs_max):
                    steer_change_pct = abs(steer_delta) / steer_abs_max
                    parts.append(
                        f"wheel turn {steer_dir_curr} {format_percent(steer_pct_curr, 0)}" +
                        (f" (Δ {format_percent(steer_change_pct, 0)})" if steer_change_pct >= 0.02 else "")
                    )
                if abs(throttle_delta) >= thr['throttle']:
                    parts.append(
                        f"increase throttle to {format_percent(curr_action['throttle'],0)}" if throttle_delta > 0
                        else f"reduce throttle to {format_percent(curr_action['throttle'],0)}"
                    )
                if abs(brake_delta) >= thr['brake']:
                    parts.append(
                        f"apply brake {format_percent(curr_action['brake'],0)}" if brake_delta > 0
                        else f"release brake to {format_percent(curr_action['brake'],0)}"
                    )
                if not parts:
                    return
                substep_note = f" (step {substep_index+1}/{total_substeps})" if (substep_index is not None and total_substeps and total_substeps > 1) else ""
                text = f"At {norm_pos:.2f} position (~{max(0.0, t_idx * seconds_per_step):.1f}s){substep_note}, " + ", ".join(parts) + "."
                steps.append({
                    't': t_idx,
                    'normalized_position': norm_pos,
                    'time_s': max(0.0, t_idx * seconds_per_step),
                    'steering_angle_deg': curr_action['steering_angle_deg'],
                    'steering_percent': curr_action['steering_percent'],
                    'steering_direction': curr_action['steering_direction'],
                    'throttle': curr_action['throttle'],
                    'brake': curr_action['brake'],
                    'deltas': {
                        'steer_delta_deg': steer_delta,
                        'throttle_delta': throttle_delta,
                        'brake_delta': brake_delta
                    },
                    'instruction_text': text,
                    'substep_index': substep_index,
                    'total_substeps': total_substeps
                })
                text_instructions.append(text)

            # Compare each recommendation to baseline
            for idx in range(len(predicted_actions_list)):
                curr_a = predicted_actions_list[idx]
                steer_delta = curr_a['steering_angle_deg'] - float(baseline[0])
                throttle_delta = curr_a['throttle'] - float(baseline[1])
                brake_delta = curr_a['brake'] - float(baseline[2])

                steer_score = abs(steer_delta) / max(thr['steer_deg'], 1e-6)
                throttle_score = abs(throttle_delta) / max(thr['throttle'], 1e-6)
                brake_score = abs(brake_delta) / max(thr['brake'], 1e-6)
                combined_score = (steer_score + throttle_score + brake_score) / 3.0

                significant = (
                    abs(steer_delta) >= thr['steer_deg'] or
                    abs(steer_delta) / steer_abs_max >= thr['steer_pct'] or
                    abs(throttle_delta) >= thr['throttle'] or
                    abs(brake_delta) >= thr['brake'] or
                    combined_score >= thr['min_combined_change_score']
                )
                if not significant:
                    continue

                if split_large_changes:
                    steer_parts = 1
                    if abs(steer_delta) > 2 * thr['steer_deg']:
                        steer_parts = min(max_substeps, int(round(abs(steer_delta) / thr['steer_deg'])))
                    throttle_parts = 1
                    if abs(throttle_delta) > 2 * thr['throttle']:
                        throttle_parts = min(max_substeps, int(round(abs(throttle_delta) / thr['throttle'])))
                    brake_parts = 1
                    if abs(brake_delta) > 2 * thr['brake']:
                        brake_parts = min(max_substeps, int(round(abs(brake_delta) / thr['brake'])))
                    total_parts = max(steer_parts, throttle_parts, brake_parts)
                    if total_parts > 1:
                        for sub_i in range(total_parts):
                            alpha = (sub_i + 1) / total_parts
                            interm = {
                                't': curr_a['t'],
                                'normalized_position': curr_a['normalized_position'],
                                'steering_angle_deg': float(baseline[0]) + steer_delta * alpha,
                                'steering_percent': steering_percent(float(baseline[0]) + steer_delta * alpha),
                                'steering_direction': steering_direction(float(baseline[0]) + steer_delta * alpha),
                                'throttle': float(baseline[1]) + throttle_delta * alpha,
                                'brake': float(baseline[2]) + brake_delta * alpha
                            }
                            create_step(curr_a['t'], curr_a, interm,
                                        steer_delta * (1 / total_parts),
                                        throttle_delta * (1 / total_parts),
                                        brake_delta * (1 / total_parts),
                                        substep_index=sub_i, total_substeps=total_parts)
                    else:
                        create_step(curr_a['t'], curr_a, curr_a, steer_delta, throttle_delta, brake_delta)
                else:
                    create_step(curr_a['t'], curr_a, curr_a, steer_delta, throttle_delta, brake_delta)

            return {
                'predicted_actions': predicted_actions_list,
                'recommended_actions': recommended_actions_abs,
                'steps': steps,
                'text_instructions': text_instructions,
                'metadata': {
                    'sequence_length': sequence_length,
                    'thresholds_used': thr,
                    'steering_range': steering_range,
                    'temperature': temperature,
                    'split_large_changes': split_large_changes,
                    'starting_normalized_position': start_norm_pos,
                    'normalized_position_feature_index': feature_index,
                    'seconds_per_step': seconds_per_step,
                    'start_speed_kmh': start_speed_kmh,
                    'context_feature_names_used': (self.context_feature_names[:10] + (['...'] if len(self.context_feature_names) > 10 else [])) if hasattr(self, 'context_feature_names') else [],
                    'predictions_are_deltas': bool(getattr(self, 'targets_are_deltas', False))
                }
            }

class TelemetryActionDataset(Dataset):
    """
    Dataset class for telemetry-to-action sequence learning
    """
    
    def __init__(self,
                 telemetry_data: List[Dict[str, Any]],
                 expert_actions: List[Dict[str, Any]],
                 enriched_contextual_data: Optional[List[Dict[str, Any]]] = None,
                 sequence_length: int = 50,
                 prediction_horizon: int = 20,
                 scaler: Optional[StandardScaler] = None,
                 context_scaler: Optional[StandardScaler] = None,
                 predict_action_deltas: bool = False):
        """
        Initialize the dataset
        
        Args:
            telemetry_data: List of telemetry records (basic telemetry only)
            expert_actions: List of corresponding expert actions
            enriched_contextual_data: List of enriched contextual features (corners, grip, etc.)
            sequence_length: Input sequence length
            prediction_horizon: Number of future actions to predict
            scaler: Optional scaler for telemetry data
            context_feature_allowlist: Optional list of context feature names to include
            reasoning_feature_allowlist: Optional list of reasoning feature names to include
            context_scaler: Optional scaler for context features
            predict_action_deltas: If True, model predicts changes (deltas) from current actions
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.predict_action_deltas = predict_action_deltas
        
        # Convert to DataFrames
        self.telemetry_df = pd.DataFrame(telemetry_data)
        self.actions_df = pd.DataFrame(expert_actions)
        
        # Handle enriched contextual data (must be list of dicts or None)
        if enriched_contextual_data:
            # Require new format: list of feature dictionaries
            if not isinstance(enriched_contextual_data[0], dict):
                raise ValueError("enriched_contextual_data must be a list of dicts; legacy mixed formats are no longer supported")
            self.enriched_df = pd.DataFrame(enriched_contextual_data)
            print(f"[INFO] Using enriched contextual data with {len(self.enriched_df.columns)} enriched features", flush=True)
            print(f"[INFO] Enriched feature names: {list(self.enriched_df.columns)[:10]}..." + 
                  (f" and {len(self.enriched_df.columns)-10} more" if len(self.enriched_df.columns) > 10 else ""), flush=True)
        else:
            self.enriched_df = None
            print("[INFO] No enriched contextual data provided - will use dummy reasoning targets", flush=True)
        
        # Ensure same length across all datasets
        if self.enriched_df is not None:
            min_len = min(len(self.telemetry_df), len(self.actions_df), len(self.enriched_df))
            self.enriched_df = self.enriched_df.iloc[:min_len]
        else:
            min_len = min(len(self.telemetry_df), len(self.actions_df))
        
        self.telemetry_df = self.telemetry_df.iloc[:min_len]
        self.actions_df = self.actions_df.iloc[:min_len]
        
        # Extract numeric features
        self.telemetry_features = self._extract_telemetry_features()
        self.action_features = self._extract_action_features()

        self._context_feature_names: List[str] = []
        self._reasoning_feature_names: List[str] = []

        self.context_features_matrix, self.reasoning_features = self._extract_context_and_reasoning_features()
        
        # Scale telemetry (primary) features
        if scaler is None:
            self.scaler = StandardScaler()
            self.telemetry_features = self.scaler.fit_transform(self.telemetry_features)
        else:
            self.scaler = scaler
            self.telemetry_features = self.scaler.transform(self.telemetry_features)

        # Separate scaler for context features to avoid range distortion
        if self.context_features_matrix.shape[1] > 0:
            if context_scaler is None:
                self.context_scaler = StandardScaler()
                self.context_features_matrix = self.context_scaler.fit_transform(self.context_features_matrix)
            else:
                self.context_scaler = context_scaler
                self.context_features_matrix = self.context_scaler.transform(self.context_features_matrix)
        else:
            self.context_scaler = None
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _extract_telemetry_features(self) -> np.ndarray:
        """Extract telemetry features using the EXACT same list as get_features_for_imitate_expert()"""
        # Import here to avoid circular imports
        from ..models.telemetry_models import TelemetryFeatures
        
        # Get the exact feature list used for training
        telemetry_features = TelemetryFeatures()
        expected_features = telemetry_features.get_features_for_imitate_expert()
        
        print(f"[INFO] Using EXACT {len(expected_features)} telemetry features from get_features_for_imitate_expert()", flush=True)
        
        # Filter to only the expected features
        available_features = [col for col in expected_features if col in self.telemetry_df.columns]
        missing_features = [col for col in expected_features if col not in self.telemetry_df.columns]
        
        if missing_features:
            print(f"[WARNING] {len(missing_features)} expected features missing from dataset: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}", flush=True)
        
        if not available_features:
            print("[ERROR] No expected telemetry features found in dataset")
            return np.zeros((len(self.telemetry_df), len(expected_features)))
        
        print(f"[INFO] Found {len(available_features)} of {len(expected_features)} expected features", flush=True)
        print(f"[INFO] Feature columns: {available_features[:5]}..." + (f" and {len(available_features)-5} more" if len(available_features) > 5 else ""), flush=True)
        
        # Extract features in the same order as expected_features list for consistency
        feature_data = []
        for feature_name in expected_features:
            if feature_name in self.telemetry_df.columns:
                feature_values = self.telemetry_df[feature_name].fillna(0).values
            else:
                # Use zeros for missing features to maintain consistent shape
                feature_values = np.zeros(len(self.telemetry_df))
            feature_data.append(feature_values)
        
        # Transpose to get shape [n_samples, n_features]
        features = np.column_stack(feature_data)
        print(f"[INFO] Extracted telemetry features shape: {features.shape} (should be {len(expected_features)} features)", flush=True)
        
        return features
    
    def _extract_action_features(self) -> np.ndarray:
        """Extract expert action features (steering, throttle, brake)"""
        action_columns = ['Physics_steer_angle', 'Physics_gas', 'Physics_brake']
        
        # Select available action features
        available_actions = [col for col in action_columns if col in self.actions_df.columns]
        
        # Steering value note:
        # If 'Physics_steer_angle' has already been preprocessed it SHOULD be normalized in [0,1]
        # with ~0.5 center. Older/legacy datasets may store a physical degree value (|value| > 2).
        # We do not alter the scale here; downstream logic detects large magnitudes (>2) to treat
        # them as degrees and approximate-normalize (see _create_basic_reasoning_features and
        # generate_expert_action_instructions). Prefer normalizing upstream for new data.

        if not available_actions:
            # Create dummy actions if not available
            actions = np.zeros((len(self.actions_df), 3))
        else:
            actions = self.actions_df[available_actions].fillna(0).values
            
            # Pad with zeros if we have fewer than 3 actions
            if actions.shape[1] < 3:
                padding = np.zeros((actions.shape[0], 3 - actions.shape[1]))
                actions = np.concatenate([actions, padding], axis=1)
        
        return actions
    
    def _extract_context_and_reasoning_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split enriched features into context (inputs) vs reasoning (targets)."""
        if self.enriched_df is None:
            print("[INFO] No enriched data; using basic reasoning only and empty context", flush=True)
            basic = self._create_basic_reasoning_features()
            return np.zeros((basic.shape[0], 0)), basic

        numeric_columns = self.enriched_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            print("[WARNING] No numeric enriched features; falling back to basic reasoning", flush=True)
            basic = self._create_basic_reasoning_features()
            return np.zeros((basic.shape[0], 0)), basic

        ctx_cols: List[str] = []
        reasoning_cols: List[str] = []

        # Import canonical catalogs
        try:
            from ..services.tire_grip_analysis_service import TireGripFeatureCatalog
        except Exception:
            TireGripFeatureCatalog = None  # type: ignore
        try:
            from ..services.corner_identification_unsupervised_service import CornerFeatureCatalog
        except Exception:
            CornerFeatureCatalog = None  # type: ignore

        catalog_ctx = set()
        catalog_reason = set()
        if TireGripFeatureCatalog is not None:
            catalog_ctx.update(getattr(TireGripFeatureCatalog, 'CONTEXT_FEATURES', []))
            catalog_reason.update(getattr(TireGripFeatureCatalog, 'REASONING_FEATURES', []))
        if CornerFeatureCatalog is not None:
            catalog_ctx.update(getattr(CornerFeatureCatalog, 'CONTEXT_FEATURES', []))

        for col in numeric_columns:
            if col in catalog_ctx:
                ctx_cols.append(col)
                continue
            if col in catalog_reason:
                reasoning_cols.append(col)
                continue
            # Fallback heuristic for any remaining numeric enriched features
            lc = col.lower()
            if any(k in lc for k in ["corner", "curvature", "radius", "arc_length", "distance_to_next_corner", "straight_after_exit", "direction", "type_numeric"]):
                ctx_cols.append(col)
            else:
                reasoning_cols.append(col)

        ctx_cols = sorted(set(ctx_cols))
        reasoning_cols = sorted(set(reasoning_cols))

        context_matrix = self.enriched_df[ctx_cols].fillna(0).values if ctx_cols else np.zeros((len(self.enriched_df), 0))
        reasoning_matrix = self.enriched_df[reasoning_cols].fillna(0).values if reasoning_cols else self._create_basic_reasoning_features()

        self._context_feature_names = ctx_cols
        self._reasoning_feature_names = reasoning_cols

        print(f"[INFO] Context features selected ({len(ctx_cols)}): {ctx_cols[:5]}" + (f" ... and {len(ctx_cols)-5} more" if len(ctx_cols) > 5 else ""), flush=True)
        print(f"[INFO] Reasoning target features selected ({len(reasoning_cols)}): {reasoning_cols[:5]}" + (f" ... and {len(reasoning_cols)-5} more" if len(reasoning_cols) > 5 else ""), flush=True)
        return context_matrix, reasoning_matrix
    
    def _create_basic_reasoning_features(self) -> np.ndarray:
        """Create basic reasoning features from telemetry when enriched data is not available"""
        reasoning_data = []
        
        for _, row in self.telemetry_df.iterrows():
            features = []
            
            # Speed-based reasoning
            speed = row.get('Physics_speed_kmh', 0)
            features.extend([
                speed / 300.0,  # Normalized speed
                1.0 if speed < 50 else 0.0,  # Slow corner indicator
                1.0 if speed > 200 else 0.0,  # High speed indicator
            ])
            
            # G-force based reasoning
            g_lat = abs(row.get('Physics_g_force_y', 0))
            g_long = row.get('Physics_g_force_x', 0)
            features.extend([
                g_lat / 3.0,  # Normalized lateral G
                abs(g_long) / 3.0,  # Normalized longitudinal G
                1.0 if g_lat > 1.5 else 0.0,  # High cornering indicator
            ])
            
            # Steering reasoning (training data now normalized 0..1 with ~0.5 center)
            steer_raw = row.get('Physics_steer_angle', 0)
            # Heuristic degree vs normalized detection:
            #  - Expected modern format:   0..1  (0.5 center)  -> convert to signed [-1,1]
            #  - Legacy format (degrees): |value| typically >> 2 (e.g. up to 500)
            #    We treat those as degrees and approximate-normalize (absolute / 500).
            if steer_raw > 2:  # legacy degrees path
                steer_norm = min(1.0, abs(steer_raw) / 500.0)
                sharp_indicator = 1.0 if abs(steer_raw) > 200 else 0.0
            else:
                # Proper normalized value 0..1
                steer_clamped = max(0.0, min(1.0, steer_raw))
                # Convert to signed -1..1 assuming 0.5 center for magnitude-based reasoning
                steer_signed = (steer_clamped - 0.5) * 2.0
                steer_norm = abs(steer_signed)  # 0..1 intensity
                sharp_indicator = 1.0 if steer_norm > 0.6 else 0.0  # threshold can be tuned
            features.extend([
                steer_norm,
                sharp_indicator,
            ])
            
            # Brake/throttle reasoning
            brake = row.get('Physics_brake', 0)
            throttle = row.get('Physics_gas', 0)
            features.extend([
                brake,  # Brake input
                throttle,  # Throttle input
                1.0 if brake > 0.5 else 0.0,  # Heavy braking indicator
                1.0 if throttle > 0.8 else 0.0,  # Full throttle indicator
            ])
            
            # Pad or truncate to ensure consistent size (30 features)
            while len(features) < 30:
                features.append(0.0)
            features = features[:30]
            
            reasoning_data.append(features)
        
        return np.array(reasoning_data)
    
    def _create_dummy_reasoning_features(self) -> np.ndarray:
        """Create dummy reasoning features filled with zeros"""
        return np.zeros((len(self.telemetry_df), 30))  # 30 dummy reasoning features
    
    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        """Create input-output sequences for training"""
        sequences = []
        
        total_len = len(self.telemetry_features)
        max_start = total_len - self.sequence_length - self.prediction_horizon
        
        for i in range(0, max_start, 5):  # Step by 5 for overlapping sequences
            # Input telemetry sequence (basic telemetry only)
            telemetry_seq = self.telemetry_features[i:i+self.sequence_length]
            
            # Target action sequence
            action_seq = self.action_features[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
            if self.predict_action_deltas:
                # Baseline: current driver's last input at encoder end time
                base_idx = i + self.sequence_length - 1
                # Extract from telemetry_df if available, else zeros
                steer_b = float(self.telemetry_df.iloc[base_idx].get('Physics_steer_angle', 0.0)) if base_idx < len(self.telemetry_df) else 0.0
                gas_b = float(self.telemetry_df.iloc[base_idx].get('Physics_gas', 0.0)) if base_idx < len(self.telemetry_df) else 0.0
                brake_b = float(self.telemetry_df.iloc[base_idx].get('Physics_brake', 0.0)) if base_idx < len(self.telemetry_df) else 0.0
                baseline = np.array([steer_b, gas_b, brake_b], dtype=np.float32)
                # Broadcast subtract baseline from each future step to form delta targets
                if action_seq.shape[1] >= 3:
                    action_seq = action_seq.copy()
                    action_seq[:, :3] = action_seq[:, :3] - baseline[None, :3]
            
            # Target reasoning sequence (enriched contextual features)
            reasoning_seq = self.reasoning_features[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
            # Context sequence aligned with encoder portion only
            context_seq = self.context_features_matrix[i:i+self.sequence_length] if self.context_features_matrix.shape[1] > 0 else None
            
            sequences.append({
                'telemetry': telemetry_seq,
                'actions': action_seq,
                'reasoning': reasoning_seq,
                'context': context_seq
            })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        sequence = self.sequences[idx]
        
        # Keep original shape: [seq_len, features] and [pred_horizon, features]
        # DataLoader will add batch dimension: [batch_size, seq_len, features]
        # Then we'll transpose in training to get: [seq_len, batch_size, features]
        telemetry = torch.FloatTensor(sequence['telemetry'])
        actions = torch.FloatTensor(sequence['actions'])
        reasoning = torch.FloatTensor(sequence['reasoning'])
        context = torch.FloatTensor(sequence['context']) if sequence.get('context') is not None else torch.empty(0)
        return telemetry, actions, reasoning, context

    @property
    def context_feature_names(self) -> List[str]:
        return self._context_feature_names

    @property
    def reasoning_feature_names(self) -> List[str]:
        return self._reasoning_feature_names

    def export_scalers(self) -> Dict[str, Any]:
        """Return scaler parameters for persistence (means/vars) without serializing full objects."""
        def pack(s: Optional[StandardScaler]):
            if s is None:
                return None
            return {
                'mean': s.mean_.tolist(),
                'scale': s.scale_.tolist(),
                'var': s.var_.tolist(),
                'n_features': len(s.mean_)
            }
        return {
            'telemetry_scaler': pack(self.scaler),
            'context_scaler': pack(self.context_scaler)
        }

class ExpertActionTrainer:
    """
    Trainer class for the Expert Action Transformer
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
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.action_loss_fn = nn.MSELoss()
        self.reasoning_loss_fn = nn.MSELoss()
        self.performance_loss_fn = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.training_history = []
    
    def train_epoch(self, dataloader: DataLoader, 
                    reasoning_weight: float = 0.5, 
                    performance_weight: float = 0.1) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_action_loss = 0
        total_reasoning_loss = 0
        total_performance_loss = 0
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Expect 4-tuple: (telemetry, actions, reasoning, context)
            telemetry, target_actions, target_reasoning, context = batch
            # Print progress every 5 batches during training for more frequent updates
            if batch_idx % 5 == 0:
                progress_pct = (batch_idx / num_batches) * 100
                print_progress(f"  Training batch {batch_idx+1}/{num_batches} ({progress_pct:.1f}%)")
                
            # Transpose to get transformer-expected dimensions
            # From [batch_size, seq_len, features] to [seq_len, batch_size, features]
            telemetry = telemetry.transpose(0, 1).to(self.device)
            target_actions = target_actions.transpose(0, 1).to(self.device)
            target_reasoning = target_reasoning.transpose(0, 1).to(self.device)
            src_context = None
            if context is not None and context.numel() > 0:
                # context shape: [batch, seq_len, ctx_features] -> transpose to [seq_len, batch, ctx_features]
                src_context = context.transpose(0, 1).to(self.device)
            
            # Create decoder input (shifted target)
            # target_actions shape: [pred_horizon, batch_size, action_features]
            decoder_input = torch.cat([
                torch.zeros(1, target_actions.shape[1], target_actions.shape[2], device=self.device),
                target_actions[:-1]
            ], dim=0)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_actions, predicted_reasoning, performance_scores = self.model(telemetry, decoder_input, src_context=src_context)
            
            # Calculate losses
            action_loss = self.action_loss_fn(predicted_actions, target_actions)
            reasoning_loss = self.reasoning_loss_fn(predicted_reasoning, target_reasoning)
            
            # Create dummy performance targets (could be based on lap times, sector times, etc.)
            performance_targets = torch.zeros_like(performance_scores)
            performance_loss = self.performance_loss_fn(performance_scores, performance_targets)
            
            # Combined loss with weights
            loss = action_loss + reasoning_weight * reasoning_loss + performance_weight * performance_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_action_loss += action_loss.item()
            total_reasoning_loss += reasoning_loss.item()
            total_performance_loss += performance_loss.item()
            total_loss += loss.item()
        
        return {
            'action_loss': total_action_loss / num_batches,
            'reasoning_loss': total_reasoning_loss / num_batches,
            'performance_loss': total_performance_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def validate(self, dataloader: DataLoader, 
                 reasoning_weight: float = 0.5, 
                 performance_weight: float = 0.1) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_action_loss = 0
        total_reasoning_loss = 0
        total_performance_loss = 0
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                telemetry, target_actions, target_reasoning, context = batch
                # Transpose to get transformer-expected dimensions
                # From [batch_size, seq_len, features] to [seq_len, batch_size, features]
                telemetry = telemetry.transpose(0, 1).to(self.device)
                target_actions = target_actions.transpose(0, 1).to(self.device)
                target_reasoning = target_reasoning.transpose(0, 1).to(self.device)
                src_context = None
                if context is not None and context.numel() > 0:
                    src_context = context.transpose(0, 1).to(self.device)
                
                # Create decoder input
                decoder_input = torch.cat([
                    torch.zeros(1, target_actions.shape[1], target_actions.shape[2], device=self.device),
                    target_actions[:-1]
                ], dim=0)
                
                # Forward pass
                predicted_actions, predicted_reasoning, performance_scores = self.model(telemetry, decoder_input, src_context=src_context)
                
                # Calculate losses
                action_loss = self.action_loss_fn(predicted_actions, target_actions)
                reasoning_loss = self.reasoning_loss_fn(predicted_reasoning, target_reasoning)
                performance_targets = torch.zeros_like(performance_scores)
                performance_loss = self.performance_loss_fn(performance_scores, performance_targets)
                
                loss = action_loss + reasoning_weight * reasoning_loss + performance_weight * performance_loss
                
                total_action_loss += action_loss.item()
                total_reasoning_loss += reasoning_loss.item()
                total_performance_loss += performance_loss.item()
                total_loss += loss.item()
        
        return {
            'action_loss': total_action_loss / num_batches,
            'reasoning_loss': total_reasoning_loss / num_batches,
            'performance_loss': total_performance_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              patience: int = 10) ->  Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of epochs
            patience: Early stopping patience
            
        Returns:
            Tuple of (trained_model, training_results_and_metrics)
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print_progress(f"\nStarting Epoch {epoch+1}/{num_epochs}")
            epoch_start_time = datetime.now()
            
            # Train
            train_metrics = self.train_epoch(train_dataloader, reasoning_weight=0.5)
            
            epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
            print_progress(f"Epoch {epoch+1} completed in {epoch_duration:.1f}s")
            
            # Validate
            if val_dataloader:
                val_metrics = self.validate(val_dataloader, reasoning_weight=0.5)
                val_loss = val_metrics['total_loss']
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print_progress(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                
                if patience_counter >= patience:
                    print_progress(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print_progress(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Store history
            epoch_history = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics if val_dataloader else None,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
        
        return {
            'training_completed': True,
            'best_val_loss': best_val_loss if val_dataloader else None,
            'training_history': self.training_history,
            'total_epochs': len(self.training_history)
        }
