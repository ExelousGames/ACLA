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
                 reasoning_features: int = 30,  # contextual reasoning features (corners, grip, etc.)
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_sequence_length: int = 100):
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
        
        self.input_features = input_features
        self.action_features = action_features
        self.reasoning_features = reasoning_features
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        
        # Input embedding layers
        self.input_embedding = nn.Linear(input_features, d_model)
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
        
        # Embed inputs
        src_embedded = self.input_embedding(src_telemetry) * math.sqrt(self.d_model)
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
                               temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expert action sequence autoregressively
        
        Args:
            src_telemetry: Source telemetry data
            sequence_length: Length of sequence to predict
            temperature: Temperature for sampling (1.0 = normal, lower = more conservative)
            
        Returns:
            Tuple of (predicted_action_sequence, predicted_reasoning_sequence, performance_scores)
        """
        self.eval()
        
        batch_size = src_telemetry.shape[1]
        device = src_telemetry.device
        
        # Initialize target sequence with zeros (or special start token)
        tgt_actions = torch.zeros(1, batch_size, self.action_features, device=device)
        predicted_sequence = []
        predicted_reasoning_sequence = []
        performance_scores = []
        
        with torch.no_grad():
            for i in range(sequence_length):
                # Forward pass
                pred_actions, pred_reasoning, perf_scores = self.forward(src_telemetry, tgt_actions)
                
                # Get the last predicted values
                last_action = pred_actions[-1:, :, :]  # [1, batch_size, action_features]
                last_reasoning = pred_reasoning[-1:, :, :]  # [1, batch_size, reasoning_features]
                last_perf = perf_scores[-1:, :, :]     # [1, batch_size, 1]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    last_action = last_action / temperature
                
                predicted_sequence.append(last_action)
                predicted_reasoning_sequence.append(last_reasoning)
                performance_scores.append(last_perf)
                
                # Update target sequence for next iteration
                tgt_actions = torch.cat([tgt_actions, last_action], dim=0)
        
        # Concatenate all predictions
        full_sequence = torch.cat(predicted_sequence, dim=0)
        full_reasoning = torch.cat(predicted_reasoning_sequence, dim=0)
        full_performance = torch.cat(performance_scores, dim=0)
        
        return full_sequence, full_reasoning, full_performance
    
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
                'd_model': self.d_model,
                'max_sequence_length': self.max_sequence_length
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
            reasoning_features=config.get('reasoning_features', 30),  # Default for backward compatibility
            d_model=config['d_model'],
            nhead=architecture['nhead'],
            num_encoder_layers=architecture['num_encoder_layers'],
            num_decoder_layers=architecture['num_decoder_layers'],
            dim_feedforward=architecture['dim_feedforward'],
            dropout=architecture['dropout'],
            max_sequence_length=config['max_sequence_length']
        )
        
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
            self.d_model = config['d_model']
            self.max_sequence_length = config['max_sequence_length']
        
        if 'model_architecture' in serialized_data:
            architecture = serialized_data['model_architecture']
            self.nhead = architecture['nhead']
            self.num_encoder_layers = architecture['num_encoder_layers']
            self.num_decoder_layers = architecture['num_decoder_layers']
            self.dim_feedforward = architecture['dim_feedforward']
            self.dropout_rate = architecture['dropout']

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
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize the dataset
        
        Args:
            telemetry_data: List of telemetry records (basic telemetry only)
            expert_actions: List of corresponding expert actions
            enriched_contextual_data: List of enriched contextual features (corners, grip, etc.)
            sequence_length: Input sequence length
            prediction_horizon: Number of future actions to predict
            scaler: Optional scaler for telemetry data
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Convert to DataFrames
        self.telemetry_df = pd.DataFrame(telemetry_data)
        self.actions_df = pd.DataFrame(expert_actions)
        
        # Handle enriched contextual data (now as separate feature dictionaries)
        if enriched_contextual_data:
            if isinstance(enriched_contextual_data[0], dict):
                # New format: list of feature dictionaries
                self.enriched_df = pd.DataFrame(enriched_contextual_data)
                print(f"[INFO] Using enriched contextual data with {len(self.enriched_df.columns)} enriched features", flush=True)
                print(f"[INFO] Enriched feature names: {list(self.enriched_df.columns)[:10]}..." + 
                      (f" and {len(self.enriched_df.columns)-10} more" if len(self.enriched_df.columns) > 10 else ""), flush=True)
            else:
                # Old format: mixed telemetry data (for backward compatibility)
                self.enriched_df = pd.DataFrame(enriched_contextual_data)
                print(f"[INFO] Using enriched contextual data (legacy format) with {len(self.enriched_df.columns)} columns", flush=True)
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
        self.reasoning_features = self._extract_reasoning_features()
        
        # Scale telemetry data
        if scaler is None:
            self.scaler = StandardScaler()
            self.telemetry_features = self.scaler.fit_transform(self.telemetry_features)
        else:
            self.scaler = scaler
            self.telemetry_features = self.scaler.transform(self.telemetry_features)
        
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
    
    def _extract_reasoning_features(self) -> np.ndarray:
        """Extract enriched contextual reasoning features (corners, grip, etc.)"""
        if self.enriched_df is not None:
            # Use all numeric columns from enriched data as reasoning features
            numeric_columns = self.enriched_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Since enriched features are already separated, use all of them
            enriched_columns = numeric_columns
            
            if enriched_columns:
                reasoning_features = self.enriched_df[enriched_columns].fillna(0).values
                print(f"[INFO] Extracted {len(enriched_columns)} reasoning features from enriched data", flush=True)
                print(f"[INFO] Reasoning features: {enriched_columns[:5]}..." + 
                      (f" and {len(enriched_columns)-5} more" if len(enriched_columns) > 5 else ""), flush=True)
                return reasoning_features
            else:
                print("[WARNING] No numeric enriched features found, using basic reasoning features", flush=True)
                # Fallback to basic derived features
                return self._create_basic_reasoning_features()
        else:
            # Create dummy reasoning features when no enriched data is provided
            print("[INFO] Creating basic reasoning features from telemetry (no enriched data provided)", flush=True)
            return self._create_basic_reasoning_features()
    
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
            
            # Steering reasoning
            steer = abs(row.get('Physics_steer_angle', 0))
            features.extend([
                steer / 500.0,  # Normalized steering angle
                1.0 if steer > 200 else 0.0,  # Sharp turn indicator
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
            
            # Target reasoning sequence (enriched contextual features)
            reasoning_seq = self.reasoning_features[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
            
            sequences.append({
                'telemetry': telemetry_seq,
                'actions': action_seq,
                'reasoning': reasoning_seq
            })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Keep original shape: [seq_len, features] and [pred_horizon, features]
        # DataLoader will add batch dimension: [batch_size, seq_len, features]
        # Then we'll transpose in training to get: [seq_len, batch_size, features]
        telemetry = torch.FloatTensor(sequence['telemetry'])  # [seq_len, features]
        actions = torch.FloatTensor(sequence['actions'])      # [pred_horizon, action_features]
        reasoning = torch.FloatTensor(sequence['reasoning'])  # [pred_horizon, reasoning_features]
        
        return telemetry, actions, reasoning

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
        
        for batch_idx, (telemetry, target_actions, target_reasoning) in enumerate(dataloader):
            # Print progress every 5 batches during training for more frequent updates
            if batch_idx % 5 == 0:
                progress_pct = (batch_idx / num_batches) * 100
                print_progress(f"  Training batch {batch_idx+1}/{num_batches} ({progress_pct:.1f}%)")
                
            # Transpose to get transformer-expected dimensions
            # From [batch_size, seq_len, features] to [seq_len, batch_size, features]
            telemetry = telemetry.transpose(0, 1).to(self.device)
            target_actions = target_actions.transpose(0, 1).to(self.device)
            target_reasoning = target_reasoning.transpose(0, 1).to(self.device)
            
            # Create decoder input (shifted target)
            # target_actions shape: [pred_horizon, batch_size, action_features]
            decoder_input = torch.cat([
                torch.zeros(1, target_actions.shape[1], target_actions.shape[2], device=self.device),
                target_actions[:-1]
            ], dim=0)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_actions, predicted_reasoning, performance_scores = self.model(telemetry, decoder_input)
            
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
            for telemetry, target_actions, target_reasoning in dataloader:
                # Transpose to get transformer-expected dimensions
                # From [batch_size, seq_len, features] to [seq_len, batch_size, features]
                telemetry = telemetry.transpose(0, 1).to(self.device)
                target_actions = target_actions.transpose(0, 1).to(self.device)
                target_reasoning = target_reasoning.transpose(0, 1).to(self.device)
                
                # Create decoder input
                decoder_input = torch.cat([
                    torch.zeros(1, target_actions.shape[1], target_actions.shape[2], device=self.device),
                    target_actions[:-1]
                ], dim=0)
                
                # Forward pass
                predicted_actions, predicted_reasoning, performance_scores = self.model(telemetry, decoder_input)
                
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
