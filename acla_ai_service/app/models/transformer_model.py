

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
    Transformer model uses current driver's telemetry to plan steps to converge to future expert state as optimal as possible.
    model is constrained by track shape, car physics, and it will learn to operate the car within
    physical and geometric limits. the primary prediction task is to output a sequence of actions for the current driver to reach
    expert state at a given normalized track distance.

    Architecture:
    - Input: Current telemetry features + contextual data (corner info, tire grip, etc.)
    - Output: Sequence actions of current driver who tries to reach expert state (velocity, location)
    - Uses attention mechanism to focus on relevant past patterns
    """
    
    def __init__(self, 
                 input_features: int = 42,  # Telemetry features from get_features_for_imitate_expert()
                 context_features: int = 31,  # Corner (16) + tire grip (4) + expert state (11)
                 action_features: int = 5,  # throttle, brake, steering, gear, speed
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 sequence_length: int = 20,
                 dropout: float = 0.1):
        """
        Initialize the Expert Action Transformer
        
        Args:
            input_features: Number of telemetry input features 
            context_features: Number of enriched contextual features (corners, grip, expert state)
            action_features: Number of action outputs to predict (throttle, brake, steer, gear, speed)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            sequence_length: Maximum sequence length for predictions
            dropout: Dropout rate
        """
        super(ExpertActionTransformer, self).__init__()
        
        # Store configuration
        self.input_features = input_features
        self.context_features = context_features 
        self.action_features = action_features
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input embeddings
        self.telemetry_embedding = nn.Linear(input_features, d_model)
        self.context_embedding = nn.Linear(context_features, d_model) if context_features > 0 else None
        
        # Positional encoding
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
        self.action_embedding = nn.Linear(action_features, d_model)
        
        # Output projection to action space
        self.action_projection = nn.Linear(d_model, action_features)
        
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
        Forward pass
        
        Args:
            telemetry: Input telemetry features [batch_size, seq_len, input_features]
            context: Contextual features [batch_size, seq_len, context_features] 
            target_actions: Target action sequence for training [batch_size, seq_len, action_features]
            target_mask: Mask for target sequence [batch_size, seq_len]
            
        Returns:
            Predicted action sequence [batch_size, seq_len, action_features]
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
        
        if target_actions is not None:
            # Training mode: use teacher forcing
            # Embed target actions (shifted right for decoder input)
            target_embedded = self.action_embedding(target_actions)  # [B, L, d_model]
            target_embedded = self.pos_encoding(target_embedded)
            
            # Create causal mask for decoder
            tgt_mask = self._generate_square_subsequent_mask(seq_len)
            
            # Decode action sequence
            decoder_output = self.transformer_decoder(
                tgt=target_embedded,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [B, L, d_model]
        else:
            # Inference mode: autoregressive generation
            decoder_output = self._generate_actions_autoregressive(memory, seq_len)
        
        # Project to action space
        output = self.action_projection(decoder_output)  # [B, L, action_features]
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _generate_actions_autoregressive(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Generate actions autoregressively during inference"""
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
    
    def predict_expert_sequence(self, 
                               telemetry: torch.Tensor,
                               context: Optional[torch.Tensor] = None,
                               sequence_length: Optional[int] = None,
                               temperature: float = 1.0,
                               deterministic: bool = False) -> torch.Tensor:
        """
        Predict a sequence of actions to reach expert state
        
        Args:
            telemetry: Input telemetry features [batch_size, input_seq_len, input_features]
            context: Optional contextual features [batch_size, input_seq_len, context_features]  
            sequence_length: Length of action sequence to predict (default: self.sequence_length)
            temperature: Temperature for sampling (higher = more random)
            deterministic: If True, use greedy decoding instead of sampling
            
        Returns:
            Predicted action sequence [batch_size, sequence_length, action_features]
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
                'input_features': self.input_features,
                'context_features': self.context_features,
                'action_features': self.action_features,
                'd_model': self.d_model,
                'sequence_length': self.sequence_length,
                'nhead': getattr(self.transformer_encoder.layers[0].self_attn, 'num_heads', 8),
                'num_layers': len(self.transformer_encoder.layers),
                'dim_feedforward': getattr(self.transformer_encoder.layers[0].linear1, 'out_features', 1024),
                'dropout': 0.1  # Default, could extract from layers if needed
            },
            'serialization_timestamp': datetime.now().isoformat()
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
        
        config = serialized_data['config']
        
        # Create new model instance with saved configuration
        model = cls(
            input_features=config['input_features'],
            context_features=config['context_features'], 
            action_features=config['action_features'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            sequence_length=config['sequence_length'],
            dropout=config.get('dropout', 0.1)
        )
        
        # Load state dict from base64 encoded bytes
        state_dict_bytes = base64.b64decode(serialized_data['state_dict'].encode('utf-8'))
        buffer = io.BytesIO(state_dict_bytes)
        state_dict = torch.load(buffer, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model

class TelemetryActionDataset(Dataset):
    """
    Dataset class for telemetry-to-action sequence learning
    """
    
    def __init__(self,
                 telemetry_data: List[Dict[str, Any]],
                 expert_actions: List[Dict[str, Any]],
                 enriched_contextual_data: Optional[List[Dict[str, Any]]] = None,
                 sequence_length: int = 20,
                 telemetry_features: Optional[List[str]] = None,
                 action_features: Optional[List[str]] = None):
        """
        Initialize the dataset
        
        Args:
            telemetry_data: List of telemetry records (basic telemetry only), drivers have different skill sets
            expert_actions: List of corresponding expert actions
            enriched_contextual_data: List of enriched contextual features (corners, grip, etc.)
            sequence_length: Length of sequences to generate
            telemetry_features: List of feature names to extract from telemetry_data
            action_features: List of action feature names to extract from expert_actions
        """
        assert len(telemetry_data) == len(expert_actions), "Telemetry and expert actions must have same length"
        
        self.telemetry_data = telemetry_data
        self.expert_actions = expert_actions  
        self.enriched_contextual_data = enriched_contextual_data or []
        self.sequence_length = sequence_length
        
        # Default feature lists
        self.telemetry_features = telemetry_features or self._get_default_telemetry_features()
        self.action_features = action_features or self._get_default_action_features()
        
        # Preprocessing
        self._preprocess_data()
        
        # Generate sequence indices
        self._generate_sequences()
    
    def _get_default_telemetry_features(self) -> List[str]:
        """Get default telemetry features for input"""
        return [
            "Graphics_normalized_car_position", "Graphics_player_pos_x", "Graphics_player_pos_y", 
            "Graphics_player_pos_z", "Graphics_current_time", "Physics_speed_kmh", "Physics_gas",
            "Physics_brake", "Physics_steer_angle", "Physics_gear", "Physics_rpm", "Physics_g_force_x",
            "Physics_g_force_y", "Physics_g_force_z", "Physics_slip_angle_front_left", "Physics_slip_angle_front_right",
            "Physics_slip_angle_rear_left", "Physics_slip_angle_rear_right", "Physics_velocity_x",
            "Physics_velocity_y", "Physics_velocity_z"
        ]
    
    def _get_default_action_features(self) -> List[str]:
        """Get default action features to predict"""
        return [
            "expert_optimal_throttle", "expert_optimal_brake", "expert_optimal_steering",
            "expert_optimal_gear", "expert_optimal_speed"
        ]
    
    def _preprocess_data(self):
        """Preprocess and normalize the data"""
        # Extract feature matrices
        self.telemetry_matrix = self._extract_features(self.telemetry_data, self.telemetry_features)
        self.action_matrix = self._extract_features(self.expert_actions, self.action_features)
        
        # Extract contextual features if available
        if self.enriched_contextual_data:
            context_features = list(self.enriched_contextual_data[0].keys()) if self.enriched_contextual_data else []
            self.context_matrix = self._extract_features(self.enriched_contextual_data, context_features)
        else:
            self.context_matrix = None
        
        # Normalize features
        self.telemetry_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        self.context_scaler = StandardScaler() if self.context_matrix is not None else None
        
        self.telemetry_matrix = self.telemetry_scaler.fit_transform(self.telemetry_matrix)
        self.action_matrix = self.action_scaler.fit_transform(self.action_matrix)
        
        if self.context_matrix is not None:
            self.context_matrix = self.context_scaler.fit_transform(self.context_matrix)
        
        print(f"[INFO] Preprocessed dataset: {self.telemetry_matrix.shape[0]} samples, "
              f"{self.telemetry_matrix.shape[1]} telemetry features, "
              f"{self.action_matrix.shape[1]} action features")
        
        if self.context_matrix is not None:
            print(f"[INFO] Context matrix: {self.context_matrix.shape[1]} contextual features")
    
    def _extract_features(self, data_list: List[Dict[str, Any]], feature_names: List[str]) -> np.ndarray:
        """Extract features from list of dictionaries"""
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
        """Generate valid sequence start indices"""
        self.sequence_indices = []
        
        # For now, generate non-overlapping sequences
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
        action_seq = torch.tensor(self.action_matrix[start_idx:end_idx], dtype=torch.float32)
        
        if self.context_matrix is not None:
            context_seq = torch.tensor(self.context_matrix[start_idx:end_idx], dtype=torch.float32)
            return telemetry_seq, context_seq, action_seq
        else:
            return telemetry_seq, action_seq
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """Get telemetry and action feature names"""
        return self.telemetry_features, self.action_features
    
    def get_scalers(self) -> Dict[str, StandardScaler]:
        """Get the fitted scalers for denormalization"""
        scalers = {
            'telemetry': self.telemetry_scaler,
            'action': self.action_scaler
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
        
        # Loss function - MSE for regression
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
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
            
            # Forward pass with teacher forcing
            # For transformer decoder, we shift target actions right for input
            decoder_input = torch.zeros_like(target_actions)
            decoder_input[:, 1:] = target_actions[:, :-1]  # Shift right
            
            predictions = self.model(
                telemetry=telemetry,
                context=context,
                target_actions=decoder_input
            )
            
            # Compute loss
            loss = self.criterion(predictions, target_actions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
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
                
                # Forward pass with teacher forcing  
                decoder_input = torch.zeros_like(target_actions)
                decoder_input[:, 1:] = target_actions[:, :-1]
                
                predictions = self.model(
                    telemetry=telemetry,
                    context=context,
                    target_actions=decoder_input
                )
                
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
        Train the model
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            epochs: Number of training epochs
            patience: Early stopping patience
            save_best: Whether to save the best model state
            
        Returns:
            Training history and final metrics
        """
        print(f"[INFO] Starting training for {epochs} epochs on {self.device}")
        print(f"[INFO] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'final_lr': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
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
                
                # Use model's prediction method (no teacher forcing)
                predictions = self.model.predict_expert_sequence(
                    telemetry=telemetry,
                    context=context,
                    sequence_length=target_actions.shape[1],
                    deterministic=True
                )
                
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
        
        return {
            'test_loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'num_samples': total_samples
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and training state"""
        return {
            'model_config': {
                'input_features': self.model.input_features,
                'context_features': self.model.context_features,
                'action_features': self.model.action_features,
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
        }