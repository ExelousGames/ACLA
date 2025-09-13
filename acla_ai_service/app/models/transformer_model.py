"""
Transformer Model for Expert Action Prediction in Racing Telemetry

This model predicts the sequence of actions needed to reach expert optimal performance
over a period of time. It analyzes current driver behavior and suggests step-by-step
improvements to match expert driving patterns.
"""

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
            input_features: Number of input telemetry features
            action_features: Number of action outputs (steering, throttle, brake)
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
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
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
        
        # Output projection
        self.output_projection = nn.Linear(d_model, action_features)
        
        # Additional layers for performance prediction
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
                tgt_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            src_telemetry: Source telemetry data [seq_len, batch_size, input_features]
            tgt_actions: Target action sequence [seq_len, batch_size, action_features]
            src_padding_mask: Padding mask for source
            tgt_padding_mask: Padding mask for target
            
        Returns:
            Tuple of (predicted_actions, performance_scores)
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
        
        # Project to action space
        predicted_actions = self.output_projection(transformer_output)
        
        # Predict performance scores
        performance_scores = self.performance_head(transformer_output)
        
        return predicted_actions, performance_scores
    
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
            Tuple of (predicted_action_sequence, performance_scores)
        """
        self.eval()
        
        batch_size = src_telemetry.shape[1]
        device = src_telemetry.device
        
        # Initialize target sequence with zeros (or special start token)
        tgt_actions = torch.zeros(1, batch_size, self.action_features, device=device)
        predicted_sequence = []
        performance_scores = []
        
        with torch.no_grad():
            for i in range(sequence_length):
                # Forward pass
                pred_actions, perf_scores = self.forward(src_telemetry, tgt_actions)
                
                # Get the last predicted action
                last_action = pred_actions[-1:, :, :]  # [1, batch_size, action_features]
                last_perf = perf_scores[-1:, :, :]     # [1, batch_size, 1]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    last_action = last_action / temperature
                
                predicted_sequence.append(last_action)
                performance_scores.append(last_perf)
                
                # Update target sequence for next iteration
                tgt_actions = torch.cat([tgt_actions, last_action], dim=0)
        
        # Concatenate all predictions
        full_sequence = torch.cat(predicted_sequence, dim=0)
        full_performance = torch.cat(performance_scores, dim=0)
        
        return full_sequence, full_performance

class TelemetryActionDataset(Dataset):
    """
    Dataset class for telemetry-to-action sequence learning
    """
    
    def __init__(self,
                 telemetry_data: List[Dict[str, Any]],
                 expert_actions: List[Dict[str, Any]],
                 sequence_length: int = 50,
                 prediction_horizon: int = 20,
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize the dataset
        
        Args:
            telemetry_data: List of telemetry records
            expert_actions: List of corresponding expert actions
            sequence_length: Input sequence length
            prediction_horizon: Number of future actions to predict
            scaler: Optional scaler for telemetry data
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Convert to DataFrames
        self.telemetry_df = pd.DataFrame(telemetry_data)
        self.actions_df = pd.DataFrame(expert_actions)
        
        # Ensure same length
        min_len = min(len(self.telemetry_df), len(self.actions_df))
        self.telemetry_df = self.telemetry_df.iloc[:min_len]
        self.actions_df = self.actions_df.iloc[:min_len]
        
        # Extract numeric features
        self.telemetry_features = self._extract_telemetry_features()
        self.action_features = self._extract_action_features()
        
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
        """Extract relevant telemetry features"""
        feature_columns = [
            'Physics_speed_kmh', 'Physics_gear', 'Physics_rpm', 'Physics_brake',
            'Physics_gas', 'Physics_steer_angle', 'Physics_slip_angle_front_left',
            'Physics_slip_angle_front_right', 'Physics_g_force_x', 'Physics_g_force_y',
            'Physics_g_force_z', 'Physics_tyre_core_temp_front_left',
            'Physics_tyre_core_temp_front_right', 'Physics_brake_temp_front_left',
            'Physics_brake_temp_front_right', 'Graphics_delta_lap_time'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in self.telemetry_df.columns]
        
        if not available_features:
            # Fallback to any numeric columns
            available_features = self.telemetry_df.select_dtypes(include=[np.number]).columns.tolist()
        
        features = self.telemetry_df[available_features].fillna(0).values
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
    
    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        """Create input-output sequences for training"""
        sequences = []
        
        total_len = len(self.telemetry_features)
        max_start = total_len - self.sequence_length - self.prediction_horizon
        
        for i in range(0, max_start, 5):  # Step by 5 for overlapping sequences
            # Input telemetry sequence
            telemetry_seq = self.telemetry_features[i:i+self.sequence_length]
            
            # Target action sequence
            action_seq = self.action_features[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon]
            
            sequences.append({
                'telemetry': telemetry_seq,
                'actions': action_seq
            })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Keep original shape: [seq_len, features] and [pred_horizon, action_features]
        # DataLoader will add batch dimension: [batch_size, seq_len, features]
        # Then we'll transpose in training to get: [seq_len, batch_size, features]
        telemetry = torch.FloatTensor(sequence['telemetry'])  # [seq_len, features]
        actions = torch.FloatTensor(sequence['actions'])      # [pred_horizon, action_features]
        
        return telemetry, actions

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
        self.performance_loss_fn = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.training_history = []
    
    def train_epoch(self, dataloader: DataLoader, performance_weight: float = 0.1) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_action_loss = 0
        total_performance_loss = 0
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (telemetry, target_actions) in enumerate(dataloader):
            # Transpose to get transformer-expected dimensions
            # From [batch_size, seq_len, features] to [seq_len, batch_size, features]
            telemetry = telemetry.transpose(0, 1).to(self.device)
            target_actions = target_actions.transpose(0, 1).to(self.device)
            
            # Create decoder input (shifted target)
            # target_actions shape: [pred_horizon, batch_size, action_features]
            decoder_input = torch.cat([
                torch.zeros(1, target_actions.shape[1], target_actions.shape[2], device=self.device),
                target_actions[:-1]
            ], dim=0)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_actions, performance_scores = self.model(telemetry, decoder_input)
            
            # Calculate losses
            action_loss = self.action_loss_fn(predicted_actions, target_actions)
            
            # Create dummy performance targets (could be based on lap times, sector times, etc.)
            performance_targets = torch.zeros_like(performance_scores)
            performance_loss = self.performance_loss_fn(performance_scores, performance_targets)
            
            # Combined loss
            loss = action_loss + performance_weight * performance_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_action_loss += action_loss.item()
            total_performance_loss += performance_loss.item()
            total_loss += loss.item()
        
        return {
            'action_loss': total_action_loss / num_batches,
            'performance_loss': total_performance_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def validate(self, dataloader: DataLoader, performance_weight: float = 0.1) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_action_loss = 0
        total_performance_loss = 0
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for telemetry, target_actions in dataloader:
                # Transpose to get transformer-expected dimensions
                # From [batch_size, seq_len, features] to [seq_len, batch_size, features]
                telemetry = telemetry.transpose(0, 1).to(self.device)
                target_actions = target_actions.transpose(0, 1).to(self.device)
                
                # Create decoder input
                decoder_input = torch.cat([
                    torch.zeros(1, target_actions.shape[1], target_actions.shape[2], device=self.device),
                    target_actions[:-1]
                ], dim=0)
                
                # Forward pass
                predicted_actions, performance_scores = self.model(telemetry, decoder_input)
                
                # Calculate losses
                action_loss = self.action_loss_fn(predicted_actions, target_actions)
                performance_targets = torch.zeros_like(performance_scores)
                performance_loss = self.performance_loss_fn(performance_scores, performance_targets)
                
                loss = action_loss + performance_weight * performance_loss
                
                total_action_loss += action_loss.item()
                total_performance_loss += performance_loss.item()
                total_loss += loss.item()
        
        return {
            'action_loss': total_action_loss / num_batches,
            'performance_loss': total_performance_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              patience: int = 10) -> Tuple[ExpertActionTransformer, Dict[str, Any]]:
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
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
                val_loss = val_metrics['total_loss']
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Store history
            epoch_history = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics if val_dataloader else None,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
        
        return self.model, {
            'training_completed': True,
            'best_val_loss': best_val_loss if val_dataloader else None,
            'training_history': self.training_history,
            'total_epochs': len(self.training_history)
        }
