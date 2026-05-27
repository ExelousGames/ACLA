"""Transformer training orchestrator.

Owns the epoch loop, mixed-precision setup, validation, checkpoint
management, and the public async entry function used by callers
(``prepare_and_train_coach_transformer_model``).

Pure orchestration: imports the model architecture from
``app.ml.transformer.model`` and the dataset from
``app.storage.datasets.telemetry_dataset``. Nothing else in the agent
or pipelines bands should reach in here — call the entry function
instead.

Extracted from app/ml/transformer/model.py in refactor/hexagonal-v5
(completes the Page-5 split started in PR #4).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.domain.expert_features import ExpertFeatureCatalog
from app.domain.telemetry import TelemetryFeatures, _safe_float
from app.domain.tire_grip_features import TireGripFeatureCatalog
from app.ml.transformer.model import (
    ExpertActionTransformer,
    _safe_number,
    make_json_safe,
)
from app.storage.datasets.telemetry_dataset import TelemetryActionDataset
from app.storage.datasets.transformer_scaler import (
    PerFeatureScaler,
    _RunningFeatureStats,
)


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

        # Detect specific GPU architecture
        self._is_amd = False
        if self._cuda:
            try:
                # Check for ROCm/HIP (AMD)
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    self._is_amd = True
                    print(f"[INFO] AMD GPU detected (ROCm {torch.version.hip})")
                else:
                    print(f"[INFO] NVIDIA GPU detected (CUDA {torch.version.cuda})")
            except Exception:
                pass

        # Enable cuDNN benchmark for faster kernels on NVIDIA GPU
        # On AMD, MIOpen handles this, but setting cudnn.benchmark doesn't hurt usually.
        # However, to be safe and explicit, we only enable it for NVIDIA.
        try:
            if self._cuda and not self._is_amd:
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
            if not self._is_amd:
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
            self.optimizer, mode='min', factor=0.5, patience=10
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
        try:
            self.model.configure_loss_weights()
            print("[INFO] Applied default loss weighting (brake/steer emphasis)")
        except RuntimeError as weight_error:
            print(f"[WARNING] Unable to configure loss weights: {weight_error}")
        
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
                for batch_inputs, batch_targets, batch_masks, batch_padding_mask in dataset.get_chunk_batches(chunk_idx):
                    # Move to device
                    batch_inputs = batch_inputs.to(self.device, non_blocking=self._cuda)
                    batch_targets = batch_targets.to(self.device, non_blocking=self._cuda)
                    batch_masks = batch_masks.to(self.device, non_blocking=self._cuda)
                    batch_padding_mask = batch_padding_mask.to(self.device, non_blocking=self._cuda)
                    
                    # Forward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._cuda and self.amp_dtype is not None):
                        target_seq_len = batch_targets.shape[1]
                        predictions = self.model(
                            unified_input=batch_inputs,
                            prediction_steps=target_seq_len,
                            use_teacher_forcing=True,
                            padding_mask=batch_padding_mask,
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
                    
                    loss = self.model.unified_loss(
                        predictions=predictions,
                        targets=batch_targets,
                        timestep_weights=batch_masks,
                    )
                    
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
                    del batch_inputs, batch_targets, batch_masks, batch_padding_mask, predictions, loss
                    
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
                    for batch_inputs, batch_targets, batch_masks, batch_padding_mask in dataset.get_chunk_batches(chunk_idx):
                        batch_inputs = batch_inputs.to(self.device, non_blocking=self._cuda)
                        batch_targets = batch_targets.to(self.device, non_blocking=self._cuda)
                        batch_masks = batch_masks.to(self.device, non_blocking=self._cuda)
                        batch_padding_mask = batch_padding_mask.to(self.device, non_blocking=self._cuda)
                        
                        target_seq_len = batch_targets.shape[1]
                        predictions = self.model(
                            unified_input=batch_inputs,
                            prediction_steps=target_seq_len,
                            use_teacher_forcing=True,
                            padding_mask=batch_padding_mask,
                        )
                        
                        loss = self.model.unified_loss(
                            predictions=predictions,
                            targets=batch_targets,
                            timestep_weights=batch_masks,
                        )
                        total_loss += loss.item()
                        num_batches += 1
                        
                        del batch_inputs, batch_targets, batch_masks, batch_padding_mask, predictions, loss
                        
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
        length_stats = segment_info.get('length_statistics') or {}
        if length_stats:
            length_summary = (
                f"min={length_stats.get('min')} | "
                f"median={length_stats.get('median', 0):.1f} | "
                f"max={length_stats.get('max')}"
            )
        else:
            length_summary = "unknown"

        print(f"[INFO] Training dataset: {segment_info['num_chunks']} chunks with variable segment lengths ({length_summary} timesteps)")
        
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

        evaluation_segments: List[Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, int]]] = []
        max_segments_to_collect = 50

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

        primary_segment_raw, processed_data, segment_metadata = evaluation_segments[0]
        selected_input, selected_target, selected_weights = processed_data

        print(
            f"[INFO] Using chunk {segment_metadata['chunk_index']} "
            f"segment {segment_metadata['segment_index']} for evaluation"
        )

        input_sequence_serialized = selected_input.tolist()

        # Prepare tensors
        input_tensor = torch.from_numpy(np.expand_dims(selected_input, axis=0)).to(self.device)
        target_tensor = torch.from_numpy(np.expand_dims(selected_target, axis=0)).to(self.device)
        weight_tensor = torch.from_numpy(np.expand_dims(selected_weights, axis=0)).to(self.device)

        target_seq_len = target_tensor.shape[1]
        feature_count = target_tensor.shape[-1]

        if weight_tensor.shape[1] != target_seq_len:
            raise ValueError(
                f"Timestep weight length {weight_tensor.shape[1]} does not match target length {target_seq_len}"
            )
        weight_tensor = weight_tensor.to(dtype=torch.float32)

        print(f"[INFO] Sequence length: {target_seq_len} | Features: {feature_count}")

        def _compute_metrics(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            weights: Optional[torch.Tensor]
        ) -> Dict[str, float]:
            diff = predictions - targets
            squared_error = diff ** 2
            abs_error = torch.abs(diff)

            if weights is not None:
                weight_broadcast = weights.to(predictions.device)
                if weight_broadcast.dim() == 2:
                    weight_broadcast = weight_broadcast.unsqueeze(-1)
                squared_error = squared_error * weight_broadcast
                abs_error = abs_error * weight_broadcast

            mse = torch.mean(squared_error).item()
            mae = torch.mean(abs_error).item()
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

        teacher_metrics = _compute_metrics(teacher_tensor, target_eval_tensor, weight_tensor)
        autoreg_metrics = _compute_metrics(autoreg_tensor, target_eval_tensor, weight_tensor)
        teacher_metrics['unified_loss'] = float(
            self.model.unified_loss(
                predictions=teacher_tensor,
                targets=target_eval_tensor,
                timestep_weights=weight_tensor
            ).item()
        )
        autoreg_metrics['unified_loss'] = float(
            self.model.unified_loss(
                predictions=autoreg_tensor,
                targets=target_eval_tensor,
                timestep_weights=weight_tensor
            ).item()
        )

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

        # Compute per-feature error summaries (weighted by timestep mask)
        def _per_feature_metrics(
            prediction: np.ndarray,
            reference: np.ndarray,
            weights: np.ndarray
        ) -> Dict[str, Dict[str, float]]:
            diff = prediction - reference
            weight_column = weights.reshape(-1, 1)
            mse = np.mean(np.square(diff) * weight_column, axis=0)
            mae = np.mean(np.abs(diff) * weight_column, axis=0)
            return {
                feature_names[idx]: {
                    'mse': float(mse[idx]),
                    'mae': float(mae[idx]),
                }
                for idx in range(len(feature_names))
            }

        teacher_metrics['per_feature'] = _per_feature_metrics(
            teacher_seq_unscaled,
            target_seq_unscaled,
            selected_weights
        )
        autoreg_metrics['per_feature'] = _per_feature_metrics(
            autoreg_seq_unscaled,
            target_seq_unscaled,
            selected_weights
        )

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
                'timestep_weight': float(selected_weights[step_idx]),
                'input_state': input_named_step,
                'input_state_unscaled': input_unscaled_named_step,
                'prediction': teacher_named,
                'prediction_unscaled': teacher_unscaled_named,
                'target_state': target_named_step,
                'target_state_unscaled': target_unscaled_named_step
            })

            autoreg_per_step.append({
                'step': step_idx,
                'timestep_weight': float(selected_weights[step_idx]),
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
            'timestep_weights': selected_weights.tolist(),
            'target_sequence': target_np.tolist(),
            'target_sequence_unscaled': target_np_unscaled.tolist(),
            'target_sequence_named': target_named,
            'target_sequence_unscaled_named': target_unscaled_named,
            'segments_sampled': [meta for _, _, meta in evaluation_segments]
        }

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


async def prepare_and_train_coach_transformer_model(
    data_cache: Any,
    segments_cache_key: str,
    *,
    use_lance_dataloader: bool = False,
) -> Dict[str, Any]:
    """Train the ExpertActionTransformer using cached telemetry segments.

    Args:
        data_cache: Cache service that provides telemetry segments via ``get_cached_data_chunks``.
        segments_cache_key: Cache key that identifies the prepared telemetry segments.
        use_lance_dataloader: When True, drive training from the Lance-native dataset
            (:class:`LanceTelemetryActionDataset`) which reads columnar telemetry
            straight from the Phase-2 typed Lance store. Default ``False`` keeps the
            legacy dict-list path; flip after running the parity script in
            ``scripts/parity_test_transformer_dataloader.py``.

    Returns:
        Dictionary containing training history, evaluation metrics, and serialized model payload.
    """

    try:
        print(f"[INFO] Creating streaming dataset from segments cache key: {segments_cache_key}")

        if use_lance_dataloader:
            # Lance-native path: requires data_cache to be a LanceTelemetryStore
            # whose cache_key is registered with SegmentsStrategy (i.e. the
            # Phase-2 migration has been run for this key).
            from app.storage.datasets.lance_telemetry_dataset import (
                LanceTelemetryActionDataset,
            )
            from app.storage.lance import LanceTelemetryStore

            if not isinstance(data_cache, LanceTelemetryStore):
                raise TypeError(
                    "use_lance_dataloader=True requires data_cache to be a "
                    f"LanceTelemetryStore; got {type(data_cache).__name__}."
                )
            dataset = LanceTelemetryActionDataset(
                store=data_cache,
                segments_cache_key=segments_cache_key,
                batch_size=32,
                min_sequence_length=3,
            )
        else:
            dataset = TelemetryActionDataset(
                data_cache=data_cache,
                segments_cache_key=segments_cache_key,
                batch_size=32,
                min_sequence_length=3,
            )

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            # Enable TF32 on Ampere+ GPUs for faster matmuls without harming stability.
            try:
                # Check for AMD GPU (ROCm) to avoid setting NVIDIA-specific flags
                is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
                if not is_amd:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        print(f"[INFO] Device: {'CUDA' if use_cuda else 'CPU'}")

        input_feature_names, action_feature_names = dataset.get_feature_names()
        input_features_count = len(input_feature_names)

        segment_info = dataset.get_segment_info()
        length_stats = segment_info.get('length_statistics') or {}
        print(
            f"[INFO] Dataset info: {input_features_count} combined input features, {len(action_feature_names)} action features"
        )
        if length_stats:
            median_length = length_stats.get('median')
            median_display = f"{median_length:.1f}" if median_length is not None else "n/a"
            print(
                f"[INFO] Segment length distribution (timesteps): min={length_stats.get('min')} | "
                f"median={median_display} | max={length_stats.get('max')}"
            )

        observed_max = length_stats.get('max')
        if observed_max is None:
            raise RuntimeError(
                "Dataset has no segments with valid length statistics; "
                "cannot derive sequence_length for the transformer."
            )
        max_sequence_length = int(observed_max)
        print(f"[INFO] Sequence length (from observed dataset max): {max_sequence_length}")

        model = ExpertActionTransformer(
            total_features_count=input_features_count,
            d_model=256,
            nhead=16,
            num_layers=20,
            sequence_length=max_sequence_length,
        )

        device = 'cuda' if use_cuda else 'cpu'
        print(f"[DEBUG] Creating trainer on device: {device}")
        trainer = ExpertActionTrainer(model, device=device, learning_rate=1e-4)

        print("[DEBUG] Starting training loop...")
        training_history = trainer.train(
            train_dataset=dataset,
            val_dataset=None,
            epochs=30,
            patience=10,
        )
        print("[DEBUG] Training loop completed successfully")

        test_metrics = trainer.evaluate(dataset)
        serialized_model = model.serialize_model()

        return {
            "success": True,
            "training_history": training_history,
            "test_metrics": test_metrics,
            "model_info": trainer.get_model_info(),
            "serialized_model": serialized_model,
        }

    except Exception as error:
        print(f"[ERROR] Transformer training failed: {error}")
        return {
            "success": False,
            "error": str(error),
        }


__all__ = [
    "ExpertActionTrainer",
    "prepare_and_train_coach_transformer_model",
]
