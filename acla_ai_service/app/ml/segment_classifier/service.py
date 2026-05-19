"""
Service for training and using an LSTM Classifier to identify behavioral segments.
Refactored to support variable length segments and learn temporal relations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterator
import asyncio
import json
import shutil
import random
import hashlib
import copy
from collections import defaultdict

from app.storage.zarr import get_shared_zarr_store
from app.domain.labels import LABEL_MAPPING
from app.domain.segment import AnnotatedSegment, PredictedSegment, SegmentFeatureCatalog

# Extracted in refactor/hexagonal-v4 — Page 5 of the architecture diagram.
# Model classes are pure (no I/O); dataset + derived-features helper own I/O.
# Re-imported here so SegmentClassifierService keeps the same internal API.
from app.ml.segment_classifier.model import CNN1DModel, FocalLoss
from app.storage.datasets.segment_dataset import (
    StreamingSegmentDataset,
    compute_derived_features,
)


class SegmentClassifierService:
    def __init__(self, models_directory: str = "models", max_length: int = 100):
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(exist_ok=True)
        self.model_path = self.models_directory / "segment_classifier.pth"
        self.mlb_path = self.models_directory / "segment_labels.joblib"
        self.scaler_path = self.models_directory / "segment_scaler.joblib"
        self.pos_weight_path = self.models_directory / "segment_pos_weight.pt"
        self.store = get_shared_zarr_store()
        self.model = None
        self.mlb = None 
        self.scaler = None
        self.pos_weight = None
        self.label_counts = {}
        
        # Device selection with explicit AMD/NVIDIA support check
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            try:
                device_name = torch.cuda.get_device_name(0)
                if hasattr(torch.version, 'hip') and torch.version.hip:
                    print(f"SegmentClassifierService: AMD GPU detected (ROCm): {device_name}")
                else:
                    print(f"SegmentClassifierService: NVIDIA GPU detected (CUDA): {device_name}")
            except Exception as e:
                print(f"SegmentClassifierService: GPU detected but failed to get name: {e}")
        else:
            self.device = torch.device("cpu")
            print("SegmentClassifierService: No GPU detected, using CPU.")
            try:
                print(f"Debug: torch.cuda.is_available()={torch.cuda.is_available()}")
                print(f"Debug: torch.version.cuda={torch.version.cuda}")
                print(f"Debug: torch.version.hip={getattr(torch.version, 'hip', 'None')}")
            except Exception as e:
                print(f"Debug: Error getting torch version info: {e}")

        self.max_length = max_length

    def _compute_segment_hash(self, segment_dict: Dict) -> str:
        """Compute deterministic hash for a segment based on its content."""
        # Create a stable string representation of key fields
        # Use session_id and timestamp if available, otherwise use telemetry data
        hash_data = ""
        if "session_id" in segment_dict:
            hash_data += str(segment_dict["session_id"])
        if "timestamp" in segment_dict:
            hash_data += str(segment_dict["timestamp"])
        if "start_index" in segment_dict:
            hash_data += str(segment_dict["start_index"])
        if "end_index" in segment_dict:
            hash_data += str(segment_dict["end_index"])
            
        # Fallback: use first few telemetry points
        if not hash_data and "telemetry_data" in segment_dict and segment_dict["telemetry_data"]:
            try:
                first_point = segment_dict["telemetry_data"][0]
                hash_data = json.dumps(first_point, sort_keys=True)
            except Exception:
                hash_data = str(segment_dict)
        
        if not hash_data:
            hash_data = json.dumps(segment_dict, sort_keys=True)
            
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    def _assign_split(self, segment_hash: str, val_split: float) -> str:
        """Deterministically assign segment to train or val based on hash."""
        # Use first 8 characters of hash to generate a number between 0 and 1
        hash_int = int(segment_hash[:8], 16)
        hash_normalized = hash_int / (16**8)
        
        return "val" if hash_normalized < val_split else "train"

    async def prepare_training_data(self, source_cache_key: str, train_cache_key: str, val_cache_key: str, val_split: float = 0.2, chunk_size: int = 100):
        """
        Splits data from source_cache_key into train and val keys with stratified, deterministic splitting.
        Uses two-pass approach:
        1. First pass: Collect label statistics per segment
        2. Second pass: Perform stratified split using deterministic hashing
        """
        print(f"Preparing training data: splitting {source_cache_key} into {train_cache_key} and {val_cache_key}")
        print("Using deterministic stratified splitting for consistent label distribution...")
        
        # Clear existing keys
        for key in [train_cache_key, val_cache_key]:
            group_path = self.store._group_path(key)
            if group_path.exists():
                shutil.rmtree(group_path)
        
        # PASS 1: Collect label statistics
        print("Pass 1: Collecting label statistics...")
        label_to_segments = defaultdict(list)  # Maps label -> list of (chunk_idx, item_idx, hash)
        chunk_index = []  # Store (chunk_data, chunk_idx)
        
        chunks = self.store.get_cached_data_chunks(source_cache_key)
        chunk_idx = 0
        
        for chunk in chunks:
            chunk_data = []
            if isinstance(chunk, list):
                chunk_data = chunk
            elif isinstance(chunk, dict) and "data" in chunk:
                chunk_data = chunk["data"]
            elif isinstance(chunk, dict) and "payload" in chunk:
                chunk_data = [chunk["payload"]]
            else:
                chunk_data = [chunk]
            
            valid_items = []
            for item_idx, d in enumerate(chunk_data):
                if not isinstance(d, dict):
                    continue
                
                # Validate
                try:
                    if "telemetry_data" not in d or not d["telemetry_data"]:
                        continue
                except Exception:
                    continue
                
                valid_items.append(d)
                
                # Compute hash and extract labels
                segment_hash = self._compute_segment_hash(d)
                
                # Extract labels (handle both 'labels' and mapped labels)
                labels = d.get("labels", [])
                if labels:
                    mapped_labels = [LABEL_MAPPING.get(str(l), str(l)) for l in labels]
                    
                    # Store segment for each label it has
                    # Use unique labels to avoid double counting
                    for lbl in set(mapped_labels):
                        label_to_segments[lbl].append((chunk_idx, len(valid_items) - 1, segment_hash))
            
            if valid_items:
                chunk_index.append((valid_items, chunk_idx))
                chunk_idx += 1
        
        print(f"Found {len(chunk_index)} chunks with {sum(len(items) for items, _ in chunk_index)} valid segments")
        print(f"Label distribution: {[(label, len(segments)) for label, segments in sorted(label_to_segments.items())]}")
        
        # PASS 2: Stratified split using deterministic hashing
        print("Pass 2: Performing stratified split...")
        
        # For each label, split segments deterministically
        train_segments_set = set()  # Set of (chunk_idx, item_idx)
        val_segments_set = set()
        
        train_label_counts = defaultdict(int)
        val_label_counts = defaultdict(int)
        
        for label, segments in label_to_segments.items():
            for chunk_idx, item_idx, segment_hash in segments:
                split = self._assign_split(segment_hash, val_split)
                
                if split == "val":
                    val_segments_set.add((chunk_idx, item_idx))
                    val_label_counts[label] += 1
                else:
                    train_segments_set.add((chunk_idx, item_idx))
                    train_label_counts[label] += 1
        
        print(f"Train segments: {len(train_segments_set)}, Val segments: {len(val_segments_set)}")
        print(f"Train label distribution: {dict(train_label_counts)}")
        print(f"Val label distribution: {dict(val_label_counts)}")
        
        # PASS 3: Write splits to storage
        print("Pass 3: Writing splits to storage...")
        train_buffer = []
        val_buffer = []
        train_idx = 1
        val_idx = 1
        
        for chunk_data, chunk_idx in chunk_index:
            for item_idx, item in enumerate(chunk_data):
                if (chunk_idx, item_idx) in train_segments_set:
                    train_buffer.append(item)
                    if len(train_buffer) >= chunk_size:
                        self.store.save_chunk(train_cache_key, train_idx, train_buffer)
                        train_buffer = []
                        train_idx += 1
                        
                elif (chunk_idx, item_idx) in val_segments_set:
                    val_buffer.append(item)
                    if len(val_buffer) >= chunk_size:
                        self.store.save_chunk(val_cache_key, val_idx, val_buffer)
                        val_buffer = []
                        val_idx += 1
        
        # Flush remainders
        if train_buffer:
            self.store.save_chunk(train_cache_key, train_idx, train_buffer)
        if val_buffer:
            self.store.save_chunk(val_cache_key, val_idx, val_buffer)
            
        print(f"Data preparation complete. Train: {len(train_segments_set)} segments, Val: {len(val_segments_set)} segments")

    async def fit_preprocessors(self, cache_key: str):
        """
        Scan data to fit preprocessors (Scaler, MLB) without loading everything.
        """
        print("Scanning data to fit preprocessors...")
        chunks = self.store.get_cached_data_chunks(cache_key)
        
        all_labels = set()
        self.label_counts = {}
        total_segments = 0
        self.scaler = StandardScaler()
        max_seq_len = 0
        
        expected_features = SegmentFeatureCatalog.get_all_available_features()
        
        has_data = False
        
        for chunk in chunks:
            chunk_data = []
            if isinstance(chunk, list):
                chunk_data = chunk
            elif isinstance(chunk, dict) and "data" in chunk:
                 chunk_data = chunk["data"]
            elif isinstance(chunk, dict) and "payload" in chunk:
                 chunk_data = [chunk["payload"]]
            else:
                 chunk_data = [chunk]
            
            for d in chunk_data:
                if not isinstance(d, dict):
                    continue
                
                try:
                    ann = AnnotatedSegment.from_dict(d)
                except Exception:
                    continue
                
                # Collect labels
                mapped_labels = ann.labels
                all_labels.update(mapped_labels)
                for l in mapped_labels:
                    self.label_counts[l] = self.label_counts.get(l, 0) + 1
                total_segments += 1
                
                if not ann.telemetry_data:
                    continue
                
                df = pd.DataFrame(ann.telemetry_data)
                
                # Align features for scaler
                current_cols = df.columns.tolist()
                if current_cols != expected_features:
                     for f in expected_features:
                         if f not in df.columns:
                             df[f] = 0
                     df = df[expected_features]
                
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Compute derived features (deltas) to match dataset structure
                df = compute_derived_features(df)

                if df.empty:
                    continue
                
                vals = df.values
                self.scaler.partial_fit(vals)
                max_seq_len = max(max_seq_len, len(vals))
                has_data = True

        if not has_data:
            raise ValueError("No valid training data found in cache.")
            
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(all_labels)])
        
        # Calculate pos_weight
        pos_weights = []
        for label in self.mlb.classes_:
            pos = self.label_counts.get(label, 0)
            neg = total_segments - pos
            if pos > 0:
                # Use sqrt dampening to prevent precision collapse on rare classes
                # Original linear weighting (neg/pos) can result in weights > 100, causing massive false positives
                weight = (neg / pos) ** 0.5
            else:
                weight = 1.0
            pos_weights.append(weight)
        
        self.pos_weight = torch.FloatTensor(pos_weights).to(self.device)
        print(f"Calculated pos_weights (dampened): {self.pos_weight}")
        
        if max_seq_len > self.max_length:
            print(f"Updating max_length from {self.max_length} to {max_seq_len}")
            self.max_length = max_seq_len
            
        print("Preprocessor fitting complete.")

    async def train_model(self, epochs=10, batch_size=32, learning_rate=0.001, val_split=0.1):
        """Train the LSTM Classifier using streaming data with train/val split."""
        from app.infra.config.pipeline import PipelineConfig
        cache_key = PipelineConfig().annotation_cache_key
        
        train_key = f"{cache_key}_train"
        val_key = f"{cache_key}_val"
        
        await self.prepare_training_data(cache_key, train_key, val_key, val_split)
        
        await self.fit_preprocessors(train_key)
        
        train_dataset = StreamingSegmentDataset(
            self.store, 
            train_key, 
            self.mlb, 
            self.scaler, 
            self.max_length,
            SegmentFeatureCatalog.get_all_available_features()
        )

        val_dataset = StreamingSegmentDataset(
            self.store, 
            val_key, 
            self.mlb, 
            self.scaler, 
            self.max_length,
            SegmentFeatureCatalog.get_all_available_features()
        )
        
        # num_workers=0 to avoid multiprocessing issues with Zarr/Pickle
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        input_dim = self.scaler.mean_.shape[0]
        output_dim = len(self.mlb.classes_)
        # Increased network size to handle larger label set (~50+ labels)
        hidden_dim = 256
        num_layers = 3
        
        self.model = CNN1DModel(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(self.device)
        criterion = FocalLoss(reduction='none', pos_weight=self.pos_weight)
        # Added weight_decay for regularization
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # Scheduler to reduce LR when validation metric plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
                
        best_val_loss = float('inf')
        best_model_state = None
        patience_limit = 3  # Early stopping patience
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            for batch_X, batch_y, batch_mask in train_loader:
                batch_X, batch_y, batch_mask = batch_X.to(self.device), batch_y.to(self.device), batch_mask.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Apply mask
                masked_loss = loss * batch_mask
                loss = masked_loss.sum() / (batch_mask.sum() * output_dim + 1e-8)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_count = 0
            with torch.no_grad():
                for val_X, val_y, val_mask in val_loader:
                    val_X, val_y, val_mask = val_X.to(self.device), val_y.to(self.device), val_mask.to(self.device)
                    outputs, _ = self.model(val_X)
                    loss = criterion(outputs, val_y)
                    
                    masked_loss = loss * val_mask
                    loss = masked_loss.sum() / (val_mask.sum() * output_dim + 1e-8)
                    
                    val_loss += loss.item()
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Scheduler Step
            scheduler.step(avg_val_loss)

            # Checkpointing and Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f"  New best model found! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"  Early stopping triggered after {patience_limit} epochs without improvement.")
                    break
        
        # Restore best model
        if best_model_state is not None:
             print(f"Restoring best model state (Val Loss: {best_val_loss:.4f})...")
             self.model.load_state_dict(best_model_state)

        # Final Evaluation Report
        print("\nGenerating final evaluation report on validation set...")
        self.model.eval()
        all_preds = []
        all_targets = []
        
        # Segment-level accumulation
        all_segment_preds = []
        all_segment_targets = []
        
        with torch.no_grad():
            for val_X, val_y, val_mask in val_loader:
                val_X, val_y, val_mask = val_X.to(self.device), val_y.to(self.device), val_mask.to(self.device)
                outputs, _ = self.model(val_X)
                
                # --- Per-Timestep Evaluation ---
                # Threshold logits (sigmoid(0) = 0.5)
                preds = (outputs > 0).float()
                
                # Filter by mask
                mask_flat = val_mask.cpu().bool().numpy().flatten()
                preds_flat = preds.cpu().numpy().reshape(-1, output_dim)
                targets_flat = val_y.cpu().numpy().reshape(-1, output_dim)
                
                if len(mask_flat) > 0:
                    all_preds.append(preds_flat[mask_flat])
                    all_targets.append(targets_flat[mask_flat])

                # --- Per-Segment Evaluation ---
                probs = torch.sigmoid(outputs)
                batch_size_curr = val_X.size(0)
                
                for i in range(batch_size_curr):
                    # Get actual length from mask
                    length = int(val_mask[i].sum().item())
                    if length == 0:
                        continue
                        
                    # Target is the same for the whole segment, just take the first valid one
                    # val_y shape: (batch, seq_len, num_classes)
                    seg_target = val_y[i, 0].cpu().numpy()
                    
                    # Predictions: Average probability over valid timesteps
                    seg_probs = probs[i, :length].mean(dim=0)
                    seg_pred = (seg_probs > 0.5).float().cpu().numpy()
                    
                    all_segment_preds.append(seg_pred)
                    all_segment_targets.append(seg_target)
        
        if all_segment_preds:
            y_seg_pred = np.array(all_segment_preds)
            y_seg_true = np.array(all_segment_targets)
            
            print("\n=== Segment-Level Classification Report (Aggregated) ===")
            # Use mappings for display
            target_names = [LABEL_MAPPING.get(str(l), str(l)) for l in self.mlb.classes_]
            
            seg_report = classification_report(
                y_seg_true,
                y_seg_pred,
                target_names=target_names,
                zero_division=0
            )
            print(seg_report)
            print("========================================================\n")

        if all_preds:
            # Concatenate
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            
            # Generate report
            report_dict = classification_report(
                y_true, 
                y_pred, 
                zero_division=0,
                output_dict=True
            )
            
            print("Validation Classification Report (Per-Timestep):")
            # Print textual report for logs
            print(classification_report(
                y_true, 
                y_pred, 
                target_names=target_names, 
                zero_division=0
            ))
            
            print("\nPer-class Metrics (Validation Set - Per Timestep):")
            for i, label in enumerate(self.mlb.classes_):
                label_key = str(i)

                if label_key in report_dict:
                    metrics = report_dict[label_key]
                    score = metrics['precision']
                    support = metrics['support']  # Number of timesteps, not segments
                    
                    label_name = LABEL_MAPPING.get(label, str(label))
                    print(f"{label_name}: Precision={score:.4f}, Support={support} timesteps")

        # Save model and artifacts
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.mlb, self.mlb_path)
        joblib.dump(self.scaler, self.scaler_path)
        if self.pos_weight is not None:
            torch.save(self.pos_weight, self.pos_weight_path)
        
        # Save config with model architecture
        config = {
            "max_length": self.max_length,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers
        }
        with open(self.models_directory / "segment_config.json", "w") as f:
            json.dump(config, f)
            
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        if self.model_path.exists() and self.mlb_path.exists() and self.scaler_path.exists():
            self.mlb = joblib.load(self.mlb_path)
            self.scaler = joblib.load(self.scaler_path)
            if self.pos_weight_path.exists():
                self.pos_weight = torch.load(self.pos_weight_path, map_location=self.device, weights_only=True)

            # Load config if exists
            hidden_dim = 256
            num_layers = 3
            config_path = self.models_directory / "segment_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.max_length = config.get("max_length", self.max_length)
                    hidden_dim = config.get("hidden_dim", hidden_dim)
                    num_layers = config.get("num_layers", num_layers)
            
            input_dim = self.scaler.mean_.shape[0]
            output_dim = len(self.mlb.classes_)
            
            self.model = CNN1DModel(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            return True
        return False

    def predict_segment(self, segment_df: pd.DataFrame) -> List[str]:
        """Predict labels for a single segment DataFrame."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or found.")

        df = segment_df.copy()
        
        # Align features with training data
        expected_features = SegmentFeatureCatalog.get_all_available_features()
        
        # 1. Ensure all expected features exist
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
                
        # 2. Select only expected features in correct order
        df = df[expected_features]
        
        # 3. Ensure numeric
        numeric_df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Feature engineering
        numeric_df = compute_derived_features(numeric_df)

        if numeric_df.empty:
            return []

        X_scaled = self.scaler.transform(numeric_df.values)
        
        # Handle max_length and padding
        original_len = len(X_scaled)
        if original_len > self.max_length:
             X_scaled = X_scaled[:self.max_length]
             original_len = self.max_length
        elif original_len < self.max_length:
             pad_len = self.max_length - original_len
             X_scaled = np.pad(X_scaled, ((0, pad_len), (0, 0)), 'constant')

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            # Apply sigmoid to get probabilities from logits
            probs_tensor = torch.sigmoid(outputs)
            
            valid_probs = probs_tensor[0, :original_len, :]
            probs = valid_probs.mean(dim=0).cpu().numpy()
            
        threshold = 0.5
        labels = []
        for i, p in enumerate(probs):
            label = self.mlb.classes_[i]
                
            if p > threshold:
                labels.append(label)
        
        return labels

    def predict_segment_probabilities(self, segment_df: pd.DataFrame) -> Dict[str, float]:
        """Predict probabilities for all labels for a single segment DataFrame."""
        if self.model is None:
            if not self.load_model():
                return {}

        df = segment_df.copy()
        
        # Align features with training data
        expected_features = SegmentFeatureCatalog.get_all_available_features()
        
        # 1. Ensure all expected features exist
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
                
        # 2. Select only expected features in correct order
        df = df[expected_features]
        
        # 3. Ensure numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 4. Feature engineering
        df = compute_derived_features(df)

        if df.empty:
            return {}

        X_scaled = self.scaler.transform(df.values)
        
        # Handle max_length and padding
        original_len = len(X_scaled)
        if original_len > self.max_length:
             X_scaled = X_scaled[:self.max_length]
             original_len = self.max_length
        elif original_len < self.max_length:
             pad_len = self.max_length - original_len
             X_scaled = np.pad(X_scaled, ((0, pad_len), (0, 0)), 'constant')

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            probs_tensor = torch.sigmoid(outputs)
            valid_probs = probs_tensor[0, :original_len, :]
            probs = valid_probs.mean(dim=0).cpu().numpy()
            
        result = {}
        for i, p in enumerate(probs):
            label = self.mlb.classes_[i]
            result[label] = float(p)
            
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    def scan_telemetry_data(self, dataframe: pd.DataFrame) -> List[PredictedSegment]:
        """
        Scan a dataframe and return found segments with labels.
        Uses full-sequence inference with Bi-LSTM and smoothing.
        """
        if self.model is None:
            if not self.load_model():
                return []
        
        df = dataframe
        
        # Align features with training data
        expected_features = SegmentFeatureCatalog.get_all_available_features()
        
        # 1. Ensure all expected features exist
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
                
        # 2. Select only expected features in correct order
        df = df[expected_features]
        
        # 3. Ensure numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if df.empty:
            return []
            
        # Feature engineering
        numeric_df = compute_derived_features(df)

        # Scale
        X_scaled = self.scaler.transform(numeric_df.values)
        
        # Inference on full sequence to allow Bi-LSTM to see full context
        # Note: For extremely long sequences (>10k steps), we might need overlapping windows,
        # but for typical telemetry sessions, full sequence is better for context.
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
            
        # Apply smoothing to probabilities to reduce jitter and enforce segment continuity
        # Rolling mean with a window of 5 steps
        probs_df = pd.DataFrame(probs, columns=self.mlb.classes_)

        probs_smoothed = probs_df.rolling(window=5, center=True, min_periods=1).mean().values
            
        # Threshold and Segment
        threshold = 0.5
        active_mask = probs_smoothed > threshold
        
        found_segments = []
        current_labels = []
        current_start = 0
        
        # Iterate through to find contiguous segments with the same label set
        for i in range(len(df)):
            # Get labels that are True for this index
            row_mask = active_mask[i]
            labels_indices = np.where(row_mask)[0]
            labels_at_i = [self.mlb.classes_[idx] for idx in labels_indices]
            labels_at_i.sort()
            
            if i == 0:
                current_labels = labels_at_i
                current_start = 0
            else:
                if labels_at_i != current_labels:
                    # Close previous segment if it had labels
                    if current_labels:
                        found_segments.append({
                            "start_index": current_start,
                            "end_index": i,
                            "labels": current_labels
                        })
                    current_labels = labels_at_i
                    current_start = i
        
        # Close final segment
        if current_labels:
            found_segments.append({
                "start_index": current_start,
                "end_index": len(df),
                "labels": current_labels
            })
        
        results = []
        for meta in found_segments:
            start = meta['start_index']
            end = meta['end_index']
            
            # Filter out very short segments (e.g. < 3 steps) as noise
            if end - start < 3:
                continue

            segment_df = df.iloc[start:end]
            
            # Extract actual data and wrap with metadata
            segment_data = segment_df.to_dict('records')
            
            predicted_segment = PredictedSegment(
                labels=meta["labels"],
                telemetry_data=segment_data,
                start_index=start,
                end_index=end
            )
            results.append(predicted_segment)
            
        return results

    async def scan_session(self, dataframe: Optional[pd.DataFrame] = None, target_labels: Optional[List[int]] = None, **kwargs) -> None:
        """
        Scan a session and find segments matching labels using LSTM.
        Identifies intervals, extracts actual segments, and saves to cache.
        """
        # Reuse the logic from scan_telemetry_data to ensure consistency
        found_segments = self.scan_telemetry_data(dataframe)
        
        # Extract and cache segments
        chunk_segments = []
        for segment in found_segments:
            if target_labels:
                if not any(label in segment.labels for label in target_labels):
                    continue

            chunk_segments.append(segment.to_dict())
            
        # Cache segments
        from app.infra.config.pipeline import PipelineConfig
        cache_key = PipelineConfig().segments_cache_key
        
        if chunk_segments:
             async def segments_generator():
                yield chunk_segments
             
             await self.store.cache_chunks_streaming(
                cache_key=cache_key,
                chunks_iterator=segments_generator()
             )

# Singleton instance
segment_classifier = SegmentClassifierService()
