"""
Service for training and using an LSTM Classifier to identify behavioral segments.
Refactored to support variable length segments and learn temporal relations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from .zarr_telemetry_store import get_shared_zarr_store
from app.models.segment_models import AnnotatedSegment, LABEL_MAPPING, PredictedSegment, SegmentFeatureCatalog

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bidirectional allows the model to see future context for each timestep
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        # Output dim * 2 because of bidirectionality
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        # Initialize hidden state with zeros if not provided
        if hidden is None:
            # num_layers * 2 for bidirectional
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        # out: (batch, seq_len, hidden_dim * 2)
        out = self.fc(out)
        return self.sigmoid(out), hidden

class StreamingSegmentDataset(IterableDataset):
    def __init__(self, store, cache_key, mlb, scaler, max_length, expected_features, mode='all', val_ratio=0.2, seed=42):
        self.store = store
        self.cache_key = cache_key
        self.mlb = mlb
        self.scaler = scaler
        self.max_length = max_length
        self.expected_features = expected_features
        self.mode = mode
        self.val_ratio = val_ratio
        self.seed = seed

    def __iter__(self):
        chunks = self.store.get_cached_data_chunks(self.cache_key)
        import random
        rng = random.Random(self.seed)

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

                if self.mode != 'all':
                    is_val = rng.random() < self.val_ratio
                    if self.mode == 'train' and is_val:
                        continue
                    if self.mode == 'val' and not is_val:
                        continue
                
                try:
                    ann = AnnotatedSegment.from_dict(d)
                except Exception:
                    continue

                if not ann.telemetry_data:
                    continue

                df = pd.DataFrame(ann.telemetry_data)
                
                # Fast path if columns match
                current_cols = df.columns.tolist()
                if current_cols != self.expected_features:
                     # Add missing
                     for f in self.expected_features:
                         if f not in df.columns:
                             df[f] = 0
                     # Drop extra and reorder
                     df = df[self.expected_features]
                
                # Ensure numeric
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                if df.empty:
                    continue
                
                seg_X = df.values
                
                # Map labels
                mapped_labels = [LABEL_MAPPING.get(l, str(l)) for l in ann.labels]
                
                # Create target
                label_vec = self.mlb.transform([mapped_labels])[0]
                seg_y = np.tile(label_vec, (len(seg_X), 1))
                
                # Scale
                scaled_X = self.scaler.transform(seg_X)
                
                # Pad
                pad_len = self.max_length - len(scaled_X)
                if pad_len > 0:
                    scaled_X = np.pad(scaled_X, ((0, pad_len), (0, 0)), 'constant')
                    seg_y = np.pad(seg_y, ((0, pad_len), (0, 0)), 'constant')
                elif pad_len < 0:
                    # Truncate
                    scaled_X = scaled_X[:self.max_length]
                    seg_y = seg_y[:self.max_length]
                
                yield torch.FloatTensor(scaled_X), torch.FloatTensor(seg_y)

class SegmentClassifierService:
    def __init__(self, models_directory: str = "models", max_length: int = 100):
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(exist_ok=True)
        self.model_path = self.models_directory / "segment_classifier.pth"
        self.mlb_path = self.models_directory / "segment_labels.joblib"
        self.scaler_path = self.models_directory / "segment_scaler.joblib"
        self.store = get_shared_zarr_store()
        self.model = None
        self.mlb = None 
        self.scaler = None
        
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

    async def fit_preprocessors(self, cache_key: str):
        """
        Scan data to fit preprocessors (Scaler, MLB) without loading everything.
        """
        print("Scanning data to fit preprocessors...")
        chunks = self.store.get_cached_data_chunks(cache_key)
        
        all_labels = set()
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
                mapped_labels = [LABEL_MAPPING.get(l, str(l)) for l in ann.labels]
                all_labels.update(mapped_labels)
                
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
        
        if max_seq_len > self.max_length:
            print(f"Updating max_length from {self.max_length} to {max_seq_len}")
            self.max_length = max_seq_len
            
        print("Preprocessor fitting complete.")

    async def train_model(self, epochs=10, batch_size=32, learning_rate=0.001, val_split=0.2):
        """Train the LSTM Classifier using streaming data with train/val split."""
        from app.config.pipeline_config import PipelineConfig
        cache_key = PipelineConfig().annotation_cache_key
        
        await self.fit_preprocessors(cache_key)
        
        train_dataset = StreamingSegmentDataset(
            self.store, 
            cache_key, 
            self.mlb, 
            self.scaler, 
            self.max_length,
            SegmentFeatureCatalog.get_all_available_features(),
            mode='train',
            val_ratio=val_split
        )

        val_dataset = StreamingSegmentDataset(
            self.store, 
            cache_key, 
            self.mlb, 
            self.scaler, 
            self.max_length,
            SegmentFeatureCatalog.get_all_available_features(),
            mode='val',
            val_ratio=val_split
        )
        
        # num_workers=0 to avoid multiprocessing issues with Zarr/Pickle
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        input_dim = self.scaler.mean_.shape[0]
        output_dim = len(self.mlb.classes_)
        hidden_dim = 64
        
        self.model = LSTMModel(input_dim, hidden_dim, output_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
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
                for val_X, val_y in val_loader:
                    val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                    outputs, _ = self.model(val_X)
                    loss = criterion(outputs, val_y)
                    val_loss += loss.item()
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Final Evaluation Report
        print("\nGenerating final evaluation report on validation set...")
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                outputs, _ = self.model(val_X)
                
                # Threshold probabilities
                preds = (outputs > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(val_y.cpu().numpy())
        
        if all_preds:
            # Concatenate and reshape to (total_samples * seq_len, num_classes)
            y_pred = np.concatenate(all_preds).reshape(-1, output_dim)
            y_true = np.concatenate(all_targets).reshape(-1, output_dim)
            
            # Generate report
            # Note: This evaluates every timestep, including padding.
            report = classification_report(
                y_true, 
                y_pred, 
                target_names=self.mlb.classes_, 
                zero_division=0
            )
            print("Validation Classification Report:")
            print(report)

        # Save model and artifacts
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.mlb, self.mlb_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save config
        config = {"max_length": self.max_length}
        with open(self.models_directory / "segment_config.json", "w") as f:
            json.dump(config, f)
            
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        if self.model_path.exists() and self.mlb_path.exists() and self.scaler_path.exists():
            self.mlb = joblib.load(self.mlb_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # Load config if exists
            config_path = self.models_directory / "segment_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.max_length = config.get("max_length", self.max_length)
            
            input_dim = self.scaler.mean_.shape[0]
            output_dim = len(self.mlb.classes_)
            hidden_dim = 64 
            
            self.model = LSTMModel(input_dim, hidden_dim, output_dim).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            return True
        return False

    def predict_segment(self, segment_df: pd.DataFrame) -> List[str]:
        """Predict labels for a single segment DataFrame."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or found.")

        numeric_df = segment_df.select_dtypes(include=['number'])
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
            # Take max probability over the segment or average?
            # Let's take the average probability across the segment
            # Only consider valid outputs (ignore padding)
            valid_outputs = outputs[0, :original_len, :]
            probs = valid_outputs.mean(dim=0).cpu().numpy()
            
        threshold = 0.5
        labels = []
        for i, p in enumerate(probs):
            if p > threshold:
                labels.append(self.mlb.classes_[i])
        
        return labels

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
        
        numeric_df = df
        if numeric_df.empty:
            return []

        # Scale
        X_scaled = self.scaler.transform(numeric_df.values)
        
        # Inference on full sequence to allow Bi-LSTM to see full context
        # Note: For extremely long sequences (>10k steps), we might need overlapping windows,
        # but for typical telemetry sessions, full sequence is better for context.
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            probs = outputs.squeeze(0).cpu().numpy()
            
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

    async def scan_session(self, dataframe: Optional[pd.DataFrame] = None, target_labels: Optional[List[str]] = None, **kwargs) -> None:
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
        from app.config.pipeline_config import PipelineConfig
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
