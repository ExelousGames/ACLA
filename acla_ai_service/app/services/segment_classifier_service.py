"""
Service for training and using an LSTM Classifier to identify behavioral segments.
Refactored to support variable length segments and learn temporal relations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import json

from .zarr_telemetry_store import get_shared_zarr_store
from app.models.segment_models import AnnotatedSegment, LABEL_MAPPING, PredictedSegment, SegmentFeatureCatalog

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        # Initialize hidden state with zeros if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        # out: (batch, seq_len, hidden_dim)
        out = self.fc(out)
        return self.sigmoid(out), hidden

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

    async def load_annotations(self) -> List[AnnotatedSegment]:
        """Load all annotations from Zarr."""
        # Import locally to avoid circular dependency
        from .full_dataset_ml_service import PipelineConfig
        cache_key = PipelineConfig().annotation_cache_key
        
        if not self.store.has_cached_data(cache_key):
            return []
        
        chunks = self.store.get_cached_data_chunks(cache_key)
        annotations = []
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
                if isinstance(d, dict):
                    annotations.append(AnnotatedSegment.from_dict(d))
                    
        return annotations

    async def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sessions and create sequences for training.
        handle variable length segments from annotations.
        """
        annotations = await self.load_annotations()
        if not annotations:
            raise ValueError("No annotations found.")

        # 1. Collect all unique labels (mapped to strings)
        all_labels = set()
        for ann in annotations:
            # Map integer labels to string labels
            mapped_labels = [LABEL_MAPPING.get(l, str(l)) for l in ann.labels]
            all_labels.update(mapped_labels)
        
        if not all_labels:
            raise ValueError("No labels found in annotations.")
            
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(all_labels)]) 
        
        # Get expected features for integrity check
        expected_features = SegmentFeatureCatalog.get_all_available_features()

        raw_segments = [] # List of (X_data, y_data)
        
        for ann in annotations:
            if not ann.telemetry_data:
                continue
                
            df = pd.DataFrame(ann.telemetry_data)
            
            # Feature Integrity Check
            current_features = df.columns.tolist()
            missing_features = [f for f in expected_features if f not in current_features]
            extra_features = [f for f in current_features if f not in expected_features]

            if missing_features:
                print(f"Warning: Segment missing features (filling with 0): {missing_features}")
                for f in missing_features:
                    df[f] = 0
            
            if extra_features:
                print(f"Warning: Segment has extra features (removing): {extra_features}")
                df = df.drop(columns=extra_features)
            
            # Check order
            current_cols = df.columns.tolist()
            if current_cols != expected_features:
                print(f"Warning: Feature order mismatch.")
                for i, (exp, act) in enumerate(zip(expected_features, current_cols)):
                    if exp != act:
                        print(f"  First mismatch at index {i}: Expected '{exp}', Got '{act}'")
                        start_ctx = max(0, i - 2)
                        end_ctx = min(len(expected_features), i + 3)
                        print(f"  Expected context: {expected_features[start_ctx:end_ctx]}")
                        print(f"  Actual context:   {current_cols[start_ctx:end_ctx]}")
                        break
                if len(expected_features) != len(current_cols):
                    print(f"  Length mismatch: Expected {len(expected_features)}, Got {len(current_cols)}")
                
                # Reorder if necessary (optional fix, but good for safety)
                df = df[expected_features]
            
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
            
            raw_segments.append((seg_X, seg_y))

        if not raw_segments:
             raise ValueError("No valid training data found.")

        # Fit Scaler on all data
        self.scaler = StandardScaler()
        all_X = np.vstack([item[0] for item in raw_segments])
        self.scaler.fit(all_X)
        
        X_sequences = []
        y_sequences = []
        
        for seg_X, seg_y in raw_segments:
            scaled_X = self.scaler.transform(seg_X)
            
            # Chunk into max_length
            length = len(scaled_X)
            for i in range(0, length, self.max_length):
                chunk_X = scaled_X[i : i + self.max_length]
                chunk_y = seg_y[i : i + self.max_length]
                
                # Pad if necessary
                if len(chunk_X) < self.max_length:
                    pad_len = self.max_length - len(chunk_X)
                    chunk_X = np.pad(chunk_X, ((0, pad_len), (0, 0)), 'constant')
                    chunk_y = np.pad(chunk_y, ((0, pad_len), (0, 0)), 'constant')
                
                X_sequences.append(chunk_X)
                y_sequences.append(chunk_y)

        return np.array(X_sequences), np.array(y_sequences)

    async def train_model(self, epochs=10, batch_size=32, learning_rate=0.001):
        """Train the LSTM Classifier."""
        print("Preparing training data...")
        X, y = await self.prepare_training_data()
        
        print(f"Training on {len(X)} sequences of length {X.shape[1]} with {X.shape[2]} features.")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        input_dim = X.shape[2]
        output_dim = y.shape[2]
        hidden_dim = 64
        
        self.model = LSTMModel(input_dim, hidden_dim, output_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        # Save model and artifacts
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.mlb, self.mlb_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        if self.model_path.exists() and self.mlb_path.exists() and self.scaler_path.exists():
            self.mlb = joblib.load(self.mlb_path)
            self.scaler = joblib.load(self.scaler_path)
            
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
        
        # Inference with chunking to handle long sessions
        self.model.eval()
        all_probs = []
        hidden = None
        chunk_size = self.max_length
        
        with torch.no_grad():
            for i in range(0, len(X_scaled), chunk_size):
                chunk = X_scaled[i : i + chunk_size]
                X_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)
                outputs, hidden = self.model(X_tensor, hidden)
                all_probs.append(outputs.squeeze(0).cpu().numpy())
                
        if not all_probs:
            return []

        probs = np.concatenate(all_probs, axis=0)
            
        # Threshold and Segment
        threshold = 0.5
        active_mask = probs > threshold
        
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
        if self.model is None:
            if not self.load_model():
                return []

        if dataframe is not None:
            df = dataframe
        else:
            return []
        
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return []

        # Scale
        X_scaled = self.scaler.transform(numeric_df.values)
        
        # Inference with chunking to handle long sessions
        self.model.eval()
        all_probs = []
        hidden = None
        chunk_size = self.max_length
        
        with torch.no_grad():
            for i in range(0, len(X_scaled), chunk_size):
                chunk = X_scaled[i : i + chunk_size]
                X_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)
                outputs, hidden = self.model(X_tensor, hidden)
                all_probs.append(outputs.squeeze(0).cpu().numpy())
                
        if not all_probs:
            return []

        probs = np.concatenate(all_probs, axis=0)
            
        # Threshold and Segment
        threshold = 0.5
        active_mask = probs > threshold
        
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
        
        # Extract and cache segments
        chunk_segments = []
        for meta in found_segments:
            if target_labels:
                if not any(label in meta["labels"] for label in target_labels):
                    continue

            start = meta['start_index']
            end = meta['end_index']
            segment_df = df.iloc[start:end]
            
            # Extract actual data and wrap with metadata
            segment_data = segment_df.to_dict('records')
            
            predicted_segment = PredictedSegment(
                labels=meta["labels"],
                telemetry_data=segment_data,
                start_index=start,
                end_index=end
            )
            
            chunk_segments.append(predicted_segment.to_dict())
            
        # Cache segments
        from .full_dataset_ml_service import PipelineConfig
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
