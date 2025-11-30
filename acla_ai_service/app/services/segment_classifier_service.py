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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out: (batch, seq_len, hidden_dim)
        out = self.fc(out)
        return self.sigmoid(out)

class SegmentClassifierService:
    def __init__(self, models_directory: str = "models"):
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

    async def load_annotations(self) -> List[Dict[str, Any]]:
        """Load all annotations from Zarr."""
        # Import locally to avoid circular dependency
        from .full_dataset_ml_service import PipelineConfig
        cache_key = PipelineConfig().annotation_cache_key
        
        if not self.store.has_cached_data(cache_key):
            return []
        
        chunks = self.store.get_cached_data_chunks(cache_key)
        annotations = []
        for chunk in chunks:
            if isinstance(chunk, list):
                annotations.extend(chunk)
            elif isinstance(chunk, dict) and "payload" in chunk:
                 # Handle wrapped payload if necessary, though store usually returns raw payload
                 annotations.append(chunk["payload"])
            else:
                 # Fallback if chunk is a single item
                 annotations.append(chunk)
        return annotations

    async def prepare_training_data(self, sequence_length: int = 100, stride: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sessions and create sequences for training.
        Returns X (samples, seq_len, features) and y (samples, seq_len, classes).
        """
        annotations = await self.load_annotations()
        if not annotations:
            raise ValueError("No annotations found.")

        # 1. Collect all unique labels
        all_labels = set()
        for ann in annotations:
            all_labels.update(ann.get("labels", []))
        
        if not all_labels:
            raise ValueError("No labels found in annotations.")
            
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(all_labels)]) # Fit on all possible labels
        
        # Group annotations by session to minimize session loading
        anns_by_session = {}
        for ann in annotations:
            s_key = ann["session_key"]
            if s_key not in anns_by_session:
                anns_by_session[s_key] = []
            anns_by_session[s_key].append(ann)

        X_sequences = []
        y_sequences = []
        raw_data_list = []
        
        for session_key, session_anns in anns_by_session.items():
            # Load session data
            chunks = self.store.get_cached_data_chunks(session_key)
            all_records = []
            for chunk in chunks:
                if isinstance(chunk, dict) and "data" in chunk:
                    all_records.extend(chunk["data"])
                elif isinstance(chunk, list):
                    all_records.extend(chunk)
            
            if not all_records:
                continue
            
            df = pd.DataFrame(all_records)
            numeric_df = df.select_dtypes(include=['number'])
            
            if numeric_df.empty:
                continue
                
            # Create target array
            y_session = np.zeros((len(df), len(self.mlb.classes_)))
            
            for ann in session_anns:
                start = ann["start_index"]
                end = ann["end_index"]
                labels = ann["labels"]
                
                # Transform labels to binary vector
                label_vec = self.mlb.transform([labels])[0]
                
                # Clip indices
                start = max(0, start)
                end = min(len(df), end)
                
                if start < end:
                    y_session[start:end] = np.maximum(y_session[start:end], label_vec)

            raw_data_list.append((numeric_df.values, y_session))

        if not raw_data_list:
             raise ValueError("No valid training data found.")

        # Fit Scaler
        self.scaler = StandardScaler()
        all_features = np.vstack([d[0] for d in raw_data_list])
        self.scaler.fit(all_features)
        
        # Create sequences
        for data_values, y_session in raw_data_list:
            scaled_data = self.scaler.transform(data_values)
            
            num_samples = len(scaled_data)
            # Create overlapping sequences
            for i in range(0, num_samples - sequence_length + 1, stride):
                X_seq = scaled_data[i : i + sequence_length]
                y_seq = y_session[i : i + sequence_length]
                
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        if not X_sequences:
            raise ValueError("Not enough data to create sequences.")

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
                outputs = self.model(batch_X)
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
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Take max probability over the segment or average?
            # Let's take the average probability across the segment
            probs = outputs.squeeze(0).mean(dim=0).cpu().numpy()
            
        threshold = 0.5
        labels = []
        for i, p in enumerate(probs):
            if p > threshold:
                labels.append(self.mlb.classes_[i])
        
        return labels

    async def scan_session(self, session_key: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None, **kwargs) -> None:
        """
        Scan a session and find segments matching labels using LSTM.
        """
        if self.model is None:
            if not self.load_model():
                return

        if dataframe is not None:
            df = dataframe
        elif session_key:
            chunks = self.store.get_cached_data_chunks(session_key)
            all_records = []
            for chunk in chunks:
                if isinstance(chunk, dict) and "data" in chunk:
                    all_records.extend(chunk["data"])
                elif isinstance(chunk, list):
                    all_records.extend(chunk)
            
            if not all_records:
                return
            
            df = pd.DataFrame(all_records)
        else:
            return
        
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return

        # Scale
        X_scaled = self.scaler.transform(numeric_df.values)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device) # (1, seq_len, features)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(X_tensor) # (1, seq_len, num_classes)
            probs = outputs.squeeze(0).cpu().numpy() # (seq_len, num_classes)
            
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
            start = meta['start_index']
            end = meta['end_index']
            segment_df = df.iloc[start:end]
            chunk_segments.append(segment_df.to_dict('records'))
            
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
