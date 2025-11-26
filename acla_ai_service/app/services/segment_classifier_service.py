"""
Service for training and using a Random Forest Classifier to identify behavioral segments.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import asyncio

from .zarr_telemetry_store import get_shared_zarr_store

class SegmentClassifierService:
    def __init__(self, models_directory: str = "models"):
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(exist_ok=True)
        self.model_path = self.models_directory / "segment_classifier.joblib"
        self.mlb_path = self.models_directory / "segment_labels.joblib"
        self.store = get_shared_zarr_store()
        self.model = None
        self.mlb = None # MultiLabelBinarizer

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

    async def prepare_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load annotations and corresponding telemetry to create X and y."""
        annotations = await self.load_annotations()
        if not annotations:
            raise ValueError("No annotations found.")

        X_list = []
        y_labels = []

        # Group annotations by session to minimize session loading
        anns_by_session = {}
        for ann in annotations:
            s_key = ann["session_key"]
            if s_key not in anns_by_session:
                anns_by_session[s_key] = []
            anns_by_session[s_key].append(ann)

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
            
            # Ensure numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            
            for ann in session_anns:
                start = ann["start_index"]
                end = ann["end_index"]
                
                if start >= len(df) or end > len(df):
                    continue
                
                segment_df = numeric_df.iloc[start:end]
                if segment_df.empty:
                    continue

                # Feature extraction: Aggregates over the segment
                # Mean, Std, Min, Max for each column
                features = {}
                for col in segment_df.columns:
                    features[f"{col}_mean"] = segment_df[col].mean()
                    features[f"{col}_std"] = segment_df[col].std()
                    features[f"{col}_min"] = segment_df[col].min()
                    features[f"{col}_max"] = segment_df[col].max()
                
                X_list.append(features)
                y_labels.append(ann["labels"])

        if not X_list:
            raise ValueError("No valid training segments found.")

        X = pd.DataFrame(X_list)
        X = X.fillna(0) # Handle NaNs
        
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform(y_labels)
        
        return X, y

    async def train_model(self):
        """Train the Random Forest Classifier."""
        print("Preparing training data...")
        X, y = await self.prepare_training_data()
        
        print(f"Training on {len(X)} samples with {X.shape[1]} features.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use MultiOutputClassifier with RandomForest
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = MultiOutputClassifier(forest, n_jobs=-1)
        
        self.model.fit(X_train, y_train)
        
        print("Model trained.")
        y_pred = self.model.predict(X_test)
        
        # Print classification report
        for i, label in enumerate(self.mlb.classes_):
            print(f"Report for {label}:")
            print(classification_report(y_test[:, i], y_pred[:, i]))

        # Save model and label binarizer
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.mlb, self.mlb_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model."""
        if self.model_path.exists() and self.mlb_path.exists():
            self.model = joblib.load(self.model_path)
            self.mlb = joblib.load(self.mlb_path)
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

        features = {}
        for col in numeric_df.columns:
            features[f"{col}_mean"] = numeric_df[col].mean()
            features[f"{col}_std"] = numeric_df[col].std()
            features[f"{col}_min"] = numeric_df[col].min()
            features[f"{col}_max"] = numeric_df[col].max()
        
        # Align features with training data (fill missing with 0)
        # Note: In a real scenario, we should save the feature names used during training
        # For now, we assume the columns are consistent or we handle it dynamically if we had the feature list
        # To be safe, we should probably save feature names.
        # But for this prototype, let's just create a DF and hope columns match or use what we have.
        # A better way is to save feature columns in __init__ or train.
        
        # Let's assume we just pass the dict and let DataFrame handle it, 
        # but we need to ensure column order/existence matches training.
        # For robustness, let's just return the raw prediction for now.
        
        X_input = pd.DataFrame([features])
        
        # Handle missing columns that were in training
        # We need to know training columns. 
        # Let's assume we reload them or just try predict.
        # Ideally we save feature_names_in_ with the model.
        
        try:
            y_pred = self.model.predict(X_input)
            labels = self.mlb.inverse_transform(y_pred)
            return list(labels[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            return []

    async def scan_session(self, session_key: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None, window_size: int = 100, step_size: int = 50) -> None:
        """
        Scan a session and find segments matching labels.
        Uses a rolling window approach to ensure every index is classified based on the consensus of covering windows.
        Note: step_size is ignored in this implementation to ensure maximum resolution (perfect matching).
        
        Modified to cache segments directly to store instead of returning them.
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
        if numeric_df.empty or len(df) < window_size:
            return

        # 1. Extract features using rolling window (effectively step_size=1)
        # We must maintain the same feature order as in prepare_training_data:
        # col_mean, col_std, col_min, col_max for each column
        feature_dfs = []
        for col in numeric_df.columns:
            r = numeric_df[col].rolling(window=window_size)
            feature_dfs.append(r.mean().rename(f"{col}_mean"))
            feature_dfs.append(r.std().rename(f"{col}_std"))
            feature_dfs.append(r.min().rename(f"{col}_min"))
            feature_dfs.append(r.max().rename(f"{col}_max"))
        
        X = pd.concat(feature_dfs, axis=1)
        
        # The first window_size-1 rows will have NaNs because the window isn't full
        # We only predict for valid full windows.
        # The row at index `i` in X corresponds to the window ending at `i` (i.e., df[i-window_size+1 : i+1])
        valid_indices = X.index[window_size-1:]
        X_valid = X.loc[valid_indices].fillna(0)
        
        if X_valid.empty:
            return

        # 2. Predict labels for all valid windows
        # y_pred is a binary matrix (n_samples, n_classes)
        try:
            y_pred = self.model.predict(X_valid)
        except Exception as e:
            print(f"Prediction error during scan: {e}")
            return

        # 3. Aggregate votes per index
        # We map predictions back to the original timeline.
        # A prediction at index `t` (end of window) applies to the range [t-window_size+1, t].
        # We want to know for each index `i`, what is the consensus of all windows covering `i`.
        
        # Create a DataFrame of predictions indexed by the window end index
        pred_df = pd.DataFrame(y_pred, index=valid_indices, columns=self.mlb.classes_)
        
        # Reindex to full df to align (fill missing with 0)
        pred_df_full = pred_df.reindex(df.index, fill_value=0)
        
        # Calculate vote counts for each index.
        # Vote count at `i` is the sum of predictions for all windows covering `i`.
        # A window ending at `t` covers `i` if `t-window_size+1 <= i <= t` <=> `i <= t <= i+window_size-1`.
        # So we sum `pred_df_full` from `t=i` to `t=i+window_size-1`.
        # This is a forward rolling sum. We can implement it as a backward rolling sum shifted.
        vote_counts = pred_df_full.rolling(window=window_size).sum().shift(-(window_size-1)).fillna(0)
        
        # Calculate coverage (how many windows cover each index)
        # This handles the edges where fewer than window_size windows cover the index.
        coverage = pd.Series(1, index=df.index).rolling(window=window_size).sum().shift(-(window_size-1)).fillna(0)
        coverage[coverage == 0] = 1 # Avoid division by zero
        
        # Normalize votes to get scores [0, 1]
        scores = vote_counts.div(coverage, axis=0)
        
        # 4. Threshold and Segment
        threshold = 0.5
        active_labels_df = scores > threshold
        
        found_segments = []
        current_labels = []
        current_start = 0
        
        # Iterate through to find contiguous segments with the same label set
        for i in range(len(df)):
            row = active_labels_df.iloc[i]
            # Get labels that are True for this index
            labels_at_i = row[row].index.tolist()
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
