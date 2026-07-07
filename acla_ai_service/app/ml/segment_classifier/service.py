"""
Service for training and using a 1D-CNN classifier to identify behavioral segments.
Refactored to support variable length segments and learn local temporal relations.
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
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterator
import asyncio
import base64
import json
import logging
import shutil
import random
import hashlib
import copy
from collections import defaultdict

from app.storage import get_shared_telemetry_store
from app.shared.labels import LABEL_MAPPING, normalize_label_id, normalize_label_ids
from app.shared.segment import AnnotatedSegment, PredictedSegment
from app.shared.segment_classifier_features import SEGMENT_CLASSIFIER_FEATURES

# Extracted in refactor/hexagonal-v4 — Page 5 of the architecture diagram.
# Model classes are pure (no I/O); dataset + derived-features helper own I/O.
# Re-imported here so SegmentClassifierService keeps the same internal API.
from app.ml.segment_classifier.model import CNN1DModel, FocalLoss
from app.storage.datasets.segment_dataset import (
    StreamingSegmentDataset,
    compute_derived_features,
)

logger = logging.getLogger(__name__)


class SegmentClassifierService:
    def __init__(self, models_directory: str = "models", max_length: int = 100):
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(exist_ok=True)
        self.model_path = self.models_directory / "segment_classifier.pth"
        self.mlb_path = self.models_directory / "segment_labels.joblib"
        self.scaler_path = self.models_directory / "segment_scaler.joblib"
        self.pos_weight_path = self.models_directory / "segment_pos_weight.pt"
        self.store = get_shared_telemetry_store()
        self.model = None
        self.mlb = None 
        self.scaler = None
        self.pos_weight = None
        self.label_counts = {}
        self.feature_names = None
        
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

    def _current_feature_names(self) -> List[str]:
        return list(SEGMENT_CLASSIFIER_FEATURES)

    def _legacy_feature_names_for_scaler(self, scaler_feature_count: Optional[int]) -> Optional[List[str]]:
        """Feature layout used by artifacts trained before gap columns were added."""
        if scaler_feature_count is None:
            return None

        current_features = self._current_feature_names()
        legacy_features = []
        replaced_gap_columns = False

        for feature in current_features:
            if feature == "Graphics_gap_ahead":
                legacy_features.append("Graphics_current_tyre_set")
                replaced_gap_columns = True
            elif feature == "Graphics_gap_behind":
                replaced_gap_columns = True
                continue
            else:
                legacy_features.append(feature)

        if replaced_gap_columns and len(legacy_features) * 2 == scaler_feature_count:
            return legacy_features
        return None

    def _scaler_feature_count(self) -> Optional[int]:
        if self.scaler is None:
            return None
        count = getattr(self.scaler, "n_features_in_", None)
        if count is not None:
            return int(count)
        mean = getattr(self.scaler, "mean_", None)
        if mean is not None:
            return int(len(mean))
        return None

    def _feature_names_for_model(self) -> List[str]:
        scaler_feature_count = self._scaler_feature_count()
        if self.feature_names:
            if scaler_feature_count is None or len(self.feature_names) * 2 == scaler_feature_count:
                return list(self.feature_names)
            self.feature_names = None

        current_features = self._current_feature_names()
        if scaler_feature_count == len(current_features) * 2:
            self.feature_names = current_features
            return list(current_features)

        legacy_features = self._legacy_feature_names_for_scaler(scaler_feature_count)
        if legacy_features is not None:
            self.feature_names = legacy_features
            return list(legacy_features)

        self.feature_names = current_features
        return list(current_features)

    def _prepare_numeric_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return numeric model features in the exact order expected by the scaler."""
        expected_features = self._feature_names_for_model()
        missing_features = [feature for feature in expected_features if feature not in dataframe.columns]
        if missing_features:
            logger.warning(
                "segment_classifier input missing %d/%d expected feature columns; filling with 0. Sample: %s",
                len(missing_features),
                len(expected_features),
                missing_features[:10],
            )
        df = dataframe.reindex(columns=expected_features, fill_value=0)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        if df.empty:
            return df
        return compute_derived_features(df)

    def _labels_ranked_by_probability(self, probabilities: np.ndarray) -> List[str]:
        """Return all labels in descending model-score order."""
        ranked_indices = np.argsort(probabilities)[::-1]
        labels = []
        seen_labels = set()
        for idx in ranked_indices:
            normalized_label = normalize_label_id(self.mlb.classes_[idx])
            if normalized_label not in seen_labels:
                labels.append(normalized_label)
                seen_labels.add(normalized_label)
        return labels

    def _print_probability_summary(
        self,
        title: str,
        probabilities: np.ndarray,
        targets: np.ndarray,
        target_names: List[str],
    ) -> None:
        """Log threshold-free validation diagnostics."""
        print(title)
        print("label | support | mean_pos_prob | mean_neg_prob | max_prob")
        for i, label_name in enumerate(target_names):
            y_true = targets[:, i].astype(bool)
            y_score = probabilities[:, i]
            support = int(y_true.sum())
            mean_pos = float(y_score[y_true].mean()) if support else 0.0
            mean_neg = float(y_score[~y_true].mean()) if np.any(~y_true) else 0.0
            max_prob = float(y_score.max()) if len(y_score) else 0.0
            print(
                f"{label_name}: Support={support}, "
                f"MeanPosProb={mean_pos:.4f}, "
                f"MeanNegProb={mean_neg:.4f}, "
                f"MaxProb={max_prob:.4f}"
            )

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
    
    def _assign_split(self, hash_value: str, val_split: float) -> str:
        """Deterministically assign an item/group to train or val based on hash."""
        # Use first 8 characters of hash to generate a number between 0 and 1
        hash_int = int(hash_value[:8], 16)
        hash_normalized = hash_int / (16**8)
        
        return "val" if hash_normalized < val_split else "train"

    def _segment_group_key(self, segment_dict: Dict[str, Any]) -> str:
        """Return the Static_track split key for an annotated segment."""
        value = segment_dict.get("Static_track")
        if value not in (None, ""):
            value = str(value).strip()
            if value:
                return value

        telemetry_data = segment_dict.get("telemetry_data")
        if isinstance(telemetry_data, list) and telemetry_data:
            first_row = telemetry_data[0]
            if isinstance(first_row, dict):
                value = first_row.get("Static_track")
                if value not in (None, ""):
                    value = str(value).strip()
                    if value:
                        return value

        for row in telemetry_data or []:
            if not isinstance(row, dict):
                continue
            value = row.get("Static_track")
            if value not in (None, ""):
                value = str(value).strip()
                if value:
                    return value

        try:
            fallback_data = json.dumps(segment_dict, sort_keys=True, default=str)
        except Exception:
            fallback_data = str(segment_dict)
        fallback_hash = hashlib.md5(fallback_data.encode()).hexdigest()
        return f"segment:{fallback_hash}"

    def _segment_session_key(self, segment_dict: Dict[str, Any]) -> str:
        """Return the session-level split key within a Static_track group."""
        for key in ("session_id", "sessionId", "sessionID", "chunk_index", "chunk_id"):
            value = segment_dict.get(key)
            if value not in (None, ""):
                value = str(value).strip()
                if value:
                    return value

        telemetry_data = segment_dict.get("telemetry_data")
        for row in telemetry_data or []:
            if not isinstance(row, dict):
                continue
            for key in ("session_id", "sessionId", "sessionID", "chunk_index", "chunk_id"):
                value = row.get(key)
                if value not in (None, ""):
                    value = str(value).strip()
                    if value:
                        return value

        try:
            fallback_data = json.dumps(segment_dict, sort_keys=True, default=str)
        except Exception:
            fallback_data = str(segment_dict)
        fallback_hash = hashlib.md5(fallback_data.encode()).hexdigest()
        return f"segment:{fallback_hash}"

    def _rebalance_track_label_splits(
        self,
        track_key: str,
        session_keys: List[str],
        session_splits: Dict[str, str],
        session_label_counts: Dict[str, Dict[Any, int]],
    ) -> None:
        """Move sessions within a track so splittable labels appear in train and val."""
        if len(session_keys) < 2:
            return

        label_to_sessions = defaultdict(list)
        for session_key in session_keys:
            for label, count in session_label_counts[session_key].items():
                if count > 0:
                    label_to_sessions[label].append(session_key)

        splittable_labels = sorted(
            label
            for label, label_session_keys in label_to_sessions.items()
            if len(label_session_keys) > 1
        )
        if not splittable_labels:
            return

        splittable_label_set = set(splittable_labels)
        max_moves = len(session_keys) * len(splittable_labels) * 2

        for _ in range(max_moves):
            train_counts = defaultdict(int)
            val_counts = defaultdict(int)
            split_session_counts = defaultdict(int)

            for session_key in session_keys:
                split = session_splits[session_key]
                split_session_counts[split] += 1
                target_counts = val_counts if split == "val" else train_counts
                for label, count in session_label_counts[session_key].items():
                    target_counts[label] += count

            move = None
            for label in splittable_labels:
                if train_counts[label] == 0:
                    source_split = "val"
                    target_split = "train"
                    source_counts = val_counts
                elif val_counts[label] == 0:
                    source_split = "train"
                    target_split = "val"
                    source_counts = train_counts
                else:
                    continue

                candidates = [
                    session_key
                    for session_key in label_to_sessions[label]
                    if session_splits[session_key] == source_split
                ]
                valid_candidates = []
                for session_key in candidates:
                    if split_session_counts[source_split] <= 1:
                        continue
                    candidate_counts = session_label_counts[session_key]
                    would_empty_label = any(
                        moved_label in splittable_label_set
                        and source_counts[moved_label] - moved_count <= 0
                        for moved_label, moved_count in candidate_counts.items()
                    )
                    if not would_empty_label:
                        valid_candidates.append(session_key)

                if valid_candidates:
                    move = min(
                        valid_candidates,
                        key=lambda session_key: hashlib.md5(
                            f"{track_key}:{label}:{session_key}:{target_split}".encode()
                        ).hexdigest(),
                    )
                    move = (move, target_split)
                    break

            if move is None:
                break

            session_key, target_split = move
            session_splits[session_key] = target_split

    async def prepare_training_data(self, source_cache_key: str, train_cache_key: str, val_cache_key: str, val_split: float = 0.2, chunk_size: int = 100):
        """
        Splits data from source_cache_key into train and val keys within each track.
        Uses two-pass approach:
        1. First pass: collect valid segments and group sessions by Static_track
        2. Second pass: assign sessions within each track using deterministic hashing,
           then rebalance labels that can appear in both train and validation
        """
        print(f"Preparing training data: splitting {source_cache_key} into {train_cache_key} and {val_cache_key}")
        print("Using deterministic per-track, per-label session splitting to avoid same-session leakage...")
        
        # Clear existing keys
        for key in [train_cache_key, val_cache_key]:
            self.store.clear_cache(key)
        
        # PASS 1: Collect label statistics
        print("Pass 1: Collecting track/session groups and label statistics...")
        track_to_session_items = defaultdict(lambda: defaultdict(list))
        track_session_label_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        label_counts = defaultdict(int)
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
                
                # Extract labels and migrate legacy IDs before any training split/cache.
                labels = normalize_label_ids(d.get("labels", []))
                item = d
                if labels != d.get("labels", []):
                    item = dict(d)
                    item["labels"] = labels

                valid_items.append(item)
                item_key = (chunk_idx, len(valid_items) - 1)

                if labels:
                    track_key = self._segment_group_key(item)
                    session_key = self._segment_session_key(item)
                    track_to_session_items[track_key][session_key].append(item_key)

                    for lbl in set(labels):
                        track_session_label_counts[track_key][session_key][lbl] += 1
                        label_counts[lbl] += 1
            
            if valid_items:
                chunk_index.append((valid_items, chunk_idx))
                chunk_idx += 1
        
        print(f"Found {len(chunk_index)} chunks with {sum(len(items) for items, _ in chunk_index)} valid segments")
        print(f"Found {len(track_to_session_items)} track groups for splitting")
        print(f"Label distribution: {[(label, count) for label, count in sorted(label_counts.items())]}")
        
        # PASS 2: Per-track session split using deterministic hashing, then label rebalancing
        print("Pass 2: Performing per-track session split with label coverage balancing...")
        
        # For each label, split segments deterministically
        train_segments_set = set()  # Set of (chunk_idx, item_idx)
        val_segments_set = set()
        
        train_label_counts = defaultdict(int)
        val_label_counts = defaultdict(int)
        
        for track_key, session_items in track_to_session_items.items():
            session_keys = sorted(
                session_items.keys(),
                key=lambda session_key: hashlib.md5(f"{track_key}:{session_key}".encode()).hexdigest(),
            )
            val_session_keys = {
                session_key
                for session_key in session_keys
                if self._assign_split(
                    hashlib.md5(f"{track_key}:{session_key}".encode()).hexdigest(),
                    val_split,
                ) == "val"
            }

            if session_keys and not val_session_keys:
                val_session_keys.add(session_keys[0])
            if len(session_keys) > 1 and len(val_session_keys) == len(session_keys):
                val_session_keys.remove(session_keys[-1])

            session_splits = {
                session_key: "val" if session_key in val_session_keys else "train"
                for session_key in session_keys
            }
            self._rebalance_track_label_splits(
                track_key,
                session_keys,
                session_splits,
                track_session_label_counts[track_key],
            )

            for session_key in session_keys:
                split = session_splits[session_key]
                target_set = val_segments_set if split == "val" else train_segments_set
                target_counts = val_label_counts if split == "val" else train_label_counts

                for item_key in session_items[session_key]:
                    target_set.add(item_key)
                for label, count in track_session_label_counts[track_key][session_key].items():
                    target_counts[label] += count
        
        print(f"Train segments: {len(train_segments_set)}, Val segments: {len(val_segments_set)}")
        print(f"Train label distribution: {dict(train_label_counts)}")
        print(f"Val label distribution: {dict(val_label_counts)}")
        unsplit_labels = [
            label
            for label in sorted(label_counts.keys())
            if train_label_counts[label] == 0 or val_label_counts[label] == 0
        ]
        if unsplit_labels:
            print(
                "Labels without both train and validation coverage "
                f"(not splittable without breaking session grouping): {unsplit_labels}"
            )
        
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
        
        self.feature_names = self._current_feature_names()
        
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
                mapped_labels = normalize_label_ids(ann.labels)
                all_labels.update(mapped_labels)
                for l in mapped_labels:
                    self.label_counts[l] = self.label_counts.get(l, 0) + 1
                total_segments += 1
                
                if not ann.telemetry_data:
                    continue
                
                df = pd.DataFrame(ann.telemetry_data)
                
                df = df.reindex(columns=self.feature_names, fill_value=0)
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
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

    async def train_model(
        self,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        val_split=0.1,
        annotation_cache_key: Optional[str] = None,
    ):
        """Train the CNN classifier using streaming data with train/val split."""
        from app.pipelines.training.config import TrainingPipelineConfig
        cache_key = annotation_cache_key or TrainingPipelineConfig().annotation_cache_key
        print(f"Training source annotation dataset: {cache_key}")
        
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
            self._feature_names_for_model()
        )

        val_dataset = StreamingSegmentDataset(
            self.store, 
            val_key, 
            self.mlb, 
            self.scaler, 
            self.max_length,
            self._feature_names_for_model()
        )
        
        # num_workers=0 — avoids subprocess Lance reader complications.
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
        all_probs = []
        all_targets = []
        
        # Segment-level accumulation
        all_segment_probs = []
        all_segment_targets = []
        
        with torch.no_grad():
            for val_X, val_y, val_mask in val_loader:
                val_X, val_y, val_mask = val_X.to(self.device), val_y.to(self.device), val_mask.to(self.device)
                outputs, _ = self.model(val_X)
                
                probs = torch.sigmoid(outputs)
                
                # Filter by mask
                mask_flat = val_mask.cpu().bool().numpy().flatten()
                probs_flat = probs.cpu().numpy().reshape(-1, output_dim)
                targets_flat = val_y.cpu().numpy().reshape(-1, output_dim)
                
                if len(mask_flat) > 0:
                    all_probs.append(probs_flat[mask_flat])
                    all_targets.append(targets_flat[mask_flat])

                # --- Per-Segment Evaluation ---
                batch_size_curr = val_X.size(0)
                
                for i in range(batch_size_curr):
                    # Get actual length from mask
                    length = int(val_mask[i].sum().item())
                    if length == 0:
                        continue
                        
                    # Target is the same for the whole segment, just take the first valid one
                    # val_y shape: (batch, seq_len, num_classes)
                    seg_target = val_y[i, 0].cpu().numpy()
                    
                    # Average probability over valid timesteps.
                    seg_probs = probs[i, :length].mean(dim=0)
                    
                    all_segment_probs.append(seg_probs.cpu().numpy())
                    all_segment_targets.append(seg_target)

        target_names = [
            LABEL_MAPPING.get(normalize_label_id(l), normalize_label_id(l))
            for l in self.mlb.classes_
        ]

        if all_segment_probs:
            y_seg_probs = np.array(all_segment_probs)
            y_seg_true = np.array(all_segment_targets)
            self._print_probability_summary(
                "\n=== Segment-Level Probability Summary (Aggregated) ===",
                y_seg_probs,
                y_seg_true,
                target_names,
            )
            print("========================================================\n")

        if all_probs:
            # Concatenate
            y_probs = np.concatenate(all_probs)
            y_true = np.concatenate(all_targets)
            self._print_probability_summary(
                "Validation Probability Summary (Per-Timestep):",
                y_probs,
                y_true,
                target_names,
            )

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
            "num_layers": num_layers,
            "feature_names": self._feature_names_for_model(),
        }
        with open(self.models_directory / "segment_config.json", "w") as f:
            json.dump(config, f)

        print(f"Model saved to {self.model_path}")

        # Push to backend so other replicas / fresh containers can hydrate
        # from /app/ml/segment_classifier/bootstrap.py on startup. Local
        # training is the source of truth — log and continue on failure.
        try:
            from app.integrations.backend.client import backend_service
            payload = self.serialize_artifacts()
            await backend_service.save_ai_model(
                model_type="segment_classifier",
                model_data=payload,
                metadata={
                    "max_length": self.max_length,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_labels": len(self.mlb.classes_) if self.mlb is not None else 0,
                    "feature_count": self._scaler_feature_count() or 0,
                },
                is_active=True,
            )
            print("[INFO] ✓ segment_classifier uploaded to backend")
        except Exception as upload_exc:
            print(f"[WARN] segment_classifier backend upload failed: {upload_exc}")

    def load_model(self):
        """Load the trained model."""
        if self.model_path.exists() and self.mlb_path.exists() and self.scaler_path.exists():
            self.feature_names = None
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
                    feature_names = config.get("feature_names")
                    scaler_feature_count = self._scaler_feature_count()
                    if (
                        isinstance(feature_names, list)
                        and all(isinstance(f, str) for f in feature_names)
                        and (
                            scaler_feature_count is None
                            or len(feature_names) * 2 == scaler_feature_count
                        )
                    ):
                        self.feature_names = feature_names

            self._feature_names_for_model()
            input_dim = self.scaler.mean_.shape[0]
            output_dim = len(self.mlb.classes_)

            self.model = CNN1DModel(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            return True
        return False

    # Artifact filenames packed into / unpacked from the backend payload.
    # pos_weight is optional to keep old classifier payloads loadable.
    _ARTIFACT_FILES_REQUIRED = (
        "segment_classifier.pth",
        "segment_labels.joblib",
        "segment_scaler.joblib",
        "segment_config.json",
    )
    _ARTIFACT_FILES_OPTIONAL = ("segment_pos_weight.pt",)

    def serialize_artifacts(self) -> Dict[str, Any]:
        """Pack the on-disk model files into a JSON-safe dict for backend upload."""
        files: Dict[str, str] = {}
        for name in self._ARTIFACT_FILES_REQUIRED:
            path = self.models_directory / name
            if not path.is_file():
                raise FileNotFoundError(f"Cannot serialize — missing required artifact: {path}")
            files[name] = base64.b64encode(path.read_bytes()).decode("ascii")

        for name in self._ARTIFACT_FILES_OPTIONAL:
            path = self.models_directory / name
            if path.is_file():
                files[name] = base64.b64encode(path.read_bytes()).decode("ascii")

        return {"format": "segment_classifier/v1", "files": files}

    def deserialize_artifacts(self, payload: Dict[str, Any]) -> None:
        """Write a backend-fetched payload back to ``self.models_directory``."""
        if not isinstance(payload, dict):
            raise ValueError(f"segment_classifier payload must be dict, got {type(payload)}")
        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("segment_classifier payload missing 'files' dict")

        for name in self._ARTIFACT_FILES_REQUIRED:
            if name not in files:
                raise ValueError(f"segment_classifier payload missing required artifact: {name}")

        self.models_directory.mkdir(parents=True, exist_ok=True)
        for name, encoded in files.items():
            (self.models_directory / name).write_bytes(base64.b64decode(encoded))

    def has_local_artifacts(self) -> bool:
        """True when every required artifact already lives on disk."""
        return all(
            (self.models_directory / name).is_file()
            for name in self._ARTIFACT_FILES_REQUIRED
        )

    def predict_segment(self, segment_df: pd.DataFrame) -> List[str]:
        """Return labels ranked by raw model probability for a single segment."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or found.")

        numeric_df = self._prepare_numeric_features(segment_df)

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
            
        return self._labels_ranked_by_probability(probs)

    def predict_segment_probabilities(self, segment_df: pd.DataFrame) -> Dict[str, float]:
        """Predict probabilities for all labels for a single segment DataFrame."""
        if self.model is None:
            if not self.load_model():
                return {}

        df = self._prepare_numeric_features(segment_df)

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
            label = normalize_label_id(self.mlb.classes_[i])
            result[label] = max(result.get(label, 0.0), float(p))
            
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    def scan_telemetry_data(self, dataframe: pd.DataFrame) -> List[PredictedSegment]:
        """
        Scan a dataframe and return found segments with labels.
        Uses full-window CNN inference with probability smoothing.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Segment classifier model not trained or found.")
        
        source_df = dataframe.reset_index(drop=True)
        numeric_df = self._prepare_numeric_features(source_df)
        if numeric_df.empty:
            return []

        # Scale
        X_scaled = self.scaler.transform(numeric_df.values)
        
        # Inference on the full window lets the 1D-CNN use local row context
        # across the window instead of treating rows independently.
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
            
        found_segments = []
        current_labels = []
        current_start = 0
        
        # Iterate through to find contiguous segments with the same top label.
        # This keeps scanning threshold-free without marking every sigmoid
        # output as active everywhere.
        for i in range(len(numeric_df)):
            top_idx = int(np.argmax(probs_smoothed[i]))
            labels_at_i = [normalize_label_id(self.mlb.classes_[top_idx])]
            
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
                "end_index": len(numeric_df),
                "labels": current_labels
            })
        
        results = []
        for meta in found_segments:
            start = meta['start_index']
            end = meta['end_index']
            
            # Filter out very short segments (e.g. < 3 steps) as noise
            if end - start < 3:
                continue

            segment_df = source_df.iloc[start:end]
            
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
        Scan a session and find segments matching labels using the CNN classifier.
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
        from app.pipelines.training.config import TrainingPipelineConfig
        cache_key = TrainingPipelineConfig().segments_cache_key
        
        if chunk_segments:
             async def segments_generator():
                yield chunk_segments
             
             await self.store.cache_chunks_streaming(
                cache_key=cache_key,
                chunks_iterator=segments_generator()
             )

# Singleton instance
segment_classifier = SegmentClassifierService()
