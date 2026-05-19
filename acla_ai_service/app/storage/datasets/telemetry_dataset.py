"""Streaming chunked telemetry dataset for transformer training.

Loads one chunk at a time from the Zarr training cache and yields
sequence batches sized for GPU training. The dataset owns its own
feature scaler (``PerFeatureScaler``), built either at init from
hints or fitted in a streaming first-pass via
``_RunningFeatureStats``.

Lives in app/storage/datasets/ alongside the segment-classifier
dataset — both are I/O-bound chunk iterators owned by the storage
band, not pure model code.

Extracted from app/ml/transformer/model.py in refactor/hexagonal-v4
(Page 5 of acla-ai-service-architecture.drawio).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from app.domain.expert_features import ExpertFeatureCatalog
from app.domain.telemetry import TelemetryFeatures, _safe_float
from app.domain.tire_grip_features import TireGripFeatureCatalog
from app.storage.datasets.transformer_scaler import PerFeatureScaler, _RunningFeatureStats


class TelemetryActionDataset(Dataset):
    """
    Simplified Large Chunk Dataset for GPU Batch Training
    
    ASSUMPTIONS:
    - Each chunk contains 10k+ segments (large chunks)
    - Load one chunk at a time during training  
    - Batch segments within chunk for efficient GPU training
    - No complex fallback mechanisms needed
    
    APPROACH:
    - __len__ returns number of chunks available
    - __getitem__ loads one chunk and returns GPU-sized batches from it
    - Simple and efficient for large-scale training
    """
    
    def __init__(self,
                 data_cache,
                 segments_cache_key: str,
                 segment_length_hint: Optional[int] = None,
                 batch_size: int = 32,
                 min_sequence_length: int = 3,
                 sequence_bucket_size: int = 16):
        """
        Initialize simplified dataset for large chunks
        
        Args:
            data_cache: Training cache instance to load chunks from
            segments_cache_key: Cache key where chunks are stored
            segment_length_hint: Optional expected length for segments (used for logging only)
            batch_size: GPU batch size for processing segments within chunks
            min_sequence_length: Minimum number of timesteps required to keep a segment
            sequence_bucket_size: Approximate timestep window used when grouping sequences by length
        """
        self.data_cache = data_cache
        self.segments_cache_key = segments_cache_key
        self.segment_length_hint = segment_length_hint
        self.batch_size = batch_size
        self.min_sequence_length = max(2, min_sequence_length)
        self.sequence_bucket_size = max(1, sequence_bucket_size)
        
        # Get basic chunk information
        self.chunk_count = self._count_chunks()
        self.unified_features = self._get_feature_names()

        # Initialize feature preprocessing
        self.feature_scaler = PerFeatureScaler(self.unified_features)
        self._features_fitted = False
        self._length_stats: Optional[Dict[str, Any]] = None
        self._processed_chunks: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        print(f"[INFO] ✓ Simplified dataset initialized: {self.chunk_count} large chunks")
        print(f"[INFO] ✓ GPU batch size: {batch_size}")
        print(f"[INFO] ✓ Features: {len(self.unified_features)}")
        if self.segment_length_hint:
            print(f"[INFO] ✓ Segment length hint: ~{self.segment_length_hint} timesteps")
        print(f"[INFO] ✓ Assuming 10k+ segments per chunk")
    
    def _count_chunks(self) -> int:
        """Count available chunks without loading them all into memory"""
        try:
            chunks_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)
            chunk_count = 0
            for _ in chunks_iterator:
                chunk_count += 1
            print(f"[INFO] Found {chunk_count} chunks available")
            return chunk_count
        except Exception as e:
            raise ValueError(f"Failed to count chunks: {str(e)}")
    
    def _get_feature_names(self) -> List[str]:
        """Extract feature names from first chunk without caching everything"""
        try:
            chunks_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)
            first_chunk = next(chunks_iterator)
            

            # Get first segment from the chunk
            first_segment = first_chunk[0]
            
            # Handle PredictedSegment structure
            if isinstance(first_segment, dict) and "telemetry_data" in first_segment:
                first_segment = first_segment["telemetry_data"]

            first_timestep = first_segment[0]
            feature_names = list(first_timestep.keys())
            
            print(f"[INFO] Extracted {len(feature_names)} feature names from chunk with {len(first_chunk)} segments")
            return feature_names
                
        except Exception as e:
            raise ValueError(f"Failed to extract feature names: {str(e)}")
    
    def _load_chunk(self, chunk_idx: int) -> List[List[Dict[str, Any]]]:
        """Load a specific chunk by index.
        
        Returns a list of segments, where each segment is a list of timestep dicts.
        """
        chunks_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)
        
        # Skip to the desired chunk
        for i, chunk_data in enumerate(chunks_iterator):
            if i == chunk_idx:
                print(f"[INFO] Loaded chunk {chunk_idx} with {len(chunk_data)} segments")
                return chunk_data
        
        raise IndexError(f"Chunk {chunk_idx} not found")
    

    @staticmethod
    def _coerce_step_index(key: Any) -> Optional[int]:
        """Attempt to coerce a dataframe column key into an integer timestep index."""
        if isinstance(key, int):
            return key
        if isinstance(key, str):
            stripped = key.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    def _ensure_length_stats(self, max_chunks: int = 3) -> None:
        """Compute basic segment length statistics if not already available."""
        if self._length_stats is not None:
            return

        sampled_lengths: List[int] = []

        try:
            chunks_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)
            for chunk_idx, chunk_data in enumerate(chunks_iterator):
                if max_chunks is not None and chunk_idx >= max_chunks:
                    break


                # Process each segment in the chunk
                for segment in chunk_data:
                    # Handle PredictedSegment structure
                    if isinstance(segment, dict) and "telemetry_data" in segment:
                        segment = segment["telemetry_data"]

                    if len(segment) >= self.min_sequence_length:
                        sampled_lengths.append(len(segment))

                if sampled_lengths and max_chunks is None:
                    break

            if sampled_lengths:
                lengths_array = np.asarray(sampled_lengths, dtype=np.int32)
                self._length_stats = {
                    'min': int(lengths_array.min()),
                    'max': int(lengths_array.max()),
                    'mean': float(lengths_array.mean()),
                    'median': float(np.median(lengths_array)),
                    'count': int(lengths_array.size),
                    'source': 'sampled',
                    'sampled_chunks': min(self.chunk_count, max_chunks if max_chunks is not None else self.chunk_count)
                }
        except Exception as exc:
            print(f"[WARNING] Unable to compute segment length statistics: {exc}")

    
    def _ensure_features_fitted(self):
        """Ensure feature scaling is fitted using sample from first chunk"""
        if self._features_fitted:
            return
        
        print(f"[INFO] Fitting feature scaling using all available chunks...")

        # Drop any previously cached scaled sequences; they'll be regenerated with the new scaler.
        self._processed_chunks.clear()

        stats = _RunningFeatureStats(len(self.unified_features))
        total_rows = 0
        all_lengths: List[int] = []

        chunk_iterator = self.data_cache.get_cached_data_chunks(self.segments_cache_key)

        for chunk_idx, chunk_data in enumerate(chunk_iterator):
            # Chunk should be a list of segments
            if not isinstance(chunk_data, list):
                print(f"[WARNING] Expected list chunk at index {chunk_idx}, got {type(chunk_data)}")
                continue
            
            chunk_rows: List[List[float]] = []

            for segment in chunk_data:
                # Handle PredictedSegment structure
                if isinstance(segment, dict) and "telemetry_data" in segment:
                    segment = segment["telemetry_data"]

                # Segment is a list of timestep dicts
                if not isinstance(segment, list) or len(segment) < self.min_sequence_length:
                    continue

                all_lengths.append(len(segment))

                for timestep_data in segment:
                    if not isinstance(timestep_data, dict):
                        continue

                    row: List[float] = []
                    for feature in self.unified_features:
                        value = timestep_data.get(feature, 0.0)
                        try:
                            row.append(float(value))
                        except (ValueError, TypeError):
                            row.append(0.0)
                    chunk_rows.append(row)

            if chunk_rows:
                chunk_matrix = np.array(chunk_rows, dtype=np.float32)
                if np.isnan(chunk_matrix).any() or np.isinf(chunk_matrix).any():
                    print(f"[WARNING] NaN/Inf detected in chunk {chunk_idx}; applying cleaning before stats update")
                    chunk_matrix = np.nan_to_num(chunk_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

                stats.update(chunk_matrix)
                total_rows += chunk_matrix.shape[0]

            print(f"[DEBUG] Processed chunk {chunk_idx}: rows accumulated={total_rows}")

        if total_rows == 0:
            raise ValueError("No data available across chunks to fit feature scaler")

        if all_lengths:
            lengths_array = np.asarray(all_lengths, dtype=np.int32)
            self._length_stats = {
                'min': int(lengths_array.min()),
                'max': int(lengths_array.max()),
                'mean': float(lengths_array.mean()),
                'median': float(np.median(lengths_array)),
                'count': int(lengths_array.size),
                'source': 'full_dataset'
            }
        else:
            self._length_stats = None

        counts, means, variances = stats.finalize()
        self.feature_scaler = PerFeatureScaler.from_feature_statistics(
            self.unified_features,
            means,
            variances,
            counts,
            scaler_factory=self.feature_scaler.scaler_factory if self.feature_scaler else None
        )
        self._features_fitted = True

        print(f"[INFO] ✓ Feature scaling fitted across {total_rows} timesteps from all chunks")
    
    def _build_matrix(self, data_list: List[Dict[str, Any]], feature_names: List[str]) -> np.ndarray:
        """Extract features and build a matrix from list of dictionaries"""
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
    
    def __len__(self) -> int:
        """Return number of chunks available"""
        return self.chunk_count
    
    def _process_segment_record(self, segment_record: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Convert one cached segment record into input/target sequences (numpy arrays).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (input_seq, target_seq) each with shape [seq_len-1, features]
            or None if the record cannot be processed.
        """
        try:
            # Handle PredictedSegment structure
            if isinstance(segment_record, dict) and "telemetry_data" in segment_record:
                segment_record = segment_record["telemetry_data"]

            # Segment is a list of timestep dicts
            if not isinstance(segment_record, list) or len(segment_record) < self.min_sequence_length:
                return None

            sequence_data: List[List[float]] = []
            phase_weights: List[float] = []

            for timestep_data in segment_record:
                feature_array: List[float] = []
                for feature_name in self.unified_features:
                    try:
                        value = float(timestep_data.get(feature_name, 0.0))
                    except (ValueError, TypeError):
                        value = 0.0
                    feature_array.append(value)
                sequence_data.append(feature_array)

                # Build phase-aware weighting mask from raw telemetry values
                driver_push = _safe_float(
                    timestep_data.get(
                        TireGripFeatureCatalog.ContextFeature.DRIVER_PUSH_TO_LIMIT.value,
                        0.0,
                    )
                ) or 0.0

                brake_signal = _safe_float(timestep_data.get("Physics_brake", 0.0)) or 0.0
                throttle_signal = _safe_float(timestep_data.get("Physics_gas", 0.0)) or 0.0
                steer_signal = _safe_float(timestep_data.get("Physics_steer_angle", 0.0)) or 0.0

                # Emphasise high braking, steering, and push-to-limit phases; deemphasise heavy throttle
                weight = 1.0
                weight += 0.8 * max(0.0, min(1.0, driver_push))
                weight += 0.7 * max(0.0, min(1.0, abs(brake_signal)))
                weight += 0.6 * max(0.0, min(1.0, abs(steer_signal)))
                weight -= 0.3 * max(0.0, min(1.0, throttle_signal))

                # Keep weights within a reasonable range to avoid instability
                weight = float(np.clip(weight, 0.5, 3.0))
                phase_weights.append(weight)

            # [timesteps, features]
            sequence_matrix = np.array(sequence_data, dtype=np.float32)
            
            # Check for NaN/Inf in raw data before scaling
            if np.isnan(sequence_matrix).any() or np.isinf(sequence_matrix).any():
                print(f"[WARNING] NaN/Inf detected in raw segment data - skipping segment")
                return None

            # Scale per-timestep using fitted scaler
            scaled_sequence = self.feature_scaler.transform(sequence_matrix)
            
            # Check for NaN/Inf after scaling
            if np.isnan(scaled_sequence).any() or np.isinf(scaled_sequence).any():
                print(f"[WARNING] NaN/Inf detected after scaling - skipping segment")
                return None

            # Teacher-forcing style next-step target
            input_sequence = scaled_sequence[:-1]   # [seq_len-1, features]
            target_sequence = scaled_sequence[1:]   # [seq_len-1, features]
            timestep_weight_array = np.asarray(phase_weights[1:], dtype=np.float32)
            return input_sequence, target_sequence, timestep_weight_array
        except Exception:
            return None


    def _get_processed_chunk(self, chunk_idx: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return cached, fully scaled sequences for a chunk (build once, reuse each epoch)."""
        cached = self._processed_chunks.get(chunk_idx)
        if cached is not None:
            return cached

        chunk_records = self._load_chunk(chunk_idx)

        processed_segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for segment_record in chunk_records:
            processed = self._process_segment_record(segment_record)
            if processed is not None:
                processed_segments.append(processed)

        self._processed_chunks[chunk_idx] = processed_segments
        print(
            f"[DEBUG] Cached scaled sequences for chunk {chunk_idx}: {len(processed_segments)} segments"
        )
        return processed_segments


    def get_chunk_batches(self, chunk_idx: int):
        """
        Generator that yields GPU-sized batches from a large chunk
        
        Args:
            chunk_idx: Index of the chunk to process
            
        Yields:
            Tuples of (batch_inputs, batch_targets, batch_timestep_weights, batch_padding_mask)
        """
        if chunk_idx >= self.chunk_count:
            raise IndexError(f"Chunk index {chunk_idx} out of range")
            
        # Ensure features are fitted
        self._ensure_features_fitted()

        processed_segments = self._get_processed_chunk(chunk_idx)

        if not processed_segments:
            return

        bucket_size = self.sequence_bucket_size
        buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for segment in processed_segments:
            seq_len = segment[0].shape[0]
            bucket_key = int((seq_len // bucket_size) * bucket_size)
            buckets.setdefault(bucket_key, []).append(segment)

        bucket_order = list(buckets.keys())
        np.random.shuffle(bucket_order)
        
        batch_inputs: List[np.ndarray] = []
        batch_targets: List[np.ndarray] = []
        batch_masks: List[np.ndarray] = []

        def _collate_and_yield():
            if not batch_inputs:
                return

            input_tensors = [torch.from_numpy(arr).float() for arr in batch_inputs]
            target_tensors = [torch.from_numpy(arr).float() for arr in batch_targets]
            mask_tensors = [torch.from_numpy(arr).float() for arr in batch_masks]

            padded_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=0.0)
            padded_targets = pad_sequence(target_tensors, batch_first=True, padding_value=0.0)
            padded_masks = pad_sequence(mask_tensors, batch_first=True, padding_value=0.0)

            lengths = torch.tensor([tensor.shape[0] for tensor in input_tensors], dtype=torch.long)
            max_len = int(padded_inputs.shape[1])
            padding_mask = torch.ones((len(input_tensors), max_len), dtype=torch.bool)
            for row_idx, seq_len in enumerate(lengths):
                padding_mask[row_idx, :seq_len] = False

            yield padded_inputs, padded_targets, padded_masks, padding_mask
        for bucket_key in bucket_order:
            bucket_segments = buckets[bucket_key]
            permutation = np.random.permutation(len(bucket_segments))

            for idx in permutation:
                input_sequence, target_sequence, timestep_weights = bucket_segments[idx]
                batch_inputs.append(input_sequence)
                batch_targets.append(target_sequence)
                batch_masks.append(timestep_weights)

                # Yield batch when we reach desired size
                if len(batch_inputs) >= self.batch_size:
                    yield from _collate_and_yield()

                    # Reset for next batch to keep padding tight within the bucket
                    batch_inputs = []
                    batch_targets = []
                    batch_masks = []

            # Flush leftovers from this bucket before moving to the next length group
            if batch_inputs:
                yield from _collate_and_yield()
                batch_inputs = []
                batch_targets = []
                batch_masks = []
        
        # Yield remaining segments as final batch
        if batch_inputs:
            yield from _collate_and_yield()

    def __getitem__(self, chunk_idx: int):
        """Return chunk index for simple iteration - actual batching done by get_chunk_batches"""
        return chunk_idx
    
    def get_feature_names(self) -> Tuple[List[str], List[str]]:
        """
        Get feature names for unified state prediction model.
        
        In the unified approach, both input and target use the same feature set,
        as we're predicting the next state which has the same structure as input state.
        
        Returns:
            Tuple of (input_features, target_features) where both are identical unified_features
        """
        return self.unified_features, self.unified_features
    
    def get_scalers(self) -> PerFeatureScaler:
        return self.feature_scaler
    
    def get_segment_info(self) -> Dict[str, Any]:
        self._ensure_length_stats()

        return {
            "num_chunks": self.chunk_count,
            "segment_length_hint": self.segment_length_hint,
            "length_statistics": self._length_stats,
            "minimum_sequence_length": self.min_sequence_length,
            "total_features": len(self.unified_features),
            "feature_names": self.unified_features,
            "tensor_shapes": {
                "input": {
                    "sequence": "variable",
                    "features": len(self.unified_features),
                },
                "target": {
                    "sequence": "variable",
                    "features": len(self.unified_features),
                }
            }
        }
    
    def get_context_feature_names(self) -> List[str]:
        """Get context feature names from unified features"""
        # Filter for context features (exclude basic telemetry features)
        context_features = []
        for feature in self.unified_features:
            if any(context_prefix in feature.lower() for context_prefix in [
                'expert_', 'tire_', 'grip_', 'distance_', 
                'velocity_alignment', 'speed_difference'
            ]):
                context_features.append(feature)
        return context_features

    # Compatibility helper used by quality checks
    def get_input_feature_names(self) -> List[str]:
        """Return the unified input feature names (for quality reports)."""
        return list(self.unified_features)
    
    @staticmethod
    def validate_segments(unified_segments: List[List[Dict[str, Any]]], min_length: int = 2) -> Dict[str, Any]:
        """
        Validate that all segments satisfy a minimum length requirement and report statistics.

        Args:
            unified_segments: List of segments to validate
            min_length: Minimum allowed length per segment

        Returns:
            Dict with validation results and descriptive statistics
        """
        errors: List[str] = []
        lengths: List[int] = []

        for idx, segment in enumerate(unified_segments):
            # Handle PredictedSegment structure
            if isinstance(segment, dict) and "telemetry_data" in segment:
                segment = segment["telemetry_data"]

            seg_length = len(segment)
            lengths.append(seg_length)
            if seg_length < min_length:
                errors.append(f"Segment {idx}: length {seg_length} < minimum {min_length}")

        lengths_array = np.asarray(lengths, dtype=np.int32) if lengths else None

        statistics = {
            'min': int(lengths_array.min()) if lengths_array is not None else None,
            'max': int(lengths_array.max()) if lengths_array is not None else None,
            'mean': float(lengths_array.mean()) if lengths_array is not None else None,
            'median': float(np.median(lengths_array)) if lengths_array is not None else None,
            'num_segments': len(unified_segments),
            'num_too_short': len(errors),
            'minimum_required': min_length,
        }

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'statistics': statistics
        }
    




__all__ = ["TelemetryActionDataset"]
