"""Utilities for converting telemetry segments into training-ready segment/explanation pairs.

The legacy transformer pipeline predicted future telemetry sequences only.
This builder now emits lightweight records that couple a cleaned telemetry
segment with the coaching explanation text (initially blank) that annotators
will supply before fine-tuning the guidance model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency; used only when tensors are present
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch not required for dataset building
    torch = None  # type: ignore

from .zarr_telemetry_store import ZarrTelemetryStore, get_shared_zarr_store


@dataclass
class PromptBuilderConfig:
    """Configuration for prompt construction."""

    system_prompt: str = (
        "You are an elite race engineer. "
        "Given a telemetry segment, explain its purpose and provide concise coaching guidance."
    )
    context_steps: int = 20  # Reserved for backward compatibility; entire segment is used.
    prediction_steps: int = 6  # Deprecated; segments no longer split into future windows.
    context_stride: int = 5  # Deprecated; sliding window traversal removed.
    max_examples: Optional[int] = None
    random_seed: int = 42
    telemetry_features: Optional[Sequence[str]] = None
    round_floats: int = 4
    min_required_std: float = 1e-3


@dataclass
class DatasetBuildStats:
    """Simple container for dataset generation statistics."""

    segments_processed: int = 0
    windows_generated: int = 0
    windows_kept: int = 0
    skipped_short_segments: int = 0
    skipped_low_variance: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "segments_processed": self.segments_processed,
            "windows_generated": self.windows_generated,
            "windows_kept": self.windows_kept,
            "skipped_short_segments": self.skipped_short_segments,
            "skipped_low_variance": self.skipped_low_variance,
        }


class TelemetryPromptDatasetBuilder:
    """Create segment/explanation training pairs from cached telemetry segments."""

    def __init__(
        self,
        data_cache: Optional[ZarrTelemetryStore] = None,
        config: Optional[PromptBuilderConfig] = None,
    ) -> None:
        self.data_cache = data_cache or get_shared_zarr_store()
        self.config = config or PromptBuilderConfig()
        self._rng = random.Random(self.config.random_seed)
        self._window_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_from_cached_segments(
        self,
        segments_cache_key: str,
        *,
        shuffle: bool = True,
        record_consumer: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], DatasetBuildStats]:
        """Generate normalized prompt/response records from cached segments.

        Args:
            segments_cache_key: Cache key that stores enriched transformer segments.
            shuffle: If True, windows are shuffled before returning.
            record_consumer: Optional callback invoked for each generated entry. When provided,
                entries are streamed to the consumer instead of being accumulated in memory.

        Returns:
            A tuple of (entries, stats) where entries is a list of dictionaries describing each
            segment window and stats contains generation metrics.
        """

        stats = DatasetBuildStats()
        examples: List[Dict[str, Any]] = []
        streaming_mode = record_consumer is not None

        if streaming_mode and shuffle:
            print(
                "[WARN] Streaming dataset generation does not support shuffling; "
                "results will be emitted in source order."
            )

        self._window_counter = 0

        segment_chunk_iterator = self.data_cache.get_cached_data_chunks(segments_cache_key)
        for chunk_index, chunk_data in enumerate(segment_chunk_iterator):
            # chunk_data is a list of segments (List[List[Dict[str, Any]]])
            if chunk_data is None or not chunk_data:
                continue
            
            # Ensure chunk_data is a list
            if not isinstance(chunk_data, list):
                continue

            # chunk_data is already a list of segments, no need to call to_dict("records")
            for segment_index, segment in enumerate(chunk_data):
                stats.segments_processed += 1

                if not segment:
                    stats.skipped_short_segments += 1  # Keeps legacy metric name for empty segments.
                    continue

                stats.windows_generated += 1

                window_id = self._next_window_id(
                    chunk_index=chunk_index,
                    segment_index=segment_index,
                    start_index=0,
                )
                window_info = {
                    "window_id": window_id,
                    "chunk_index": chunk_index,
                    "segment_index": segment_index,
                    "start_index": 0,
                }

                prompt_record = self._build_dataset_entry(
                    context_timesteps=segment,
                    window_info=window_info,
                )
                if prompt_record:
                    if streaming_mode:
                        if record_consumer:
                            record_consumer(prompt_record)
                        else:
                            raise ValueError(
                                "record_consumer callback is required in streaming mode."
                            )
                    else:
                        examples.append(prompt_record)
                    stats.windows_kept += 1

                if self.config.max_examples and stats.windows_kept >= self.config.max_examples:
                    break

            if self.config.max_examples and stats.windows_kept >= self.config.max_examples:
                break

        if shuffle and not streaming_mode:
            self._rng.shuffle(examples)

        total_examples = stats.windows_kept
        stats_summary = stats.to_dict()
        stats_summary.update({
            "total_examples": total_examples,
        })
        print("[INFO] Telemetry Prompt dataset generated:", json.dumps(stats_summary, indent=2))
        return (examples if not streaming_mode else []), stats

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    def _build_dataset_entry(
        self,
        *,
        context_timesteps: List[Dict[str, Any]],
        window_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Create a prompt/response record using the entire telemetry segment as context."""

        if not context_timesteps:
            return None

        explanation = ""
        raw_steps = len(context_timesteps)

        telemetry_feature_list = (
            list(self.config.telemetry_features)
            if self.config.telemetry_features is not None
            else None
        )

        metadata = {
            "window_id": window_info.get("window_id"),
            "source_indices": {
                "chunk_index": window_info.get("chunk_index"),
                "record_index": window_info.get("record_index"),
                "segment_index": window_info.get("segment_index"),
                "start_index": window_info.get("start_index"),
            },
            "config": {
                "context_steps": raw_steps,
                "prediction_steps": 0,
                "telemetry_features": telemetry_feature_list,
            },
            "annotation_complete": bool(explanation),
            "system_prompt": self.config.system_prompt,
        }
        if telemetry_feature_list is None:
            metadata["config"].pop("telemetry_features")

        system_prompt = metadata.get("system_prompt") or ""
        prompt_body = (
            "Provide coaching explanation for the following telemetry segment.\n"
            "Telemetry segment (ordered timesteps):\n"
            f"{json.dumps(context_timesteps, indent=2, ensure_ascii=False, default=self._json_default)}"
        )

        clean_metadata = {
            key: value
            for key, value in metadata.items()
            if key != "system_prompt"
        }
        window_payload = {"context": context_timesteps}
        if window_payload:
            clean_metadata["window"] = window_payload

        record: Dict[str, Any] = {
            "prompt": prompt_body,
            "response": explanation,
        }
        if system_prompt:
            record["system_prompt"] = system_prompt
        if clean_metadata:
            record["metadata"] = clean_metadata
        return record

    def _next_window_id(
        self,
        *,
        chunk_index: int,
        segment_index: int,
        start_index: int,
    ) -> str:
        """Generate a reproducible identifier for each sliding window."""

        window_id = (
            f"chunk{chunk_index:04d}-segment{segment_index:04d}-start{start_index:05d}-"
            f"window{self._window_counter:07d}"
        )
        self._window_counter += 1
        return window_id

    @staticmethod
    def _clone_timesteps(timesteps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cloned: List[Dict[str, Any]] = []
        for step in timesteps:
            if isinstance(step, dict):
                cloned.append({key: value for key, value in step.items()})
        return cloned

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (set, tuple)):
            return list(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")