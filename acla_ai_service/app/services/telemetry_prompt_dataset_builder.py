"""Utilities for converting telemetry segments into prompt/response pairs for
large language model fine-tuning.

The legacy transformer pipeline predicted future telemetry sequences only.
This builder prepares instruction-tuning data that teaches an LLM to output
both future telemetry and a human-readable explanation when provided with
partial coaching notes and recent telemetry context.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .Training_data_cache_service import TrainingOptimizedCache, training_cache

PLAN_SUMMARY_FEATURES = (
    "Physics_gas",
    "Physics_brake",
    "Physics_steer_angle",
    "Physics_speed_kmh",
    "Physics_gear",
)


@dataclass
class PromptBuilderConfig:
    """Configuration for prompt construction."""

    system_prompt: str = (
        "You are an elite Assetto Corsa Competizione race engineer. "
        "Given recent telemetry and a predicted numeric driving plan, "
        "translate the plan into concise coaching commentary."
    )
    context_steps: int = 20
    prediction_steps: int = 6
    context_stride: int = 5
    max_examples: Optional[int] = None
    random_seed: int = 42
    telemetry_features: Sequence[str] = field(
        default_factory=lambda: (
            "Physics_speed_kmh",
            "Physics_rpm",
            "Physics_gear",
            "Physics_brake",
            "Physics_gas",
            "Physics_steer_angle",
            "Graphics_delta_lap_time",
            "Graphics_normalized_car_position",
            "Graphics_current_sector_index",
        )
    )
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
    """Create prompt/response training pairs from cached telemetry segments."""

    def __init__(
        self,
        data_cache: Optional[TrainingOptimizedCache] = None,
        config: Optional[PromptBuilderConfig] = None,
    ) -> None:
        self.data_cache = data_cache or training_cache
        self.config = config or PromptBuilderConfig()
        self._rng = random.Random(self.config.random_seed)
        self._window_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_from_cached_segments(
        self,
        segments_cache_key: str,
        output_path: Path,
        *,
        annotations: Optional[Dict[str, Dict[str, Any]]] = None,
        shuffle: bool = True,
    ) -> DatasetBuildStats:
        """Generate an instruction-tuning dataset from cached segments.

        Args:
            segments_cache_key: Cache key that stores enriched transformer segments.
            output_path: Destination JSONL file path; parent directory is created.
            annotations: Optional mapping of window_id -> annotation payload.
            shuffle: If True, windows are shuffled before writing to disk.

        Returns:
            DatasetBuildStats with generation metrics.
        """

        stats = DatasetBuildStats()
        windows: List[Dict[str, Any]] = []
        self._window_counter = 0

        chunk_iterator = self.data_cache.get_cached_data_chunks(segments_cache_key)
        for chunk_index, chunk_df in enumerate(chunk_iterator):
            if chunk_df is None or chunk_df.empty:
                continue

            chunk_records = chunk_df.to_dict("records")
            for record_index, record in enumerate(chunk_records):
                segments = record.get("data")
                if not isinstance(segments, list):
                    continue

                for segment_index, segment in enumerate(segments):
                    stats.segments_processed += 1
                    timesteps = self._extract_timesteps(segment)

                    if len(timesteps) < (self.config.context_steps + self.config.prediction_steps):
                        stats.skipped_short_segments += 1
                        continue

                    # Slide across the segment to generate multiple windows
                    end_limit = len(timesteps) - self.config.prediction_steps
                    for start_index in range(0, end_limit - self.config.context_steps + 1, self.config.context_stride):
                        stats.windows_generated += 1
                        context_slice = timesteps[start_index : start_index + self.config.context_steps]
                        future_slice = timesteps[
                            start_index + self.config.context_steps : start_index + self.config.context_steps + self.config.prediction_steps
                        ]

                        if not self._passes_variance_check(context_slice, future_slice):
                            stats.skipped_low_variance += 1
                            continue

                        window_id = self._next_window_id(
                            chunk_index=chunk_index,
                            record_index=record_index,
                            segment_index=segment_index,
                            start_index=start_index,
                        )
                        window_info = {
                            "window_id": window_id,
                            "chunk_index": chunk_index,
                            "record_index": record_index,
                            "segment_index": segment_index,
                            "start_index": start_index,
                        }
                        annotation_payload = annotations.get(window_id) if annotations else None

                        entry = self._build_dataset_entry(
                            context_slice,
                            future_slice,
                            window_info,
                            annotation_payload,
                        )
                        if entry:
                            windows.append(entry)
                            stats.windows_kept += 1

                        if self.config.max_examples and stats.windows_kept >= self.config.max_examples:
                            break

                    if self.config.max_examples and stats.windows_kept >= self.config.max_examples:
                        break

                if self.config.max_examples and stats.windows_kept >= self.config.max_examples:
                    break

            if self.config.max_examples and stats.windows_kept >= self.config.max_examples:
                break

        if shuffle:
            self._rng.shuffle(windows)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as jsonl_file:
            for entry in windows:
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        stats_summary = stats.to_dict()
        stats_summary.update({
            "output_path": str(output_path),
            "total_examples": len(windows),
        })
        print("[INFO] Prompt dataset generated:", json.dumps(stats_summary, indent=2))
        return stats

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    def _build_dataset_entry(
        self,
        context_timesteps: Sequence[Dict[str, Any]],
        future_timesteps: Sequence[Dict[str, Any]],
        window_info: Dict[str, Any],
        annotation: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context_payload = self._format_timesteps(context_timesteps)
        future_payload = self._format_timesteps(future_timesteps)

        if not context_payload or not future_payload:
            return None

        annotation = annotation or {}
        driver_note = annotation.get("driver_note")
        explanation = annotation.get("coaching_explanation")
        if not driver_note or not explanation:
            return None
        plan_summary = self._summarize_future_plan(future_payload)

        user_content = (
            "Task: Convert the numeric improvement plan into natural coaching commentary.\n"
            f"Driver request: {driver_note}\n\n"
            f"Recent telemetry window (last {self.config.context_steps} steps):\n"
            f"{json.dumps(context_payload, indent=2, ensure_ascii=False)}\n\n"
            f"Predicted improvement plan (next {self.config.prediction_steps} steps):\n"
            f"{json.dumps(plan_summary, indent=2, ensure_ascii=False)}\n\n"
            "Respond with a JSON object containing:\n"
            "- `coaching_summary`: concise natural-language advice (2-3 sentences).\n"
            "- Optional `key_focus`: list of bullet strings summarising critical adjustments."
        )

        assistant_payload = {
            "coaching_summary": explanation,
        }
        key_focus = annotation.get("key_focus")
        if key_focus:
            assistant_payload["key_focus"] = key_focus

        metadata = {
            "window_id": window_info.get("window_id"),
            "annotation": {
                "driver_note": driver_note,
                "coaching_explanation": explanation,
                "updated_at": annotation.get("updated_at") if annotation else None,
            },
            "source_indices": {
                "chunk_index": window_info.get("chunk_index"),
                "record_index": window_info.get("record_index"),
                "segment_index": window_info.get("segment_index"),
                "start_index": window_info.get("start_index"),
            },
            "config": {
                "context_steps": self.config.context_steps,
                "prediction_steps": self.config.prediction_steps,
                "telemetry_features": list(self.config.telemetry_features),
            },
            "plan_summary": plan_summary,
            "annotation_complete": bool(explanation),
        }

        entry = {
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
            ],
            "metadata": metadata,
            "window": {
                "context": context_payload,
                "future": future_payload,
            },
        }
        return entry

    def _next_window_id(
        self,
        *,
        chunk_index: int,
        record_index: int,
        segment_index: int,
        start_index: int,
    ) -> str:
        """Generate a reproducible identifier for each sliding window."""

        window_id = (
            f"chunk{chunk_index:04d}-record{record_index:05d}-segment{segment_index:04d}-start{start_index:05d}-"
            f"window{self._window_counter:07d}"
        )
        self._window_counter += 1
        return window_id

    def _format_timesteps(self, timesteps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for relative_index, timestep in enumerate(timesteps):
            filtered = {}
            for feature in self.config.telemetry_features:
                value = timestep.get(feature)
                if value is None:
                    continue

                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    filtered[feature] = round(float(value), self.config.round_floats)
                else:
                    filtered[feature] = value

            if not filtered:
                continue

            filtered["relative_index"] = relative_index
            formatted.append(filtered)
        return formatted

    def format_timesteps(self, timesteps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Public helper for formatting timesteps consistent with the dataset builder."""

        return self._format_timesteps(timesteps)

    def _summarize_future_plan(self, future_payload: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for index, step in enumerate(future_payload):
            if not isinstance(step, dict):
                continue

            actions: Dict[str, Any] = {}
            for feature in PLAN_SUMMARY_FEATURES:
                value = step.get(feature)
                if value is None:
                    continue
                actions[feature] = value

            if actions:
                summary.append({"step": index + 1, "actions": actions})

        return summary

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
    def _coerce_step_index(key: Any) -> Optional[int]:
        if isinstance(key, int):
            return key
        if isinstance(key, str) and key.strip().isdigit():
            return int(key.strip())
        return None

    def _extract_timesteps(self, segment_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        step_items: List[Tuple[int, Dict[str, Any]]] = []
        for key, value in segment_record.items():
            if not isinstance(value, dict):
                continue
            step_idx = self._coerce_step_index(key)
            if step_idx is None:
                continue
            step_items.append((step_idx, value))

        step_items.sort(key=lambda item: item[0])
        return [step for _, step in step_items]

    def _passes_variance_check(
        self,
        context_timesteps: Sequence[Dict[str, Any]],
        future_timesteps: Sequence[Dict[str, Any]],
    ) -> bool:
        """Skip windows that are effectively static (hard for LLM to learn from)."""
        combined = context_timesteps + future_timesteps
        feature_matrix: List[List[float]] = []
        for step in combined:
            row: List[float] = []
            for feature in self.config.telemetry_features:
                row.append(self._coerce_float(step.get(feature)))
            feature_matrix.append(row)

        matrix = np.asarray(feature_matrix, dtype=np.float32)
        if not np.isfinite(matrix).all():
            matrix = np.nan_to_num(matrix)

        variances = matrix.var(axis=0)
        return bool(np.any(variances > self.config.min_required_std))
