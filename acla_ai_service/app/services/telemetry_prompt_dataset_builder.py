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


@dataclass
class PromptBuilderConfig:
    """Configuration for prompt construction."""

    system_prompt: str = (
        "You are an elite Assetto Corsa Competizione race engineer. "
        "Given partial notes and recent telemetry, forecast the next telemetry "
        "steps and continue the coaching explanation."
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_from_cached_segments(
        self,
        segments_cache_key: str,
        output_path: Path,
        *,
        shuffle: bool = True,
    ) -> DatasetBuildStats:
        """Generate an instruction-tuning dataset from cached segments.

        Args:
            segments_cache_key: Cache key that stores enriched transformer segments.
            output_path: Destination JSONL file path; parent directory is created.
            shuffle: If True, windows are shuffled before writing to disk.

        Returns:
            DatasetBuildStats with generation metrics.
        """

        stats = DatasetBuildStats()
        windows: List[Dict[str, Any]] = []

        chunk_iterator = self.data_cache.get_cached_data_chunks(segments_cache_key)
        for chunk_index, chunk_df in enumerate(chunk_iterator):
            if chunk_df is None or chunk_df.empty:
                continue

            chunk_records = chunk_df.to_dict("records")
            for record in chunk_records:
                segments = record.get("data")
                if not isinstance(segments, list):
                    continue

                for segment in segments:
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

                        entry = self._build_dataset_entry(context_slice, future_slice)
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
    ) -> Optional[Dict[str, Any]]:
        context_payload = self._format_timesteps(context_timesteps)
        future_payload = self._format_timesteps(future_timesteps)

        if not context_payload or not future_payload:
            return None

        partial_note = self._generate_partial_human_note(context_timesteps)
        explanation = self._generate_future_explanation(context_timesteps, future_timesteps)

        user_content = (
            "Task: Forecast telemetry and continue the coaching note.\n"
            f"Human note (partial): {partial_note}\n\n"
            f"Recent telemetry window (last {self.config.context_steps} steps):\n"
            f"{json.dumps(context_payload, indent=2, ensure_ascii=False)}\n\n"
            f"Provide the next {self.config.prediction_steps} telemetry steps in JSON "
            "alongside a clear explanation extending the human note."
        )

        assistant_payload = {
            "future_telemetry": future_payload,
            "explanation": explanation,
        }

        entry = {
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
            ]
        }
        return entry

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

    def _generate_partial_human_note(self, context_timesteps: Sequence[Dict[str, Any]]) -> str:
        last_step = context_timesteps[-1]
        speed = self._coerce_float(last_step.get("Physics_speed_kmh"))
        brake = self._coerce_float(last_step.get("Physics_brake"))
        gas = self._coerce_float(last_step.get("Physics_gas"))
        steer = self._coerce_float(last_step.get("Physics_steer_angle"))
        delta = self._coerce_float(last_step.get("Graphics_delta_lap_time"))

        phrases: List[str] = []
        if brake > 0.7:
            phrases.append("heavy brake entry")
        elif gas > 0.7:
            phrases.append("strong throttle push")
        else:
            phrases.append("balanced pedal input")

        if abs(steer) > 0.45:
            direction = "left" if steer > 0 else "right"
            phrases.append(f"aggressive turn to the {direction}")
        else:
            phrases.append("mid-corner stability focus")

        if delta > 0.05:
            phrases.append("losing time to target lap ...")
        elif delta < -0.05:
            phrases.append("gaining time relative to delta ...")
        else:
            phrases.append("matching reference pace ...")

        if speed > 200:
            phrases.append("approaching high-speed zone ...")
        elif speed < 80:
            phrases.append("car slowed for apex ...")

        return "Coach observation (partial): " + ", ".join(phrases)

    def _generate_future_explanation(
        self,
        context_timesteps: Sequence[Dict[str, Any]],
        future_timesteps: Sequence[Dict[str, Any]],
    ) -> str:
        last_context = context_timesteps[-1]
        first_future = future_timesteps[0]
        final_future = future_timesteps[-1]

        speed_change = self._describe_delta(
            last_context.get("Physics_speed_kmh"), final_future.get("Physics_speed_kmh"), "speed"
        )
        brake_change = self._describe_delta(
            last_context.get("Physics_brake"), first_future.get("Physics_brake"), "brake input"
        )
        steer_change = self._describe_delta(
            last_context.get("Physics_steer_angle"), first_future.get("Physics_steer_angle"), "steering"
        )

        notes = [speed_change, brake_change, steer_change]
        notes = [note for note in notes if note]
        if not notes:
            notes.append("maintain current attitude while smoothing throttle application")

        return "; ".join(notes)

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

    def _describe_delta(self, start_value: Any, end_value: Any, label: str) -> Optional[str]:
        start = self._coerce_float(start_value)
        end = self._coerce_float(end_value)
        delta = end - start

        if abs(delta) < 0.01:
            return None

        direction = "increase" if delta > 0 else "decrease"
        magnitude = abs(delta)
        if label == "speed":
            return f"expect {direction} in speed by {magnitude:.1f} km/h"
        if label == "brake input":
            return f"anticipate {direction} in brake pedal to {end:.2f}"
        if label == "steering":
            return f"steering will {direction} towards {end:.2f}"
        return f"{label} change of {delta:.2f}"