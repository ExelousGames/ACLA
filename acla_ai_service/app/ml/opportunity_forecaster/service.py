"""Opportunity forecaster for future successful overtake / defense actions."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app.shared.circuit_sections import CIRCUIT_SECTION_RANGES
from app.shared.labels import LABEL_MAPPING
from app.shared.telemetry import MAX_CARS


FORECAST_LABELS = ("O1", "O3", "O4", "O5", "OD1", "OD2")
NO_OPPORTUNITY = "NO_OPPORTUNITY"
NORMALIZED_POSITION_COLUMN = "Graphics_normalized_car_position"
TIME_COLUMN = "Graphics_current_time"

BASE_FEATURES = (
    NORMALIZED_POSITION_COLUMN,
    TIME_COLUMN,
    "Physics_speed_kmh",
    "Physics_gas",
    "Physics_brake",
    "Physics_steer_angle",
    "Physics_g_force_x",
    "Physics_g_force_y",
    "Graphics_gap_ahead",
    "Graphics_gap_behind",
    "expert_time_difference",
    "speed_difference",
    "trajectory_offset",
    "driver_push_to_limit",
)


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _circular_delta(start: float, end: float) -> float:
    delta = float(end) - float(start)
    if delta < -0.5:
        delta += 1.0
    elif delta > 0.5:
        delta -= 1.0
    return delta


def _overlap_no_wrap(lo: float, hi: float, r_lo: float, r_hi: float) -> float:
    return max(0.0, min(hi, r_hi) - max(lo, r_lo))


def _split_wrapped_range(lo: float, hi: float) -> List[Tuple[float, float]]:
    lo = float(lo) % 1.0
    hi = float(hi) % 1.0
    if hi >= lo:
        return [(lo, hi)]
    return [(lo, 1.0), (0.0, hi)]


def _range_overlap(lo: float, hi: float, r_lo: float, r_hi: float) -> float:
    overlap = 0.0
    for seg_lo, seg_hi in _split_wrapped_range(lo, hi):
        if r_hi >= r_lo:
            overlap += _overlap_no_wrap(seg_lo, seg_hi, r_lo, r_hi)
        else:
            overlap += _overlap_no_wrap(seg_lo, seg_hi, r_lo, 1.0)
            overlap += _overlap_no_wrap(seg_lo, seg_hi, 0.0, r_hi)
    return overlap


def _range_span(lo: float, hi: float) -> float:
    span = (float(hi) - float(lo)) % 1.0
    return max(span, 1e-6)


def estimate_future_position_range(
    telemetry_rows: List[Dict[str, Any]],
    horizon_seconds: float,
) -> Optional[Tuple[float, float]]:
    df = pd.DataFrame(telemetry_rows)
    pos = _safe_numeric_series(df, NORMALIZED_POSITION_COLUMN)
    if pos.empty:
        return None

    current_pos = float(pos.iloc[-1]) % 1.0
    if len(pos) < 2:
        return current_pos, current_pos

    times = _safe_numeric_series(df, TIME_COLUMN)
    progress_delta = _circular_delta(float(pos.iloc[0]), float(pos.iloc[-1]))
    if len(times) >= 2:
        time_delta = float(times.iloc[-1] - times.iloc[0])
        if time_delta > 100.0:
            time_delta /= 1000.0
    else:
        time_delta = float(len(pos) - 1)

    if time_delta <= 0:
        return current_pos, current_pos

    progress_rate = max(0.0, progress_delta / time_delta)
    projected_end = (current_pos + progress_rate * max(0.0, float(horizon_seconds))) % 1.0
    return current_pos, projected_end


def match_circuit_section(
    telemetry_rows: List[Dict[str, Any]],
    horizon_seconds: float,
) -> Dict[str, Any]:
    projected = estimate_future_position_range(telemetry_rows, horizon_seconds)
    if projected is None:
        return {
            "projected_position_range": None,
            "top_matches": [],
            "is_ambiguous": False,
            "best_match": None,
        }

    lo, hi = projected
    span = _range_span(lo, hi)
    matches: List[Dict[str, Any]] = []
    for label_id, section_range in CIRCUIT_SECTION_RANGES.items():
        r_lo, r_hi = section_range
        overlap = _range_overlap(lo, hi, r_lo, r_hi)
        if overlap <= 0:
            continue
        matches.append({
            "label_id": label_id,
            "name": LABEL_MAPPING.get(label_id, label_id),
            "section_range": [float(r_lo), float(r_hi)],
            "overlap_fraction": float(overlap / span),
        })

    matches.sort(key=lambda item: item["overlap_fraction"], reverse=True)
    top = matches[:3]
    ambiguous_margin = 0.05
    is_ambiguous = (
        len(top) >= 2
        and (top[0]["overlap_fraction"] - top[1]["overlap_fraction"]) < ambiguous_margin
    )
    return {
        "projected_position_range": [float(lo), float(hi)],
        "top_matches": top,
        "is_ambiguous": is_ambiguous,
        "best_match": None if is_ambiguous or not top else top[0],
    }


class OpportunityForecasterService:
    def __init__(self, models_directory: str = "models"):
        self.models_directory = Path(models_directory).resolve()
        self.models_directory.mkdir(exist_ok=True)
        self.model_path = self.models_directory / "opportunity_forecaster.joblib"
        self.scaler_path = self.models_directory / "opportunity_forecaster_scaler.joblib"
        self.config_path = self.models_directory / "opportunity_forecaster_config.json"
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def extract_features(self, telemetry_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        df = pd.DataFrame(telemetry_rows or [])
        features: Dict[str, float] = {"sample_count": float(len(df))}

        for column in BASE_FEATURES:
            values = _safe_numeric_series(df, column)
            if values.empty:
                features[f"{column}_mean"] = 0.0
                features[f"{column}_last"] = 0.0
                features[f"{column}_min"] = 0.0
                features[f"{column}_max"] = 0.0
                features[f"{column}_delta"] = 0.0
                continue

            features[f"{column}_mean"] = float(values.mean())
            features[f"{column}_last"] = float(values.iloc[-1])
            features[f"{column}_min"] = float(values.min())
            features[f"{column}_max"] = float(values.max())
            features[f"{column}_delta"] = float(values.iloc[-1] - values.iloc[0]) if len(values) > 1 else 0.0

        opponent_distances: List[np.ndarray] = []
        if {"Graphics_player_pos_x", "Graphics_player_pos_y"}.issubset(df.columns):
            px = pd.to_numeric(df["Graphics_player_pos_x"], errors="coerce").to_numpy(dtype=float)
            py = pd.to_numeric(df["Graphics_player_pos_y"], errors="coerce").to_numpy(dtype=float)
            for slot in range(1, MAX_CARS + 1):
                x_col = f"Car_{slot}_pos_x"
                y_col = f"Car_{slot}_pos_y"
                if x_col not in df.columns or y_col not in df.columns:
                    continue
                ox = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
                oy = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
                active = np.isfinite(ox) & np.isfinite(oy) & ((ox != 0.0) | (oy != 0.0))
                if not active.any():
                    continue
                distance = np.sqrt((ox[active] - px[active]) ** 2 + (oy[active] - py[active]) ** 2)
                if distance.size:
                    opponent_distances.append(distance)

        if opponent_distances:
            all_distances = np.concatenate(opponent_distances)
            features["nearest_opponent_min_distance_m"] = float(np.nanmin(all_distances))
            features["nearest_opponent_mean_distance_m"] = float(np.nanmean(all_distances))
            features["active_opponent_slots"] = float(len(opponent_distances))
        else:
            features["nearest_opponent_min_distance_m"] = 0.0
            features["nearest_opponent_mean_distance_m"] = 0.0
            features["active_opponent_slots"] = 0.0

        return features

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        if not self.feature_names:
            self.feature_names = sorted(features)
        return np.asarray([[float(features.get(name, 0.0)) for name in self.feature_names]], dtype=float)

    def train(self, examples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        feature_rows: List[Dict[str, float]] = []
        labels: List[str] = []
        for example in examples:
            rows = example.get("telemetry_data") or example.get("telemetry_rows") or []
            label = str(example.get("label") or example.get("target_label") or NO_OPPORTUNITY)
            if label not in FORECAST_LABELS:
                label = NO_OPPORTUNITY
            feature_rows.append(self.extract_features(rows))
            labels.append(label)

        if not feature_rows:
            raise ValueError("No opportunity forecast training examples provided")

        self.feature_names = sorted({name for row in feature_rows for name in row})
        x = np.asarray(
            [[float(row.get(name, 0.0)) for name in self.feature_names] for row in feature_rows],
            dtype=float,
        )
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)
        self.model = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(x_scaled, labels)
        self.save_artifacts()
        return {
            "status": "success",
            "samples": len(labels),
            "classes": list(self.model.classes_),
            "feature_count": len(self.feature_names),
        }

    def save_artifacts(self) -> None:
        if self.model is None or self.scaler is None:
            raise ValueError("Cannot save opportunity forecaster before training")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        self.config_path.write_text(
            json.dumps({"feature_names": self.feature_names}, indent=2),
            encoding="utf-8",
        )

    def load_model(self) -> bool:
        if not (self.model_path.exists() and self.scaler_path.exists() and self.config_path.exists()):
            return False
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.feature_names = list(config.get("feature_names", []))
        return True

    def has_local_artifacts(self) -> bool:
        return self.model_path.exists() and self.scaler_path.exists() and self.config_path.exists()

    def serialize_artifacts(self) -> Dict[str, Any]:
        files = {}
        for path in (self.model_path, self.scaler_path, self.config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing opportunity forecaster artifact: {path}")
            files[path.name] = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"format": "opportunity_forecaster/v1", "files": files}

    def deserialize_artifacts(self, payload: Dict[str, Any]) -> None:
        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("opportunity_forecaster payload missing 'files' dict")
        required = (self.model_path.name, self.scaler_path.name, self.config_path.name)
        for name in required:
            if name not in files:
                raise ValueError(f"opportunity_forecaster payload missing required artifact: {name}")
        self.models_directory.mkdir(parents=True, exist_ok=True)
        for name, encoded in files.items():
            (self.models_directory / name).write_bytes(base64.b64decode(encoded))

    def forecast(
        self,
        telemetry_rows: List[Dict[str, Any]],
        horizon_seconds: float = 10.0,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        section_match = match_circuit_section(telemetry_rows, horizon_seconds)
        if self.model is None or self.scaler is None:
            if not self.load_model():
                return {
                    "status": "success",
                    "model_status": "not_trained",
                    "horizon_seconds": float(horizon_seconds),
                    "opportunities": [],
                    "circuit_section_match": section_match,
                }

        features = self.extract_features(telemetry_rows)
        x = self._vectorize(features)
        x_scaled = self.scaler.transform(x)
        probabilities = self.model.predict_proba(x_scaled)[0]
        class_probs = sorted(
            zip(self.model.classes_, probabilities),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        best_section = section_match.get("best_match")
        opportunities = []
        for label_id, probability in class_probs:
            if label_id == NO_OPPORTUNITY or label_id not in FORECAST_LABELS:
                continue
            item = {
                "label_id": str(label_id),
                "label_name": LABEL_MAPPING.get(str(label_id), str(label_id)),
                "parent_label": "OD" if str(label_id).startswith("OD") else "O",
                "probability": float(probability),
            }
            if best_section:
                item["circuit_section_id"] = best_section["label_id"]
                item["circuit_section_name"] = best_section["name"]
            opportunities.append(item)
            if len(opportunities) >= max(1, int(top_k)):
                break

        return {
            "status": "success",
            "model_status": "ready",
            "horizon_seconds": float(horizon_seconds),
            "opportunities": opportunities,
            "circuit_section_match": section_match,
        }


opportunity_forecaster = OpportunityForecasterService()


__all__ = [
    "FORECAST_LABELS",
    "NO_OPPORTUNITY",
    "OpportunityForecasterService",
    "estimate_future_position_range",
    "match_circuit_section",
    "opportunity_forecaster",
]
