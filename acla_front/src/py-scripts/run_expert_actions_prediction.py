#!/usr/bin/env python3
"""Run expert action prediction using imitate_expert_learning_service.

This script expects two arguments:
    1. Path to a JSON file containing the serialized imitation learning model data
    2. Path to a JSON file containing telemetry samples (array of dicts or JSONL)

It loads the model, processes telemetry with FeatureProcessor, and outputs
predicted expert actions along with a normalized-position sweep for visualization.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from imitate_expert_learning_service import ExpertImitateLearningService, FeatureProcessor  # noqa: E402


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Support JSON and JSONL files
    if path.suffix.lower() == '.jsonl':
        records: List[Any] = []
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with path.open('r', encoding='utf-8') as handle:
        content = handle.read().strip()
        if not content:
            return []
        return json.loads(content)


def _ensure_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    raise ValueError('Telemetry data must be a list or dict')


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _build_position_series(service: ExpertImitateLearningService, points: int = 51) -> List[Dict[str, Any]]:
    try:
        positions = np.linspace(0.0, 1.0, points)
        predictions = service.position_learner.predict_expert_actions_at_position(positions)
    except Exception:
        return []

    series: List[Dict[str, Any]] = []
    if isinstance(predictions, list):
        for pos, prediction in zip(positions, predictions):
            if isinstance(prediction, dict):
                entry = {k: _serialize_value(v) for k, v in prediction.items()}
            else:
                entry = {'value': _serialize_value(prediction)}
            entry['normalized_position'] = float(pos)
            series.append(entry)
    elif isinstance(predictions, dict):
        entry = {k: _serialize_value(v) for k, v in predictions.items()}
        entry['normalized_position'] = float(positions[0])
        series.append(entry)
    return series


def main() -> None:
    if len(sys.argv) < 3:
        print(json.dumps({
            'status': 'error',
            'message': 'Usage: run_expert_actions_prediction.py <model_json_path> <telemetry_json_path>'
        }))
        sys.exit(1)

    model_path = Path(sys.argv[1]).expanduser().resolve()
    telemetry_path = Path(sys.argv[2]).expanduser().resolve()

    try:
        model_payload = _load_json_file(model_path)
        telemetry_payload = _load_json_file(telemetry_path)

        if not model_payload:
            raise ValueError('Model data is empty')
        if not telemetry_payload:
            raise ValueError('Telemetry data is empty')

        service = ExpertImitateLearningService()
        service.deserialize_imitation_model(model_payload)

        telemetry_df = _ensure_dataframe(telemetry_payload)
        sample_count = len(telemetry_df)
        if sample_count == 0:
            raise ValueError('Telemetry dataframe is empty after conversion')

        # Limit to most recent samples to keep processing lightweight
        max_samples = 2000
        if sample_count > max_samples:
            telemetry_df = telemetry_df.tail(max_samples)
            sample_count = len(telemetry_df)

        processor = FeatureProcessor(telemetry_df)
        processed_df = processor.general_cleaning_for_analysis()
        if processed_df is None or processed_df.empty:
            raise ValueError('Processed telemetry dataframe is empty')

        predictions = service.predict_expert_actions(processed_df)
        optimal_actions = predictions.get('optimal_actions', {}) if isinstance(predictions, dict) else predictions

        normalized_positions: List[float] = []
        if 'Graphics_normalized_car_position' in processed_df.columns:
            normalized_positions = [
                float(_serialize_value(val))
                for val in processed_df['Graphics_normalized_car_position'].tolist()
            ]

        position_series = _build_position_series(service)

        result = {
            'status': 'success',
            'prediction': {k: _serialize_value(v) for k, v in optimal_actions.items()} if isinstance(optimal_actions, dict) else optimal_actions,
            'metadata': {
                'telemetry_samples_used': sample_count,
                'has_normalized_positions': bool(normalized_positions),
            },
            'normalized_positions': normalized_positions,
            'position_series': position_series,
        }
        print(json.dumps(result, separators=(',', ':')))

    except Exception as error:
        print(json.dumps({
            'status': 'error',
            'message': str(error),
            'traceback': traceback.format_exc()
        }, separators=(',', ':')))
        sys.exit(1)


if __name__ == '__main__':
     main()
