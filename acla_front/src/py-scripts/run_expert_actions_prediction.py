#!/usr/bin/env python3
"""Run expert action prediction using imitate_expert_learning_service.

This script expects two arguments:
    1. Path to a JSON file containing the serialized imitation learning model data
    2. Path to a JSON file containing telemetry samples (array of dicts or JSONL)

It loads the model, processes telemetry with FeatureProcessor, and outputs
predicted expert actions along with a normalized-position sweep for visualization.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import traceback
import types
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
except Exception:  # pragma: no cover - optional dependency at runtime
    DecisionTreeRegressor = None  # type: ignore[assignment]
    DecisionTreeClassifier = None  # type: ignore[assignment]

warnings.filterwarnings('ignore', category=FutureWarning)

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(',', ':')) + '\n')
    sys.stdout.flush()


@contextlib.contextmanager
def _capture_stdout() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


def _ensure_package(package_name: str, path: Optional[Path] = None) -> None:
    """Dynamically register a namespace package for relative imports."""
    module = sys.modules.get(package_name)
    path_str = str(path) if path is not None else None

    if module is None:
        module = types.ModuleType(package_name)
        module.__path__ = []  # type: ignore[attr-defined]
        if path_str is not None:
            module.__path__.append(path_str)  # type: ignore[attr-defined]
        sys.modules[package_name] = module
        return

    if path_str is not None:
        existing_path = list(getattr(module, '__path__', []))
        if path_str not in existing_path:
            existing_path.append(path_str)
            module.__path__ = existing_path  # type: ignore[attr-defined]

REPO_ROOT = CURRENT_DIR.parents[2] if len(CURRENT_DIR.parents) > 2 else None
if REPO_ROOT is not None:
    ai_service_app_dir = REPO_ROOT / 'acla_ai_service' / 'app'
    if ai_service_app_dir.exists():
        ai_service_app_path = str(ai_service_app_dir)
        if ai_service_app_path not in sys.path:
            sys.path.insert(0, ai_service_app_path)

front_dir: Optional[Path] = CURRENT_DIR.parents[1] if len(CURRENT_DIR.parents) > 1 else None
src_dir: Path = CURRENT_DIR.parent

_ensure_package('acla_front', front_dir)
_ensure_package('acla_front.src', src_dir)
_ensure_package('acla_front.src.py_scripts', CURRENT_DIR)
_ensure_package('acla_front.src.models', CURRENT_DIR)

_MODULE_LOGS: List[str] = []


def _emit_log(raw: str, stage: str) -> None:
    if not raw:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
    tagged = f"{stage}: {line}"
    _MODULE_LOGS.append(tagged)
    _emit({'status': 'log', 'stage': stage, 'message': line})


def _patch_decision_tree_monotonic() -> None:
    """Ensure pickled DecisionTree models with monotonic constraints deserialize cleanly."""

    def _ensure_property(tree_cls: Optional[type]) -> None:
        if tree_cls is None or hasattr(tree_cls, 'monotonic_cst'):
            return

        attr_name = '_monotonic_cst'

        def _getter(self: Any) -> Any:  # type: ignore[misc]
            return getattr(self, attr_name, None)

        def _setter(self: Any, value: Any) -> None:  # type: ignore[misc]
            setattr(self, attr_name, value)

        setattr(tree_cls, 'monotonic_cst', property(_getter, _setter))

    _ensure_property(DecisionTreeRegressor)
    _ensure_property(DecisionTreeClassifier)


_patch_decision_tree_monotonic()

telemetry_module_name = 'acla_front.src.models.telemetry_models'
telemetry_module_path = CURRENT_DIR / 'telemetry_models.py'
if telemetry_module_path.exists() and telemetry_module_name not in sys.modules:
    telemetry_spec = importlib.util.spec_from_file_location(telemetry_module_name, telemetry_module_path)
    if telemetry_spec is None or telemetry_spec.loader is None:
        raise ImportError(f'Unable to load telemetry_models module from {telemetry_module_path}')
    telemetry_module = importlib.util.module_from_spec(telemetry_spec)
    sys.modules[telemetry_module_name] = telemetry_module
    with _capture_stdout() as telemetry_stdout:
        telemetry_spec.loader.exec_module(telemetry_module)
    captured = telemetry_stdout.getvalue().strip()
    _emit_log(captured, 'import.telemetry_models')

module_name = 'acla_front.src.py_scripts.imitate_expert_learning_service'

module_path = CURRENT_DIR / 'imitate_expert_learning_service.py'
spec = importlib.util.spec_from_file_location(module_name, module_path)
if spec is None or spec.loader is None:
    raise ImportError(f'Unable to load imitate_expert_learning_service module from {module_path}')

imitate_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = imitate_module
with _capture_stdout() as imitate_stdout:
    spec.loader.exec_module(imitate_module)
captured_import_logs = imitate_stdout.getvalue().strip()
_emit_log(captured_import_logs, 'import.imitate_expert_learning_service')

ExpertImitateLearningService = imitate_module.ExpertImitateLearningService
FeatureProcessor = imitate_module.FeatureProcessor


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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
        _emit({'status': 'error', 'message': 'Usage: run_expert_actions_prediction.py <model_json_path> <telemetry_json_path>'})
        return

    model_path = Path(sys.argv[1]).expanduser().resolve()
    telemetry_path = Path(sys.argv[2]).expanduser().resolve()

    runtime_capture: Optional[io.StringIO] = None

    try:
        with _capture_stdout() as captured_stdout:
            runtime_capture = captured_stdout

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

        runtime_logs = runtime_capture.getvalue().strip() if runtime_capture is not None else ''
        _emit_log(runtime_logs, 'runtime')

        result: Dict[str, Any] = {
            'status': 'success',
            'prediction': {k: _serialize_value(v) for k, v in optimal_actions.items()} if isinstance(optimal_actions, dict) else optimal_actions,
            'metadata': {
                'telemetry_samples_used': sample_count,
                'has_normalized_positions': bool(normalized_positions),
            },
            'normalized_positions': normalized_positions,
            'position_series': position_series,
        }

        if _MODULE_LOGS:
            result['logs'] = list(_MODULE_LOGS)

        _emit(result)
        return

    except Exception as error:
        runtime_logs = runtime_capture.getvalue().strip() if runtime_capture is not None else ''
        _emit_log(runtime_logs, 'runtime')

        error_payload: Dict[str, Any] = {
            'status': 'error',
            'message': str(error),
            'traceback': traceback.format_exc()
        }

        if _MODULE_LOGS:
            error_payload['logs'] = list(_MODULE_LOGS)

        _emit(error_payload)
        return


if __name__ == '__main__':
    main()
