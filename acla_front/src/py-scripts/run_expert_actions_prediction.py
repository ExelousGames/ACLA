#!/usr/bin/env python3
"""Run expert action prediction using imitate_expert_learning_service.

This script expects two arguments:
    1. Path to a JSON file containing the serialized imitation learning model data
    2. JSON string containing telemetry samples (array of dicts or JSONL-style string)

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


def _import_module_from_path(module_name: str, module_path: Path, stage: str) -> types.ModuleType:
    """Import module from an explicit path and capture stdout for structured logs."""

    existing = sys.modules.get(module_name)
    if isinstance(existing, types.ModuleType):
        return existing

    if not module_path.exists():
        raise FileNotFoundError(f"Required module file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    with _capture_stdout() as captured_stdout:
        spec.loader.exec_module(module)

    captured = captured_stdout.getvalue().strip()
    _emit_log(captured, stage)
    return module

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

_MODULE_LOGS: List[str] = []


def _emit_log(raw: str, stage: str, request_id: Optional[str] = None) -> None:
    if not raw:
        return
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload: Dict[str, Any] = {'status': 'log', 'stage': stage, 'message': line}
        if request_id is not None:
            payload['request_id'] = request_id
        else:
            tagged = f"{stage}: {line}"
            _MODULE_LOGS.append(tagged)
        _emit(payload)

telemetry_module_name = 'acla_front.src.py_scripts.telemetry_models'
telemetry_module_path = CURRENT_DIR / 'telemetry_models.py'
telemetry_module = _import_module_from_path(telemetry_module_name, telemetry_module_path, 'import.telemetry_models')

module_name = 'acla_front.src.py_scripts.imitate_expert_learning_service'

module_path = CURRENT_DIR / 'imitate_expert_learning_service.py'
imitate_module = _import_module_from_path(module_name, module_path, 'import.imitate_expert_learning_service')

ExpertImitateLearningService = imitate_module.ExpertImitateLearningService
if not hasattr(telemetry_module, 'FeatureProcessor'):
    raise AttributeError('FeatureProcessor not found in telemetry_models module')
FeatureProcessor = getattr(telemetry_module, 'FeatureProcessor')


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


def _load_json_payload(raw: str) -> Any:
    payload = raw.strip()
    if not payload:
        return []

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        records: List[Any] = []
        for line in payload.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        if not records:
            raise
        return records


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


def _build_position_series(optimal_actions: Any, normalized_position: Optional[float]) -> List[Dict[str, Any]]:
    if not isinstance(optimal_actions, dict):
        return []

    entry = {k: _serialize_value(v) for k, v in optimal_actions.items()}
    if normalized_position is not None:
        entry['normalized_position'] = float(normalized_position)
    return [entry]


def _normalize_telemetry_payload(payload: Any) -> Any:
    if payload is None:
        return []
    if isinstance(payload, str):
        return _load_json_payload(payload)
    return payload


def _prepare_service(model_path: Path) -> ExpertImitateLearningService:
    model_payload = _load_json_file(model_path)
    if not model_payload:
        raise ValueError('Model data is empty')

    runtime_capture: Optional[io.StringIO] = None
    service: Optional[ExpertImitateLearningService] = None
    try:
        with _capture_stdout() as captured_stdout:
            runtime_capture = captured_stdout

            service = ExpertImitateLearningService()
            service.deserialize_imitation_model(model_payload)

        captured = runtime_capture.getvalue().strip() if runtime_capture is not None else ''
        _emit_log(captured, 'init.deserialize')
    except Exception:
        captured = runtime_capture.getvalue().strip() if runtime_capture is not None else ''
        _emit_log(captured, 'init.deserialize')
        raise

    if service is None:
        raise RuntimeError('Failed to initialize expert imitation learning service')

    return service


def _run_prediction(
    service: ExpertImitateLearningService,
    telemetry_payload: Any,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    runtime_capture: Optional[io.StringIO] = None
    try:
        with _capture_stdout() as captured_stdout:
            runtime_capture = captured_stdout

            telemetry_data = _normalize_telemetry_payload(telemetry_payload)
            if not telemetry_data:
                raise ValueError('Telemetry data is empty')

            telemetry_df = _ensure_dataframe(telemetry_data)
            if len(telemetry_df) == 0:
                raise ValueError('Telemetry dataframe is empty after conversion')

            processor = FeatureProcessor(telemetry_df)
            processed_df = processor.general_cleaning_for_analysis()
            if processed_df is None or processed_df.empty:
                raise ValueError('Processed telemetry dataframe is empty')

            if len(processed_df) > 1:
                print(f"[INFO] Processed telemetry generated {len(processed_df)} rows; using the first row for prediction.")
                processed_df = processed_df.iloc[[0]].copy()

            predictions = service.predict_expert_actions(processed_df)
            optimal_actions = predictions.get('optimal_actions', {}) if isinstance(predictions, dict) else predictions

            normalized_position: Optional[float] = None
            if 'Graphics_normalized_car_position' in processed_df.columns:
                normalized_position = float(_serialize_value(processed_df['Graphics_normalized_car_position'].iloc[0]))

            normalized_positions: List[float] = [normalized_position] if normalized_position is not None else []
            position_series = _build_position_series(optimal_actions, normalized_position)

        runtime_logs = runtime_capture.getvalue().strip() if runtime_capture is not None else ''
        _emit_log(runtime_logs, 'runtime', request_id)

        result: Dict[str, Any] = {
            'status': 'success',
            'prediction': {k: _serialize_value(v) for k, v in optimal_actions.items()} if isinstance(optimal_actions, dict) else optimal_actions,
            'metadata': {
                'telemetry_samples_used': 1,
                'has_normalized_positions': bool(normalized_positions),
            },
            'normalized_positions': normalized_positions,
            'position_series': position_series,
        }

        if request_id is not None:
            result['request_id'] = request_id

        if _MODULE_LOGS:
            result['logs'] = list(_MODULE_LOGS)

        return result
    except Exception:
        runtime_logs = runtime_capture.getvalue().strip() if runtime_capture is not None else ''
        _emit_log(runtime_logs, 'runtime', request_id)
        raise


def _run_streaming(service: ExpertImitateLearningService) -> None:
    ready_payload: Dict[str, Any] = {'status': 'ready'}
    if _MODULE_LOGS:
        ready_payload['logs'] = list(_MODULE_LOGS)
    _emit(ready_payload)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            _emit({'status': 'error', 'message': f'Invalid JSON payload: {error}'})
            continue

        action = message.get('action', 'predict')
        request_token = message.get('request_id')
        request_id = str(request_token) if request_token is not None else None

        if action == 'predict':
            if 'telemetry' not in message:
                error_payload: Dict[str, Any] = {'status': 'error', 'message': 'Missing telemetry payload'}
                if request_id is not None:
                    error_payload['request_id'] = request_id
                _emit(error_payload)
                continue

            telemetry_payload = message.get('telemetry')
            try:
                result = _run_prediction(service, telemetry_payload, request_id)
                _emit(result)
            except Exception as error:
                error_payload = {
                    'status': 'error',
                    'message': str(error),
                    'traceback': traceback.format_exc(),
                }
                if request_id is not None:
                    error_payload['request_id'] = request_id
                if _MODULE_LOGS:
                    error_payload['logs'] = list(_MODULE_LOGS)
                _emit(error_payload)
        elif action == 'shutdown':
            payload: Dict[str, Any] = {'status': 'shutdown'}
            if request_id is not None:
                payload['request_id'] = request_id
            _emit(payload)
            break
        elif action == 'ping':
            payload = {'status': 'pong'}
            if request_id is not None:
                payload['request_id'] = request_id
            _emit(payload)
        else:
            error_payload = {'status': 'error', 'message': f'Unknown action: {action}'}
            if request_id is not None:
                error_payload['request_id'] = request_id
            _emit(error_payload)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _emit({'status': 'error', 'message': 'Usage: run_expert_actions_prediction.py <model_json_path> [<telemetry_json_payload> | --stream]'} )
        return

    stream_mode = False
    if '--stream' in args:
        stream_mode = True
        args.remove('--stream')

    if not args:
        _emit({'status': 'error', 'message': 'Model path argument is required'})
        return

    model_path = Path(args[0]).expanduser().resolve()

    try:
        service = _prepare_service(model_path)
    except Exception as error:
        error_payload: Dict[str, Any] = {
            'status': 'error',
            'message': str(error),
            'traceback': traceback.format_exc(),
        }
        if _MODULE_LOGS:
            error_payload['logs'] = list(_MODULE_LOGS)
        _emit(error_payload)
        return

    if stream_mode:
        _run_streaming(service)
        return

    if len(args) < 2:
        _emit({'status': 'error', 'message': 'Usage: run_expert_actions_prediction.py <model_json_path> <telemetry_json_payload>'})
        return

    telemetry_payload_raw = args[1]

    try:
        result = _run_prediction(service, telemetry_payload_raw)
        _emit(result)
    except Exception as error:
        error_payload = {
            'status': 'error',
            'message': str(error),
            'traceback': traceback.format_exc(),
        }
        if _MODULE_LOGS:
            error_payload['logs'] = list(_MODULE_LOGS)
        _emit(error_payload)


if __name__ == '__main__':
    main()
