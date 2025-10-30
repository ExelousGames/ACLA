#!/usr/bin/env python3
"""Append telemetry samples to JSONL via a persistent streaming session.

Usage:
    python append_telemetry_data.py <file_path> --stream

The script must be launched in streaming mode. It accepts newline-delimited
JSON commands from stdin and replies with acknowledgements that include any
supplied ``request_id``.

Supported actions:
    {"action": "append", "payload": {"data": <any>}, "request_id": <string>}
    {"action": "ping", "request_id": <string>}
    {"action": "shutdown", "request_id": <string>}

Each append returns {"status": "ok", "written": 1}. Errors surface with
{"status": "error", "message": str}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _ensure_directory(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(file_path: Path, data: Any) -> None:
    _ensure_directory(file_path)
    with file_path.open('a', encoding='utf-8') as handle:
        json.dump(data, handle, separators=(",", ":"))
        handle.write('\n')
        handle.flush()


def _extract_append_data(message: Dict[str, Any]) -> Any:
    payload = message.get('payload')
    if isinstance(payload, dict) and 'data' in payload:
        return payload['data']
    if 'data' in message:
        return message['data']
    raise ValueError('Missing "data" field in append command')


def _handle_stream(file_path: Path) -> None:
    ready_payload: Dict[str, Any] = {'status': 'ready'}
    _emit(ready_payload)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message: Dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as error:
            _emit({'status': 'error', 'message': f'Invalid JSON payload: {error}'})
            continue

        action = message.get('action', 'append')
        request_token = message.get('request_id')
        request_id = str(request_token) if request_token is not None else None

        try:
            if action == 'append':
                append_data = _extract_append_data(message)
                _append_jsonl(file_path, append_data)
                response: Dict[str, Any] = {'status': 'ok', 'written': 1}
                if request_id is not None:
                    response['request_id'] = request_id
                _emit(response)
            elif action == 'ping':
                response = {'status': 'pong'}
                if request_id is not None:
                    response['request_id'] = request_id
                _emit(response)
            elif action == 'shutdown':
                response = {'status': 'shutdown'}
                if request_id is not None:
                    response['request_id'] = request_id
                _emit(response)
                break
            else:
                raise ValueError(f'Unknown action: {action}')
        except Exception as error:  # pylint: disable=broad-except
            error_payload: Dict[str, Any] = {
                'status': 'error',
                'message': str(error),
            }
            if request_id is not None:
                error_payload['request_id'] = request_id
            _emit(error_payload)


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('Usage: python append_telemetry_data.py <file_path> --stream', file=sys.stderr)
        return 1

    if '--stream' not in args:
        print('Streaming mode is required. Launch with the --stream flag.', file=sys.stderr)
        return 1

    args = [arg for arg in args if arg != '--stream']

    if not args:
        print('Usage: python append_telemetry_data.py <file_path> --stream', file=sys.stderr)
        return 1

    target_path = Path(args[0]).expanduser()
    _handle_stream(target_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())