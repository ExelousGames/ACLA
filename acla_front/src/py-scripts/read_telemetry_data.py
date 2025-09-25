#!/usr/bin/env python3
"""
Python script to read all telemetry data from a JSONL file.
This script is called from the Electron frontend to read telemetry data
for uploading or analysis.

Usage: python read_telemetry_data.py <file_path>
"""

import sys
import json
import os
import time
from pathlib import Path

PROGRESS_EVENT = "progress"
COMPLETE_EVENT = "complete"
ERROR_EVENT = "error"

def _emit(obj):
    """Emit a JSON line immediately (stdout is line-buffered in Electron integration)."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def read_telemetry_data(file_path):
    """Stream telemetry data with progress events then final completion.

    Protocol:
      {"type":"progress","read":<int>,"total":<int|null>} repeated
      {"type":"complete","data":[...]} once at end
      {"type":"error","message":str} on failure (followed by empty complete)
    """
    start_time = time.time()
    try:
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            _emit({"type": COMPLETE_EVENT, "data": []})
            return

        # Attempt to get total lines quickly (could be large; fallback if too big)
        total_lines = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as tf:
                for _ in tf:
                    total_lines += 1
        except Exception:
            total_lines = None  # Unknown

        data = []
        read_lines = 0
        last_emit = 0.0
        progress_interval_sec = 0.1  # throttle progress events
        emit_every_n = 200  # also emit every N lines regardless of time

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Skip invalid JSON line but report once per issue
                        _emit({"type": "warn", "line": line_number, "message": f"Invalid JSON: {e}"})
                        continue
                read_lines += 1
                now = time.time()
                if read_lines % emit_every_n == 0 or (now - last_emit) >= progress_interval_sec:
                    _emit({"type": PROGRESS_EVENT, "read": read_lines, "total": total_lines})
                    last_emit = now

        # Final progress emit (ensure UI sees 100%)
        _emit({"type": PROGRESS_EVENT, "read": read_lines, "total": total_lines})
        _emit({"type": COMPLETE_EVENT, "data": data, "elapsed_ms": int((time.time() - start_time) * 1000)})

    except Exception as e:
        _emit({"type": ERROR_EVENT, "message": str(e)})
        _emit({"type": COMPLETE_EVENT, "data": []})
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_telemetry_data.py <file_path>", file=sys.stderr)
        sys.exit(1)
    
    file_path = sys.argv[1]
    read_telemetry_data(file_path)