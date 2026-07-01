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
CHUNK_EVENT = "chunk"
COMPLETE_EVENT = "complete"
ERROR_EVENT = "error"

def _emit(obj):
    """Emit a JSON line immediately (stdout is line-buffered in Electron integration)."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def read_telemetry_data(file_path):
    """Stream telemetry data with progress events then final completion.

    Protocol:
      {"type":"chunk","data":[...]} repeated
      {"type":"progress","read":<int>,"total":<int|null>,"bytesRead":<int>,"totalBytes":<int>} repeated
      {"type":"complete","data":[],"elapsed_ms":<int>} once at end
      {"type":"error","message":str} on failure (followed by empty complete)
    """
    start_time = time.time()
    try:
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            _emit({"type": COMPLETE_EVENT, "data": []})
            return

        file_size = file_path_obj.stat().st_size
        
        chunk_size = 2000 # Send in chunks of 2000 lines
        current_chunk = []
        
        read_lines = 0
        bytes_read = 0
        last_emit = 0.0
        progress_interval_sec = 0.1  # throttle progress events

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Approximation of bytes read (utf-8 encoding might vary but this is close enough for progress)
                line_len = len(line.encode('utf-8')) 
                bytes_read += line_len
                
                line_stripped = line.strip()
                if line_stripped:
                    try:
                        obj = json.loads(line_stripped)
                        current_chunk.append(obj)
                    except json.JSONDecodeError:
                        pass
                
                read_lines += 1
                
                if len(current_chunk) >= chunk_size:
                    _emit({"type": CHUNK_EVENT, "data": current_chunk})
                    current_chunk = []
                    
                    now = time.time()
                    if (now - last_emit) >= progress_interval_sec:
                        _emit({
                            "type": PROGRESS_EVENT, 
                            "read": read_lines, 
                            "total": None, 
                            "bytesRead": bytes_read, 
                            "totalBytes": file_size
                        })
                        last_emit = now

        # Emit remaining chunk
        if current_chunk:
            _emit({"type": CHUNK_EVENT, "data": current_chunk})

        # Final progress
        _emit({
            "type": PROGRESS_EVENT, 
            "read": read_lines, 
            "total": None, 
            "bytesRead": bytes_read, 
            "totalBytes": file_size
        })
        
        # Complete event with empty data (since we sent chunks)
        _emit({"type": COMPLETE_EVENT, "data": [], "elapsed_ms": int((time.time() - start_time) * 1000)})

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