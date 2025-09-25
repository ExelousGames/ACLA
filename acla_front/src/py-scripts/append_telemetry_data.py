#!/usr/bin/env python3
"""
Python script to append telemetry data to a JSONL file.
This script is called from the Electron frontend to write telemetry data
without keeping it all in memory.

Usage: python append_telemetry_data.py <file_path> <json_data>
"""

import sys
import json
import os
from pathlib import Path

def append_telemetry_data(file_path, json_data):
    """
    Append telemetry data to a JSONL file.
    Creates the directory and file if they don't exist.
    """
    try:
        # Parse the JSON data
        data = json.loads(json_data)
        
        # Ensure the directory exists
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Append the data as a single line (JSONL format)
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))
            f.write('\n')
        
        print(f"Successfully appended data to {file_path}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON data - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python append_telemetry_data.py <file_path> <json_data>", file=sys.stderr)
        sys.exit(1)
    
    file_path = sys.argv[1]
    json_data = sys.argv[2]
    
    append_telemetry_data(file_path, json_data)