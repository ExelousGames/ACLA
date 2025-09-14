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
from pathlib import Path

def read_telemetry_data(file_path):
    """
    Read all telemetry data from a JSONL file and return as JSON array.
    Returns empty array if file doesn't exist.
    """
    try:
        file_path_obj = Path(file_path)
        
        # Check if file exists
        if not file_path_obj.exists():
            print(json.dumps([]))
            return
        
        # Read all lines from the JSONL file
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_number}: {e}", file=sys.stderr)
                        continue
        
        # Output the data as a JSON array
        print(json.dumps(data))
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        print(json.dumps([]))  # Return empty array on error
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_telemetry_data.py <file_path>", file=sys.stderr)
        sys.exit(1)
    
    file_path = sys.argv[1]
    read_telemetry_data(file_path)