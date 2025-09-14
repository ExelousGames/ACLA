#!/usr/bin/env python3
"""
Python script to delete a telemetry data file.
This script is called from the Electron frontend to clean up
temporary telemetry files after upload or cancellation.

Usage: python delete_telemetry_file.py <file_path>
"""

import sys
import os
from pathlib import Path

def delete_telemetry_file(file_path):
    """
    Delete a telemetry file safely.
    """
    try:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            print(f"File {file_path} does not exist, nothing to delete")
            return
        
        # Verify it's a telemetry file (safety check)
        if not (file_path_obj.name.startswith('telemetry_') and file_path_obj.suffix == '.jsonl'):
            print(f"Error: File {file_path} doesn't appear to be a telemetry file", file=sys.stderr)
            sys.exit(1)
        
        # Delete the file
        file_path_obj.unlink()
        print(f"Successfully deleted telemetry file: {file_path}")
        
    except Exception as e:
        print(f"Error deleting telemetry file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_telemetry_file.py <file_path>", file=sys.stderr)
        sys.exit(1)
    
    file_path = sys.argv[1]
    delete_telemetry_file(file_path)
