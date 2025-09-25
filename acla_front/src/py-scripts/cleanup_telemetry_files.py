#!/usr/bin/env python3
"""
Python script to clean up old telemetry files.
This script removes temporary telemetry files older than a specified age.

Usage: python cleanup_telemetry_files.py [max_age_hours]
"""

import sys
import os
import time
from pathlib import Path

def cleanup_telemetry_files(temp_dir="../session_recording/temp", max_age_hours=24):
    """
    Remove telemetry files older than max_age_hours.
    """
    try:
        temp_path = Path(temp_dir)
        
        if not temp_path.exists():
            print(f"Temp directory {temp_dir} does not exist")
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        removed_count = 0
        total_size_removed = 0
        
        # Clean up both .jsonl and .csv files (in case there are old CSV files)
        for pattern in ["telemetry_*.jsonl", "telemetry_*.csv", "acc_*.csv"]:
            for file_path in temp_path.glob(pattern):
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        removed_count += 1
                        total_size_removed += file_size
                        print(f"Removed old telemetry file: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}", file=sys.stderr)
        
        size_mb = total_size_removed / (1024 * 1024)
        print(f"Cleanup complete. Removed {removed_count} old telemetry files ({size_mb:.2f} MB freed).")
        
    except Exception as e:
        print(f"Error during cleanup: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    max_age_hours = 24  # Default to 24 hours
    
    if len(sys.argv) > 1:
        try:
            max_age_hours = float(sys.argv[1])
        except ValueError:
            print("Error: max_age_hours must be a number", file=sys.stderr)
            sys.exit(1)
    
    cleanup_telemetry_files(max_age_hours=max_age_hours)