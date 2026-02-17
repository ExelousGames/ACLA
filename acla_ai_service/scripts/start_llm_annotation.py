#!/usr/bin/env python3
"""
Launcher script for the LLM Dataset Annotation Streamlit app.
Run this script to start the UI: python scripts/start_llm_annotation.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Calculate paths
    # This script is in acla_ai_service/scripts/
    script_dir = Path(__file__).resolve().parent
    # Project root is acla_ai_service/
    project_root = script_dir.parent
    # UI script is in acla_ai_service/ui/
    ui_script = project_root / "ui" / "llm_dataset_annotation.py"
    
    if not ui_script.exists():
        print(f"Error: Could not find UI script at: {ui_script}")
        sys.exit(1)
        
    print(f"Starting LLM Annotation UI...")
    print(f"Working Directory: {project_root}")
    print(f"Target Script: {ui_script}")
    
    # Build command: python -m streamlit run [script_path] [args]
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_script)]
    
    # Forward any command line arguments to streamlit
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
        
    try:
        # Run streamlit, with CWD set to project root so relative paths (like logs/) work correctly
        subprocess.run(cmd, cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("\nShutdown requested. Exiting...")
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
