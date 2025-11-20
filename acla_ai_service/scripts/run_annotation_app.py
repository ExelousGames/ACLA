"""Convenience script for launching the Telemetry Prompt Annotation UI."""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def main():
    """Launch the Streamlit annotation app."""
    
    parser = argparse.ArgumentParser(description="Launch the Telemetry Prompt Annotation UI.")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="",
        help="Path to the JSONL dataset file to annotate. Can be set via ANNOTATION_DATASET_PATH.",
    )
    parser.add_argument(
        "--dataset-dir",
        dest="dataset_dir",
        default="",
        help="Directory containing datasets. Can be set via ANNOTATION_DATASET_DIR.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve().parents[1] / "ui" / "telemetry_prompt_annotation_app.py"
    if not script_path.exists():
        print(f"Error: Annotation UI script not found at {script_path}")
        sys.exit(1)

    print(f"Launching Telemetry Annotation UI from {script_path}...")
    
    env = os.environ.copy()
    if args.dataset_path:
        env["ANNOTATION_DATASET_PATH"] = args.dataset_path
    if args.dataset_dir:
        env["ANNOTATION_DATASET_DIR"] = args.dataset_dir
    
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", str(script_path),
                "--server.address=0.0.0.0",
                "--server.headless=true"
            ],
            env=env,
            check=True
        )
    except KeyboardInterrupt:
        print("\nAnnotation UI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Annotation UI exited with error: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()

