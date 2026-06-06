import sys
import subprocess
from pathlib import Path
import argparse
import logging
from datetime import datetime
import faulthandler

faulthandler.enable()

logger = logging.getLogger("run_full_pipeline")

def setup_logging(log_file: Path) -> None:
    """Configure logging to write to both the specified file and console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

def log_message(message: str, level: int = logging.INFO) -> None:
    """Log a message to configured handlers."""
    if logger.handlers:
        logger.log(level, message)
    else:
        print(message)

def main() -> None:
    log_message("\n" + "="*50)
    log_message(" Segment Annotation Pipeline")
    log_message("="*50)
    log_message("Launching Streamlit app for data preparation, annotation, and training...")
    log_message("Run Data Preparation from the left-most Pipeline card to download,")
    log_message("process, and enrich telemetry before selecting annotation sources.")
    log_message("Close the Streamlit app (Ctrl+C in terminal) when finished.")

    app_path = Path(__file__).resolve().parents[1] / "ui" / "segment_annotation_app.py"

    if not app_path.exists():
        log_message(f"Error: Could not find segment_annotation_app.py at {app_path}", level=logging.ERROR)
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True
        )
    except KeyboardInterrupt:
        log_message("\nStreamlit app closed by user. Continuing...")
    except subprocess.CalledProcessError as e:
        log_message(f"\nStreamlit app exited with code {e.returncode}. Continuing...", level=logging.WARNING)

    log_message("\n" + "="*50)
    log_message(" Pipeline Execution Completed")
    log_message(" (Data preparation and training now live in Streamlit.)")
    log_message("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ACC full telemetry pipeline")
    parser.add_argument("--log-file", type=str, help="Optional path to a log file. Defaults to logs/full_pipeline_<timestamp>.log")
    parser.add_argument("--log-dir", type=str, help="Directory to store generated log file when --log-file is not provided")
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
    else:
        default_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else Path(__file__).resolve().parents[1] / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = default_dir / f"full_pipeline_{timestamp}.log"

    setup_logging(log_path)
    logger.info("Pipeline logs will be written to %s", log_path)

    main()
