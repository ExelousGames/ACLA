import asyncio
import sys
import subprocess
from pathlib import Path
import argparse
import logging
from datetime import datetime
import faulthandler

faulthandler.enable()

# Add parent directory to path to allow imports
# parents[0] = scripts
# parents[1] = acla_ai_service (or /app in docker)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipelines.training.full_dataset import Full_dataset_TelemetryMLService
from app.pipelines.training.pipeline import prepare_training_data
from app.infra.config.pipeline import PipelineConfig

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

def confirm_step(step_name):
    while True:
        response = input(f"\nDo you want to execute {step_name}? (y/n) [y]: ").strip().lower()
        if response in ["", "y", "yes"]:
            logger.info("Confirmed execution for %s", step_name)
            return True
        elif response in ["n", "no"]:
            logger.info("Skipping execution for %s", step_name)
            return False
        else:
            log_message("Invalid input. Please enter 'y' or 'n'.", level=logging.WARNING)

async def main():
    steps = ["prepare_data", "annotate"]

    log_message("\nSelect start step:")
    for i, step in enumerate(steps):
        log_message(f"{i + 1}. {step}")
    
    while True:
        try:
            choice = input(f"\nEnter step number (1-{len(steps)}) [default 1]: ").strip()
            if not choice:
                start_step = steps[0]
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(steps):
                start_step = steps[idx]
                break
            else:
                log_message(f"Please enter a number between 1 and {len(steps)}", level=logging.WARNING)
        except ValueError:
            log_message("Invalid input. Please enter a number.", level=logging.WARNING)

    logger.info("Starting pipeline from step: %s", start_step)
    log_message("Initializing services...")
    
    pipeline_config = PipelineConfig()

    # Pass pipeline_config to service so they share the same cache keys
    service = Full_dataset_TelemetryMLService(logger=logger, pipeline_config=pipeline_config)

    try:
        start_index = steps.index(start_step)
    except ValueError:
        log_message(f"Invalid start step: {start_step}", level=logging.ERROR)
        return

    # Step 1: Prepare Training Data
    if start_index <= 0:
        if confirm_step("Step 1: Prepare Training Data"):
            log_message("\n" + "="*50)
            log_message(" Step 1: Prepare Training Data")
            log_message("="*50)

            result = await prepare_training_data(
                telemetry_store=service.telemetry_store,
                cache_config=service.cache_config,
                backend_service=service.backend_service,
                imitate_expert_feature_names=service._imitate_expert_feature_names,
                top_laps_count=1,
            )
            if not result.get("success"):
                log_message(f"Error in prepare_training_data: {result.get('error')}", level=logging.ERROR)
                return

            log_message("Step 1 completed successfully.")
        else:
            log_message("Skipping Step 1.")

    # Step 2: Run Segment Annotation App (also hosts the Training tab)
    if start_index <= 1:
        if confirm_step("Step 2: Run Segment Annotation App"):
            log_message("\n" + "="*50)
            log_message(" Step 2: Run Segment Annotation App")
            log_message("="*50)
            log_message("Launching Streamlit app for annotation + training...")
            log_message("Use the 'LLM Pipeline (Claude)' tab to draft/approve LLM data,")
            log_message("then the '🏋️ Training' tab to train the classifier, transformer,")
            log_message("and LLM (individually or via 'Run all').")
            log_message("Close the Streamlit app (Ctrl+C in terminal) when finished.")

            # Locate the annotation app script
            # We are in acla_ai_service/scripts/run_full_pipeline.py
            # App is in acla_ai_service/ui/segment_annotation_app.py
            app_path = Path(__file__).resolve().parents[1] / "ui" / "segment_annotation_app.py"

            if not app_path.exists():
                log_message(f"Error: Could not find segment_annotation_app.py at {app_path}", level=logging.ERROR)
                return

            try:
                # Run streamlit as a subprocess
                subprocess.run(
                    [sys.executable, "-m", "streamlit", "run", str(app_path)],
                    check=True
                )
            except KeyboardInterrupt:
                log_message("\nStreamlit app closed by user. Continuing...")
            except subprocess.CalledProcessError as e:
                # Streamlit might exit with non-zero if killed, but we want to continue if user is done
                log_message(f"\nStreamlit app exited with code {e.returncode}. Continuing...", level=logging.WARNING)

            log_message("Step 2 completed.")
        else:
            log_message("Skipping Step 2.")

    log_message("\n" + "="*50)
    log_message(" Pipeline Execution Completed")
    log_message(" (Training now lives in the Streamlit '🏋️ Training' tab.)")
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

    asyncio.run(main())
