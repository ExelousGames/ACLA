import os
import sys
import subprocess
from pathlib import Path

def main():
    # Assuming this script is in acla_ai_service/scripts/
    # And the app is in acla_ai_service/ui/
    script_path = Path(__file__).resolve().parents[1] / "ui" / "llm_training_app.py"
    if not script_path.exists():
        print(f"Error: Training UI script not found at {script_path}")
        sys.exit(1)

    print(f"Launching LLM Training UI from {script_path}...")
    
    env = os.environ.copy()
    
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
        print("\nTraining UI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Training UI exited with error: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
