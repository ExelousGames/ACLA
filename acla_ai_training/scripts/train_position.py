"""Train the local YOLO position-regression extension.

The input folder and its subfolders are scanned for .position.json files holding
reference-lap/centerline XYZ, game, map name, normalized lap position, and signed
offset targets beside their images. Game/map names are one-hot encoded, with the
category order saved in each checkpoint for inference.
A positions.json index is also accepted. Individual frames are shuffled and split
when training starts, regardless of their source sessions.
Preparation aligns images to telemetry using Graphics_analyst_session_time.
No driver telemetry is read during training.
Uses the first GPU by default; use --device cpu for CPU training.
Uploads best.pt to the backend by default; use --no-upload for local-only runs
or --upload-name to change the backend name (default: track-position).

Usage: python scripts/train_position.py --manifest storage/session_downloads --model best.pt --output position-runs/run-1
Run --help for training options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["train-position", *sys.argv[1:]]))
