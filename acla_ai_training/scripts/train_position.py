"""Train the standalone position-regression CNN from scratch.

All image encoder and regression weights are initialized and trained together.
No pretrained or segmentation checkpoint is used.

The input folder and its subfolders are scanned for .position.json files holding
reference-lap XYZ, game, map name, normalized lap position, and signed
offset targets beside their images. Any stored centerline inputs and targets
are ignored. The model takes only reference-lap XYZ and predicts its signed
delta from the driver. Game/map names are one-hot encoded, with the
category order saved in each checkpoint for inference.
A positions.json index is also accepted. Individual frames are shuffled and split
when training starts, regardless of their source sessions.
Preparation aligns images to telemetry using Graphics_analyst_session_time.
No driver telemetry is read during training.
Validation reports mean 3D distance and direction errors grouped by actual
reference-lap offset length: <0.1, 0.1-0.25, 0.25-0.5, 0.5-1, 1-2, 2-5, >=5.
Ranges include their lower bound and exclude their upper bound, in source
coordinate units. Direction errors use degrees. Zero-length vectors are excluded
from direction averages, with valid counts reported. Empty averages show n/a.
Uses the first GPU by default; use --device cpu for CPU training.
Exports only signed position deltas in best.onnx, with best.onnx.json for inference, preserving
best.pt and last.pt for Python use. Uploads best.onnx to the backend by default;
use --no-upload to export locally without publishing
or --upload-name to change the backend name (default: track-position).

Usage: python scripts/train_position.py --manifest storage/session_downloads --output position-runs/run-1
Run --help for training options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["train-position", *sys.argv[1:]]))
