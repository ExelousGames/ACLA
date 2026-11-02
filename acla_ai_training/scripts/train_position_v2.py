"""Train position_model_v2: pretrained ResNet-18 + spatial ConvLSTM + XYZ regression.

Uses the same prepared datasets, options, validation metrics, and ONNX export as
train_position.py. The original script continues to train the CNN/GRU model.
Training uses every frame unless --density-position-radius R is set. Density
filtering removes endpoints only when driver world XYZ is within R source units
of a retained endpoint in the same game/map and wrapped heading, pitch, roll
differences are each within --density-orientation-deg D (default: 5 degrees).
Larger thresholds remove more examples. A spatial index finds nearby positions;
--seed controls representative selection across all sessions before splitting
the selected endpoints into training and validation. Whole-session splitting
balances selected counts; at least two sessions per map must retain endpoints.
Both splits build temporal windows from their sessions' original frames,
preserving steps, order, and padding. Files are retained. Driver XYZ is selection
metadata, never a model input; filtering requires finite driver_position triples
in the prepared files. split.json saves thresholds, total candidate/selected
counts, selected paths, and full train_history/val_history for temporal runs.
Omit the radius to retain all frames.
V2 freezes all ImageNet ResNet-18 weights and BatchNorm state, training only the
channel projection, ConvLSTM, and regression head. ImageNet RGB normalization is in the model.
The first run downloads ResNet-18 weights unless cached; checkpoint loading and
export do not download weights. The default temporal window contains four frames.
Uploads best.onnx as position_model_v2; use --no-upload to export locally.

Usage: python scripts/train_position_v2.py --manifest storage/session_downloads --output position-runs/v2-run-1
Density: python scripts/train_position_v2.py --manifest storage/session_downloads --output position-runs/v2-density-1 --density-position-radius 0.5 --density-orientation-deg 5
Upload: python scripts/train_position_v2.py --upload-model position-runs/v2-run-1/best.onnx
Run --help for training options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["train-position-v2", *sys.argv[1:]]))
