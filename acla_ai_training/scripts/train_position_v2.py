"""Train position_model_v2: pretrained ResNet-18 + spatial ConvLSTM + XYZ regression.

Uses the same prepared datasets, options, validation metrics, and ONNX export as
train_position.py. The original script continues to train the CNN/GRU model.
V2 fine-tunes all encoder, ConvLSTM, and regression parameters while keeping
pretrained BatchNorm statistics fixed. ImageNet RGB normalization is in the model.
The first run downloads ResNet-18 weights unless cached; checkpoint loading and
export do not download weights. The default temporal window contains four frames.
Uploads best.onnx as position_model_v2; use --no-upload to export locally.

Usage: python scripts/train_position_v2.py --manifest storage/session_downloads --output position-runs/v2-run-1
Upload: python scripts/train_position_v2.py --upload-model position-runs/v2-run-1/best.onnx
Run --help for training options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["train-position-v2", *sys.argv[1:]]))
