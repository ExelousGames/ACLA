"""Split Labelme annotations, train segmentation, and upload an ONNX export.

Usage: python scripts/train_labelme.py path/to/annotations [--device 0]
An explicit annotation root is required; run --help for split and training options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["train-labelme", *sys.argv[1:]]))
