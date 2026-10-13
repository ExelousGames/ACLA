"""Split nested Labelme annotations, save the sample lists, and start training.

Usage: python scripts/train_labelme.py [path/to/annotations] [--device 0]
Defaults to storage/annotation_images; run --help for split and training options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["train-labelme", *sys.argv[1:]]))
