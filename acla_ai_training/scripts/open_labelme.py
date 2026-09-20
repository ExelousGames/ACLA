"""Start Labelme; headless Linux containers serve the editor at localhost:6080.

Usage: python scripts/open_labelme.py [path/to/images] [--labels path/to/labels.txt] [--browser]
Use Create LineStrip for left_boundary/right_boundary polylines; Ctrl+N for region polygons.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["annotate", *sys.argv[1:]]))
