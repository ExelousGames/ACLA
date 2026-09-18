"""Start Labelme with the project's polygon annotation settings.

Usage: python scripts/open_labelme.py path/to/images [--labels path/to/labels.txt]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["annotate", *sys.argv[1:]]))
