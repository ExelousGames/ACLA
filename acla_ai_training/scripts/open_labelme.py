"""Restart Labelme; headless Linux containers serve the editor at localhost:6080.

Usage: python scripts/open_labelme.py [path/to/images] [--labels path/to/labels.txt] [--browser]
Use Ctrl+N to create region polygons.
Use Ctrl+L to create polylines.
Use Ctrl+Shift+L to initialize a polygon from a polyline.
Set its total width in pixels with Edit > Set Polyline Polygon Width.
Track, grass, fence, curb, sand, sky, and racing track white line regions may
continue behind car or car pack polygons where their hidden extent is clear.
Annotate those regions continuously and annotate the cars separately.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["annotate", *sys.argv[1:]]))
