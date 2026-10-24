"""Restart Labelme; headless Linux containers serve the editor at localhost:6080.

Usage: python scripts/open_labelme.py [path/to/images] [--labels path/to/labels.txt] [--browser]
Click Custom Annotate to add editable polygons from the newest backend model.
Click YOLO26x Annotate to add predicted polygon regions labeled "other".
Relabel those regions as needed using Edit Label before training.
Both reuse local model weights before downloading; Custom checks the latest version.
Use Ctrl+N to create region polygons.
Use Ctrl+L to create polylines.
Use Ctrl+Shift+L to initialize a polygon from a polyline.
Corners are rounded with multiple editable points along both edges.
Set its total width in pixels with Edit > Set Polyline Polygon Width.
Use the left/right variants of curb, grass, fence, sand, Outfield asphalt road,
racing track white line, forest, buildings, and pit lane when labeling a specific side.
The original labels remain available.
Track, grass, fence, curb, sand, sky, and racing track white line regions may
continue behind car or car pack polygons where their hidden extent is clear.
Annotate those regions continuously and annotate the cars separately.
Enable View > Position Context for a separate top-down map window. Prepared
.position.json files supply the centerline, reference lap and driver trajectory
across 100 frames before and after the current image; adjust the range and axes
in that window. The current frame's world X/Y/Z coordinates appear below the map
for all three paths. It follows image navigation without changing annotations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["annotate", *sys.argv[1:]]))
