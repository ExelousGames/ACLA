"""Train the standalone position-regression CNN from scratch.

All image encoder and regression weights are initialized and trained together.
No pretrained or segmentation checkpoint is used.

The input folder and its subfolders are scanned for .position.json files holding
reference-lap XYZ, game, map name, normalized lap position, and signed
offset targets beside their images. Any stored centerline inputs and targets
are ignored. The model takes only reference-lap XYZ and predicts its signed
delta from the driver. Game/map names are one-hot encoded, with the
category order saved in each checkpoint for inference.
A positions.json index is also accepted. After image extraction, whole sessions
from each game/map are assigned to validation with a frame count as close as
possible to 20% by default, with the rest used for training. The seed randomizes
equally balanced session choices. Every map appears in both sets, with at least
two sessions required per map. Sessions never cross splits; frames are shuffled
within each split. The maps and image assignments are saved in split.json.
Preparation aligns images to telemetry using Graphics_analyst_session_time.
No driver telemetry is read during training.
Training adds mean component Smooth L1 and a mean direction penalty 1-cos(angle).
The direction penalty is weighted by min(target_length / 0.1, 1)^2, fading near
zero while keeping full weight at lengths >=0.1 in source coordinate units.
Zero target offsets are excluded from the direction average; vector norms are
clamped to 1e-8 for numerical stability.
Each epoch prints training and validation MAE in raw coordinates. Training MAE
averages absolute XYZ errors over all frames during the training pass and is
saved as train_mae in results.json and checkpoint metrics.
Validation shows a separate section for each game/map, with frame count and MAE.
Each section reports mean 3D distance and direction errors grouped by actual
reference-lap offset length: <0.1, 0.1-0.25, 0.25-0.5, 0.5-1, 1-2, 2-5, >=5.
Per-map metrics are saved as val_by_map alongside overall metrics in results.json
and checkpoints. Overall validation MAE selects the best checkpoint.
Ranges include their lower bound and exclude their upper bound, in source
coordinate units. Direction errors use degrees. Zero-length vectors are excluded
from direction averages, with valid counts reported. Empty averages show n/a.
Near zero, a large direction error can accompany a small distance error.
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
