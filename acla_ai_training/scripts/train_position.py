"""Train the original position_model: CNN/GRU offset regression from scratch.

The CNN learns spatial image features from position targets, with a GRU for
temporal windows and a signed XYZ regression head. All parameters train from
scratch. Use scripts/train_position_v2.py for pretrained ResNet-18 + ConvLSTM.
By default, a causal four-frame window predicts the final frame's offset.
Use --sequence-length N to change the window; 1 processes only the current frame.
Windows use numeric image suffixes (frame_000123.jpg), oldest first, within the
same session/video directory. They contain only the current and earlier prepared
frames, repeating the first frame when history is short. Sampling cadence should
match inference; time gaps are not inferred from frame numbers. Batch counts
windows, so image memory/compute grows with sequence length; lower --batch if needed.

The input folder and its subfolders are scanned for .position.json files holding
reference-lap XYZ, world orientation, game, map name, normalized lap position, and signed
car-local offset targets beside their images. Any stored centerline inputs and
targets are ignored. The model keeps reference-lap world XYZ as input and predicts
the signed offset from the driver to the reference in local XYZ: right, forward,
up. Preparation applies inverse world rotation (heading, pitch, roll in radians)
to the reference-minus-driver vector, then negates ACC's left-positive local X
to make X point right. Orientation is a required input: world heading, pitch, roll
in radians, saved as orientation in each position file. The model converts these
angles to sine/cosine internally. Regenerate datasets missing orientation with
scripts/prepare_position_dataset.py and retrain older checkpoints.
The preparation workflow currently supports ACC only and skips iRacing sessions.
Game/map names are one-hot encoded, with the
category order saved in each checkpoint for inference.
A positions.json index is also accepted. After image extraction, whole sessions
from each game/map are assigned to validation with a frame count as close as
possible to 20% by default, with the rest used for training. The seed randomizes
equally balanced session choices. Every map appears in both sets, with at least
two sessions required per map. Sessions never cross splits; windows are built
within each split before training shuffles them. The maps, image assignments,
and sequence length are saved in split.json.
Training uses every frame unless --density-position-radius R is set. Density
filtering keeps coverage of driver world XYZ and orientation separately per
game/map: an endpoint is redundant only within radius R (source units) of a
retained endpoint and within --density-orientation-deg D in each wrapped heading,
pitch, roll angle (default D=5 degrees). Larger thresholds remove more examples.
A spatial index avoids all-pairs comparisons; --seed controls representative
selection. Only training endpoints are filtered, after building full temporal
windows: original history, order, padding, and validation are preserved. No files
are deleted. Driver XYZ is selection metadata, never a model input. Filtering
requires finite driver_position triples in the prepared files. split.json saves
the thresholds, candidate/selected counts, selected paths, and full train_history
for temporal runs. Omit --density-position-radius to keep every training frame.
Preparation aligns images to telemetry using Graphics_analyst_session_time.
Training reads the prepared orientation values, without reopening source telemetry.
Before letterboxing, each training-frame load has a 50% chance of HUD
CoarseDropout: 1-9 randomly selected panels are fully covered with random RGB
colors or pixel noise, using regions normalized from a 1252 x 720 annotation.
Reference inputs, orientation, context, and XYZ targets stay unchanged. Validation and
inference use original frames; obscured versions are generated on each load.
Training uses mean component Smooth L1 between predicted and target XYZ offsets.
Each epoch prints training and validation MAE in local coordinates, in source units. Training MAE
averages absolute XYZ errors over all frames during the training pass and is
saved as train_mae in results.json and checkpoint metrics.
Validation shows a separate section for each game/map, with frame count and MAE.
Each section reports mean 3D distance and direction errors grouped by actual
reference-lap offset length: <0.1, 0.1-0.25, 0.25-0.5, 0.5-1, 1-2, 2-5, >=5.
Per-map metrics are saved as val_by_map alongside overall metrics in results.json
and checkpoints. Overall validation MAE selects the best checkpoint.
Each map also saves val_errors_by_normalized_position: 20 sections covering 5%
of the lap each, with normalized start/end, frame count, MAE, per-axis MAE, and
mean 3D distance error. Sections include their start and exclude their end,
except the final section includes 1.0. weakness_rank orders populated sections
by descending mean distance error (1 is weakest); empty errors/ranks are null.
Ranges include their lower bound and exclude their upper bound, in source
coordinate units. Direction errors use degrees. Zero-length vectors are excluded
from direction averages, with valid counts reported. Empty averages show n/a.
Near zero, a large direction error can accompany a small distance error.
Uses the first GPU by default; use --device cpu for CPU training.
Temporal ONNX uses schemaVersion 7: images [1,T,3,H,W], references [1,T,1,3],
context [1,T,C], and orientation [1,T,3], with the history order/padding contract in best.onnx.json.
The caller buffers frames; GRU state resets per window. Clear history on session,
recording, game/map changes or seeks. Single-frame exports use schemaVersion 6 and orientation [1,3].
Exports only car-local position deltas in best.onnx, preserving
best.pt and last.pt for Python use. Uploads best.onnx to the backend as
track-position by default; use --no-upload to export locally without publishing.
Use --upload-model path/to/best.onnx to skip training and publish an existing
ONNX model under its saved model name; --manifest and --output are not needed.

Usage: python scripts/train_position.py --manifest storage/session_downloads --output position-runs/run-1
Density: python scripts/train_position.py --manifest storage/session_downloads --output position-runs/density-1 --density-position-radius 0.5 --density-orientation-deg 5
Upload: python scripts/train_position.py --upload-model position-runs/run-1/best.onnx
Run --help for training options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.image_segmentation.__main__ import main


if __name__ == "__main__":
    sys.exit(main(["train-position", *sys.argv[1:]]))
