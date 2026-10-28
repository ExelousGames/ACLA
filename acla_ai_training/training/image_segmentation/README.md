# Racing-image region and boundary segmentation

This local Python workflow uses Labelme to draw region polygons and left/right
boundary polylines, then trains Ultralytics YOLO instance segmentation on them.
Run commands from `acla_ai_training/`.

## Install

Labelme and Ultralytics are included in the training workspace's common
requirements. All three training Docker builds also include a virtual desktop
and noVNC for browser access. From the repository root, rebuild and recreate the
training container so the dependencies and browser port are available:

```bash
docker compose --env-file .dev.env --env-file .env.secrets \
  -f docker-compose.dev.yaml -f docker-compose.cpu.yaml up -d --build --no-deps ai_training
```

Use your existing NVIDIA or AMD override instead of `docker-compose.cpu.yaml`
when applicable. A container restart alone does not install dependencies or add
the port mapping.

For annotation outside Docker, install Labelme in a desktop Python 3.11+ environment:

```bash
python -m pip install -r training/image_segmentation/requirements-labelme.txt
```

Labelme is a Qt desktop application. The launcher automatically serves it through
noVNC on Linux when no display is set; local desktop sessions open a normal
window. `--browser` forces noVNC even when a display is set. Labelme 6 is used for
Python 3.11 compatibility; the launcher uses its command-line interface.

## Draw polygons and boundary polylines

Put screenshots or extracted video frames in a folder with any number of nested
subfolders. The training launcher mixes individual annotated images from all
folders before splitting them into training and validation samples.

```text
storage/annotation_images/
  session_a/frame_001.png
  session_b/frame_001.png
```

Launch Labelme in the training container:

```bash
docker exec -it acla_ai_training_c python3 /app/scripts/open_labelme.py
```

Open [Labelme in your browser](http://localhost:6080/vnc.html?autoconnect=true&resize=remote),
then click **Open Dir** to choose a folder. The file picker browses the container:
host `acla_ai_training/storage/...` is `/app/storage/...`. Bind-mount other host
folders before selecting them. Saved annotation JSON files persist beside the
images in the mounted folder.

Close Labelme or press Ctrl+C in the terminal to stop its virtual desktop and
browser server. Closing the browser tab only disconnects the viewer. Run one
Labelme browser session per container. It uses port 6080 independently of the
Streamlit UI on 8501. The unauthenticated desktop is published only on
`127.0.0.1`; keep that host binding. For a remote Docker host, forward port 6080
over SSH (`ssh -L 6080:127.0.0.1:6080 user@docker-host`).

You can also pass a parent folder directly. Labelme scans all its nested session
folders recursively and shows the images in one file list:

```bash
docker exec -it acla_ai_training_c python3 /app/scripts/open_labelme.py \
  /app/storage/annotation_images
```

On a local desktop, run
`python scripts/open_labelme.py [path/to/images]` from `acla_ai_training/`.

In Labelme, press **Ctrl+N** (Create Polygons), click along the region boundary,
and double-click to finish. Choose a label, then **Ctrl+S** to save. Use **D** / **A**
for the next / previous image. JSON annotations are saved beside the images.

For frames from `prepare_position_dataset.py`, enable **View > Position Context**
to open a separate floating top-down map window. It follows the current image and
reads the adjacent `.position.json` files in frame order, showing the **centerline**,
**reference lap**, and **driver trajectory** with dots at the current frame.
The default range is **100 prepared frames before and after**; change **Frames
before / after** in the map window to adjust it. Only the current session and video
folder are included. Invalid or unavailable neighboring frames leave gaps in the
paths; an image without position data clears the map.

The **Axes** selector defaults to **X / Z**; choose the horizontal plane matching
your saved coordinate convention. These are world-coordinate paths, independent
of the image's camera perspective. The map never adds annotation shapes or changes
the position files. Close it or toggle **View > Position Context** to hide it;
Labelme remembers its window visibility and layout between launches.

Click **Custom Annotate** in the toolbar (or **Edit > Custom Annotate**) to add
polygons for the current image using the newest segmentation ONNX model uploaded
to the backend. Each click compares the latest model ID, size, and SHA-256 with
the local cache before downloading. Matching weights are reused; missing or
corrupt weights are downloaded, verified, and cached under
`storage/image_segmentation/backend_models/`. Training
must have uploaded a model first. This uses the same backend URL and AI service
credentials as training uploads; for a local desktop, also set
`PYTHONPATH=../acla_ai_service` so the shared backend client can be imported.
Local installations need Ultralytics, ONNX Runtime, and Labelme. Republish an older
`.pt` backend model with the `upload` command below to export it as ONNX.

Click **YOLO26x Annotate** in the same toolbar or Edit menu to use pretrained
`yolo26x-seg.pt` without the backend. It checks for local weights in
`storage/image_segmentation/pretrained/`, the current working directory, and
Ultralytics' configured weights directory before downloading to the pretrained
cache. Subsequent clicks reuse the loaded model. Detected regions are added as
polygons labeled `other`; YOLO's class names are not assigned. Select a generated
polygon and use **Edit Label** to relabel it as needed before exporting or
training. Regions can be saved and reopened to finish labeling later.
This pretrained model does not know custom track regions; use **Custom Annotate**
for those.

Generated contours are simplified with Douglas-Peucker using a two-pixel
tolerance to reduce the number of editable points. If that would collapse a
small or thin region, only collinear points are removed.

Existing shapes are preserved, and predictions are added as one undoable edit.
Review the generated polygons with **Edit Shapes**; the normal auto-save setting
also applies to generated annotations. A custom model's labels must be present
in the editor's labels file (use `--labels` when needed). Backend or prediction errors
leave the current annotations unchanged.

For an open polyline, press **Ctrl+L**, click along the line, then double-click to
finish and choose its label. Polylines use Labelme's default display style.
Polygons remain region annotations.

To initialize a polygon from a centerline, choose **Polygon from Polyline**
(**Ctrl+Shift+L**), click along the centerline, and double-click to finish. Bends
are rounded locally, with multiple vertices along the inner and outer edges to
preserve the stroke width. Redundant vertices are removed to keep the outline
easy to edit. Move the generated vertices independently with
**Edit Shapes**. Set the total width with **Edit > Set Polyline Polygon Width…**
(default **15 image pixels**, independent of zoom). Rounding is limited by the
adjacent segment lengths, and the polygon is clipped to the image edges.
Consecutive duplicate points are ignored. Choose the label as usual; the saved
annotation is a polygon.

The class IDs follow the order in `labels.txt`:

| ID | Label | Region |
| --- | --- | --- |
| 0 | track | Full track surface within the image, including reasonably inferred parts hidden by cars |
| 1 | curb | Full curb regions, including reasonably inferred parts hidden by cars or car packs |
| 2 | grass | Full grass regions, including reasonably inferred parts hidden by cars or car packs |
| 3 | car | One polygon per visible car |
| 4 | other | Other regions you explicitly want the model to identify |
| 5 | fence | Full fence regions, including reasonably inferred parts hidden by cars or car packs |
| 6 | car pack | Regions annotated as a pack of cars |
| 7 | sand | Full sand regions, including reasonably inferred parts hidden by cars or car packs |
| 8 | Outfield asphalt road | Visible asphalt road regions outside the track |
| 9 | sky | Full sky regions, including reasonably inferred parts hidden by cars or car packs |
| 10 | car interior | Visible car interior regions |
| 11 | racing track white line | Full white line markings, including reasonably inferred parts hidden by cars or car packs |
| 12 | forest | Visible forest regions |
| 13 | buildings | Visible building regions |
| 14 | overhead truss structure | Visible overhead truss structures |
| 15 | curb left | Left-side curb regions |
| 16 | curb right | Right-side curb regions |
| 17 | grass left | Left-side grass regions |
| 18 | grass right | Right-side grass regions |
| 19 | fence left | Left-side fence regions |
| 20 | fence right | Right-side fence regions |
| 21 | sand left | Left-side sand regions |
| 22 | sand right | Right-side sand regions |
| 23 | outfield asphalt road left | Left-side asphalt road regions outside the track |
| 24 | outfield asphalt road right | Right-side asphalt road regions outside the track |
| 25 | racing track white line left | Left-side white line markings |
| 26 | racing track white line right | Right-side white line markings |
| 27 | forest left | Left-side forest regions |
| 28 | forest right | Right-side forest regions |
| 29 | buildings left | Left-side building regions |
| 30 | buildings right | Right-side building regions |

The left/right variants are appended so existing labels keep their class IDs.
Choose a side-specific label when annotating a particular side of the track from
the driver's perspective. The original labels remain available for unsided regions
and existing annotations. Side-specific variants follow the same visibility and
occlusion policy as their original region class below.

For `track`, `grass`, `fence`, `curb`, `sand`, `sky`, and `racing track white line`,
trace the continuous outer boundaries, including portions behind a `car` or
`car pack` where the region's continuation is reasonably clear. Do not cut holes
or split these polygons around cars or car packs. This is amodal segmentation:
each mask describes the region's extent, including occupied areas. For `track`,
this also applies to other foreground objects. Annotate cars and car packs
separately; their polygons may overlap these region polygons. Use nearby video
frames to resolve uncertain boundaries when possible, and avoid inventing hidden
turns or edges without supporting evidence.

For the other region classes, trace visible boundaries and draw disconnected visible
pieces as separate polygons. Annotate all instances of the chosen classes in each
image. Unannotated pixels are background; `other` is an explicit class, not an
automatic background label. Each shape becomes a separate instance. Holes and
Labelme group IDs are not merged. Use polygons for regions; rectangles and AI mask
shapes are not supported by this exporter.

Apply this policy consistently to training and validation images. Review existing
visible-only annotations of these regions and prepare a new dataset after revising them;
training does not automatically fill missing sections in the source polygons.

To add specific regions such as gravel or barriers, append labels to a copy of
`labels.txt` and pass that file with `--labels path/to/labels.txt` to both
`annotate` and `prepare`. Preserve the class order for existing datasets/models.
Keep the original images alongside their JSON files (image bytes are not embedded).

## Split and start training

Run the companion launcher with an explicit annotation directory in the training container:

```bash
docker exec -it acla_ai_training_c python3 /app/scripts/train_labelme.py /app/storage/annotation_images
```

Or run it locally from `acla_ai_training/`:

```bash
python scripts/train_labelme.py storage/annotation_images
```

The annotation directory is required on every run, including when reusing a saved
split. On the first run it recursively finds Labelme JSON annotations under the
selected directory, shuffles all images together with seed 42, and assigns
80% to training and 20% to validation. Folder boundaries do not affect the split.
Only annotated images are included; metadata JSON and images without annotations
are skipped. An explicitly saved annotation with no shapes is a background sample.
At least two annotated images are required, with at least one in each split.

The sample lists are saved in `storage/image_segmentation/split.json` as `train`
and `val` arrays of annotation paths relative to the recorded source folder.
Each annotation points to its original image through Labelme's `imagePath`.
The source path is relative to the split file so the same lists work on the host
and through the `/app` Docker mount. Later runs load these exact lists; they do
not reshuffle or automatically add new images. Missing or invalid listed samples
stop preparation instead of silently changing the dataset.

Supply a different source, validation fraction, or seed when creating a split:

```bash
python scripts/train_labelme.py storage/my_annotations \
  --split-file storage/image_segmentation/my_split.json \
  --val-fraction 0.2 --seed 42 --device 0 --epochs 100 --batch 8
```

Use `--rebuild-split` to replace the saved lists with a new split of all current
annotations, including newly annotated images. This can change previous sample
assignments. The selected directory must match the saved source unless you use
`--rebuild-split` or choose another `--split-file`.
`--val-fraction` and `--seed` apply only when creating or rebuilding a split.
Use `--prepare-only` to save the lists and export the dataset without training.

Each run validates and exports exactly the listed samples to a new
`storage/image_segmentation/dataset_*/yolo/` directory, then passes its `data.yaml`
to the existing trainer. Fresh exports include annotation edits and avoid stale
images or label caches. Earlier exports are retained; with `--split-file`, exports
are placed beside that file. Training options below also apply to this launcher,
along with `--labels` and `--polyline-width` from dataset preparation.

## Prepare the dataset

If you already maintain separate training and validation folders, you can still
prepare them explicitly instead of using the automatic split launcher:

```bash
python -m training.image_segmentation prepare \
  --train storage/image_segmentation/annotations/train \
  --val storage/image_segmentation/annotations/val \
  --polyline-width 8 \
  --output storage/image_segmentation/yolo
```

This recursively reads annotated images, validates labels, shape geometry and
image dimensions, then copies images and writes normalized YOLO polygon labels.
YOLO segmentation cannot train directly on open lines, so the exporter strokes
each polyline into a narrow polygon, clipped to the image. It does not close the
path across its endpoints or fill the track between boundaries. `--polyline-width`
sets the stroke thickness in original-image pixels (default 8, minimum 2). Increase
it for high-resolution images if the boundaries become too thin after resizing to
the training `--imgsz`. Repeated points are allowed if at least two distinct pixel
positions remain. Boundaries may cross, touch, retrace themselves, or form loops.
The exporter preserves enclosed gaps in the stroke mask and keeps each boundary
as one instance; no splitting or stroke-width adjustment is needed for intersections.

The trained model predicts separate left/right boundary **masks**, not ordered
polyline points. Source JSON keeps the original editable polylines. Prepare a new
dataset and retrain to learn these classes; existing checkpoints do not gain them
by changing the labels file.

Images without JSON are skipped; an explicitly saved JSON with no shapes becomes
a background example. Both splits must contain annotations. Repeated references
to the same source image are rejected, but copied/near-identical frames must still
be kept in the same split by the dataset author.

The result contains `data.yaml`, `images/train`, `images/val`, `labels/train`, and
`labels/val`. Session subfolders are preserved. Source images/annotations are not
modified. An existing output directory is refused; choose a new output path when
preparing a revised dataset. Relative paths in `data.yaml` work on both the host
and its `/app` Docker mount.

## Train

Both `train` and `train-labelme` default to pretrained YOLO26 nano segmentation
(`yolo26n-seg.pt`). YOLO26 requires Ultralytics 8.4 or newer; rebuild an older
training container or update its training dependencies before running.

In the training Python environment:

```bash
python -m training.image_segmentation train \
  --data storage/image_segmentation/yolo/data.yaml \
  --model yolo26n-seg.pt --epochs 100 --imgsz 768 --batch 8 --device cpu
```

Or use the rebuilt training container (GPU 0):

```bash
docker exec -it acla_ai_training_c python -m training.image_segmentation train \
  --data /app/storage/image_segmentation/yolo/data.yaml \
  --model yolo26n-seg.pt --epochs 100 --imgsz 768 --batch 8 --device 0
```

The default is the first configured NVIDIA/ROCm GPU (`--device 0`); select
`--device cpu` for CPU training or `--device mps` for Apple Silicon.
Pretrained weights download on first use to
`storage/image_segmentation/pretrained/`, independent of the working directory. Supply a local
segmentation checkpoint to avoid that download, or `--model yolo26n-seg.yaml`
to initialize without pretrained weights.

The trainer sets `overlap_mask=False` for all classes so region, car, and car pack
masks remain separate, preserving annotated regions beneath cars and car packs in
the training targets. Ultralytics' default merges masks with smaller masks on top,
which can remove those hidden regions.
Keep this setting when training or validating these annotations outside this CLI.

When a dataset contains labels ending in ` left` or ` right`, or the custom
`left_boundary` or `right_boundary` labels, the trainer also
sets `fliplr=0`, `flipud=0`, and `copy_paste=0` to preserve side labels. Horizontal
flips otherwise mirror the regions without swapping their class IDs. For custom
`left_boundary` or `right_boundary` datasets, it also uses `mask_ratio=1` to avoid
further downsampling thin mask targets after resizing the input image; this uses
more mask memory. Preserve these settings when training side-specific datasets
outside this CLI.

Each validation pass prints the full box and mask metric header separately from
the progress bar, so narrow terminals wrap the labels instead of cutting them off.

Checkpoints and metrics go under `storage/image_segmentation/runs/train*/`, including
`weights/best.pt` and `weights/last.pt`. `--project` and `--name` change the run
location. In Docker these defaults live under `/app/storage/image_segmentation/`
and persist in the host's `acla_ai_training/storage/image_segmentation/` directory.
After successful training, both `train` and `train-labelme` (including
`scripts/train_labelme.py`) export the actual run's `best.pt` to `best.onnx` and
upload only that ONNX graph to `POST /ai-model/ultralytics`. If there is no best
checkpoint, `last.pt` is exported to `last.onnx` instead. Exports use ONNX opset 17,
fixed batch size 1, and the training image size, with raw segmentation outputs.
Local checkpoints and ONNX exports are preserved; `.pt` files are never uploaded.
Use `--upload-name track-segments` to set the backend model name
(default: `track-segments`), or `--no-upload` to
train offline without exporting or publishing.

## Predict signed position differences from images

`train-position` trains a standalone position CNN from scratch. It combines
learned image features with reference-lap XYZ, normalized lap position, and one-hot encoded game
and map names. Reference-lap XYZ is looked up during preparation at the frame's
normalized lap position. The output is exactly three signed values:

```text
reference_delta_xyz = inverse_world_rotation @ (reference_lap_xyz(s) - driver_xyz)
```

Targets use car-local **X right, Y forward, Z up**, keeping the source coordinate
units. Preparation uses `Physics_heading`, `Physics_pitch`, and `Physics_roll`
in radians to invert the car's world rotation directly in the corrected world
XYZ frame (Z up). The body-to-world matrix is `Rz(heading) Rx(pitch) Ry(roll)`;
world-offset row vectors multiply by this matrix, then negate ACC's left-positive
local X to obtain right/forward/up offsets.
All three angles must be finite; missing rotation is an error.
Normalized position `s` is a lookup key and a model input in `[0, 1]`.
Reference-lap and driver XYZ must use the same world frame and scale before
subtraction. This preparation currently supports ACC only: iRacing sessions are
skipped because their recorded position and heading fields use different frames.

Driver XYZ and world rotation are required **only during preparation to construct training labels**.
Each image's `.position.json` contains its filename, session ID, reference coordinates,
game, map name, normalized lap position, local targets, and `coordinate_frame: car_local`.
It also retains `driver_position` for the annotation map; the training loader ignores it.
The model receives the image, one reference-lap XYZ triple, and encoded context;
targets provide supervision and error metrics. Inference requires the same inputs.
Prepared files may still contain centerline coordinates and targets for compatibility
with the preparation workflow. Training ignores those rows; files with only the
reference-lap row are also accepted. Centerline data is not an inference input or output.
Reference lookup uses linear interpolation, including across the start/finish seam when needed;
missing axes and invalid or missing reference curves fail instead of becoming
zero coordinates.

The image encoder uses five convolution/GroupNorm/ReLU blocks with
16, 32, 64, 96, and 128 channels and strides (2, 2, 2, 1, 1), respectively.
A 4×4 adaptive pooling grid preserves coarse
image layout, followed by a 128-feature linear/ReLU position encoder. By default,
a 128-unit causal GRU reads four frames of image features, standardized reference
coordinates, and context, in chronological order. An MLP combines its final memory
with the current reference coordinates and context to predict the current signed
offsets. All encoder, GRU, and regression weights are randomly
initialized and trained together. No pretrained weights, segmentation model,
or segmentation labels are used.
Reference inputs are standardized using training-set statistics. Lap progress
stays in `[0, 1]`; game and map names use separate one-hot vectors. Sorted category
vocabularies are built from the prepared dataset and saved in each checkpoint.
The context order is normalized position, game vector, then map-name vector.
Inference reuses the saved order and rejects names absent from the vocabulary.
Outputs stay in raw coordinate units, with an unconstrained linear output layer.
Training uses mean component Smooth L1 between predicted and target XYZ offsets.
Before letterboxing, each training-frame load has a 50% chance of HUD
CoarseDropout. It selects 1-9 of the nine annotated HUD panels and fully covers
each with either a random solid RGB color or random pixel noise (equal chance).
Panel bounds are normalized from the 1252 x 720 annotation and scaled to the
source frame. Original and newly obscured frames are mixed during training;
reference inputs, context, and XYZ targets stay unchanged. Validation and
inference always use original frames.
Image preparation then letterboxes the entire frame;
there are no flips, crops, mosaics, or other geometric training augmentations
that would invalidate its position target.

### Align frames to telemetry

Run the preparation command without arguments for step-by-step choices of
sessions, sampling, and alignment. It downloads the selected sessions
without splitting them, extracts images and aligns telemetry in memory, and
uses the backend reference-lap model and centerline maps to prepare numeric inputs
and targets:

```bash
python scripts/prepare_position_dataset.py
# Unattended equivalent with explicit sessions and sampling:
python scripts/prepare_position_dataset.py \
  --sessions SESSION_ID_A SESSION_ID_B --fps 2
```

This creates `frame_000000.jpg` and `frame_000000.position.json` together in each
session's `frames/video_000/` folder. The position file holds that image's reference
XYZ, game, map name, normalized lap position, and precomputed offsets. Training
discovers these files recursively from the input folder; preparation creates no aggregate `positions.json` index or
`position-data` directory. No telemetry or reference snapshots are written to the
trainer output, and no per-image telemetry files are saved.
Sessions with existing local folders are
skipped without downloading or processing them; only new sessions enter the output.
The [workspace guide](../../README.md#prepare-images-telemetry-and-position-references)
describes session selection, backend configuration, alignment offsets, and
coordinate extraction.

Once preparation succeeds, open each reported `annotation_directories` folder
with `python scripts/open_labelme.py PATH`. Save Labelme JSON beside the images.
Pass the folder containing the image/position pairs to `train-position` below.
The position trainer reads images, reference XYZ, game/map context, lap progress, and
prepared targets. It does not read source telemetry or Labelme polygons; its
image features are learned directly from the position targets.

For other images, prepare both reference XYZ inputs and signed offset targets
synchronized to each image. Provide at least two sessions for each game/map;
a single map is sufficient. Save each position file
beside its image, using the same stem with `.position.json`, for example
`frame_000120.position.json`:

```json
{
  "session_id": "session-a",
  "image": "frame_000120.jpg",
  "game": "acc",
  "map_name": "Spa-Francorchamps",
  "normalized_position": 0.25,
  "references": [[0.2, 0.0, 0.5], [0.25, 0.0, 0.5]],
  "coordinate_frame": "car_local",
  "target_definition": "diag(-1, 1, 1) @ inverse_world_rotation @ (reference_position - current_position)",
  "driver_position": [0.13, -0.02, 0.41],
  "target": [[-0.07, 0.02, 0.09], [-0.12, 0.02, 0.09]]
}
```

Folder input discovers `.position.json` files recursively and groups their frames
by session for a split balanced by frame count within each game/map; Labelme annotations are ignored. The image
filename is relative to its position file, so these pairs can be moved together. A dataset index instead
contains sessions with `id` and `frames`, where each frame has a `position_file`
path relative to the index. Inline frame data and preset session `split`
assignments are rejected; regenerate these datasets with the preparation command.

Each `references` and `target` array must have shape `(2, 3)` and finite values.
Row 0 is the reference lap; row 1 is the centerline. Targets are the corresponding
reference-minus-driver vectors rotated into ACC local coordinates, with the
left-positive local X negated to produce right/forward/up
(the example above has zero rotation). The world-space target is no longer saved.
The trainer does no reference lookup or target calculation. `game` and `map_name`
must be nonempty strings and `normalized_position` must be finite and in `[0, 1]`.
Files without `coordinate_frame: car_local` must be regenerated. Preparation skips
existing session folders, so move the affected folders out of the download cache
before selecting those sessions again; preserve any annotations you need.

Existing ACC local datasets prepared with negative heading/pitch/roll need only
the lateral sign migration. The command validates all files, backs up the original
JSON files, then negates target X for both reference and centerline and records
the corrected `target_definition`. It preserves images, annotations, references,
driver coordinates, and target Y/Z. Repeated runs skip corrected files. Older
rotation conventions must be regenerated first.

```bash
python scripts/migrate_position_targets.py storage/session_downloads
python scripts/migrate_position_targets.py storage/session_downloads --apply \
  --backup storage/backups/position-targets-left-positive.tar.gz
```

Training rejects files without the corrected definition. Start a fresh training
run after migration and publish its ONNX export; existing checkpoints and models
retain their old output convention.

During preparation, reference selection requires an exact track/car/grip bucket
in the active top-lap store. Circuit maps must match `source_track_key`; display
names are not used for matching. Centerline points come from `samples.middle_line`.
After selection, model inputs `game` and `map_name` come from the map detail's
`game` and `circuit_name`; lap progress comes from aligned telemetry's
`Graphics_normalized_car_position`.
Both curves need at least two distinct normalized positions, all three axes, and
adequately sampled complete laps. When running outside Docker, expose the shared
service package with `PYTHONPATH=../acla_ai_service` as for other training commands.

### Train and predict

```bash
python scripts/train_position.py \
  --manifest storage/session_downloads \
  --output storage/image_segmentation/position-runs/run-1 \
  --epochs 100 --imgsz 768 --batch 8 --device 0
```

In the training container, use
`docker exec -it acla_ai_training_c python3 /app/scripts/train_position.py`
with the same arguments. The script forwards to
`python -m training.image_segmentation train-position`.

Use a new output directory; every run trains from scratch. The first GPU
is the default if `--device` is omitted; use `--device cpu` for CPU training.
After image extraction and before the first epoch, training assigns whole sessions
from each `(game, map_name)` to validation, targeting `--val-fraction 0.2` of that
map's frames using `--seed 42` by default. It chooses the closest achievable frame
count, accounting for unequal session sizes; equally balanced session choices
are randomized by the seed. The rest go to training, so every map appears in both
sets. Each set receives at least one session per map; maps with fewer than two
sessions are rejected. Sessions must each belong to one game/map. Frames from the
same session never cross splits, and all frames are retained. An exact target
fraction may be impossible with whole sessions. Images from all maps and sessions are shuffled
within each split. Temporal windows are assembled in chronological order before
training batches are reshuffled each epoch.
The training and validation maps are printed before the first epoch.
The run's `split.json` records the seed, target frame fraction, sequence length, map lists, and image paths.
Prepared position files and the dataset index are not changed.

`--sequence-length 4` is the default: each example contains the current frame and
up to three preceding prepared frames from the same session and video directory.
Frames are ordered by their numeric filename suffix (`frame_2` precedes `frame_10`);
filenames within a recording must share a prefix and have unique numeric suffixes.
At the start of a recording, repeat its first frame on the left until the window
is full. Images, references, and context all follow the same window; only the
final frame's target contributes to loss and metrics. No history crosses a
training/validation split, session, recording folder, or game/map boundary.
Memory starts at zero for each window. `--sequence-length 1` disables the GRU
and preserves single-frame training, checkpoint loading, and schema 4 export.
Single-frame Python models remain the default when constructing `PositionModel`
directly; the training CLI enables four-frame memory explicitly.

Windows use consecutive **available prepared frames**, not a fixed elapsed time.
Use the same sampling cadence during training and inference. Frame-number gaps
are not used to infer elapsed time or automatically reset history; put separate
recordings in separate directories. `--batch` counts windows, so four-frame
training encodes four times as many images per batch; reduce it if GPU memory is
limited. HUD augmentation still runs independently for each real frame load;
repeated padding reuses the same augmented frame.

Training also writes `best.pt`, `last.pt`, and
`results.json`, including `train_mae` and validation MAE for every reference/axis
in car-local coordinates and source units. Each epoch prints training and validation MAE together.
Training MAE averages absolute XYZ errors over all training frames using the
predictions from the training pass, before each optimizer update. Frames receive
equal weight even when the final batch is smaller.
Each epoch also prints and saves mean 3D distance error
(`||prediction - target||`) and mean direction error (the angle between the
predicted and actual offset vectors, from 0 to 180 degrees), grouped by the
actual reference-lap offset length (`||target||`): `<0.1`, `0.1-0.25`,
`0.25-0.5`, `0.5-1`, `1-2`, `2-5`, and `>=5`. Ranges include their lower
bound and exclude their upper bound. Length and distance use source coordinate
units. `val_errors_by_offset_length` stores each range's frame count,
mean distance error, mean direction error in degrees, and valid direction count.
The console shows a separate validation section for each `(game, map_name)`,
including its frame count, MAE, and offset-length breakdown. `val_by_map` in
`results.json` and checkpoint metrics saves these sections with `game`, `map_name`,
`count`, `val_mae`, `val_mae_per_axis`, and `val_errors_by_offset_length`.
Each map also saves `val_errors_by_normalized_position`, an array of 20 sections
in lap order: `[0.00, 0.05)`, `[0.05, 0.10)`, ..., `[0.95, 1.00]`.
These use the prepared normalized lap position, include the lower bound, and
exclude the upper bound except for `1.0` in the last section. Each section contains:

- `normalized_position_start` and `normalized_position_end`: the section's lap bounds.
- `count`: the number of validation frames in this section for this game/map.
- `val_mae` and `val_mae_per_axis`: mean absolute XYZ error in car-local source units.
- `mean_distance_error`: mean 3D distance between predicted and target offsets, in source units.
- `weakness_rank`: populated sections ranked by descending mean distance error within
  this game/map; `1` identifies the weakest section. Ties follow lap order.

Section metrics weight each frame equally across batches and reset each epoch.
Sections without validation frames retain `count: 0` and `null` errors and rank;
they do not indicate perfect prediction. Higher errors identify weaker sections;
these regression metrics are not percentage accuracy. Check sample counts when
comparing sections. This breakdown is saved in `results.json`, checkpoint metrics,
and the exported best model's metadata.
The top-level metrics remain aggregated across all validation frames.
Direction averages exclude pairs where either vector is zero. Empty ranges and
undefined direction averages are printed as `n/a` and saved as `null`.
Reported angles are unweighted; near-zero offsets can have a large angle error
despite a small distance error, so interpret the two together.
The best checkpoint is selected by mean validation MAE. It then
exports that checkpoint to `best.onnx` and `best.onnx.json`, including with
`--no-upload`. The `.pt` checkpoints remain available for Python inference.

After export, `best.onnx` is uploaded to the backend using the same authentication
and binary storage as segmentation training. The name is `track-position`;
use `--no-upload` to train and export offline.
Position uploads contain one self-contained ONNX graph with
only the signed position outputs. They are stored with
`task: position` and `metadata.artifactFormat: onnx` and do not enter the Track
Vision segmentation model selection.

The backend must accept `.onnx` on `POST /ai-model/ultralytics` and preserve the
filename and metadata when serving the model. Its current `.pt` restriction and
the frontend's local Python export path require follow-up changes; use
`--no-upload` until the backend accepts ONNX. The frontend should download the
published ONNX bytes directly into ONNX Runtime and use the supplied metadata.
It no longer needs to reconstruct the PyTorch model or understand its internal
encoder/head architecture.

If export or upload fails, training exits with an error and keeps its local
checkpoints and metrics. Export and publish the saved checkpoint without
retraining with:

```bash
python -m training.image_segmentation upload-position \
  storage/image_segmentation/position-runs/run-1/best.pt --name track-position
```

Passing `best.onnx` instead publishes the existing export without exporting
again. Publication reads metadata embedded in the ONNX file, so the JSON sidecar
is not required for upload.

The training script can also publish an existing ONNX model through the same
uploader, skipping all training steps:

```bash
python scripts/train_position.py \
  --upload-model storage/image_segmentation/position-runs/run-1/best.onnx
```

`--manifest` and `--output` are not required in this mode. `--upload-model`
cannot be combined with `--no-upload`.

Use the same backend environment configuration described under **Upload a trained
model** below. Successful uploads print the backend model ID.

#### Track Vision ONNX contract

The graph uses ONNX opset 17 and fixed batch size 1 at the checkpoint's training
image size. All inputs and outputs are float32. `metadata.inference` describes
the names, exact shapes, preprocessing, and context order; its `schemaVersion`
is 5 for temporal models (4 for single-frame models). Reference and delta tensors
contain only the reference-lap XYZ triple for each represented frame.
References remain in world coordinates; deltas use local right/forward/up axes,
identified by `metadata.coordinateFrame` and `metadata.coordinateAxes`.
The graph has no segmentation `predictions` or `prototypes` outputs.
The interface is independent of the informational checkpoint `format` label.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Input `images` | `[1, T, 3, imgsz, imgsz]` | RGB, NTCHW, divided by 255. Letterbox each full frame, round resized dimensions ties-to-even, center with floor offsets, and pad with RGB 114. |
| Input `references` | `[1, T, 1, 3]` | Reference-lap XYZ for each frame in source coordinates. Learned standardization is inside the graph. |
| Input `context` | `[1, T, 1 + len(game) + len(map_name)]` | Each frame's normalized lap position, then game and map-name one-hot vectors in `metadata.categories` order. |
| Output `deltas` | `[1, 1, 3]` | Final frame's car-local reference offset: right, forward, up. |

`T` is fixed to `metadata.inference.temporal.sequenceLength`. Keep a rolling
window of aligned images, references, and contexts in **oldest-to-newest** order.
Repeat the earliest frame on the left when history is short. Clear the window
on a new session/recording, game/map change, or seek. The graph starts GRU state
from zero on every call and exposes no recurrent state input/output; all memory
comes from the supplied window. The export unrolls the GRU into ordinary tensor
operators. For schema 4, omit the `T` dimension from every input; output shape is
unchanged. Existing frontend/backend consumers need schema 5 support before
using temporal exports; this change is confined to the training workspace.

The frontend must validate finite reference coordinates, lap position in `[0, 1]`,
and known category values; Python's input validation is not part of the graph.
The shared model store's required `classNames` field contains the three regression
axis names from `outputNames`, rather than segmentation classes.
Category vocabularies, coordinate/output definitions, and training
metrics accompany the graph in the upload metadata. The same metadata (without
the user-selected upload `name`) is saved in `best.onnx.json` and as JSON in the
ONNX custom metadata property `acla.track-position`.

For inference, load your reference-lap XYZ points and use game/map names from the
model's saved vocabulary:

```python
import json
from pathlib import Path
from training.image_segmentation.position_data import PositionCurve
from training.image_segmentation.position_model import PositionModel

reference = PositionCurve.from_points(json.loads(Path("reference-lap.json").read_text()))
model = PositionModel.load(Path("position-runs/run-1/best.pt"))
result = model.predict_position(
    Path("frame.jpg"), 0.25, reference,
    game="acc", map_name="Monza",
    history=[(Path("previous-frame.jpg"), 0.249)],  # Previous frames, oldest first.
)
# {"reference_delta": [dx, dy, dz]}
```

For an already loaded top-lap entry, use `PositionCurve.from_top_lap(entry)`.
Position checkpoints include only the image encoder, position encoder, optional GRU, regression
head, reference normalization, and category vocabulary. Load them through
`PositionModel.load`. Use `upload-position` to export and store their ONNX graph; the segmentation `upload`
command continues to accept segmentation checkpoints only. Uploading stores the
model for later download; the live API does not yet run position inference.
Temporal checkpoints use `acla-position-cnn-gru-v1` and save `sequence_length`;
single-frame checkpoints use `acla-position-cnn-v3`. Existing corrected single-frame
checkpoints without `sequence_length` load as length 1. Previous checkpoints that used
centerline inputs or YOLO encoders must be retrained from scratch using the same prepared position
dataset. Format labels remain informational; loading checks the saved context
and architecture, including the reference input width.

Accuracy needs evaluation on synchronized held-out sessions, especially when
camera view, car, track, or reference conditions change. Compare against a
position-only baseline to establish whether the images improve prediction.

Related research supports the components of this approach, rather than this
exact racing target:

- [PoseNet (ICCV 2015)](https://arxiv.org/abs/1505.07427) regresses camera pose
  from a single RGB image, demonstrating image-to-continuous-position learning.
- [FiLM (AAAI 2018)](https://arxiv.org/abs/1709.07871) conditions visual features
  on additional information. It offers a possible alternative to the simple
  feature concatenation used here; its experiments concern visual reasoning.
- [TransFuser (CVPR 2021)](https://arxiv.org/abs/2104.09224) combines image and
  LiDAR representations for driving and waypoint prediction. It is evidence
  for multimodal fusion, not a study of this exact XYZ input.

## Upload a trained model

The publication component in `publication.py` reads class names in class-ID order,
training parameters, and metrics from the saved checkpoint, exports it to ONNX,
then streams the ONNX bytes through the existing authenticated backend uploader.
Upload metadata includes `artifactFormat: onnx`. The shared backend client uses
`BACKEND_SERVER_IP`, `BACKEND_PROXY_PORT`,
`AI_SERVICE_USERNAME`, and `AI_SERVICE_PASSWORD`. Docker configures these from
the training service's environment. When running locally, make the shared
runtime package available with `export PYTHONPATH=../acla_ai_service` from
`acla_ai_training`, and configure the backend environment variables there.

To export and publish an existing trained checkpoint without retraining or
starting the local API:

```bash
python -m training.image_segmentation upload \
  storage/image_segmentation/runs/train/weights/best.pt --name track-segments
```

Use the actual run directory (`train`, `train2`, etc.). Successful uploads print
the backend model ID. An export or upload failure exits with a nonzero status and
preserves the local checkpoint; after resolving the failure, use the command above
to export and upload it without retraining. A timeout may occur after the backend
saved the record, so check its model list before repeating an uncertain upload.

### Upload through the local FastAPI endpoint

For clients that supply an ONNX export and their own metadata, the existing local
training API remains available. Start it from the training Python environment:

```bash
python -m uvicorn training.api.app:app --host 127.0.0.1 --port 8002
```

The API uses the shared backend client and its configured backend address,
`AI_SERVICE_USERNAME`, and `AI_SERVICE_PASSWORD` to authenticate. Use the same
backend environment configuration as the other training publishers.

Save `model-metadata.json` with the class names from the trained dataset's
`data.yaml`, preserving class-ID order (array index 0 is class 0):

```json
{
  "name": "track-segments",
  "task": "segment",
  "classNames": ["track", "curb", "grass", "car", "other", "fence", "car pack", "sand", "Outfield asphalt road"],
  "metadata": {
    "artifactFormat": "onnx",
    "baseModel": "yolo26n-seg.pt",
    "epochs": 100,
    "trainingRunId": "train"
  }
}
```

Upload the ONNX export and metadata to FastAPI:

```bash
curl --fail-with-body http://127.0.0.1:8002/models/ultralytics/upload \
  -F "file=@storage/image_segmentation/runs/train/weights/best.onnx" \
  -F "metadata=<model-metadata.json"
```

Use the actual run directory (`train`, `train2`, etc.) and training parameters.
The route accepts only a non-empty `.onnx` file up to 512 MiB and a JSON metadata
field up to 1 MiB; `.pt` uploads are rejected. It forwards binary models to
`POST /ai-model/ultralytics` and returns the saved backend record with HTTP 201,
including `_id`, `modelFileId`, and
`sha256`. It does not load or execute the model. Each successful upload
creates a new record, even when the name matches an existing model.

Backend storage does not activate a model or add live inference. Validation and
backend errors return non-success HTTP statuses. This raw-file endpoint does not
inspect graph contents; the training publication component and `upload`
command load local trusted checkpoints to obtain their metadata and export ONNX
automatically.

References: [Labelme](https://github.com/wkentaro/labelme),
[noVNC](https://github.com/novnc/noVNC),
[Ultralytics polygon format](https://docs.ultralytics.com/datasets/segment/),
[Ultralytics mask overlap setting](https://docs.ultralytics.com/usage/cfg/#train-settings),
[Ultralytics ONNX export](https://docs.ultralytics.com/integrations/onnx/),
[Ultralytics segmentation training](https://docs.ultralytics.com/tasks/segment/).
