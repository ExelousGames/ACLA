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

The class IDs follow the order in `labels.txt`:

| ID | Label | Region |
| --- | --- | --- |
| 0 | track | Full track surface within the image, including reasonably inferred parts hidden by cars |
| 1 | curb | Each visible curb region |
| 2 | grass | Each visible grass region |
| 3 | car | One polygon per visible car |
| 4 | other | Other regions you explicitly want the model to identify |
| 5 | fence | Visible fence regions |
| 6 | car pack | Regions annotated as a pack of cars |
| 7 | sand | Visible sand regions |
| 8 | Outfield asphalt road | Visible asphalt road regions outside the track |

For `track`, trace the continuous outer boundaries, including the road behind cars
or other foreground objects where its continuation is reasonably clear. Do not cut
holes or split the track polygon around those objects. This is amodal road
segmentation: the mask describes the track's extent, including occupied areas.
Annotate cars separately; their polygons may overlap the track polygon. Use nearby
video frames to resolve uncertain boundaries when possible, and avoid inventing
hidden turns or edges without supporting evidence.

For the other region classes, trace visible boundaries and draw disconnected visible
pieces as separate polygons. Annotate all instances of the chosen classes in each
image. Unannotated pixels are background; `other` is an explicit class, not an
automatic background label. Each shape becomes a separate instance. Holes and
Labelme group IDs are not merged. Use polygons for regions; rectangles and AI mask
shapes are not supported by this exporter.

Apply this policy consistently to training and validation images. Review existing
visible-only track annotations and prepare a new dataset after revising them;
training does not automatically fill missing sections in the source polygons.

To add specific regions such as gravel or barriers, append labels to a copy of
`labels.txt` and pass that file with `--labels path/to/labels.txt` to both
`annotate` and `prepare`. Preserve the class order for existing datasets/models.
Keep the original images alongside their JSON files (image bytes are not embedded).

## Split and start training

Run the companion launcher in the training container:

```bash
docker exec -it acla_ai_training_c python3 /app/scripts/train_labelme.py
```

Or run it locally from `acla_ai_training/`:

```bash
python scripts/train_labelme.py
```

On the first run it recursively finds Labelme JSON annotations under
`storage/annotation_images`, shuffles all images together with seed 42, and assigns
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
assignments. If the source is omitted, the saved source folder is reused.
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

In the training Python environment:

```bash
python -m training.image_segmentation train \
  --data storage/image_segmentation/yolo/data.yaml \
  --model yolo11n-seg.pt --epochs 100 --imgsz 640 --batch 8 --device cpu
```

Or use the rebuilt training container (GPU 0):

```bash
docker exec -it acla_ai_training_c python -m training.image_segmentation train \
  --data /app/storage/image_segmentation/yolo/data.yaml \
  --model yolo11n-seg.pt --epochs 100 --imgsz 640 --batch 8 --device 0
```

The default is CPU; select `--device 0` for a configured NVIDIA/ROCm GPU or `mps`
for Apple Silicon. Pretrained weights download on first use to
`storage/image_segmentation/pretrained/`, independent of the working directory. Supply a local
segmentation checkpoint to avoid that download, or `--model yolo11n-seg.yaml`
to initialize without pretrained weights.

The trainer sets `overlap_mask=False` so track and car masks remain separate,
preserving the track beneath cars in the training targets. Ultralytics' default
merges masks with smaller masks on top, which can remove those hidden track areas.
Keep this setting when training or validating these annotations outside this CLI.

When a custom dataset contains `left_boundary` or `right_boundary`, the trainer also
sets `fliplr=0`, `flipud=0`, and `copy_paste=0` to preserve side labels. Horizontal
flips otherwise mirror the edges without swapping their class IDs. It uses
`mask_ratio=1` to avoid further downsampling thin mask targets after resizing the
input image; this uses more mask memory. Preserve these settings when training
boundary datasets outside this CLI.

Checkpoints and metrics go under `storage/image_segmentation/runs/train*/`, including
`weights/best.pt` and `weights/last.pt`. `--project` and `--name` change the run
location. In Docker these defaults live under `/app/storage/image_segmentation/`
and persist in the host's `acla_ai_training/storage/image_segmentation/` directory.
After successful training, both `train` and `train-labelme` (including
`scripts/train_labelme.py`) upload the actual run's `best.pt` to
`POST /ai-model/ultralytics`, falling back to `last.pt` if there is no best
checkpoint. Local checkpoints are preserved. Use `--upload-name track-segments`
to set the backend model name (default: `track-segments`), or `--no-upload` to
train offline without publishing.

## Upload a trained model

The publication component in `publication.py` reads class names in class-ID order,
training parameters, and metrics from the saved checkpoint, then streams its
unchanged bytes through the existing authenticated backend uploader. The shared
backend client uses `BACKEND_SERVER_IP`, `BACKEND_PROXY_PORT`,
`AI_SERVICE_USERNAME`, and `AI_SERVICE_PASSWORD`. Docker configures these from
the training service's environment. When running locally, make the shared
runtime package available with `export PYTHONPATH=../acla_ai_service` from
`acla_ai_training`, and configure the backend environment variables there.

To publish an existing trained checkpoint without retraining or starting the
local API:

```bash
python -m training.image_segmentation upload \
  storage/image_segmentation/runs/train/weights/best.pt --name track-segments
```

Use the actual run directory (`train`, `train2`, etc.). Successful uploads print
the backend model ID. An upload failure exits with a nonzero status and preserves
the local weights; after resolving the failure, use the command above to upload
them without retraining. A timeout may occur after the backend saved the record,
so check its model list before repeating an uncertain upload.

### Upload through the local FastAPI endpoint

For clients that supply a checkpoint and their own metadata, the existing local
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
    "baseModel": "yolo11n-seg.pt",
    "epochs": 100,
    "trainingRunId": "train"
  }
}
```

Upload the checkpoint and metadata to FastAPI:

```bash
curl --fail-with-body http://127.0.0.1:8002/models/ultralytics/upload \
  -F "file=@storage/image_segmentation/runs/train/weights/best.pt" \
  -F "metadata=<model-metadata.json"
```

Use the actual run directory (`train`, `train2`, etc.) and training parameters.
The route accepts a non-empty `.pt` file up to 512 MiB and a JSON metadata field
up to 1 MiB. It forwards binary weights to `POST /ai-model/ultralytics` and returns
the saved backend record with HTTP 201, including `_id`, `modelFileId`, and
`sha256`. It does not load or execute the checkpoint. Each successful upload
creates a new record, even when the name matches an existing model.

Backend storage does not activate a model or add live inference. Validation and
backend errors return non-success HTTP statuses. This raw-file endpoint does not
inspect checkpoint contents; the training publication component and `upload`
command load local trusted checkpoints to obtain their metadata automatically.

References: [Labelme](https://github.com/wkentaro/labelme),
[noVNC](https://github.com/novnc/noVNC),
[Ultralytics polygon format](https://docs.ultralytics.com/datasets/segment/),
[Ultralytics mask overlap setting](https://docs.ultralytics.com/usage/cfg/#train-settings),
[Ultralytics segmentation training](https://docs.ultralytics.com/tasks/segment/).
