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

Put screenshots or extracted video frames in these folders, keeping all frames
from a recording/session in the same split. Reserve separate sessions for
validation; adjacent frames split between training and validation inflate scores.

```text
storage/image_segmentation/annotations/
  train/session_a/frame_001.png
  val/session_b/frame_001.png
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
  /app/storage/image_segmentation/annotations/train
```

Use the `val` folder for validation images. On a local desktop, run
`python scripts/open_labelme.py [path/to/images]` from `acla_ai_training/`.

In Labelme, press **Ctrl+N** (Create Polygons), click along the region boundary,
and double-click to finish. Choose a label, then **Ctrl+S** to save. Use **D** / **A**
for the next / previous image. JSON annotations are saved beside the images.

For track edges, choose **Create LineStrip**, click successive points along
one edge, and press **Enter** to finish. Choose `left_boundary` or `right_boundary`
from the driver's perspective looking forward along the track. These are open
polylines: do not connect the last point back to the first. Two points suffice
for a straight edge; add points to follow curves. Labelme stores these shapes as
`linestrip`; two-point `line` shapes are also accepted during preparation.

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
| 8 | left_boundary | Polyline along the left track edge in the direction of travel |
| 9 | right_boundary | Polyline along the right track edge in the direction of travel |
| 10 | Outfield asphalt road | Visible asphalt road regions outside the track |

The boundary classes are appended so existing class IDs stay unchanged. Trace
each continuous edge as its own polyline; split disconnected or uncertain sections
into separate shapes. Left/right identify the side of the track, even when a turn
places both visible edges on the same side of the image. Annotate both boundaries
where they can be identified, along with the existing track polygon.

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
Labelme group IDs are not merged. Use polygons for regions and polylines for
boundaries; rectangles and AI mask shapes are not supported by this exporter.

Apply this policy consistently to training and validation images. Review existing
visible-only track annotations and prepare a new dataset after revising them;
training does not automatically fill missing sections in the source polygons.

To add specific regions such as gravel or barriers, append labels to a copy of
`labels.txt` and pass that file with `--labels path/to/labels.txt` to both
`annotate` and `prepare`. Preserve the class order for existing datasets/models.
Keep the original images alongside their JSON files (image bytes are not embedded).

## Prepare the dataset

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
positions remain. Strokes that enclose holes are rejected; split those paths into
separate shapes.

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
for Apple Silicon. Pretrained weights download on first use. Supply a local
segmentation checkpoint to avoid that download, or `--model yolo11n-seg.yaml`
to initialize without pretrained weights.

The trainer sets `overlap_mask=False` so track and car masks remain separate,
preserving the track beneath cars in the training targets. Ultralytics' default
merges masks with smaller masks on top, which can remove those hidden track areas.
Keep this setting when training or validating these annotations outside this CLI.

When the dataset contains `left_boundary` or `right_boundary`, the trainer also
sets `fliplr=0`, `flipud=0`, and `copy_paste=0` to preserve side labels. Horizontal
flips otherwise mirror the edges without swapping their class IDs. It uses
`mask_ratio=1` to avoid further downsampling thin mask targets after resizing the
input image; this uses more mask memory. Preserve these settings when training
boundary datasets outside this CLI.

Checkpoints and metrics go under `models/image_segmentation/train*/`, including
`weights/best.pt` and `weights/last.pt`. `--project` and `--name` change the run
location. Publish a selected checkpoint with the local FastAPI endpoint below.

## Upload a trained model

Start the existing local training API from the training Python environment:

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
  "classNames": ["track", "curb", "grass", "car", "other", "fence", "car pack", "sand", "left_boundary", "right_boundary"],
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
  -F "file=@models/image_segmentation/train/weights/best.pt" \
  -F "metadata=<model-metadata.json"
```

Use the actual run directory (`train`, `train2`, etc.) and training parameters.
The route accepts a non-empty `.pt` file up to 512 MiB and a JSON metadata field
up to 1 MiB. It forwards binary weights to `POST /ai-model/ultralytics` and returns
the saved backend record with HTTP 201, including `_id`, `modelFileId`, and
`sha256`. It does not load or execute the checkpoint. Each successful upload
creates a new record, even when the name matches an existing model.

Uploads are explicit; training still saves checkpoints locally. Backend storage
does not activate a model or add live inference. Validation and backend errors
return non-success HTTP statuses; a timeout may occur after the backend saved
the record, so check the backend model list before repeating an uncertain upload.

References: [Labelme](https://github.com/wkentaro/labelme),
[noVNC](https://github.com/novnc/noVNC),
[Ultralytics polygon format](https://docs.ultralytics.com/datasets/segment/),
[Ultralytics mask overlap setting](https://docs.ultralytics.com/usage/cfg/#train-settings),
[Ultralytics segmentation training](https://docs.ultralytics.com/tasks/segment/).
