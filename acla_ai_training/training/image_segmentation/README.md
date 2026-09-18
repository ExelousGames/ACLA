# Racing-image polygon segmentation

This local Python workflow uses Labelme to draw polygons and Ultralytics YOLO
instance segmentation to train on them. Run commands from `acla_ai_training/`.

## Install

Ultralytics is included in the training workspace's common requirements and all
three training Docker builds. Rebuild your training image to install it, or use
the existing `requirements.cpu.txt` / `requirements.nvidia.txt` for local Python.
The Dockerfiles include OpenCV's system libraries.

Install Labelme separately in a desktop Python 3.11+ environment:

```bash
python -m pip install -r training/image_segmentation/requirements-labelme.txt
```

Labelme is a Qt desktop application and needs a graphical display. Run annotation
on your desktop, then train in the headless container or a local training Python
environment. It does not appear inside the Streamlit telemetry UI. Labelme 6 is
used for Python 3.11 compatibility; the launcher uses its command-line interface.

## Draw polygons

Put screenshots or extracted video frames in these folders, keeping all frames
from a recording/session in the same split. Reserve separate sessions for
validation; adjacent frames split between training and validation inflate scores.

```text
storage/image_segmentation/annotations/
  train/session_a/frame_001.png
  val/session_b/frame_001.png
```

Open each session folder to label its images:

```bash
python scripts/open_labelme.py storage/image_segmentation/annotations/train/session_a
python scripts/open_labelme.py storage/image_segmentation/annotations/val/session_b
```

In Labelme, press **Ctrl+N** (Create Polygons), click along the region boundary,
and double-click to finish. Choose a label, then **Ctrl+S** to save. Use **D** / **A**
for the next / previous image. JSON annotations are saved beside the images.

The class IDs follow the order in `labels.txt`:

| ID | Label | Region |
| --- | --- | --- |
| 0 | track | Visible driving surface |
| 1 | curb | Each visible curb region |
| 2 | grass | Each visible grass region |
| 3 | car | One polygon per visible car |
| 4 | other | Other regions you explicitly want the model to identify |

Trace visible boundaries and annotate all instances of the chosen classes in each
image. Unannotated pixels are background; `other` is an explicit class, not an
automatic background label. Each polygon becomes a separate instance. Holes and
Labelme group IDs are not merged; draw separate visible pieces as separate polygons.
Use polygons rather than rectangles, lines, or AI mask shapes.

To add specific regions such as gravel or barriers, append labels to a copy of
`labels.txt` and pass that file with `--labels path/to/labels.txt` to both
`annotate` and `prepare`. Preserve the class order for existing datasets/models.
Keep the original images alongside their JSON files (image bytes are not embedded).

## Prepare the dataset

```bash
python -m training.image_segmentation prepare \
  --train storage/image_segmentation/annotations/train \
  --val storage/image_segmentation/annotations/val \
  --output storage/image_segmentation/yolo
```

This recursively reads annotated images, validates labels, polygon geometry and
image dimensions, then copies images and writes normalized YOLO polygon labels.
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

Checkpoints and metrics go under `models/image_segmentation/train*/`, including
`weights/best.pt` and `weights/last.pt`. `--project` and `--name` change the run
location. These are local image models; no live-service model publication or
inference route is configured by this workflow.

References: [Labelme](https://github.com/wkentaro/labelme),
[Ultralytics polygon format](https://docs.ultralytics.com/datasets/segment/),
[Ultralytics segmentation training](https://docs.ultralytics.com/tasks/segment/).
