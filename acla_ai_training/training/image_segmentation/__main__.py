"""Run with python -m training.image_segmentation {annotate,prepare,train}."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from . import DEFAULT_LABELS, PACKAGE_DIR, WORKSPACE_DIR, read_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotate and train racing-image segmentation.")
    commands = parser.add_subparsers(dest="command", required=True)

    annotate = commands.add_parser("annotate", help="Open Labelme on the desktop or in a browser.")
    annotate.add_argument(
        "images", type=Path, nargs="?",
        help="Directory of images (including nested folders); omit to choose with Open Dir in Labelme.",
    )
    annotate.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    annotate.add_argument(
        "--browser", action="store_true",
        help="Serve Labelme through noVNC on port 6080 (automatic on Linux without a display).",
    )

    prepare = commands.add_parser("prepare", help="Convert Labelme train/val folders to a YOLO dataset.")
    prepare.add_argument("--train", type=Path, required=True)
    prepare.add_argument("--val", type=Path, required=True)
    prepare.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    prepare.add_argument(
        "--polyline-width", type=int, default=8,
        help="Polyline stroke thickness in original-image pixels (minimum 2, default 8).",
    )
    prepare.add_argument("--output", type=Path, default=WORKSPACE_DIR / "storage/image_segmentation/yolo")

    train = commands.add_parser("train", help="Train an Ultralytics segmentation model.")
    train.add_argument("--data", type=Path, required=True, help="Prepared data.yaml file.")
    train.add_argument("--model", default="yolo11n-seg.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="cpu", help="cpu, 0 for the first GPU, or mps.")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--project", type=Path, default=WORKSPACE_DIR / "models/image_segmentation")
    train.add_argument("--name", default="train")
    args = parser.parse_args(argv)

    if args.command == "annotate":
        if args.images is not None and not args.images.is_dir():
            parser.error(f"Image directory does not exist: {args.images}")
        read_labels(args.labels)
        if importlib.util.find_spec("labelme") is None:
            parser.error(
                "Rebuild the ai_training Docker image, or install Labelme locally with: "
                "python -m pip install -r training/image_segmentation/requirements-labelme.txt"
            )
        image_args = [str(args.images.resolve())] if args.images is not None else []
        command = [
            sys.executable, str(PACKAGE_DIR / "labelme_editor.py"), *image_args,
            "--labels", str(args.labels.resolve()),
            "--config", str(PACKAGE_DIR / "labelme.yaml"),
        ]
        if args.browser or (
            sys.platform == "linux"
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        ):
            from .browser import run_browser

            try:
                return run_browser(command)
            except (OSError, RuntimeError) as exc:
                parser.error(str(exc))
        return subprocess.call(command)

    if args.command == "prepare":
        from .dataset import prepare_dataset

        try:
            data = prepare_dataset(
                args.train, args.val, args.output,
                labels_file=args.labels, polyline_width=args.polyline_width,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Dataset ready: {data}")
        return 0

    from .trainer import train_model

    train_model(
        args.data, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        project=args.project, name=args.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
