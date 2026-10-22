"""Local image annotation, segmentation training, and position-regression training."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import httpx

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

    train = argparse.ArgumentParser(add_help=False)
    train.add_argument("--model", default="yolo26n-seg.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=768)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="0", help="0 for the first GPU (default), cpu, or mps.")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--project", type=Path, default=WORKSPACE_DIR / "storage/image_segmentation/runs")
    train.add_argument("--name", default="train")
    train.add_argument("--no-upload", action="store_true", help="Keep the checkpoint local without uploading.")
    train.add_argument("--upload-name", default="track-segments", help="Model name in the backend store.")
    prepared = commands.add_parser("train", parents=[train], help="Train a prepared YOLO dataset.")
    prepared.add_argument("--data", type=Path, required=True, help="Prepared data.yaml file.")

    position = commands.add_parser("train-position", help="Train a standalone signed XYZ position model from scratch.")
    position.add_argument("--manifest", type=Path, help="Required for training: recursively scan a folder for .position.json files, or read a positions.json index; split whole sessions within each game/map, balancing frame counts.")
    position.add_argument("--output", type=Path, help="Required for training: new directory for position checkpoints, ONNX, and metrics.")
    position.add_argument("--epochs", type=int, default=100)
    position.add_argument("--imgsz", type=int, default=768)
    position.add_argument("--batch", type=int, default=8)
    position.add_argument("--device", default="0", help="0 for the first GPU (default), cpu, or mps.")
    position.add_argument("--workers", type=int, default=0)
    position.add_argument("--val-fraction", type=float, default=0.2, help="Target validation share of frames per game/map (default: 0.2), using whole sessions; at least two sessions per map are required.")
    position.add_argument("--seed", type=int, default=42, help="Random session split, frame shuffle, and training seed (default: 42).")
    position_upload = position.add_mutually_exclusive_group()
    position_upload.add_argument("--no-upload", action="store_true", help="Export ONNX locally without uploading.")
    position_upload.add_argument("--upload-model", type=Path, help="Upload an existing position .onnx model without training.")

    automatic = commands.add_parser(
        "train-labelme", parents=[train], help="Split nested Labelme annotations and train.",
    )
    automatic.add_argument(
        "images", type=Path,
        help="Required annotation root; recursively read Labelme annotations from this directory.",
    )
    automatic.add_argument(
        "--split-file", type=Path, default=WORKSPACE_DIR / "storage/image_segmentation/split.json",
        help="Persistent JSON lists of training and validation annotations.",
    )
    automatic.add_argument("--val-fraction", type=float, default=0.2, help="New split validation share (default: 0.2).")
    automatic.add_argument("--seed", type=int, default=42, help="New split shuffle seed (default: 42).")
    automatic.add_argument("--rebuild-split", action="store_true", help="Replace saved lists using all current annotations.")
    automatic.add_argument("--prepare-only", action="store_true", help="Save the split and export data without training.")
    automatic.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    automatic.add_argument("--polyline-width", type=int, default=8)

    upload = commands.add_parser("upload", help="Export and upload a segmentation model as ONNX without retraining.")
    upload.add_argument("checkpoint", type=Path, help="Local trained .pt checkpoint to export as ONNX.")
    upload.add_argument("--name", default="track-segments", help="Model name in the backend store.")
    upload_position = commands.add_parser("upload-position", help="Export/publish a position model as ONNX without retraining.")
    upload_position.add_argument("checkpoint", type=Path, help="Local position .pt checkpoint or exported .onnx artifact.")
    upload_position.add_argument("--name", default="track-position", help="Model name in the backend store.")
    args = parser.parse_args(argv)

    if args.command == "train-position" and args.upload_model is not None:
        args.command = "upload-position"
        args.checkpoint = args.upload_model
        args.name = "track-position"

    if args.command == "train-position":
        missing = [flag for flag in ("manifest", "output") if getattr(args, flag) is None]
        if missing:
            parser.error("the following arguments are required for training: " + ", ".join(f"--{flag}" for flag in missing))

        from .position_training import train_position_model

        try:
            checkpoint = train_position_model(
                args.manifest, output=args.output, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers,
                val_fraction=args.val_fraction, seed=args.seed,
                upload=not args.no_upload,
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            parser.error(str(exc))
        print(f"Position checkpoint ready: {checkpoint}")
        return 0

    if args.command in ("upload", "upload-position"):
        if args.command == "upload-position":
            from .position_publication import upload_position_checkpoint as upload_checkpoint
        else:
            from .publication import upload_checkpoint

        try:
            record = asyncio.run(upload_checkpoint(args.checkpoint, name=args.name))
        except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as exc:
            parser.error(f"Backend upload failed: {exc}")
        print(f"Model uploaded to backend: {record['_id']}")
        return 0

    if args.command == "annotate":
        from .processes import stop_existing_labelme

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
        try:
            stop_existing_labelme()
        except (OSError, RuntimeError) as exc:
            parser.error(str(exc))
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

    if args.command == "train-labelme":
        from .splits import prepare_training_dataset

        try:
            args.data = prepare_training_dataset(
                args.images, split_file=args.split_file, val_fraction=args.val_fraction,
                seed=args.seed, rebuild_split=args.rebuild_split,
                labels_file=args.labels, polyline_width=args.polyline_width,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Dataset ready: {args.data}", flush=True)
        if args.prepare_only:
            return 0

    from .trainer import train_model

    try:
        train_model(
            args.data, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, workers=args.workers,
            project=args.project, name=args.name,
            upload=not args.no_upload, upload_name=args.upload_name,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
