"""Extract numbered JPEG frames from each video in a folder.

Usage: python scripts/videos_to_images.py [path/to/videos] [path/to/images]
Omit a folder to open the browser UI at http://localhost:8501.
"""

import argparse
import math
from pathlib import Path
import subprocess
import sys


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos_folder", type=Path, nargs="?",
        help="Folder containing videos (not recursive); opens the browser UI if omitted.",
    )
    parser.add_argument(
        "output_folder", type=Path, nargs="?",
        help="Destination folder; created if missing. Opens the browser UI if omitted.",
    )
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        metavar="N",
        help="Save every Nth frame, starting with the first (default: 1, all frames).",
    )
    sampling.add_argument(
        "--fps",
        type=float,
        help="Images per second (e.g. 2 or 0.5), capped at each video's source frame rate.",
    )
    args = parser.parse_args()
    if args.every_n_frames < 1:
        parser.error("--every-n-frames must be at least 1")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be a positive, finite number")
    return args


def extract_frames(videos_folder: Path, output_folder: Path, *, every_n_frames=1, fps=None):
    """Yield (video name, saved count, destination) after each completed video."""
    if every_n_frames < 1:
        raise ValueError("Frame interval must be at least 1")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("Images per second must be a positive, finite number")
    videos_folder = videos_folder.expanduser().resolve()
    output_folder = output_folder.expanduser().resolve()
    if not videos_folder.is_dir():
        raise ValueError(f"Video folder does not exist or is not a directory: {videos_folder}")
    videos = sorted(
        path for path in videos_folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise ValueError(f"No supported videos found in {videos_folder}")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install it with: python -m pip install opencv-python"
        ) from exc

    try:
        for video in videos:
            # Keep the extension so clip.mp4 and clip.mov have separate outputs.
            destination = output_folder / video.name
            if destination.exists() and (
                not destination.is_dir() or any(destination.iterdir())
            ):
                raise OSError(f"Output already exists and is not an empty directory: {destination}")

            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened():
                    raise OSError(f"Cannot open video: {video}")
                if fps is not None:
                    source_fps = capture.get(cv2.CAP_PROP_FPS)
                    if not math.isfinite(source_fps) or source_fps <= 0:
                        raise OSError(f"Cannot determine source frame rate: {video}")
                destination.mkdir(parents=True, exist_ok=True)
                frame_index = 0
                saved = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    if fps is not None:
                        # Select the first source frame at or after each sample time.
                        should_save = frame_index * fps >= saved * source_fps
                    else:
                        should_save = frame_index % every_n_frames == 0
                    if should_save:
                        image_path = destination / f"frame_{frame_index:06d}.jpg"
                        if not cv2.imwrite(str(image_path), frame):
                            raise OSError(f"Cannot write image: {image_path}")
                        saved += 1
                    frame_index += 1
                if saved == 0:
                    raise OSError(f"No frames could be read from: {video}")
            finally:
                capture.release()
            yield video.name, saved, destination
    except cv2.error as exc:
        raise OSError(str(exc)) from exc


def main() -> None:
    args = parse_args()
    if args.videos_folder is None or args.output_folder is None:
        app_path = Path(__file__).resolve().parents[1] / "ui" / "videos_to_images_app.py"
        try:
            code = subprocess.call([
                sys.executable, "-m", "streamlit", "run", str(app_path),
                "--server.address=0.0.0.0", "--server.port=8501",
                "--server.headless=true", "--browser.gatherUsageStats=false",
                "--", *sys.argv[1:],
            ])
        except KeyboardInterrupt:
            return
        sys.exit(code)

    try:
        for name, saved, destination in extract_frames(
            args.videos_folder, args.output_folder,
            every_n_frames=args.every_n_frames, fps=args.fps,
        ):
            print(f"{name}: saved {saved} images to {destination}")
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
