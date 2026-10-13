"""Browser-based folder selection and video frame extraction for Docker."""

from pathlib import Path
import sys

import streamlit as st

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from scripts.videos_to_images import extract_frames, parse_args


def folder_picker(label: str, key: str, default: Path) -> Path | None:
    st.subheader(label)
    st.session_state.setdefault(key, str(default))
    entered = st.text_input("Folder path", key=key).strip()
    if not entered:
        return None
    folder = Path(entered).expanduser().resolve()

    def navigate(destination: Path) -> None:
        st.session_state[key] = str(destination)

    st.button(
        "Up one folder", key=f"{key}_up", disabled=folder == folder.parent,
        on_click=navigate, args=(folder.parent,),
    )
    try:
        if folder.is_dir():
            children = sorted(path for path in folder.iterdir() if path.is_dir())
            selected = st.selectbox(
                "Subfolders", children, index=None, format_func=lambda path: path.name,
                key=f"{key}_children_{folder}", placeholder="Choose a subfolder",
            )
            st.button(
                "Open selected folder", key=f"{key}_open", disabled=selected is None,
                on_click=navigate, args=(selected,),
            )
    except OSError as exc:
        st.error(f"Cannot browse folder: {exc}")
    st.caption(f"Selected: {folder}")
    return folder


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Videos to images", layout="wide")
    st.title("Videos to images")
    st.caption(
        "Choose folders available inside the container. Host files must be mounted "
        "in Docker; the training folder is available at /app."
    )
    input_column, output_column = st.columns(2)
    with input_column:
        videos_folder = folder_picker(
            "Input videos", "videos_folder", args.videos_folder or TRAINING_ROOT / "storage",
        )
    with output_column:
        output_folder = folder_picker(
            "Output images", "output_folder", TRAINING_ROOT / "storage" / "video_frames",
        )
        st.caption("You can enter a new output folder path; it will be created on extraction.")

    sampling = st.radio(
        "Sampling", ["Every N frames", "Images per second"],
        index=1 if args.fps is not None else 0, horizontal=True,
    )
    every_n_frames, fps = 1, None
    if sampling == "Images per second":
        fps = st.number_input(
            "Images per second", min_value=0.0, value=args.fps or 2.0,
        )
    else:
        every_n_frames = st.number_input(
            "Save every Nth frame", min_value=1, value=args.every_n_frames, step=1,
        )
    st.caption(
        "Only videos directly inside the input folder are processed. "
        "Each video gets its own output subfolder; existing images are never overwritten."
    )
    if st.button(
        "Extract images", type="primary", disabled=videos_folder is None or output_folder is None,
    ):
        try:
            with st.spinner("Extracting images…"):
                total = 0
                for name, saved, destination in extract_frames(
                    videos_folder, output_folder, every_n_frames=every_n_frames, fps=fps,
                ):
                    total += saved
                    st.write(f"{name}: saved {saved} images to {destination}")
            st.success(f"Finished: saved {total} images to {output_folder}")
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
