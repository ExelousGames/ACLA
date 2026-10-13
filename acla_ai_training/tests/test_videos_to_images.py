from pathlib import Path
import sys
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from scripts import videos_to_images


APP_PATH = Path(__file__).resolve().parents[1] / "ui" / "videos_to_images_app.py"


@pytest.fixture
def video_folder(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    writer = cv2.VideoWriter(
        str(folder / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (32, 32),
    )
    assert writer.isOpened()
    try:
        for index in range(4):
            writer.write(np.full((32, 32, 3), index * 50, dtype=np.uint8))
    finally:
        writer.release()
    return folder


def test_no_paths_launches_docker_browser_ui(monkeypatch):
    launch = Mock(return_value=0)
    monkeypatch.setattr(videos_to_images.subprocess, "call", launch)
    monkeypatch.setattr(sys, "argv", ["videos_to_images.py", "--fps", "0.5"])
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.delenv("DISPLAY", raising=False)

    with pytest.raises(SystemExit) as result:
        videos_to_images.main()

    assert result.value.code == 0
    command = launch.call_args.args[0]
    assert command[:5] == [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    assert "--server.address=0.0.0.0" in command
    assert "--server.port=8501" in command
    assert "--server.headless=true" in command
    assert command[-3:] == ["--", "--fps", "0.5"]


def test_cli_extracts_without_ui_and_refuses_overwrite(video_folder, tmp_path, monkeypatch):
    output = tmp_path / "frames"
    launch = Mock()
    monkeypatch.setattr(videos_to_images.subprocess, "call", launch)
    monkeypatch.setattr(sys, "argv", [
        "videos_to_images.py", str(video_folder), str(output), "--every-n-frames", "2",
    ])

    videos_to_images.main()

    launch.assert_not_called()
    images = sorted((output / "clip.avi").glob("*.jpg"))
    assert [path.name for path in images] == ["frame_000000.jpg", "frame_000002.jpg"]
    assert all(cv2.imread(str(path)).shape == (32, 32, 3) for path in images)
    before = [path.read_bytes() for path in images]
    with pytest.raises(SystemExit, match="Output already exists"):
        videos_to_images.main()
    assert [path.read_bytes() for path in images] == before


def test_browser_navigates_and_extracts_without_display(video_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(APP_PATH), str(tmp_path)])
    monkeypatch.setitem(sys.modules, "tkinter", None)
    monkeypatch.delenv("DISPLAY", raising=False)
    output = tmp_path / "new_output"
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception

    app.selectbox(key=f"videos_folder_children_{tmp_path}").select(video_folder).run()
    app.button(key="videos_folder_open").click().run()
    assert app.text_input(key="videos_folder").value == str(video_folder)
    app.button(key="videos_folder_up").click().run()
    assert app.text_input(key="videos_folder").value == str(tmp_path)
    app.text_input(key="videos_folder").set_value(str(video_folder))
    app.text_input(key="output_folder").set_value(str(output)).run()
    app.radio[0].set_value("Images per second").run()
    app.number_input[0].set_value(2.0).run()
    assert not output.exists()
    next(button for button in app.button if button.label == "Extract images").click().run()

    assert not app.exception
    assert not app.error
    assert "saved 2 images" in app.success[0].value
    assert len(list((output / "clip.avi").glob("*.jpg"))) == 2


def test_browser_reports_invalid_input_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(APP_PATH)])
    output = tmp_path / "output"
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_input(key="videos_folder").set_value(str(tmp_path / "missing"))
    app.text_input(key="output_folder").set_value(str(output)).run()
    next(button for button in app.button if button.label == "Extract images").click().run()

    assert not app.exception
    assert "Video folder does not exist" in app.error[0].value
    assert not output.exists()
