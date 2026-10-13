from contextlib import nullcontext
from unittest.mock import MagicMock

import pandas as pd

from ui.segment_tabs import manual, shared


class _SessionState(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_lap_annotation_page_displays_its_title(monkeypatch):
    streamlit = MagicMock()
    streamlit.session_state = _SessionState()
    streamlit.columns.side_effect = [
        [nullcontext(), nullcontext()],
        [nullcontext(), nullcontext()],
        [nullcontext(), nullcontext()],
    ]
    streamlit.selectbox.return_value = "session-1"
    streamlit.spinner.return_value = nullcontext()
    streamlit.slider.return_value = (0, 1)

    store = MagicMock()
    monkeypatch.setattr(manual, "st", streamlit)
    monkeypatch.setattr(manual, "get_available_sessions", lambda _key: [])
    monkeypatch.setattr(manual, "load_annotations", lambda *_args: [])
    monkeypatch.setattr(
        manual,
        "load_session_data",
        lambda *_args: pd.DataFrame({"speed_kmh": [100]}),
    )
    monkeypatch.setattr(shared, "get_store", lambda: store)
    monkeypatch.setattr(manual, "render_feature_visualization", lambda *_args: None)
    monkeypatch.setattr(manual, "render_manual_track_map", lambda *_args: None)
    monkeypatch.setattr(manual, "render_manual_annotation_manager", lambda *_args: None)
    monkeypatch.setattr(manual, "render_manual_lap_agent", lambda *_args: None)

    manual.render_manual_annotation(
        "annotations",
        "sessions",
        ["session-1"],
    )

    streamlit.subheader.assert_called_once_with("Lap Annotation")


def test_annotation_build_load_and_save_discard_track_labels(monkeypatch):
    frame = pd.DataFrame({"Static_track": ["silverstone"], "speed": [40.0]})
    segment = shared.build_segment(
        frame, start=0, end=1,
        label_ids=["silverstone", "silverstone1", "MSP", "ST1"],
    )
    assert segment.labels == ["MSP", "ST1"]
    assert segment.telemetry_data == frame.to_dict(orient="records")

    # Existing in-memory or stored annotations must not reintroduce old labels.
    segment.labels = ["silverstone", "silverstone1", "MSP", "ST1"]
    store = MagicMock()
    store.get_chunk.return_value = [segment.to_dict()]
    monkeypatch.setattr(shared, "get_store", lambda: store)
    assert shared.load_annotations("session", "key")[0].labels == ["MSP", "ST1"]
    assert shared.load_session_segments("key", "session")[0].labels == ["MSP", "ST1"]
    shared.save_annotations("session", [segment], "key", silent=True)
    expected = segment.to_dict()
    expected["labels"] = ["MSP", "ST1"]
    store.save_chunk.assert_called_once_with("key", "session", [expected])
