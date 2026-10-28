from unittest.mock import Mock

import pandas as pd
import pytest

from ui.segment_tabs.components import manual_track_map


class _Column:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Streamlit:
    def __init__(self):
        self.figure = None
        self.checkboxes = []

    def caption(self, *_args, **_kwargs):
        pass

    def columns(self, count):
        return [_Column() for _ in range(count)]

    def checkbox(self, label, value=False, **_kwargs):
        self.checkboxes.append(label)
        return value

    def selectbox(self, _label, options, **_kwargs):
        return options[0]

    def plotly_chart(self, figure, **_kwargs):
        self.figure = figure

    def info(self, *_args, **_kwargs):
        pass


@pytest.mark.parametrize("start, end", [(0, 3), (1, 1)])
def test_manual_map_preserves_telemetry_axes(monkeypatch, start, end):
    fake_streamlit = _Streamlit()
    dataframe = pd.DataFrame({
        "Graphics_player_pos_x": [1, 2, 3],
        "Graphics_player_pos_y": [10, 20, 30],
        "Graphics_player_pos_z": [100, 200, 300],
    })
    original = dataframe.copy()
    interaction = {"trace": dataframe.rename(columns=lambda name: name.replace("Graphics_player_pos", "player"))}
    overlay = Mock()

    monkeypatch.setattr(manual_track_map, "st", fake_streamlit)
    monkeypatch.setattr(manual_track_map, "track_sections_available", lambda _df: False)
    monkeypatch.setattr(
        manual_track_map,
        "render_opponent_interaction_panel",
        lambda *_args, **_kwargs: interaction,
    )
    monkeypatch.setattr(
        manual_track_map,
        "add_interaction_overlay",
        overlay,
    )

    manual_track_map.render_manual_track_map(dataframe, start, end, "session-1")

    assert "Flip Y/Z" not in fake_streamlit.checkboxes
    trajectory = next(
        trace for trace in fake_streamlit.figure.data
        if trace.name == "Player Trajectory"
    )
    assert list(trajectory.y) == dataframe["Graphics_player_pos_y"].iloc[start:end].tolist()
    assert list(trajectory.z) == dataframe["Graphics_player_pos_z"].iloc[start:end].tolist()
    for marker, index in (("Start", start), ("End", max(start, end - 1))):
        trace = next(trace for trace in fake_streamlit.figure.data if trace.name == f"Player, {marker}")
        assert list(trace.y) == [dataframe["Graphics_player_pos_y"].iloc[index]]
        assert list(trace.z) == [dataframe["Graphics_player_pos_z"].iloc[index]]
    assert overlay.call_args.args[1] is interaction
    pd.testing.assert_frame_equal(dataframe, original)
