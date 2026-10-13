import pandas as pd

from ui.segment_tabs.components import manual_track_map


class _Column:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Streamlit:
    def __init__(self):
        self.figure = None

    def caption(self, *_args, **_kwargs):
        pass

    def columns(self, count):
        return [_Column() for _ in range(count)]

    def checkbox(self, label, value=False, **_kwargs):
        return label == "Flip Y/Z"

    def selectbox(self, _label, options, **_kwargs):
        return options[0]

    def plotly_chart(self, figure, **_kwargs):
        self.figure = figure

    def info(self, *_args, **_kwargs):
        pass


def test_manual_trajectory_can_flip_y_and_z_without_mutating_telemetry(monkeypatch):
    fake_streamlit = _Streamlit()
    dataframe = pd.DataFrame({
        "Graphics_player_pos_x": [1, 2, 3],
        "Graphics_player_pos_y": [10, 20, 30],
        "Graphics_player_pos_z": [100, 200, 300],
    })
    original = dataframe.copy()

    monkeypatch.setattr(manual_track_map, "st", fake_streamlit)
    monkeypatch.setattr(manual_track_map, "track_sections_available", lambda _df: False)
    monkeypatch.setattr(
        manual_track_map,
        "render_opponent_interaction_panel",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        manual_track_map,
        "add_interaction_overlay",
        lambda *_args, **_kwargs: None,
    )

    manual_track_map.render_manual_track_map(dataframe, 0, 3, "session-1")

    trajectory = next(
        trace for trace in fake_streamlit.figure.data
        if trace.name == "Player Trajectory"
    )
    pd.testing.assert_series_equal(
        pd.Series(trajectory.y),
        pd.Series([100, 200, 300]),
    )
    pd.testing.assert_series_equal(
        pd.Series(trajectory.z),
        pd.Series([10, 20, 30]),
    )
    pd.testing.assert_frame_equal(dataframe, original)
