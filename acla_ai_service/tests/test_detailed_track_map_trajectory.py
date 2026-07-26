import pandas as pd

from ui.segment_tabs.components import detailed_track_map


class _SessionState(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class _Column:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Streamlit:
    def __init__(self):
        self.session_state = _SessionState()
        self.figure = None

    def subheader(self, *_args, **_kwargs):
        pass

    def caption(self, *_args, **_kwargs):
        pass

    def columns(self, count):
        return [_Column() for _ in range(count)]

    def checkbox(self, _label, value=False, **_kwargs):
        return value

    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index]

    def plotly_chart(self, figure, **_kwargs):
        self.figure = figure

    def info(self, *_args, **_kwargs):
        pass


def test_track_map_uses_exclusive_end_and_marks_last_telemetry_index(monkeypatch):
    fake_streamlit = _Streamlit()
    dataframe = pd.DataFrame({
        "Graphics_player_pos_x": list(range(29)),
        "Graphics_player_pos_y": [index * 2 for index in range(29)],
        "Graphics_player_pos_z": [index * 3 for index in range(29)],
    })

    monkeypatch.setattr(detailed_track_map, "st", fake_streamlit)
    monkeypatch.setattr(detailed_track_map, "track_sections_available", lambda _df: False)
    monkeypatch.setattr(
        detailed_track_map,
        "render_opponent_interaction_panel",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        detailed_track_map,
        "add_interaction_overlay",
        lambda *_args, **_kwargs: None,
    )

    detailed_track_map.render_track_map(dataframe, 8, 29, "session-1")

    figure = fake_streamlit.figure
    trajectory = next(trace for trace in figure.data if trace.name.startswith("Player Trajectory"))
    end_marker = next(trace for trace in figure.data if trace.name == "Player, End")

    assert list(trajectory.customdata) == list(range(8, 29))
    assert any(28 in row for row in end_marker.customdata)
    assert figure.layout.title.text == "Positions (Start: 8, End: 28) (3D)"
