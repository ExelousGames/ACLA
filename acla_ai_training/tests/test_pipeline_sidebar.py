import pytest
from streamlit.testing.v1 import AppTest

from training.pipelines.manifest import registry


def _render_sidebar():
    import streamlit as st
    from ui.segment_tabs.pipeline_sidebar import render_pipeline_sidebar

    with st.sidebar:
        render_pipeline_sidebar(None)


@pytest.mark.parametrize("has_remaining_pipeline", [False, True])
def test_remove_selected_pipeline_clears_route_and_preserves_data(
    monkeypatch, tmp_path, has_remaining_pipeline,
):
    monkeypatch.setattr(registry, "_STORAGE_ROOT", tmp_path / "pipelines")
    if has_remaining_pipeline:
        registry.create_pipeline("pipeline_a")
    registry.create_pipeline("pipeline_b")
    dataset = tmp_path / "annotations.json"
    dataset.write_text('{"annotations": []}')

    app = AppTest.from_function(_render_sidebar)
    app.query_params.update(pipeline="pipeline_b", view="detailed", node="node_b")
    route_state = {
        "active_view": "detailed",
        "pipeline_routed_view": "detailed",
        "pipeline_active_node_id": "node_b",
        "pipeline_training_node": "training_b",
        "pipeline_annotation_key": "annotations_b",
        "pipeline_session_key": "sessions_b",
    }
    for key, value in route_state.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception
    assert app.selectbox(key="pipeline_picker").value == "pipeline_b"
    assert app.sidebar.button(key="remove_pipeline").label == "Remove pipeline"
    assert all(
        button.key != "remove_pipeline"
        for expander in app.expander for button in expander.button
    )

    app.button(key="remove_pipeline").click().run()

    assert not app.exception
    assert not app.error
    assert registry.load("pipeline_b") is None
    assert dataset.read_text() == '{"annotations": []}'
    assert "view" not in app.query_params
    assert "node" not in app.query_params
    for key in route_state:
        assert key not in app.session_state
    if has_remaining_pipeline:
        assert registry.list_pipelines() == ["pipeline_a"]
        assert app.selectbox(key="pipeline_picker").value == "pipeline_a"
        assert app.session_state["active_pipeline_id"] == "pipeline_a"
        assert app.query_params["pipeline"] == ["pipeline_a"]
    else:
        assert registry.list_pipelines() == []
        assert "active_pipeline_id" not in app.session_state
        assert "pipeline" not in app.query_params
        assert not any(button.key == "remove_pipeline" for button in app.button)
        assert any("No pipelines yet" in caption.value for caption in app.caption)
