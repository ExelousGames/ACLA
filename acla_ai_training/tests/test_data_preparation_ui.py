import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from segment_tabs import _training_runner, pipeline_view
from training.pipelines.training.config import TrainingPipelineConfig


def _render_preparation():
    from types import SimpleNamespace
    import streamlit as st
    from segment_tabs.pipeline_view import _render_data_preparation_card
    from training.pipelines.training.config import TrainingPipelineConfig

    store = SimpleNamespace(
        has_cached_data=lambda key: key in st.session_state["cache_keys"],
        get_cache_metadata=lambda key: None,
    )
    _render_data_preparation_card(store, TrainingPipelineConfig())


@pytest.fixture
def preparation_app(monkeypatch):
    monkeypatch.setattr(pipeline_view.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(pipeline_view, "read_marker", lambda job: None)
    monkeypatch.setattr(_training_runner, "read_marker", lambda job: None)
    app = AppTest.from_function(_render_preparation)
    app.session_state["cache_keys"] = []
    return app


@pytest.mark.parametrize("available,disabled", [
    ([], [False, True, True]),
    (["cleaned"], [False, False, True]),
    (["cleaned", "top"], [False, False, False]),
])
def test_preparation_cards_follow_cache_prerequisites(preparation_app, available, disabled):
    cfg = TrainingPipelineConfig()
    keys = {"cleaned": cfg.processed_session_data_cache_key, "top": cfg.top_laps_cache_key}
    preparation_app.session_state["cache_keys"] = [keys[name] for name in available]
    preparation_app.run()
    assert not preparation_app.exception
    assert [heading.value for heading in preparation_app.subheader] == [
        "1. Data Cleanup", "2. Select Top Laps", "3. Add Expert Features",
    ]
    assert [button.disabled for button in preparation_app.button] == disabled


@pytest.mark.parametrize("index,job,stage", [
    (0, "prepare_data", "cleanup"),
    (1, "select_top_laps", "top_laps"),
    (2, "add_expert_features", "enrich"),
])
@pytest.mark.parametrize("workers", [1, 3])
def test_preparation_buttons_run_only_the_selected_stage(monkeypatch, preparation_app, index, job, stage, workers):
    cfg = TrainingPipelineConfig()
    preparation_app.session_state["cache_keys"] = [cfg.processed_session_data_cache_key, cfg.top_laps_cache_key]
    spawn = Mock()
    monkeypatch.setattr(pipeline_view, "spawn", spawn)
    preparation_app.run()
    preparation_app.number_input[0].set_value(workers).run()
    preparation_app.button[index].click().run()
    assert not preparation_app.exception
    spawn.assert_called_once_with(job, [
        sys.executable, "-u", "-m", "training.pipelines.training.prepare_training_data", "--stage", stage,
        "--workers", str(workers),
    ])


def test_running_stage_disables_other_preparation_actions(monkeypatch, preparation_app):
    cfg = TrainingPipelineConfig()
    preparation_app.session_state["cache_keys"] = [cfg.processed_session_data_cache_key, cfg.top_laps_cache_key]
    monkeypatch.setattr(pipeline_view, "read_marker", lambda job: {"pid": 123} if job == "select_top_laps" else None)
    monkeypatch.setattr(pipeline_view, "pid_alive", lambda pid: True)
    preparation_app.run()
    assert not preparation_app.exception
    assert all(button.disabled for button in preparation_app.button)
    assert preparation_app.number_input[0].disabled


@pytest.mark.parametrize("cpus,expected", [(8, 4), (2, 2), (None, 1)])
def test_worker_control_defaults_to_available_cpus(monkeypatch, preparation_app, cpus, expected):
    monkeypatch.setattr(pipeline_view.os, "cpu_count", lambda: cpus)
    preparation_app.run()
    assert not preparation_app.exception
    workers = preparation_app.number_input[0]
    assert workers.value == expected
    assert workers.min == 1
    assert workers.max == (cpus or 1)
