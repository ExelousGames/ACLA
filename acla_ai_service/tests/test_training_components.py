from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


UI_DIR = Path(__file__).resolve().parents[1] / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from app.pipelines.manifest import node_kinds
from app.pipelines.manifest.models import TrainingNode
from segment_tabs.training_components import (
    TRAINING_COMPONENTS,
    TRAINING_ROUTES,
    TrainingComponent,
    TrainingComponentContext,
    build_training_component_registry,
)
from segment_tabs.training_components.classifier import (
    ClassifierTrainingComponent,
)
from segment_tabs.training_components.opportunity_forecaster import (
    OpportunityForecasterTrainingComponent,
)
from segment_tabs.training_components.segment_cropper import (
    SegmentCropperTrainingComponent,
)
from segment_tabs.training_components.transformer import (
    TransformerTrainingComponent,
)
from segment_tabs.training_components._shared import resolve_input_key


class _CompleteComponent(TrainingComponent):
    def open(self, context: TrainingComponentContext) -> None:
        pass


def _form_streamlit(columns: list[MagicMock]) -> MagicMock:
    streamlit = MagicMock()
    streamlit.form.return_value = nullcontext()
    streamlit.columns.return_value = columns
    streamlit.form_submit_button.return_value = True
    return streamlit


def test_training_component_requires_open_implementation():
    class IncompleteComponent(TrainingComponent):
        pass

    with pytest.raises(TypeError):
        IncompleteComponent()


def test_training_component_context_is_frozen():
    context = TrainingComponentContext(node=None, input_key=None)

    with pytest.raises(FrozenInstanceError):
        context.input_key = "changed"


def test_registry_covers_exactly_the_manifest_training_kinds():
    training_specs = node_kinds.list_by_category("training")

    assert {spec.kind for spec in training_specs} == {
        "classifier",
        "segment_cropper",
        "transformer",
        "opportunity_forecaster",
    }
    assert TRAINING_ROUTES == {spec.ui_route for spec in training_specs}
    assert "llm_training" not in TRAINING_ROUTES
    assert {
        route: type(component)
        for route, component in TRAINING_COMPONENTS.items()
    } == {
        "classifier": ClassifierTrainingComponent,
        "segment_cropper": SegmentCropperTrainingComponent,
        "transformer": TransformerTrainingComponent,
        "opportunity_forecaster": OpportunityForecasterTrainingComponent,
    }


def test_registry_rejects_duplicate_and_missing_registrations():
    complete = [
        (spec.kind, _CompleteComponent())
        for spec in node_kinds.list_by_category("training")
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate training component registration: classifier",
    ):
        build_training_component_registry(
            [*complete, ("classifier", _CompleteComponent())]
        )

    with pytest.raises(ValueError, match="missing: opportunity_forecaster"):
        build_training_component_registry(complete[:-1])


def test_unknown_pipeline_training_node_can_be_deleted(monkeypatch):
    from segment_tabs import pipeline_view

    unknown = TrainingNode(
        id="llm_train_v1",
        kind="llm_training",
        input_ref="llm_v1.output",
    )
    known = TrainingNode(
        id="classifier_v1",
        kind="classifier",
        input_ref="children_v1",
    )
    pipeline = SimpleNamespace(trainings=[known, unknown])
    streamlit = MagicMock()
    streamlit.button.return_value = True
    save_pipeline = MagicMock()
    monkeypatch.setattr(pipeline_view, "st", streamlit)
    monkeypatch.setattr(pipeline_view, "save_pipeline", save_pipeline)

    pipeline_view._render_training_card(pipeline, unknown, store=None)

    warning = streamlit.warning.call_args.args[0]
    assert "`llm_training` is no longer registered" in warning
    delete_button = streamlit.button.call_args
    assert delete_button.args[0] == "🗑 Delete unsupported component"
    assert delete_button.kwargs.get("disabled", False) is False
    assert pipeline.trainings == [known]
    save_pipeline.assert_called_once_with(pipeline)
    streamlit.rerun.assert_called_once_with()


def test_pipeline_training_dispatch_passes_node_and_resolved_input(monkeypatch):
    import segment_annotation_app as shell

    node = TrainingNode(
        id="classifier-round-1",
        kind="classifier",
        input_ref="labels",
    )
    pipeline = SimpleNamespace(
        training=MagicMock(return_value=node),
        resolve_source_key=MagicMock(return_value="resolved-labels"),
    )
    component = MagicMock(spec=TrainingComponent)
    streamlit = SimpleNamespace(
        session_state={"pipeline_training_node": node.id},
        warning=MagicMock(),
    )
    monkeypatch.setattr(shell, "st", streamlit)
    monkeypatch.setattr(shell, "get_training_component", lambda route: component)

    shell._open_training_component(pipeline, "classifier")

    component.open.assert_called_once_with(
        TrainingComponentContext(node=node, input_key="resolved-labels")
    )


def test_deep_link_and_view_only_training_dispatch(monkeypatch):
    import segment_annotation_app as shell

    node = TrainingNode(id="cropper", kind="segment_cropper", input_ref="laps")
    pipeline = SimpleNamespace(
        training=MagicMock(return_value=node),
        resolve_source_key=MagicMock(return_value="resolved-laps"),
    )
    component = MagicMock(spec=TrainingComponent)
    streamlit = SimpleNamespace(
        query_params={"view": "segment_cropper", "node": node.id},
        session_state={},
        warning=MagicMock(),
    )
    monkeypatch.setattr(shell, "st", streamlit)
    monkeypatch.setattr(shell, "get_training_component", lambda route: component)

    shell._restore_route_from_query(pipeline)
    shell._open_training_component(pipeline, "segment_cropper")

    component.open.assert_called_once_with(
        TrainingComponentContext(node=node, input_key="resolved-laps")
    )

    component.reset_mock()
    streamlit.query_params = {"view": "segment_cropper"}
    streamlit.session_state["pipeline_training_node"] = "stale-node"
    shell._restore_route_from_query(pipeline)
    shell._open_training_component(pipeline, "segment_cropper")

    component.open.assert_called_once_with(
        TrainingComponentContext(node=None, input_key=None)
    )


@pytest.mark.parametrize(
    ("component_type", "input_label"),
    [
        (ClassifierTrainingComponent, "Input annotation dataset"),
        (SegmentCropperTrainingComponent, "Complete session dataset"),
        (TransformerTrainingComponent, "Input annotation dataset"),
        (OpportunityForecasterTrainingComponent, "Input annotation dataset"),
    ],
)
@pytest.mark.parametrize(
    ("input_key", "expected_key"),
    [(None, ""), ("pipeline-dataset", "pipeline-dataset")],
)
def test_training_components_display_only_the_resolved_pipeline_input(
    monkeypatch,
    component_type,
    input_label,
    input_key,
    expected_key,
):
    module = sys.modules[component_type.__module__]
    show_input_location = MagicMock()
    monkeypatch.setattr(module, "st", MagicMock())
    monkeypatch.setattr(module, "show_node_context", MagicMock())
    monkeypatch.setattr(module, "show_input_location", show_input_location)
    monkeypatch.setattr(module, "render_card", MagicMock())
    node = TrainingNode(
        id="training-node",
        kind="classifier",
        input_ref="source" if input_key else "",
    )
    context = TrainingComponentContext(node=node, input_key=input_key)

    component_type().open(context)

    assert resolve_input_key(context) == expected_key
    show_input_location.assert_called_once_with(input_label, expected_key)


@pytest.mark.parametrize(
    ("component_type", "column_count"),
    [
        (ClassifierTrainingComponent, 4),
        (SegmentCropperTrainingComponent, 3),
        (TransformerTrainingComponent, 0),
        (OpportunityForecasterTrainingComponent, 2),
    ],
)
def test_training_components_disable_start_and_do_not_spawn_without_input(
    monkeypatch,
    component_type,
    column_count,
):
    module = sys.modules[component_type.__module__]
    streamlit = _form_streamlit([MagicMock() for _ in range(column_count)])
    streamlit.session_state = {}
    spawn = MagicMock()
    monkeypatch.setattr(module, "st", streamlit)
    monkeypatch.setattr(module, "spawn", spawn)
    if component_type is ClassifierTrainingComponent:
        get_available_sessions = MagicMock()
        monkeypatch.setattr(module, "get_available_sessions", get_available_sessions)

    component_type._render_form("")

    assert streamlit.form_submit_button.call_args.kwargs["disabled"] is True
    assert not streamlit.text_input.called
    spawn.assert_not_called()
    if component_type is ClassifierTrainingComponent:
        get_available_sessions.assert_not_called()


def test_classifier_form_defaults_and_command(monkeypatch):
    from segment_tabs.training_components import classifier as module

    columns = [MagicMock() for _ in range(4)]
    columns[0].number_input.return_value = 10
    columns[1].number_input.return_value = 32
    columns[2].number_input.return_value = 1e-3
    columns[3].slider.return_value = 0.1
    streamlit = _form_streamlit(columns)
    streamlit.session_state = {}
    streamlit.multiselect.return_value = ["session-a", "session-b"]
    spawn = MagicMock()
    monkeypatch.setattr(module, "st", streamlit)
    monkeypatch.setattr(module, "get_available_sessions", lambda key: ["session-a", "session-b"])
    monkeypatch.setattr(module, "spawn", spawn)

    ClassifierTrainingComponent._render_form("labels")

    assert columns[0].number_input.call_args.kwargs["value"] == 10
    assert columns[1].number_input.call_args.kwargs["value"] == 32
    assert columns[2].number_input.call_args.kwargs["value"] == 1e-3
    assert columns[3].slider.call_args.args[3:] == (0.1, 0.05)
    assert streamlit.form_submit_button.call_args.kwargs["disabled"] is False
    assert not streamlit.text_input.called
    spawn.assert_called_once_with(
        "classifier",
        [
            sys.executable,
            "-u",
            str(module.TRAINING_ENTRYPOINTS / "train_segment_classifier.py"),
            "--epochs",
            "10",
            "--batch-size",
            "32",
            "--lr",
            "0.001",
            "--val-split",
            "0.1",
            "--annotation-key",
            "labels",
            "--session-id",
            "session-a",
            "--session-id",
            "session-b",
        ],
    )


def test_segment_cropper_form_defaults_and_command(monkeypatch):
    from segment_tabs.training_components import segment_cropper as module

    columns = [MagicMock() for _ in range(3)]
    columns[0].number_input.return_value = 10
    columns[1].number_input.return_value = 8
    columns[2].number_input.return_value = 1e-3
    streamlit = _form_streamlit(columns)
    spawn = MagicMock()
    monkeypatch.setattr(module, "st", streamlit)
    monkeypatch.setattr(module, "spawn", spawn)

    SegmentCropperTrainingComponent._render_form("complete-sessions")

    assert columns[0].number_input.call_args.kwargs["value"] == 10
    assert columns[1].number_input.call_args.kwargs["value"] == 8
    assert columns[2].number_input.call_args.kwargs["value"] == 1e-3
    assert streamlit.form_submit_button.call_args.kwargs["disabled"] is False
    assert not streamlit.text_input.called
    spawn.assert_called_once_with(
        "segment_cropper",
        [
            sys.executable,
            "-u",
            str(module.TRAINING_ENTRYPOINTS / "train_segment_cropper.py"),
            "--epochs",
            "10",
            "--batch-size",
            "8",
            "--lr",
            "0.001",
            "--annotation-key",
            "complete-sessions",
        ],
    )


def test_transformer_form_default_and_command(monkeypatch):
    from segment_tabs.training_components import transformer as module

    streamlit = _form_streamlit([])
    spawn = MagicMock()
    monkeypatch.setattr(module, "st", streamlit)
    monkeypatch.setattr(module, "spawn", spawn)

    TransformerTrainingComponent._render_form("segments")

    assert streamlit.form_submit_button.call_args.kwargs["disabled"] is False
    assert not streamlit.text_input.called
    spawn.assert_called_once_with(
        "transformer",
        [
            sys.executable,
            "-u",
            str(module.TRAINING_ENTRYPOINTS / "train_transformer_guidance.py"),
            "--annotation-key",
            "segments",
        ],
    )


def test_opportunity_forecaster_form_defaults_and_command(monkeypatch):
    from segment_tabs.training_components import opportunity_forecaster as module

    columns = [MagicMock() for _ in range(2)]
    columns[0].slider.return_value = 0.5
    columns[1].number_input.return_value = 5000
    streamlit = _form_streamlit(columns)
    spawn = MagicMock()
    monkeypatch.setattr(module, "st", streamlit)
    monkeypatch.setattr(module, "spawn", spawn)

    OpportunityForecasterTrainingComponent._render_form("opportunities")

    assert columns[0].slider.call_args.kwargs["value"] == 0.5
    assert columns[1].number_input.call_args.kwargs["value"] == 5000
    assert streamlit.form_submit_button.call_args.kwargs["disabled"] is False
    assert not streamlit.text_input.called
    spawn.assert_called_once_with(
        "opportunity_forecaster",
        [
            sys.executable,
            "-u",
            str(module.TRAINING_ENTRYPOINTS / "train_opportunity_forecaster.py"),
            "--annotation-key",
            "opportunities",
            "--input-fraction",
            "0.5",
            "--max-negatives",
            "5000",
        ],
    )
