from types import SimpleNamespace

import pandas as pd

from app.local_annotation_agent.workflow import deterministic
from app.local_annotation_agent.workflow.deterministic_engine import (
    HalfOpenRange,
    PredicateEvaluation,
    RequirementBranchEvaluation,
    RequirementEvaluation,
)
from ui.segment_tabs import batch_subsegment
from ui.segment_tabs.shared import build_segment


def test_batch_children_inherit_main_labels_but_not_parent_segment_type(monkeypatch):
    saved_children = []

    def fake_build_segment(_df, **kwargs):
        child = SimpleNamespace(labels=kwargs["label_ids"])
        saved_children.append(child)
        return child

    monkeypatch.setattr(batch_subsegment, "build_segment", fake_build_segment)
    monkeypatch.setattr(batch_subsegment, "save_annotations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        batch_subsegment,
        "st",
        SimpleNamespace(session_state={"current_annotations": []}),
    )

    parent = SimpleNamespace(id="parent-1", labels=["RM", "ST2"])
    result = SimpleNamespace(label_annotations=[
        {
            "label_id": "RM7",
            "start_index": 1,
            "end_index": 5,
            "reasoning": "matched sub-label",
        },
        {
            "label_id": "ST14",
            "start_index": 1,
            "end_index": 5,
            "reasoning": "matched segment type",
        },
    ])

    count = batch_subsegment._persist_children_for_parent(
        parent,
        result,
        "session-1",
        "annotations",
        pd.DataFrame(index=range(6)),
    )

    assert count == 1
    assert saved_children[0].labels == ["RM", "RM7", "ST14"]


def test_deterministic_end_row_is_embedded_with_exclusive_persistence_bound(
    monkeypatch,
):
    final_row = 4
    branch = RequirementBranchEvaluation(0, [
        PredicateEvaluation(
            True,
            "trajectory: True",
            "fact",
            HalfOpenRange(2, final_row + 1),
        ),
    ])
    matched = RequirementEvaluation(
        True, branch.branch, branch.passed, [], [branch],
    )

    def fake_evaluate(label_ids, _context, _scope):
        if "RM7" in label_ids:
            return deterministic.LabelEvaluation(["RM7"], {"RM7": matched})
        return deterministic.LabelEvaluation([], {})

    monkeypatch.setattr(deterministic, "evaluate_labels", fake_evaluate)
    df = pd.DataFrame({"row": range(6)})
    result = deterministic.calculate_detailed_annotation(
        df,
        parent_start=0,
        parent_end=len(df),
        parent_main_labels=["RM"],
    )
    proposal = next(
        item for item in result.label_annotations if item["label_id"] == "RM7"
    )

    segment = build_segment(
        df,
        start=proposal["start_index"],
        end=proposal["end_index"],
        label_ids=[proposal["label_id"]],
    )

    assert segment.telemetry_data[-1]["row"] == final_row
    assert segment.end_index == final_row + 1
    assert segment.segment_length == len(segment.telemetry_data) == (
        segment.end_index - segment.start_index
    )


def test_delete_selected_parent_subsegments_preserves_annotations_outside_range(monkeypatch):
    parent_1 = SimpleNamespace(id="parent-1", parent_id=None)
    parent_2 = SimpleNamespace(id="parent-2", parent_id=None)
    selected_child_1 = SimpleNamespace(id="child-1", parent_id="parent-1")
    selected_child_2 = SimpleNamespace(id="child-2", parent_id="parent-1")
    unselected_child = SimpleNamespace(id="child-3", parent_id="parent-2")
    orphaned_child = SimpleNamespace(id="child-4", parent_id="missing-parent")
    annotations = [
        parent_1,
        parent_2,
        selected_child_1,
        unselected_child,
        selected_child_2,
        orphaned_child,
    ]
    saved = []

    monkeypatch.setattr(
        batch_subsegment,
        "st",
        SimpleNamespace(session_state={"current_annotations": annotations}),
    )
    monkeypatch.setattr(
        batch_subsegment,
        "save_annotations",
        lambda *args, **kwargs: saved.append((args, kwargs)),
    )

    deleted = batch_subsegment._delete_selected_parent_subsegments(
        "session-1",
        "annotations",
        {"parent-1"},
    )

    remaining = [parent_1, parent_2, unselected_child, orphaned_child]
    assert deleted == 2
    assert batch_subsegment.st.session_state["current_annotations"] == remaining
    assert saved == [
        (("session-1", remaining, "annotations"), {"silent": True}),
    ]


def test_delete_selected_parent_subsegments_does_not_save_without_matches(monkeypatch):
    parent = SimpleNamespace(id="parent-1", parent_id=None)
    child = SimpleNamespace(id="child-1", parent_id="parent-1")
    annotations = [parent, child]
    saved = []

    monkeypatch.setattr(
        batch_subsegment,
        "st",
        SimpleNamespace(session_state={"current_annotations": annotations}),
    )
    monkeypatch.setattr(
        batch_subsegment,
        "save_annotations",
        lambda *args, **kwargs: saved.append((args, kwargs)),
    )

    assert batch_subsegment._delete_selected_parent_subsegments(
        "session-1",
        "annotations",
        set(),
    ) == 0
    assert batch_subsegment._delete_selected_parent_subsegments(
        "session-1",
        "annotations",
        {"missing-parent"},
    ) == 0
    assert batch_subsegment.st.session_state["current_annotations"] == annotations
    assert saved == []
