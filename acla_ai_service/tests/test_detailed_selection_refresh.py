from types import SimpleNamespace

from ui.segment_tabs.detailed import (
    _annotation_index_limits,
    _resolve_loaded_annotation_selection,
    _segments_to_positioned_dataframe,
)
from ui.segment_tabs.components import detailed_annotation_manager


def _segment(segment_id, parent_id=None):
    return SimpleNamespace(
        id=segment_id,
        parent_id=parent_id,
        start_index=0,
        end_index=10,
        labels=[],
        telemetry_data=[],
    )


def test_resolve_loaded_annotation_selection_keeps_current_root_index():
    annotations = [_segment("parent-1"), _segment("parent-2")]

    assert _resolve_loaded_annotation_selection(annotations, 1) == 1


def test_resolve_loaded_annotation_selection_uses_next_available_root_index():
    annotations = [
        _segment("parent-1"),
        _segment("child-1", parent_id="parent-1"),
        _segment("parent-3"),
    ]

    assert _resolve_loaded_annotation_selection(annotations, 1) == 2


def test_resolve_loaded_annotation_selection_uses_last_root_when_past_end():
    annotations = [
        _segment("parent-1"),
        _segment("child-1", parent_id="parent-1"),
        _segment("parent-2"),
    ]

    assert _resolve_loaded_annotation_selection(annotations, 99) == 2


def test_resolve_loaded_annotation_selection_uses_first_root_without_prior_selection():
    annotations = [_segment("parent-1"), _segment("parent-2")]

    assert _resolve_loaded_annotation_selection(annotations, None) == 0


def test_positioned_dataframe_ends_at_last_telemetry_index():
    segment = _segment("parent-1")
    segment.start_index = 8
    segment.end_index = 29
    segment.telemetry_data = [
        {"speed": index}
        for index in range(8, 29)
    ]

    dataframe = _segments_to_positioned_dataframe([segment])

    assert dataframe.index[-1] == 28
    assert dataframe.loc[28, "speed"] == 28
    assert 29 not in dataframe.index


def test_annotation_end_limit_allows_the_exclusive_dataframe_boundary():
    assert _annotation_index_limits(1251) == (0, 1251)


def test_clear_annotation_manager_state_removes_session_specific_fields(monkeypatch):
    session_state = {
        "detailed_form_start_0": 10,
        "detailed_form_labels_0_Main Labels": ["Braking"],
        "detailed_calc_feat_0": "speed_kmh",
        "detailed_roc_smooth_0": 5,
        "manage_subsegment_selector": "old segment",
        "sub_start_new": 12,
        "sub_end_new": 18,
        "sub_labels_new": ["MSP"],
        "sub_notes_new": "old notes",
        "detailed_interaction_focus_car": {"slot": 2},
        "detailed_opponent_interaction_target": {"slot": 2},
        "detailed_annotation_selector": 0,
        "detailed_graph_ids": [0, 1],
    }
    monkeypatch.setattr(
        detailed_annotation_manager,
        "st",
        SimpleNamespace(session_state=session_state),
    )
    monkeypatch.setattr(
        detailed_annotation_manager,
        "clear_agent_annotation_review_state",
        lambda: session_state.pop("agent_annot_result", None),
    )
    session_state["agent_annot_result"] = object()

    detailed_annotation_manager.clear_annotation_manager_state()

    assert session_state == {
        "detailed_annotation_selector": 0,
        "detailed_graph_ids": [0, 1],
    }
