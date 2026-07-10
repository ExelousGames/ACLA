from types import SimpleNamespace

from ui.segment_tabs.detailed import _resolve_loaded_annotation_selection


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
