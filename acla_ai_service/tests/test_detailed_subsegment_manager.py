from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from ui.segment_tabs.components import detailed_subsegment_manager


class _SessionState(dict):
    __getattr__ = dict.__getitem__


def test_manage_subsegment_does_not_render_parent_segment_section(monkeypatch):
    parent = SimpleNamespace(
        id="parent-1",
        parent_id=None,
        start_index=2,
        end_index=12,
        labels=["RM"],
    )
    streamlit = MagicMock()
    streamlit.session_state = _SessionState(
        current_annotations=[parent],
        detailed_annotation_selector=0,
    )
    streamlit.selectbox.return_value = "Create New Sub-Segment"
    streamlit.columns.return_value = [nullcontext(), nullcontext()]
    streamlit.button.return_value = False
    monkeypatch.setattr(detailed_subsegment_manager, "st", streamlit)

    detailed_subsegment_manager.render_subsegment_manager(
        range(20),
        "session-1",
        "annotations",
    )

    rendered_markdown = [call.args[0] for call in streamlit.markdown.call_args_list]
    assert all("Parent Segment" not in text for text in rendered_markdown)


def test_manage_subsegment_places_range_after_id(monkeypatch):
    parent = SimpleNamespace(
        id="parent-1",
        parent_id=None,
        start_index=2,
        end_index=12,
        labels=["PARENT"],
    )
    child = SimpleNamespace(
        id="child-1",
        parent_id="parent-1",
        start_index=4,
        end_index=8,
        labels=["CHILD"],
    )
    streamlit = MagicMock()
    streamlit.session_state = _SessionState(
        current_annotations=[parent, child],
        detailed_annotation_selector=0,
    )
    streamlit.selectbox.return_value = "Create New Sub-Segment"
    streamlit.columns.return_value = [nullcontext(), nullcontext()]
    streamlit.button.return_value = False
    monkeypatch.setattr(detailed_subsegment_manager, "st", streamlit)
    monkeypatch.setattr(
        detailed_subsegment_manager,
        "get_display_labels",
        lambda labels: labels,
    )

    detailed_subsegment_manager.render_subsegment_manager(
        range(20),
        "session-1",
        "annotations",
    )

    assert streamlit.selectbox.call_args.kwargs["options"] == [
        "Create New Sub-Segment",
        "0: (4-8) CHILD",
    ]
