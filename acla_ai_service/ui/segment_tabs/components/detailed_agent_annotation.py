"""Dispatcher for the VLM sub-segment discovery feature.

Renders both backends as independent expanders, followed by the shared
staged-review panel:

- 🖥️  Local VLM   → ``detailed_agent_annotation_local``
- ☁️  Claude       → ``detailed_agent_annotation_claude``
- Staged review   → ``_agent_annotation_shared.render_staged_review``

Both backends write into ``st.session_state['agent_annot_result']`` so
the staged-review panel is last-run-wins.
"""

from ._agent_annotation_shared import (
    collect_parent_info,
    render_followup_chat,
    render_staged_review,
)
from .detailed_agent_annotation_claude import render_agent_annotation_claude
from .detailed_agent_annotation_local import render_agent_annotation_local


def render_agent_annotation(
    df,
    form_start,
    form_end,
    form_labels,
    session_id,
    selected_annotation_key,
):
    """Render both VLM backends and the shared staged-review panel."""
    render_agent_annotation_local(
        df, form_start, form_end, form_labels, session_id, selected_annotation_key,
    )
    render_agent_annotation_claude(
        df, form_start, form_end, form_labels, session_id, selected_annotation_key,
    )

    parent_id, _, _ = collect_parent_info(form_labels)
    render_staged_review(
        parent_id, session_id, selected_annotation_key, form_start, form_end,
        df=df,
    )
    render_followup_chat()
