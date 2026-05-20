import streamlit as st
from typing import Any, Dict, List

from app.domain.labels import LABEL_MAPPING


def _humanize(label_ids: List[str]) -> List[str]:
    return [f"{LABEL_MAPPING.get(str(l), l)} ({l})" for l in label_ids]


def render_unit_labels_section(unit: Dict[str, Any]) -> None:
    """Render a training unit's labels (parent + children separately)."""
    if unit["kind"] == "parent_with_children":
        st.subheader("Unit Labels (parent + children)")
        st.info(f"ℹ️ Parent segment with **{unit['child_count']}** child(ren).")
        st.markdown("**Parent labels:**")
        st.write(_humanize(unit["parent_label_ids"]) or ["(none)"])
        for i, child_labels in enumerate(unit["children_label_ids"], start=1):
            st.markdown(f"**Child {i} labels:**")
            st.write(_humanize(child_labels) or ["(none)"])
    else:
        st.subheader("Unit Labels (isolated segment)")
        st.write(_humanize(unit["parent_label_ids"]) or ["(none)"])
