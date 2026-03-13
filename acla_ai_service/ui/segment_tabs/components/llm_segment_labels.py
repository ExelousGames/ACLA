import streamlit as st
from app.models.segment_models import LABEL_MAPPING
from typing import List

def render_segment_labels_section(labels: List[str], sub_segment_count: int = 0):
    """Render the segment labels section."""
    st.subheader("Segment Labels")
    
    if sub_segment_count > 0:
        st.info(f"ℹ️ **Contains {sub_segment_count} Sub-segment(s):** Labels from sub-segments have been included below automatically.")
    
    # Lookup full names
    full_labels = []
    human_readable_labels = []
    for l in labels:
        full_name = LABEL_MAPPING.get(str(l), l)
        full_labels.append(f"{full_name} ({l})")
        human_readable_labels.append(full_name)
        
    st.write(full_labels)
