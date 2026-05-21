import streamlit as st
import pandas as pd

from ..shared import get_display_labels, save_annotations

def render_list_view(session_id, selected_annotation_key):
    # List View
    if st.toggle("Show Current Session Annotations List"):
        st.subheader("Current Session Annotations List")
        if st.session_state.current_annotations:
            display_data = []
            
            # Determine filter range if a segment is selected
            filter_range = None
            selected_option = st.session_state.get("detailed_annotation_selector", "Create New")
            if selected_option != "Create New":
                sel_ann = st.session_state.current_annotations[selected_option]
                filter_range = (sel_ann.start_index, sel_ann.end_index)

            for i, ann in enumerate(st.session_state.current_annotations):
                # Apply filter if set
                if filter_range:
                    if not (ann.start_index >= filter_range[0] and ann.end_index <= filter_range[1]):
                        continue

                d = ann.to_dict()
                d["Annotation ID"] = i
                d["labels"] = ", ".join(get_display_labels(ann.labels))
                if "telemetry_data" in d:
                    del d["telemetry_data"]
                display_data.append(d)
            
            if display_data:
                st.dataframe(pd.DataFrame(display_data).set_index("Annotation ID"), width='stretch')
            else:
                st.info("No segments found inside the selected annotation.")
        else:
            st.info("No annotations added yet.")
        
    if st.button("Force Save All", key="detailed_force_save"):
        save_annotations(session_id, st.session_state.current_annotations, selected_annotation_key)
