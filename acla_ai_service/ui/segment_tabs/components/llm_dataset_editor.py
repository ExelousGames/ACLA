import streamlit as st
import json
import time
from typing import List, Any
from segment_tabs.components.llm_training_data_generation import generate_training_prompt, get_human_readable_labels, get_available_prompts

def render_dataset_editor_tab(
    labels: List[Any],
    existing_record: dict,
    prompt_widget_key: str,
    desc_widget_key: str,
    segment_id: str,
    output_path: str,
    total_segments: int,
    save_training_pair_fn,
    load_all_annotations_df_fn
):
    st.subheader("Manual Annotation")
    
    available_prompts = get_available_prompts(labels)
    
    def on_template_change():
        idx = st.session_state[f"template_select_{segment_id}"]
        st.session_state[prompt_widget_key] = generate_training_prompt(labels, idx)

    st.selectbox(
        "Select Prompt Template:", 
        range(len(available_prompts)), 
        format_func=lambda x: f"Template {x+1}",
        key=f"template_select_{segment_id}",
        on_change=on_template_change
    )
    
    default_prompt = generate_training_prompt(labels, st.session_state.get(f"template_select_{segment_id}", 0))
    
    if f"pending_{desc_widget_key}" in st.session_state:
        st.session_state[desc_widget_key] = st.session_state.pop(f"pending_{desc_widget_key}")
    
    if prompt_widget_key not in st.session_state:
        if existing_record:
            st.session_state[prompt_widget_key] = existing_record.get("prompt", default_prompt)
        else:
            st.session_state[prompt_widget_key] = default_prompt

    if desc_widget_key not in st.session_state:
        if existing_record:
            st.session_state[desc_widget_key] = existing_record.get("completion", "")
        else:
            st.session_state[desc_widget_key] = ""

    prompt_text = st.text_area(
        "Input Prompt (Instruction):",
        value=st.session_state[prompt_widget_key],
        height=100,
        key=prompt_widget_key
    )

    description = st.text_area(
        "Target Output (Description):", 
        value=st.session_state[desc_widget_key],
        height=150,
        key=desc_widget_key
    )
    
    if st.button("Save Training Pair", type="primary"):
        current_label_ids = [str(l) for l in labels]
        human_readable_labels = get_human_readable_labels(labels)
        save_training_pair_fn(segment_id, prompt_text, description, human_readable_labels, current_label_ids, str(output_path))
        st.success("Saved!")
        time.sleep(0.5)
        st.session_state.current_index = min(total_segments - 1, st.session_state.current_index + 1)
        st.rerun()

    st.divider()

    st.subheader("Dataset Editor")
    st.markdown("Edit existing training pairs. Changes are saved when you click 'Save Dataset Changes'.")
    
    df = load_all_annotations_df_fn(str(output_path))
    if not df.empty and "segment_id" in df.columns:
        segment_df = df[df["segment_id"] == segment_id].copy()
        
        if not segment_df.empty:
            segment_df.insert(0, "Delete", False)
            edited_segment_df = st.data_editor(
                segment_df,
                num_rows="dynamic",
                width="stretch",
                height=500,
                key=f"dataset_editor_{segment_id}",
                disabled=("segment_id", "labels", "label_ids", "timestamp")
            )
            if st.button("Save Dataset Changes", type="secondary"):
                other_segments_df = df[df["segment_id"] != segment_id].copy()
                
                with open(output_path, "w") as f:
                    # Write back other segments
                    for _, row in other_segments_df.iterrows():
                        row_dict = row.dropna().to_dict()
                        f.write(json.dumps(row_dict) + "\n")
                    
                    # Write updated segment data
                    for _, row in edited_segment_df.iterrows():
                        if row.get("Delete", False):
                            continue
                        # Remove Delete column before saving
                        if "Delete" in row:
                            row = row.drop("Delete")
                        row_dict = row.dropna().to_dict()
                        f.write(json.dumps(row_dict) + "\n")
                        
                st.success("Successfully updated dataset!")
                time.sleep(1)
                st.rerun()
        else:
            st.info("No annotations found for this segment yet.")
    else:
        st.info("The dataset is currently empty. Add pairs using the above form.")