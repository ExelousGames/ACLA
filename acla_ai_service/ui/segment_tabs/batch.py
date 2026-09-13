import streamlit as st
import time
import pandas as pd
import copy

from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_available_sessions, load_session_segments,
    LABEL_NAME_TO_ID,
    LABEL_CATEGORIES
)

def render_rule_based_annotation(df, selected_annotation_key):
    """
    Renders the Rule-Based Annotation section.
    Allows user to select a feature, a value, and a label.
    If any point in the segment matches the feature value, the label is applied.
    """
    st.header("Rule-Based Annotation")
    st.write("Automatically apply a label to segments where a selected feature matches a specific value.")
    
    # Ensure annotations exist
    if not st.session_state.get("current_annotations"):
        st.info("No segments available to process.")
        return

    # 1. Select Feature
    if df.empty:
        st.warning("Dataframe is empty.")
        return
        
    features = sorted(df.columns.tolist())
    selected_feature = st.selectbox("Select Feature", features, key="rule_feature_select")
    
    # 2. Input Value
    col_dtype = df[selected_feature].dtype
    
    # Heuristic to detect if "object" column is actually numeric
    is_numeric_col = pd.api.types.is_numeric_dtype(col_dtype)
    if not is_numeric_col and col_dtype == 'object':
        try:
            # Check first non-null value
            first_valid = df[selected_feature].dropna().iloc[0]
            if isinstance(first_valid, (int, float, complex)) or (isinstance(first_valid, str) and first_valid.replace('.','',1).isdigit()):
                 is_numeric_col = True
        except:
            pass

    # Unified Input Logic: Offer suggestions (selectbox) for ANY column type if low cardinality
    target_value_str = ""
    try:
         # Limit unique values for performance
         unique_raw = df[selected_feature].dropna().unique()
         type_info = f"{col_dtype} - treated as Numeric" if is_numeric_col else str(col_dtype)

         if len(unique_raw) < 1000:
             # Convert all to string for display/selection
             # Sort carefully if numeric to ensure logical ordering in dropdown
             if is_numeric_col:
                 # Sort numerically first, then stringify
                 try:
                    # Preserve numeric sort order!
                    unique_vals = [str(v) for v in sorted(unique_raw)]
                 except:
                    # Fallback to string sort
                    unique_vals = sorted([str(v) for v in unique_raw])
             else:
                 unique_vals = sorted([str(v) for v in unique_raw])
             
             # Use selectbox for "closest matching" feel via search
             # index=None makes it start with no selection (placeholder)
             target_value_str = st.selectbox(
                 f"Select Target Value (Type: {type_info})", 
                 unique_vals, 
                 index=None,
                 placeholder="Type to search...",
                 key="rule_value_select"
            )
             
             # If user hasn't selected anything yet, default to empty string so logic downstream handles it gracefully
             if target_value_str is None:
                 target_value_str = ""
         else:
             target_value_str = st.text_input(f"Target Value (Type: {type_info})", key="rule_value_input")
    except Exception:
         # Fallback if error getting unique values
         target_value_str = st.text_input(f"Target Value (Type: {col_dtype})", key="rule_value_input_fallback")
    
    # 3. Select Label
    # LABEL_NAME_TO_ID maps "Label Name" -> "Label ID"
    label_names = sorted(list(LABEL_NAME_TO_ID.keys()))
    selected_label_name = st.selectbox("Select Label", label_names, key="rule_label_select")
    
    # 4. Apply Rule Logic
    col_rule_1, col_rule_2 = st.columns([1, 4])
    with col_rule_1:
        apply_clicked = st.button("Apply Rule", key="rule_apply_btn")
    
    with col_rule_2:
        if st.session_state.get("last_rule_snapshot"):
            if st.button("Undo Last Change", key="rule_undo_btn"):
                st.session_state.current_annotations = st.session_state.last_rule_snapshot
                st.session_state.last_rule_snapshot = None # Clear after undo
                st.success("Reverted last rule application.")
                
                # Persist reversion
                if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                    save_annotations(
                        st.session_state.last_session_id,
                        st.session_state.current_annotations,
                        st.session_state.last_annotation_key,
                        silent=False
                    )
                time.sleep(1)
                st.rerun()

    if apply_clicked:
        if target_value_str == "":
            st.warning("Please enter a target value.")
            return
            
        # target_val preparation
        target_val = target_value_str
        try:
            if is_numeric_col:
                target_val = float(target_value_str)
            elif pd.api.types.is_bool_dtype(col_dtype):
                 if target_value_str.lower() in ['true', 't', '1', 'yes']:
                     target_val = True
                 elif target_value_str.lower() in ['false', 'f', '0', 'no']:
                     target_val = False
        except ValueError:
            st.error(f"Could not convert '{target_value_str}' for column type.")
            return

        selected_label_id = LABEL_NAME_TO_ID[selected_label_name]
        
        # Save snapshot for undo
        st.session_state.last_rule_snapshot = copy.deepcopy(st.session_state.current_annotations)
        
        count_updated = 0
        count_matching_value = 0
        total_segments = len(st.session_state.current_annotations)
        progress_bar = st.progress(0)
        
        for i, ann in enumerate(st.session_state.current_annotations):
            # Check bounds
            start = ann.start_index if ann.start_index is not None else 0
            end = ann.end_index if ann.end_index is not None else len(df)
            
            if start < 0 or end > len(df) or start >= end:
                continue
                
            segment_data = df.iloc[start:end]
            segment_series = segment_data[selected_feature]
            
            # If we are treating as numeric but column is object, try to convert segment series
            if is_numeric_col and segment_series.dtype == 'object':
                 segment_series = pd.to_numeric(segment_series, errors='coerce')

            # Comparison
            try:
                # Basic any() check
                # For numeric, direct equality can be tricky with floats, but fits "== input value" request.
                if (segment_series == target_val).any():
                    count_matching_value += 1
                    current_labels = set(ann.labels)
                    if selected_label_id not in current_labels:
                        ann.labels.append(selected_label_id)
                        count_updated += 1
            except Exception:
                continue
            
            progress_bar.progress((i + 1) / total_segments)
            
        if count_updated > 0:
            st.success(f"Updated {count_updated} segments with label '{selected_label_name}'.")
            # Save functionality
            if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                save_annotations(
                    st.session_state.last_session_id,
                    st.session_state.current_annotations,
                    st.session_state.last_annotation_key,
                    silent=False
                )
                time.sleep(1) # Give time for user to see success
                st.rerun()
        elif count_matching_value > 0:
            st.warning(f"Found {count_matching_value} matching segments, but they already have the label '{selected_label_name}'.")
        else:
            st.info(f"No segments matched the value '{target_value_str}' for feature '{selected_feature}'.")

    _render_parent_label_propagation(selected_annotation_key)


def _render_parent_label_propagation(selected_annotation_key):
    """Add each parent's labels to its direct children using ``parent_id``."""
    st.markdown("---")
    st.subheader("Add Parent Labels to Children")
    st.caption("Matches each child segment to its parent by parent_id and appends any missing parent labels.")

    annotations = st.session_state.get("current_annotations") or []
    if not annotations:
        st.info("No segments available.")
        return

    parents_by_id = {
        getattr(ann, "id", None): ann
        for ann in annotations
        if getattr(ann, "id", None)
    }
    children_with_parent = [
        ann for ann in annotations
        if getattr(ann, "parent_id", None)
    ]
    eligible_children = [
        child for child in children_with_parent
        if getattr(child, "parent_id", None) in parents_by_id
        and _missing_parent_labels(child, parents_by_id)
    ]
    missing_parents = sum(
        1 for child in children_with_parent
        if getattr(child, "parent_id", None) not in parents_by_id
    )

    st.caption(
        f"{len(children_with_parent)} child segment(s), "
        f"{len(eligible_children)} need parent labels."
    )
    if missing_parents:
        st.caption(f"{missing_parents} child segment(s) reference a missing parent.")

    apply_clicked = st.button(
        "Add Parent Labels to Children",
        key="rule_parent_labels_apply_btn",
        disabled=not eligible_children,
    )
    if not apply_clicked:
        return

    st.session_state.last_rule_snapshot = copy.deepcopy(annotations)

    children_updated = 0
    labels_added = 0
    for child in children_with_parent:
        added_for_child = _missing_parent_labels(child, parents_by_id)
        if not added_for_child:
            continue

        child.labels = list(getattr(child, "labels", []) or []) + added_for_child
        children_updated += 1
        labels_added += len(added_for_child)

    if children_updated == 0:
        st.info("All children already include their parent labels.")
        return

    st.success(
        f"Added {labels_added} parent label(s) to {children_updated} child segment(s)."
    )
    if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
        save_annotations(
            st.session_state.last_session_id,
            st.session_state.current_annotations,
            selected_annotation_key,
            silent=False,
        )
        time.sleep(1)
        st.rerun()


def _missing_parent_labels(child, parents_by_id):
    parent = parents_by_id.get(getattr(child, "parent_id", None))
    if parent is None:
        return []

    child_labels = list(getattr(child, "labels", []) or [])
    child_label_set = set(child_labels)
    missing_labels = []
    for label in getattr(parent, "labels", []) or []:
        if label in child_label_set:
            continue
        missing_labels.append(label)
        child_label_set.add(label)
    return missing_labels


def _render_rule_session_data_table(
    df,
    session_id,
    selected_session_key,
    available_sessions,
):
    """Show raw telemetry rows for any session while staying on Rule-Based Annotation."""
    st.markdown("---")
    st.subheader("Session Data")

    if not available_sessions:
        st.info("No sessions available to display.")
        return

    table_index = 0
    if session_id in available_sessions:
        table_index = available_sessions.index(session_id)

    col_select, _ = st.columns([1, 3])
    with col_select:
        table_session_id = st.selectbox(
            "View Session Data",
            options=available_sessions,
            index=table_index,
            key="rule_session_data_selector",
        )

    if table_session_id == session_id:
        table_df = df
    else:
        with st.spinner(f"Loading session {table_session_id} data..."):
            table_df = load_session_data(selected_session_key, table_session_id)

    if table_df.empty:
        st.warning("Selected session has no data.")
        return

    st.caption(
        f"{table_session_id} | {len(table_df):,} rows | "
        f"{len(table_df.columns):,} columns"
    )
    st.dataframe(table_df, hide_index=False, width="stretch", height=420)


def render_bulk_label_utils(selected_annotation_key):
    """
    Renders bulk utilities like removing a specific label from all segments.
    """
    st.header("Bulk Label Management")
    st.write("Perform operations on all segments in the current session.")
    
    # Ensure current annotations exist
    if not st.session_state.get("current_annotations"):
        st.info("No segments available.")
        return

    # --- Remove Specific Label ---
    st.subheader("Remove Label from All Segments")
    st.caption("Select a label to remove from every segment where it exists.")
    
    label_names = sorted(list(LABEL_NAME_TO_ID.keys()))
    if not label_names:
        st.info("No labels configured in LABEL_NAME_TO_ID.")
        return

    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_label_name = st.selectbox("Select Label to Remove", label_names, key="bulk_remove_label_select")
    
    if st.button("Remove Label from All Segments", key="bulk_remove_label_btn", type="primary"):
        selected_label_id = LABEL_NAME_TO_ID[selected_label_name]
        count_removed = 0
        
        for ann in st.session_state.current_annotations:
            if selected_label_id in ann.labels:
                # Remove all instances of the label (just in case of duplicates)
                ann.labels = [lid for lid in ann.labels if lid != selected_label_id]
                count_removed += 1
        
        if count_removed > 0:
            st.success(f"Removed label '{selected_label_name}' from {count_removed} segments.")
            # Save functionality
            if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                save_annotations(
                    st.session_state.last_session_id,
                    st.session_state.current_annotations,
                    st.session_state.last_annotation_key,
                    silent=False
                )
                time.sleep(1) # Give time for user to see success
                st.rerun()
        else:
            st.info(f"Label '{selected_label_name}' was not found in any segment.")


def render_classifier_auto_annotation(df, selected_annotation_key, session_id):
    """
    Renders the Segment Classifier Auto-Annotation section.
    Allows user to scan a range using the temporal behavior detector.
    """
    st.header("Classifier Auto-Annotation")
    st.write("Automatically identify behavior segments and their sub-label ranges.")
    st.warning("⚠️ Warning: Any existing segments within the selected range will be removed and replaced by the newly identified segments.")

    try:
        from app.ml.segment_classifier.service import segment_classifier
    except ImportError:
        try:
            from app.ml.segment_classifier.service import segment_classifier
        except ImportError:
            segment_classifier = None

    if not segment_classifier:
        st.error("SegmentClassifierService could not be imported.")
        return

    total_rows = len(df)
    if total_rows > 1:
        range_slider = st.slider(
            "Select Data Range to Scan (Row Indices)", 
            0, total_rows-1, (0, total_rows-1), step=1, 
            key="classifier_range_slider"
        )
    else:
        st.write("Not enough data to scan.")
        return

    if st.button("Identify Segments with Classifier", type="primary", key="classifier_scan_btn"):
        with st.spinner("Loading model and scanning telemetry data..."):
            try:
                if not segment_classifier.load_model():
                    st.error("Could not load trained model. Ensure it is trained first.")
                    return
                
                scan_df = df.iloc[range_slider[0]:range_slider[1] + 1].copy()
                found_segments = segment_classifier.detect_segments(scan_df)
                
                new_annotations = []
                count_added = 0
                for seg in found_segments:
                    start_idx = seg.start_index + range_slider[0]
                    end_idx = seg.end_index + range_slider[0]
                    
                    from app.shared.segment import AnnotatedSegment
                    seg_rows = df.iloc[start_idx:end_idx].to_dict(orient="records")
                    new_ann = AnnotatedSegment(
                        labels=[seg.label],
                        segment_length=end_idx - start_idx,
                        start_index=start_idx,
                        end_index=end_idx,
                        notes=f"Temporal detector score: {seg.score:.3f}",
                        chunk_index=session_id,
                        telemetry_data=seg_rows,
                    )
                    new_annotations.append(new_ann)
                    count_added += 1
                    for child in seg.subsegments:
                        child_start = child.start_index + range_slider[0]
                        child_end = child.end_index + range_slider[0]
                        new_annotations.append(AnnotatedSegment(
                            labels=[seg.label, child.label],
                            segment_length=child_end - child_start,
                            start_index=child_start,
                            end_index=child_end,
                            notes=f"Temporal detector score: {child.score:.3f}",
                            parent_id=new_ann.id,
                            chunk_index=session_id,
                            telemetry_data=df.iloc[child_start:child_end].to_dict(orient="records"),
                        ))
                        count_added += 1
                
                if count_added > 0:
                    if "current_annotations" not in st.session_state or st.session_state.current_annotations is None:
                        st.session_state.current_annotations = []
                    else:
                        filtered_annotations = []
                        removed_count = 0
                        for ann in st.session_state.current_annotations:
                            start = ann.start_index if ann.start_index is not None else 0
                            end = ann.end_index if ann.end_index is not None else len(df) - 1
                            
                            if start < range_slider[1] + 1 and end > range_slider[0]:
                                removed_count += 1
                            else:
                                filtered_annotations.append(ann)
                                
                        st.session_state.current_annotations = filtered_annotations
                        if removed_count > 0:
                            st.info(f"Removed {removed_count} existing segments in the selected range.")
                        
                    st.session_state.current_annotations.extend(new_annotations)
                    st.session_state.current_annotations.sort(key=lambda x: (x.start_index if x.start_index is not None else 0))
                    
                    st.success(f"Successfully identified and added {count_added} temporal segments.")
                    
                    if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                         save_annotations(
                             st.session_state.last_session_id,
                             st.session_state.current_annotations,
                             st.session_state.last_annotation_key,
                             silent=False
                         )
                         time.sleep(1)
                         st.rerun()
                else:
                    st.info("No behavior segments were identified in the selected range.")
                    
            except Exception as e:
                st.error(f"Error classifying segments: {str(e)}")


def _load_batch_session(
    selected_annotation_key,
    selected_session_key,
    available_sessions,
    *,
    seed_from_source_segments=False,
):
    """Shared session selector + dataframe load for every batch page.

    Returns ``(df, session_id)`` or ``(None, None)`` if the page should
    short-circuit (no data / nothing selected).
    """
    annotated_sessions = set(get_available_sessions(selected_annotation_key))

    def format_session_option(s):
        status = "✅" if s in annotated_sessions else "⭕"
        return f"{status} {s}"

    current_session = (
        st.session_state.get("batch_session_selector")
        or st.session_state.get("detailed_session_selector")
    )
    index = 0
    if current_session and current_session in available_sessions:
        index = available_sessions.index(current_session)

    col_sel1, _ = st.columns([1, 3])
    with col_sel1:
        session_id = st.selectbox(
            "Select Session for Batch Analysis",
            options=available_sessions,
            format_func=format_session_option,
            index=index,
            key="batch_session_selector",
        )

    with st.spinner(f"Loading session {session_id}..."):
        df = load_session_data(selected_session_key, session_id)
        if ("last_session_id" not in st.session_state or
            st.session_state.last_session_id != session_id or
            "last_annotation_key" not in st.session_state or
            st.session_state.last_annotation_key != selected_annotation_key):
            saved_annotations = load_annotations(session_id, selected_annotation_key)
            if seed_from_source_segments and not saved_annotations:
                source_segments = load_session_segments(selected_session_key, session_id)
                st.session_state.current_annotations = copy.deepcopy(source_segments)
            else:
                st.session_state.current_annotations = saved_annotations
            st.session_state.last_session_id = session_id
            st.session_state.last_annotation_key = selected_annotation_key

    if df.empty:
        st.warning("Selected session has no data.")
        return None, None

    if "Static_track" in df.columns:
        track_name = df["Static_track"].iloc[0]
        st.markdown(f"**Track:** {track_name}")

    return df, session_id


def render_batch_bulk_label(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_bulk_label_utils(selected_annotation_key)


def render_batch_rule_based(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
        seed_from_source_segments=True,
    )
    if df is None:
        return
    render_rule_based_annotation(df, selected_annotation_key)
    _render_rule_session_data_table(
        df, session_id, selected_session_key, available_sessions,
    )


def render_batch_classifier(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_classifier_auto_annotation(df, selected_annotation_key, session_id)
