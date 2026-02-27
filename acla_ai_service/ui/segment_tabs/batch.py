import streamlit as st
import time
import threading
import pandas as pd
import copy
from streamlit.runtime.scriptrunner import add_script_run_ctx

from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_available_sessions,
    LABEL_MAPPING, LABEL_NAME_TO_ID,
    LABEL_CATEGORIES, MAIN_LABEL_GUIDELINES
)

try:
    from ..gemini_analyzer import GeminiAnalyzer
except ImportError:
    # Fallback if relative import fails structure
    try:
        from ui.gemini_analyzer import GeminiAnalyzer
    except ImportError:
        GeminiAnalyzer = None

try:
    from ..services.batch_annotation_service import BatchAnnotationService, StreamlitBatchObserver, BatchFileJobManager
except ImportError:
    try:
        from ui.services.batch_annotation_service import BatchAnnotationService, StreamlitBatchObserver, BatchFileJobManager
    except ImportError:
        BatchAnnotationService = None
        StreamlitBatchObserver = None
        BatchFileJobManager = None

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
            end = ann.end_index if ann.end_index is not None else len(df) - 1
            
            if start < 0 or end >= len(df) or start > end:
                continue
                
            segment_data = df.iloc[start : end + 1]
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


def render_batch_api_file_job_ui(df, selected_annotation_key, process_indices, context_padding=200):
    """
    Renders the UI for Batch API File Job mode.
    - Prepares all requests with base64-encoded images in JSONL format
    - Uploads JSONL file via File API
    - Submits batch job referencing the uploaded file
    - Polls for status
    - Downloads and processes results when complete
    """
    st.markdown("##### Batch API Mode (50% Cost Savings)")
    st.info(
        "**How it works:** All segments are prepared as a JSONL file with base64-encoded graphs, "
        "uploaded via File API, then submitted as a batch job. Results are typically available within "
        "minutes to hours (up to 24h SLO). You get 50% discount compared to sequential API calls."
    )
    
    if not BatchFileJobManager:
        st.error("BatchFileJobManager could not be imported.")
        return
    
    # Initialize state for batch file job
    if "batch_file_manager" not in st.session_state:
        st.session_state.batch_file_manager = None
    if "batch_file_logs" not in st.session_state:
        st.session_state.batch_file_logs = []
    
    # UI Components
    batch_progress_bar = st.progress(0.0)
    batch_status_text = st.empty()
    st.markdown("##### Batch Job Log")
    with st.container(height=400):
        batch_log_area = st.empty()
        
        # Restore logs display
        if st.session_state.batch_file_logs:
            batch_log_area.code("\n".join(st.session_state.batch_file_logs), language="text", line_numbers=True)
    
    # Create observer for this UI
    observer = StreamlitBatchObserver(
        batch_progress_bar, 
        batch_status_text, 
        batch_log_area, 
        st.session_state.batch_file_logs
    )
    
    # Get or create manager
    manager = st.session_state.batch_file_manager
    if manager:
        manager.observer = observer
    
    # Determine current state
    current_status = manager.state.status if manager else "idle"
    
    # Display current job info
    if manager and manager.state.job_name:
        st.markdown(f"**Job Name:** `{manager.state.job_name}`")
        st.markdown(f"**Status:** {manager.state.status}")
        if manager.state.total_requests:
            st.markdown(f"**Requests:** {manager.state.total_requests}")
    
    # Options for batch job
    st.checkbox(
        "Include Graphs", 
        value=True, 
        key="batch_include_graphs",
        help="When unchecked, sends text-only requests without generating graphs. Faster and cheaper, but less accurate analysis."
    )
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Submit button - only show when idle or failed
        submit_disabled = current_status in ["preparing", "uploading", "submitted", "polling"]
        if st.button("Submit Batch Job", key="batch_file_submit_btn", disabled=submit_disabled):
            api_key = GeminiAnalyzer.get_api_key()
            if not api_key:
                st.error("API Key required.")
            else:
                # Initialize manager
                analyzer = GeminiAnalyzer(api_key)
                manager = BatchFileJobManager(analyzer, observer)
                st.session_state.batch_file_manager = manager
                st.session_state.batch_file_logs = []
                manager.observer.logs = st.session_state.batch_file_logs
                
                # Define Task Config
                track_config = {
                    "player_x": "Graphics_player_pos_x",
                    "player_y": "Graphics_player_pos_y",
                    "expert_x": "expert_optimal_player_pos_x",
                    "expert_y": "expert_optimal_player_pos_y"
                }
                
                # Prepare and submit
                include_graphs = st.session_state.get("batch_include_graphs", True)
                success = manager.prepare_and_submit(
                    process_indices=process_indices,
                    annotations=st.session_state.current_annotations,
                    df=df,
                    track_config=track_config,
                    label_mapping=LABEL_MAPPING,
                    label_name_to_id=LABEL_NAME_TO_ID,
                    main_guidelines=MAIN_LABEL_GUIDELINES,
                    label_categories=LABEL_CATEGORIES,
                    include_graphs=include_graphs,
                    context_padding=context_padding
                )
                
                if success:
                    st.success("Batch job submitted! Click 'Poll Status' to check progress.")
                st.rerun()
    
    with col2:
        # Poll button - only show when polling
        poll_disabled = current_status not in ["polling", "submitted", "uploading"]
        if st.button("Poll Status", key="batch_file_poll_btn", disabled=poll_disabled):
            if manager:
                status = manager.poll_status()
                if status.get("finished"):
                    if status.get("success"):
                        st.success("Batch job completed!")
                    else:
                        st.error(f"Batch job failed: {status.get('error', 'Unknown error')}")
                st.rerun()
    
    with col3:
        # Cancel button - only show when running
        cancel_disabled = current_status not in ["polling", "submitted", "uploading"]
        if st.button("Cancel Job", key="batch_file_cancel_btn", disabled=cancel_disabled, type="secondary"):
            if manager:
                manager.cancel_job()
                st.rerun()
    
    # Results section - show when completed
    if current_status == "completed" and manager:
        st.markdown("---")
        st.markdown("##### Results")
        
        col_results1, col_results2 = st.columns([1, 1])
        
        with col_results1:
            if st.button("Fetch & Apply Results", key="batch_file_apply_btn"):
                results = manager.get_results()
                
                if results:
                    skip_if_sublabels = st.session_state.get("batch_skip_sublabels", True)
                    auto_update = st.session_state.get("batch_auto_update", True)
                    main_labels_set = set(LABEL_CATEGORIES.get("Main Labels", []))
                    
                    success_count = 0
                    error_count = 0
                    
                    for result in results:
                        seg_idx = result.get("segment_index", -1)
                        
                        if seg_idx < 0 or seg_idx >= len(st.session_state.current_annotations):
                            error_count += 1
                            continue
                        
                        if result.get("error"):
                            observer.on_log(f"⚠️ Segment #{seg_idx}: {result['error']}")
                            error_count += 1
                            continue
                        
                        # Process labels
                        suggested_labels = result.get("parsed_labels", [])
                        new_label_ids = []
                        
                        for sugg in suggested_labels:
                            lname = sugg.get("label")
                            lid = LABEL_NAME_TO_ID.get(lname)
                            if not lid and lname in LABEL_MAPPING:
                                lid = lname
                            if lid:
                                new_label_ids.append(lid)
                        
                        if auto_update and new_label_ids:
                            ann = st.session_state.current_annotations[seg_idx]
                            current_set = set(ann.labels)
                            existing_sub_labels = [l for l in current_set if l not in main_labels_set]
                            
                            if skip_if_sublabels and len(existing_sub_labels) > 0:
                                observer.on_log(f"⏭️ Segment #{seg_idx}: Skipped (existing sub-labels)")
                                continue
                            
                            # Apply new labels
                            kept_labels = [l for l in ann.labels if l in main_labels_set]
                            updated_labels = list(set(kept_labels).union(set(new_label_ids)))
                            ann.labels = updated_labels
                            
                            # Update notes
                            summary = result.get("summary", "")
                            if summary:
                                timestamp = time.strftime("%H:%M:%S")
                                new_note = f"\n\n[Batch API {timestamp}]:\n{summary}"
                                if hasattr(ann, 'notes') and ann.notes:
                                    ann.notes += new_note
                                else:
                                    ann.notes = new_note.strip()
                            
                            observer.on_log(f"✅ Segment #{seg_idx}: Applied {len(new_label_ids)} labels")
                            success_count += 1
                        else:
                            observer.on_log(f"ℹ️ Segment #{seg_idx}: {len(new_label_ids)} labels found (not applied)")
                    
                    # Save all annotations
                    if success_count > 0 and "last_session_id" in st.session_state:
                        save_annotations(
                            st.session_state.last_session_id,
                            st.session_state.current_annotations,
                            st.session_state.last_annotation_key,
                            silent=False
                        )
                    
                    observer.on_complete(success_count, error_count, "Results processed")
                    st.success(f"Processed {success_count} segments, {error_count} errors.")
                else:
                    st.warning("No results available.")
                
                st.rerun()
        
        with col_results2:
            if st.button("View Raw Results", key="batch_file_view_btn"):
                results = manager.get_results()
                if results:
                    st.json(results)
    
    # Reset button
    if current_status in ["completed", "failed", "cancelled"]:
        if st.button("Reset / New Job", key="batch_file_reset_btn"):
            st.session_state.batch_file_manager = None
            st.session_state.batch_file_logs = []
            st.rerun()
    
    # Auto-poll when waiting
    if current_status == "polling":
        st.info("Job is running... Click 'Poll Status' to check progress, or wait for auto-poll.")
        # Optional: Auto-poll every 30 seconds
        # time.sleep(30)
        # st.rerun()


def render_batch_auto_annotation(df, selected_annotation_key):
    """
    Renders the Batch Auto-Annotation section using BatchAnnotationService.
    Decoupled integration: UI controls service, observes progress.
    Uses @st.fragment for efficient partial reruns without recursion issues.
    """
    st.header("Batch Auto-Annotation (Gemini/Service)")
    st.write("Automatically analyze and update labels for a range of segments using AI Service.")
    
    # Ensure annotations exist
    if not st.session_state.get("current_annotations"):
            st.info("No segments available to process. Please create segments first.")
            return

    if not BatchAnnotationService:
        st.error("BatchAnnotationService could not be imported.")
        return

    total_segments = len(st.session_state.current_annotations)
    # Slider to select range of segments
    if total_segments > 1:
        batch_range = st.slider("Select Segment Range", 0, total_segments-1, (0, total_segments-1), step=1)
    else:
        batch_range = (0, 0)
        st.write("1 segment available.")

    process_indices = list(range(batch_range[0], batch_range[1]+1))
    st.write(f"Selected {len(process_indices)} segments for analysis.")

    if GeminiAnalyzer:
        # Context Padding Input
        context_padding_val = st.number_input(
            "Context Padding (surrounding data points)", 
            min_value=50, 
            max_value=10000, 
            value=2000, 
            step=100,
            key="batch_context_padding_input",
            help="Controls how much track data around the segment is included in the context analysis."
        )

        # Mode Selection: Sequential vs Batch API
        st.markdown("---")
        batch_mode = st.radio(
            "Processing Mode",
            options=["Sequential API", "Batch API File Job (50% cheaper)"],
            index=0,
            horizontal=True,
            key="batch_processing_mode",
            help="Sequential: Process one segment at a time with immediate results. "
                 "Batch API: Upload JSONL file and submit as batch job, 50% discount, results available within 24h."
        )
        
        # Service and State Initialization
        if "batch_service_instance" not in st.session_state:
            st.session_state.batch_service_instance = None
        
        # Checkbox for auto-update (outside fragment so it persists)
        st.checkbox("Auto-update Labels & Notes", value=True, key="batch_auto_update")
        st.checkbox("Skip if sub-labels exist", value=True, key="batch_skip_sublabels")
        
        if batch_mode == "Batch API File Job (50% cheaper)":
            # Render Batch API File Job UI
            render_batch_api_file_job_ui(df, selected_annotation_key, process_indices, context_padding=context_padding_val)
        else:
            # Render Sequential API UI (existing fragment-based approach)
            # Use fragment for progress monitoring to avoid full script reruns
            @st.fragment
            def batch_progress_fragment():
                """Fragment that handles progress updates with partial reruns."""
                # UI Components for Progress
                st.markdown("##### Batch Progress")
                batch_progress_bar = st.progress(0.0)
                batch_status_text = st.empty()
                st.markdown("##### Batch Log")
                with st.container(height=400):
                    batch_log_area = st.empty()
                
                # Retrieve existing service if any
                service = st.session_state.get("batch_service_instance")
                
                # Ensure service has logs attribute initialized (important for persistence)
                initial_logs = []
                if service:
                     # Backward compatibility: initialize logs list on service if missing
                     if not hasattr(service, "logs"):
                         service.logs = []
                         # Try to recover logs from old observer if it exists
                         if hasattr(service, "observer") and service.observer and hasattr(service.observer, "logs"):
                              try:
                                  current_logs = service.observer.logs
                                  if isinstance(current_logs, list):
                                      service.logs.extend(current_logs)
                              except: pass
                     
                     # Use the service's logs list as the source of truth
                     initial_logs = service.logs

                # Create Observer linked to these UI components
                # Re-create observer on every run to bind to current st elements
                # Pass persistent logs list so updates are synced
                observer = StreamlitBatchObserver(batch_progress_bar, batch_status_text, batch_log_area, initial_logs)
                
                if service:
                    # Restore state from previous observer attached to the service
                    if hasattr(service, "observer") and service.observer:
                        try:
                            # prev_logs is service.logs because we linked it via reference.
                            # We pass None to restore_state so it doesn't overwrite logs reference.
                            prev_progress = getattr(service.observer, "current_progress", 0.0)
                            prev_status = getattr(service.observer, "current_status", "")
                            
                            observer.restore_state(None, prev_progress, prev_status)
                        except Exception as e:
                            print(f"Error restoring batch observer state: {e}")

                    # Update observer to point to new UI elements (important for re-renders)
                    service.observer = observer 
                    is_running = service.is_running
                else:
                    is_running = False

                col_ctrl1, col_ctrl2 = st.columns([1, 4])
                
                with col_ctrl1:
                    # Callback for Start Button
                    def on_start_batch_click():
                        st.session_state.start_batch_requested = True
                    
                    # Render Buttons based on state
                    if not is_running:
                        st.button("Start Batch Analysis", key="start_batch_gemini_svc_btn", on_click=on_start_batch_click)
                    elif st.button("Stop Batch", key="stop_batch_gemini_svc_btn"):
                        if service:
                            service.stop()
                            # Wait briefly?
                            time.sleep(0.5)
                            st.rerun(scope="fragment")
                    
                    # Check for Start Request (set via callback)
                    if st.session_state.get("start_batch_requested", False) and not is_running:
                        # Clear flag immediately
                        st.session_state.start_batch_requested = False
                        
                        api_key = GeminiAnalyzer.get_api_key()
                        if not api_key:
                            st.error("API Key required.")
                        else:
                            # Initialize Service
                            analyzer = GeminiAnalyzer(api_key)
                            service = BatchAnnotationService(analyzer, observer)
                            # Set running flag immediately to prevent race condition/UI flickering on rerun
                            service.is_running = True
                            st.session_state.batch_service_instance = service
                            
                            # Define Task Config
                            track_config = {
                                "player_x": "Graphics_player_pos_x",
                                "player_y": "Graphics_player_pos_y",
                                "expert_x": "expert_optimal_player_pos_x",
                                "expert_y": "expert_optimal_player_pos_y"
                            }
                            
                            # Add track name from Static_track for reference image loading
                            if "Static_track" in df.columns:
                                track_name = df["Static_track"].iloc[0] if not df.empty else None
                                if track_name:
                                    track_config["track_name"] = track_name

                            # Capture the skip setting outside the thread
                            skip_if_sublabels = st.session_state.get("batch_skip_sublabels", True)

                            # Callback for Persisting Data (runs in worker thread)
                            def persistence_callback(idx, new_label_ids, response_text):
                                if not st.session_state.get("batch_auto_update", False):
                                    print(f"[Batch] Auto-update disabled. Skipping persistence for segment {idx}")
                                    return
                                
                                try:
                                    # Verify index range
                                    if idx < 0 or idx >= len(st.session_state.current_annotations):
                                        print(f"[Batch] Index {idx} out of range for persistence.")
                                        return

                                    ann = st.session_state.current_annotations[idx]
                                    
                                    # IDENTIFY SUB-LABELS
                                    # We consider Main Labels to be the top-level keys in LABEL_CATEGORIES['Main Labels']
                                    # Everything else is a sub-label
                                    main_labels_set = set(LABEL_CATEGORIES.get("Main Labels", []))
                                    current_set = set(ann.labels)
                                    
                                    # Find existing sub-labels
                                    existing_sub_labels = [l for l in current_set if l not in main_labels_set]
                                    
                                    if skip_if_sublabels and len(existing_sub_labels) > 0:
                                        # Skip update for this segment
                                        print(f"[Batch] Skipping segment {idx} due to existing sub-labels: {existing_sub_labels}")
                                        return

                                    # PREPARE NEW LABELS
                                    # If we are NOT skipping, we must REMOVE existing sub-labels before adding new ones
                                    # Keep only main labels from the original set
                                    kept_labels = [l for l in ann.labels if l in main_labels_set]
                                    
                                    # Add the NEW labels from AI (which might be sub-labels or main labels)
                                    # We use set to avoid duplicates, but we base it on kept_labels + new_label_ids
                                    updated_labels = list(set(kept_labels).union(set(new_label_ids)))
                                    
                                    ann.labels = updated_labels
                                    print(f"[Batch] Updating segment {idx} with labels: {updated_labels}")
                                    
                                    # Update Notes (Append)
                                    timestamp = time.strftime("%H:%M:%S")
                                    new_note = f"\\n\\n[Auto-Analysis {timestamp}]:\\n{response_text}"
                                    
                                    if hasattr(ann, 'notes') and ann.notes:
                                        ann.notes += new_note
                                    else:
                                        ann.notes = new_note.strip()
                                    
                                    # Save directly to store
                                    if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                                        save_annotations(
                                            st.session_state.last_session_id,
                                            st.session_state.current_annotations,
                                            st.session_state.last_annotation_key,
                                            silent=True
                                        )
                                        print(f"[Batch] Saved annotations for segment {idx}")
                                except Exception as e:
                                    print(f"Error in persistence_callback: {e}")
                            
                            # Start Thread
                            thread = threading.Thread(
                                target=service.run_batch,
                                args=(
                                    process_indices,
                                    st.session_state.current_annotations,
                                    df,
                                    track_config,
                                    LABEL_MAPPING,
                                    LABEL_NAME_TO_ID,
                                    MAIN_LABEL_GUIDELINES,
                                    LABEL_CATEGORIES,
                                    persistence_callback,
                                    context_padding_val
                                )
                            )
                            # Attach script context so st.* calls in service (if any) or observer work
                            add_script_run_ctx(thread)
                            thread.start()
                            st.rerun(scope="fragment")

                # Auto-Rerun for Updates if running - using fragment scope to avoid recursion
                if is_running:
                    time.sleep(1)
                    st.rerun(scope="fragment")
                elif service and not is_running:
                    # Clean up service reference or leave it for log viewing?
                    # Reset button
                    if st.button("Clear / Reset", key="reset_batch_svc"):
                        st.session_state.batch_service_instance = None
                        st.rerun(scope="fragment")
            
            # Call the fragment
            batch_progress_fragment()
            
    else:
        st.warning("Gemini Analyzer unavailable. Check imports.")


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


def render_batch_view(selected_annotation_key, selected_session_key, available_sessions):
    """
    Renders the Batch Annotation Tab View (including session selection).
    """
    
    # Shared helper for formatting session options
    annotated_sessions = set(get_available_sessions(selected_annotation_key))

    def format_session_option(s):
        status = "✅" if s in annotated_sessions else "⭕"
        return f"{status} {s}"

    # Calculate index to maintain selection across reruns
    index = 0
    # Re-use the same session selector key as detailed view if we want to sync them?
    # Or separate? Let's use separate or shared? 
    # The detailed view uses "detailed_session_selector".
    # User might want to switch tabs and keep session.
    # Let's try to sync by initializing from session_state if available.
    
    current_session = st.session_state.get("detailed_session_selector") 
    # If not set, try shared or default
    
    if current_session and current_session in available_sessions:
        index = available_sessions.index(current_session)
        
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        session_id = st.selectbox(
            "Select Session for Batch Analysis", 
            options=available_sessions,
            format_func=format_session_option,
            index=index,
            key="batch_session_selector" 
            # Note: Using different key than detailed view, 
            # but we could sync them manually if needed.
            # Ideally tabs share session selection context but Streamlit tabs re-run.
        )
    
    # Sync logic: if batch session changes, maybe we want to update other views?
    # For now, let's just load data.

    with st.spinner(f"Loading session {session_id}..."):
        df = load_session_data(selected_session_key, session_id)
        
        # Load existing annotations for this session if we switched sessions
        # We reuse the same state variables as detailed view for annotations: 
        # "current_annotations", "last_session_id"
        if ("last_session_id" not in st.session_state or 
            st.session_state.last_session_id != session_id or 
            "last_annotation_key" not in st.session_state or
            st.session_state.last_annotation_key != selected_annotation_key):
            
                st.session_state.current_annotations = load_annotations(session_id, selected_annotation_key)
                st.session_state.last_session_id = session_id
                st.session_state.last_annotation_key = selected_annotation_key
    
    if df.empty:
        st.warning("Selected session has no data.")
        return

    from .shared import get_store
    store = get_store()
    metadata = store.get_cache_metadata(selected_session_key)
    
    if "Static_track" in df.columns:
         track_name = df["Static_track"].iloc[0]
         st.markdown(f"**Track:** {track_name}")

    render_bulk_label_utils(selected_annotation_key)
    render_rule_based_annotation(df, selected_annotation_key)
    render_batch_auto_annotation(df, selected_annotation_key)
