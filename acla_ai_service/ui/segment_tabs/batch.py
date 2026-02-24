import streamlit as st
import time
import threading
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
    from ..services.batch_annotation_service import BatchAnnotationService, StreamlitBatchObserver
except ImportError:
    try:
        from ui.services.batch_annotation_service import BatchAnnotationService, StreamlitBatchObserver
    except ImportError:
        BatchAnnotationService = None
        StreamlitBatchObserver = None

def render_batch_auto_annotation(df, selected_annotation_key):
    """
    Renders the Batch Auto-Annotation section using BatchAnnotationService.
    Decoupled integration: UI controls service, observes progress.
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
        # Service and State Initialization
        if "batch_service_instance" not in st.session_state:
            st.session_state.batch_service_instance = None
        
        # UI Components for Progress
        st.markdown("##### Batch Progress")
        batch_progress_bar = st.progress(0.0)
        batch_status_text = st.empty()
        st.markdown("##### Batch Log")
        batch_log_area = st.empty()
        
        # Create Observer linked to these UI components
        # Re-create observer on every run to bind to current st elements
        observer = StreamlitBatchObserver(batch_progress_bar, batch_status_text, batch_log_area)
        
        # Retrieve existing service if any
        service = st.session_state.batch_service_instance
        if service:
            # Update observer to point to new UI elements (important for re-renders)
            service.observer = observer 
            is_running = service.is_running
        else:
            is_running = False

        col_ctrl1, col_ctrl2 = st.columns([1, 4])
        
        with col_ctrl1:
            # Checkbox for auto-update
            st.checkbox("Auto-update Labels & Notes", value=True, key="batch_auto_update")

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
                    st.rerun()
            
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

                    # Callback for Persisting Data (runs in worker thread)
                    def persistence_callback(idx, new_label_ids, response_text):
                        if not st.session_state.get("batch_auto_update", False):
                            return
                        
                        try:
                            # Verify index range
                            if idx < 0 or idx >= len(st.session_state.current_annotations):
                                return

                            ann = st.session_state.current_annotations[idx]
                            
                            # Update Labels (Union)
                            current_set = set(ann.labels)
                            new_set = set(new_label_ids)
                            updated_labels = list(current_set.union(new_set))
                            ann.labels = updated_labels
                            
                            # Update Notes (Append)
                            timestamp = time.strftime("%H:%M:%S")
                            new_note = f"\n\n[Auto-Analysis {timestamp}]:\n{response_text}"
                            
                            if ann.notes:
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
                            persistence_callback
                        )
                    )
                    # Attach script context so st.* calls in service (if any) or observer work
                    add_script_run_ctx(thread)
                    thread.start()
                    st.rerun()

        # Auto-Rerun for Updates if running
        if is_running:
            time.sleep(1)
            st.rerun()
        elif service and not is_running:
                # Clean up service reference or leave it for log viewing?
                # Reset button
                if st.button("Clear / Reset", key="reset_batch_svc"):
                    st.session_state.batch_service_instance = None
                    st.rerun()
            
    else:
        st.warning("Gemini Analyzer unavailable. Check imports.")


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

    render_batch_auto_annotation(df, selected_annotation_key)
