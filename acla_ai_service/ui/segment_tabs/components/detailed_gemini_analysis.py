import streamlit as st
import os
import time
import traceback

from ..shared import (
    LABEL_CATEGORIES, MAIN_LABEL_GUIDELINES,
    LABEL_MAPPING, LABEL_NAME_TO_ID
)

try:
    from ...gemini_analyzer import GeminiAnalyzer, GRAPH_DEFINITIONS
except ImportError:
    try:
        from ui.gemini_analyzer import GeminiAnalyzer, GRAPH_DEFINITIONS
    except ImportError:
        GeminiAnalyzer = None
        GRAPH_DEFINITIONS = []

def render_gemini_analysis(df, form_start, form_end, form_labels):
    with st.expander("Advanced AI Analysis (Gemini)"):
        if GeminiAnalyzer:
            # API Key management handled by GeminiAnalyzer
            gemini_api_key = GeminiAnalyzer.get_api_key()
            
            if gemini_api_key:
                context_padding_val = st.number_input(
                    "Context Padding (surrounding data points)", 
                    min_value=50, 
                    max_value=10000, 
                    value=2000, 
                    step=100,
                    help="Controls how much track data around the segment is included in the context map."
                )

                # Trajectory Overlay Controls
                st.markdown("##### Track Map Overlay Settings")
                
                overlay_config = None
                
                # Get track name to find reference image
                track_name = None
                if "Static_track" in df.columns and not df.empty:
                    track_name = df["Static_track"].iloc[0]
                
                if track_name:
                    # Try to find track map path
                    try:
                        from app.models.segment_models import LABEL_IMAGE_MAP
                        
                        track_map_filename = LABEL_IMAGE_MAP.get(track_name)
                        if track_map_filename:
                            # Search for the file
                            possible_paths = [
                                os.path.join(os.getcwd(), "acla_ai_service/ui/source"),
                                os.path.join(os.getcwd(), "ui/source"),
                                os.path.join(os.path.dirname(__file__), "../../source"),
                                "ui/source",
                                "source"
                            ]
                            
                            track_map_path = None
                            for base_path in possible_paths:
                                full_path = os.path.join(base_path, track_map_filename)
                                if os.path.exists(full_path):
                                    track_map_path = full_path
                                    break
                            
                            if track_map_path:
                                st.success(f"Found track map: {track_name}")
                                
                                # Overlay adjustment controls
                                col_ov1, col_ov2 = st.columns(2)
                                
                                with col_ov1:
                                    offset_x = st.slider(
                                        "Horizontal Offset", 
                                        min_value=-500, 
                                        max_value=500, 
                                        value=0,
                                        step=1,
                                        key="gemini_overlay_x"
                                    )
                                    
                                    rotation = st.slider(
                                        "Rotation (degrees)", 
                                        min_value=-180, 
                                        max_value=180, 
                                        value=0,
                                        step=1,
                                        key="gemini_overlay_rotation"
                                    )
                                    
                                    alpha = st.slider(
                                        "Trajectory Opacity", 
                                        min_value=0.1, 
                                        max_value=1.0, 
                                        value=0.8,
                                        step=0.1,
                                        key="gemini_overlay_alpha"
                                    )
                                
                                with col_ov2:
                                    offset_y = st.slider(
                                        "Vertical Offset", 
                                        min_value=-500, 
                                        max_value=500, 
                                        value=0,
                                        step=1,
                                        key="gemini_overlay_y"
                                    )
                                    
                                    scale_x = st.slider(
                                        "Scale X", 
                                        min_value=0.01, 
                                        max_value=3.0, 
                                        value=1.0,
                                        step=0.01,
                                        key="gemini_overlay_scale_x"
                                    )
                                    
                                    scale_y = st.slider(
                                        "Scale Y", 
                                        min_value=0.01, 
                                        max_value=3.0, 
                                        value=1.0,
                                        step=0.01,
                                        key="gemini_overlay_scale_y"
                                    )
                                
                                overlay_config = {
                                    "enabled": True,
                                    "track_map_path": track_map_path,
                                    "offset_x": offset_x,
                                    "offset_y": offset_y,
                                    "rotation": rotation,
                                    "scale_x": scale_x,
                                    "scale_y": scale_y,
                                    "alpha": alpha
                                }
                                
                                # Preview Section
                                st.markdown("---")
                                st.markdown("##### Preview Overlay")
                                
                                preview_col1, preview_col2 = st.columns([3, 1])
                                
                                with preview_col2:
                                    preview_auto = st.checkbox(
                                        "Auto-update preview",
                                        value=True,
                                        key="gemini_overlay_auto_preview",
                                        help="Automatically update preview when sliders change"
                                    )
                                    
                                    if not preview_auto:
                                        if st.button("Update Preview", key="gemini_overlay_preview_btn"):
                                            st.session_state.gemini_preview_trigger = time.time()
                                
                                # Generate preview
                                should_preview = preview_auto or st.session_state.get("gemini_preview_trigger", 0) > 0
                                
                                if should_preview:
                                    with st.spinner("Generating overlay preview..."):
                                        try:
                                            # Prepare data for preview
                                            analysis_df = df.iloc[int(form_start):int(form_end)]
                                            
                                            # Prepare context dataframe
                                            padding = context_padding_val
                                            start_idx_ctx = max(0, int(form_start) - padding)
                                            end_idx_ctx = min(len(df), int(form_end) + padding)
                                            preview_context_df = df.iloc[start_idx_ctx:end_idx_ctx]
                                            
                                            # Track config
                                            preview_track_config = {
                                                "player_x": "Graphics_player_pos_x",
                                                "player_y": "Graphics_player_pos_y",
                                                "expert_x": "expert_optimal_player_pos_x",
                                                "expert_y": "expert_optimal_player_pos_y"
                                            }
                                            
                                            # Create analyzer instance just for preview
                                            preview_analyzer = GeminiAnalyzer(gemini_api_key)
                                            
                                            # Generate overlay image
                                            overlay_img = preview_analyzer.create_trajectory_overlay(
                                                analysis_df,
                                                preview_track_config,
                                                context_df=preview_context_df,
                                                track_map_path=track_map_path,
                                                overlay_config=overlay_config
                                            )
                                            
                                            if overlay_img:
                                                with preview_col1:
                                                    st.image(overlay_img, caption="Trajectory Overlay Preview", width='stretch')
                                                    st.caption("💡 Adjust sliders to align trajectory with track layout")
                                            else:
                                                st.error("Failed to generate overlay preview")
                                                
                                        except Exception as e:
                                            st.error(f"Preview error: {str(e)}")
                                            st.code(traceback.format_exc())
                            else:
                                st.warning(f"Track map file '{track_map_filename}' not found in source directories.")
                        else:
                            st.info(f"No track map configured for '{track_name}'. Track map overlay is always needed for best analysis.")
                    except ImportError:
                        st.warning("Could not import LABEL_IMAGE_MAP")
                else:
                    st.info("Track name not found in session data. Track map overlay is disabled.")

                if st.button("Identify Sub-Labels with Gemini", key="gemini_identify_btn"):
                    if form_start >= form_end:
                        st.error("Invalid range selected.")
                    else:
                        with st.spinner("Preparing graphs and asking Gemini..."):
                            try:
                                analyzer = GeminiAnalyzer(gemini_api_key)
                                
                                # 1. Prepare Data
                                analysis_df = df.iloc[int(form_start):int(form_end)]
                                
                                # 2. Gather Feature Graphs
                                # Use the predefined GRAPH_DEFINITIONS for consistent analysis
                                graph_definitions = GRAPH_DEFINITIONS
                                
                                # Prepare context dataframe (user controlled padding)
                                padding = context_padding_val
                                start_idx_ctx = max(0, int(form_start) - padding)
                                end_idx_ctx = min(len(df), int(form_end) + padding)
                                context_df = df.iloc[start_idx_ctx:end_idx_ctx]

                                # 3. Track Config
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
                                
                                # 4. Current Labels
                                current_labels_display = form_labels
                                
                                # 5. Contextual Sub-labels (e.g. MS -> MS1..MS30)
                                sub_label_context = []
                                
                                # Always include "Segment Type" context 
                                if "Segment Type" in LABEL_CATEGORIES:
                                    other_ids = LABEL_CATEGORIES["Segment Type"]
                                    if other_ids:
                                        # Add the main guideline for Segment Type if available
                                        if "Segment Type" in MAIN_LABEL_GUIDELINES:
                                            sub_label_context.append(f"Guideline for 'Segment Type': {MAIN_LABEL_GUIDELINES['Segment Type']}")

                                        other_docs = []
                                        for child_id in other_ids:
                                            child_name = LABEL_MAPPING.get(child_id, child_id)
                                            other_docs.append(f"- {child_id}: {child_name}")
                                        
                                        block = f"Available 'Segment Type' Labels:\n" + "\n".join(other_docs)
                                        sub_label_context.append(block)

                                for lname in current_labels_display:
                                    lid = LABEL_NAME_TO_ID.get(lname)
                                    
                                    # Add Description if available
                                    if lid and lid in MAIN_LABEL_GUIDELINES:
                                        desc = MAIN_LABEL_GUIDELINES[lid]
                                        sub_label_context.append(f"Context/Instruction for '{lname}' ({lid}): {desc}")

                                    # If the selected label is a parent category (e.g. 'MS', 'RM')
                                    if lid and lid in LABEL_CATEGORIES:
                                        child_ids = LABEL_CATEGORIES[lid]
                                        if child_ids:
                                            # Format list of children
                                            child_docs = []
                                            for child_id in child_ids:
                                                child_name = LABEL_MAPPING.get(child_id, child_id)
                                                child_docs.append(f"- {child_id}: {child_name}")
                                            
                                            block = f"Sub-labels for '{lname}' ({lid}):\n" + "\n".join(child_docs)
                                            sub_label_context.append(block)

                                # Analyze
                                result = analyzer.analyze_segment(
                                    analysis_df, 
                                    graph_definitions=graph_definitions, 
                                    track_config=track_config, 
                                    current_labels=current_labels_display,
                                    available_sub_labels_context=sub_label_context,
                                    context_df=context_df,
                                    context_padding=context_padding_val,
                                    overlay_config=overlay_config
                                )
                                
                                if isinstance(result, dict):
                                    st.markdown("### Gemini Analysis Results")
                                    st.markdown(result.get("response", "No response text found."))
                                    
                                    st.markdown("---")
                                    st.markdown("#### Analysis Details (Images & Prompt)")
                                    
                                    st.subheader("Generated Prompt")
                                    st.code(result.get("prompt", "No prompt found"), language="text")
                                    
                                    st.subheader("Analyzed Images")
                                    images = result.get("images", [])
                                    if images:
                                        for idx, img in enumerate(images):
                                            st.image(img, caption=f"Graph {idx+1}", width="stretch")
                                    else:
                                        st.info("No images were generated for this analysis.")
                                else:
                                     st.markdown("### Gemini Analysis Results")
                                     st.markdown(result)
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                st.error(traceback.format_exc())