import streamlit as st
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Callable

from segment_tabs.components.llm_training_data_generation import generate_training_prompt
from segment_tabs.shared import _run_async
from app.domain.labels import LABEL_MAPPING
def render_batch_ai_annotation_tab(
    segments: List[Dict[str, Any]], 
    existing_annotations: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    available_sessions: List[str] = None,
    save_training_pair_fn: Callable = None
):
    """
    Renders the batch AI annotation functionality for generating descriptions for multiple segments.
    """
    st.header("Batch AI Annotation")
    st.markdown("Use this tab to process and generate dataset descriptions for multiple segments at once using AI.")
    
    if not segments:
        st.warning("No segments available for batch annotation.")
        return
        
    model_id = st.text_input("Local Model ID or Path (Batch)", value="mistralai/Ministral-3-14B-Reasoning-2512")
    gguf_file = st.text_input("GGUF Filename (if using a GGUF repo)", key="batch_gguf")
    use_4bit = st.checkbox("Use 4-bit Quantization (bitsandbytes)", value=False, key="batch_4bit")

    # Add label filtering
    st.subheader("Filter Segments")
    
    if available_sessions:
        selected_sessions = st.multiselect(
            "Only process segments from these sessions:",
            options=available_sessions,
            help="Leave empty to process segments from all available sessions."
        )
    else:
        selected_sessions = []

    all_label_ids = list(LABEL_MAPPING.keys())
    format_label = lambda x: f"{x} - {LABEL_MAPPING.get(x, 'Unknown')}"
    
    selected_labels = st.multiselect(
        "Only process segments containing (any of) these labels:",
        options=all_label_ids,
        format_func=format_label,
        help="Leave empty to process all available segments."
    )
    
    # Filter segments
    filtered_segments = segments
    if selected_sessions:
        filtered_segments = [
            s for s in filtered_segments 
            if s.get("session_id") in selected_sessions
        ]
        
    if selected_labels:
        filtered_segments = [
            s for s in filtered_segments 
            if any(l in selected_labels for l in s.get("labels", []))
        ]

    st.write(f"Total segments available (after filtering): **{len(filtered_segments)}**")
    
    batch_limit = st.number_input("Batch Limit (number of segments to process)", min_value=1, max_value=len(filtered_segments) or 1, value=min(10, len(filtered_segments)) if filtered_segments else 1)

    if st.button("Start Batch Processing", key="btn_batch_start", disabled=len(filtered_segments) == 0):
        try:
            from app.llm.local_llm import LocalLLMConfig, GenerationRequest
            from app.services.llm.telemetry_llm_orchestrator import TelemetryLLMOrchestrator
            
            with st.spinner(f"Loading/Using model {model_id} via hf_local..."):
                llm_config = LocalLLMConfig(base_model=model_id, load_in_4bit=use_4bit, gguf_file=gguf_file if gguf_file else None)
                orchestrator = TelemetryLLMOrchestrator(
                    llm_config=llm_config,
                    adapter_directory=Path("models/llm_adapters"),
                    dataset_directory=Path("models/llm_datasets")
                )

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                segments_to_process = filtered_segments[:int(batch_limit)]
                
                success_count = 0
                error_count = 0
                
                for i, segment in enumerate(segments_to_process):
                    segment_id = str(segment.get("id"))
                    labels = segment.get("labels", [])
                    label_ids = [str(l) for l in labels]
                    
                    status_text.text(f"Processing {i+1}/{len(segments_to_process)}: Segment {segment_id}")
                    
                    current_prompt = generate_training_prompt(labels, 0)
                    
                    req = GenerationRequest(
                        user_prompt=current_prompt,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                    
                    result = _run_async(orchestrator.generate_inference, provider="hf_local", model_id=model_id, request_data=req)
                    
                    if result.get("status") == "success":
                        generated_draft = result.get("result", "")
                        
                        # Save the annotation using the shared saving function if provided
                        if save_training_pair_fn:
                            save_training_pair_fn(
                                segment_id=segment_id,
                                prompt=current_prompt,
                                completion=generated_draft,
                                labels=labels,
                                label_ids=label_ids,
                                output_path=output_path
                            )
                        else:
                            record = {
                                "segment_id": segment_id,
                                "prompt": current_prompt,
                                "completion": generated_draft,
                                "labels": labels,
                                "label_ids": label_ids,
                                "timestamp": time.time()
                            }
                            with open(output_path, "a") as f:
                                f.write(json.dumps(record) + "\n")
                                
                        # Update the in-memory state
                        record_for_tracking = {
                            "segment_id": segment_id,
                            "prompt": current_prompt,
                            "completion": generated_draft,
                            "labels": labels,
                            "label_ids": label_ids,
                            "timestamp": time.time()
                        }
                        existing_annotations[segment_id] = [record_for_tracking]
                        success_count += 1
                    else:
                        st.error(f"Failed to generate for {segment_id}: {result.get('message', 'Unknown error')}")
                        error_count += 1
                        
                    progress_bar.progress((i + 1) / len(segments_to_process))
                
                st.success(f"Batch processing complete! Successfully annotated {success_count} segments. Errors: {error_count}")
        except Exception as e:
            import traceback
            st.error(f"Error during AI generation: {e}")
            st.code(traceback.format_exc())
