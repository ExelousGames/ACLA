import streamlit as st
from pathlib import Path
from typing import List, Any
from segment_tabs.components.llm_training_data_generation import generate_training_prompt, get_available_prompts

def render_ai_assisted_annotation_tab(
    labels: List[Any],
    existing_record: dict,
    prompt_widget_key: str, 
    desc_widget_key: str
):
    st.subheader("AI Assisted Annotation")
    st.markdown("Use a local or cloud LLM to draft a description based on the prompt.")

    available_prompts = get_available_prompts(labels)
    selected_template_idx = 0

    # Always generate a fresh prompt based on current labels to ensure AI gets the canonical generated base
    current_prompt = generate_training_prompt(labels, selected_template_idx)

    st.text_area("Prompt to Use:", value=current_prompt, height=100, disabled=True)

    model_id = st.text_input("Local Model ID or Path", value="mistralai/Ministral-3-14B-Reasoning-2512")
    gguf_file = st.text_input("GGUF Filename (if using a GGUF repo)")
    use_4bit = st.checkbox("Use 4-bit Quantization (bitsandbytes)", value=False)

    if st.button("Generate Draft", key=f"generate_ai_{st.session_state.current_index}"):
        try:
            from segment_tabs.shared import _run_async
            from app.llm.local_llm import LocalLLMConfig
            from app.pipelines.chat.orchestrator import TelemetryLLMOrchestrator
            from app.llm.local_llm import GenerationRequest
            
            with st.spinner(f"Loading/Using model {model_id} via hf_local..."):
                llm_config = LocalLLMConfig(base_model=model_id, load_in_4bit=use_4bit, gguf_file=gguf_file if gguf_file else None)
                orchestrator = TelemetryLLMOrchestrator(
                    llm_config=llm_config,
                    adapter_directory=Path("models/llm_adapters"),
                    dataset_directory=Path("models/llm_datasets")
                )

                req = GenerationRequest(
                    user_prompt=current_prompt,
                    max_new_tokens=256,
                    temperature=0.7
                )
                
                result = _run_async(orchestrator.generate_inference, provider="hf_local", model_id=model_id, request_data=req)
                
                if result.get("status") == "success":
                    generated_draft = result.get("result", "")
                    st.session_state[f"pending_{desc_widget_key}"] = generated_draft
                    st.success("Draft generated! Switch to Dataset Editor to review, edit, and save.")
                    st.text_area("Generated Draft:", value=generated_draft, height=150, disabled=True)
                else:
                    st.error(f"Failed to generate: {result.get('message', 'Unknown error')}")
        except Exception as e:
            import traceback
            st.error(f"Error during AI generation: {e}")
            st.code(traceback.format_exc())
