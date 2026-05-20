import streamlit as st
from pathlib import Path
from typing import Any, Dict

from app.pipelines.training.dataset_builder import MODES, render_labels_text


def _draft_for_mode(orchestrator, model_id: str, labels_text: str, mode: str) -> Dict[str, Any]:
    from app.llm.local_llm import GenerationRequest
    from segment_tabs.shared import _run_async

    req = GenerationRequest(
        user_prompt=f"{mode} user, {labels_text}",
        max_new_tokens=256,
        temperature=0.7,
    )
    return _run_async(
        orchestrator.generate_inference,
        provider="hf_local",
        model_id=model_id,
        request_data=req,
    )


def render_ai_assisted_annotation_tab(
    unit: Dict[str, Any],
    crit_widget_key: str,
    guide_widget_key: str,
):
    st.subheader("AI Assisted Annotation")
    st.markdown(
        "Use a local or cloud LLM to draft both completions. Drafts land in "
        "the editor tab where you can review and save them."
    )

    labels_text = render_labels_text(
        unit["parent_label_ids"], unit["children_label_ids"],
    )
    st.text_area("Labels text used as prompt context",
                 value=labels_text, height=80, disabled=True)

    model_id = st.text_input(
        "Local Model ID or Path",
        value="mistralai/Ministral-3-14B-Reasoning-2512",
    )
    gguf_file = st.text_input("GGUF Filename (if using a GGUF repo)")
    use_4bit = st.checkbox(
        "Use 4-bit Quantization (bitsandbytes)", value=False,
    )

    if st.button("Generate Drafts (critique + guide)",
                 key=f"generate_ai_{st.session_state.current_index}"):
        try:
            from app.llm.local_llm import LocalLLMConfig
            from app.pipelines.chat.orchestrator import TelemetryLLMOrchestrator

            with st.spinner(f"Loading/Using model {model_id} via hf_local..."):
                llm_config = LocalLLMConfig(
                    base_model=model_id,
                    load_in_4bit=use_4bit,
                    gguf_file=gguf_file if gguf_file else None,
                )
                orchestrator = TelemetryLLMOrchestrator(
                    llm_config=llm_config,
                    adapter_directory=Path("models/llm_adapters"),
                    dataset_directory=Path("models/llm_datasets"),
                )

                drafts: Dict[str, str] = {}
                for mode in MODES:
                    result = _draft_for_mode(orchestrator, model_id, labels_text, mode)
                    if result.get("status") == "success":
                        drafts[mode] = result.get("result", "")
                    else:
                        st.error(
                            f"{mode} draft failed: "
                            f"{result.get('message', 'Unknown error')}"
                        )
                        return

                st.session_state[f"pending_{crit_widget_key}"] = drafts.get("critique", "")
                st.session_state[f"pending_{guide_widget_key}"] = drafts.get("guide", "")
                st.success("Drafts generated — switch to the Dataset Editor tab.")
                st.text_area("Critique draft",
                             value=drafts.get("critique", ""), height=120, disabled=True)
                st.text_area("Guide draft",
                             value=drafts.get("guide", ""), height=120, disabled=True)
        except Exception as e:
            import traceback
            st.error(f"Error during AI generation: {e}")
            st.code(traceback.format_exc())
