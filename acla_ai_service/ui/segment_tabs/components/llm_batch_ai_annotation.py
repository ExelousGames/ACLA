import streamlit as st
from pathlib import Path
from typing import Any, Callable, Dict, List

from app.domain.labels import LABEL_MAPPING
from app.pipelines.training.dataset_builder import MODES, render_labels_text
from segment_tabs.shared import _run_async


def _unit_all_label_ids(unit: Dict[str, Any]) -> List[str]:
    out = list(unit["parent_label_ids"])
    for child in unit["children_label_ids"]:
        out.extend(child)
    return out


def render_batch_ai_annotation_tab(
    units: List[Dict[str, Any]],
    existing_units: Dict[str, Dict[str, Any]],
    output_path: str,
    available_sessions: List[str] = None,
    save_training_unit_fn: Callable = None,
):
    """Generate critique + guide drafts for many units in one pass."""
    st.header("Batch AI Annotation")
    st.markdown("Draft critique + guide completions for many units at once.")

    if not units:
        st.warning("No units available for batch annotation.")
        return

    model_id = st.text_input(
        "Local Model ID or Path (Batch)",
        value="mistralai/Ministral-3-14B-Reasoning-2512",
    )
    gguf_file = st.text_input(
        "GGUF Filename (if using a GGUF repo)", key="batch_gguf",
    )
    use_4bit = st.checkbox(
        "Use 4-bit Quantization (bitsandbytes)", value=False, key="batch_4bit",
    )

    st.subheader("Filter Units")

    if available_sessions:
        selected_sessions = st.multiselect(
            "Only process units from these sessions:",
            options=available_sessions,
            help="Leave empty to process units from all sessions.",
        )
    else:
        selected_sessions = []

    all_label_ids = list(LABEL_MAPPING.keys())
    selected_labels = st.multiselect(
        "Only process units containing (any of) these labels:",
        options=all_label_ids,
        format_func=lambda x: f"{x} - {LABEL_MAPPING.get(x, 'Unknown')}",
        help="Leave empty to process all units.",
    )

    filtered = units
    if selected_sessions:
        filtered = [u for u in filtered if u.get("session_id") in selected_sessions]
    if selected_labels:
        filtered = [
            u for u in filtered
            if any(l in selected_labels for l in _unit_all_label_ids(u))
        ]

    st.write(f"Total units available (after filtering): **{len(filtered)}**")

    if not filtered:
        return

    batch_limit = st.number_input(
        "Batch Limit (number of units to process)",
        min_value=1,
        max_value=len(filtered),
        value=min(10, len(filtered)),
    )

    if st.button("Start Batch Processing", key="btn_batch_start"):
        try:
            from app.llm.local_llm import GenerationRequest, LocalLLMConfig
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

                progress = st.progress(0)
                status = st.empty()
                target_units = filtered[: int(batch_limit)]

                success = 0
                errors = 0
                for i, unit in enumerate(target_units):
                    status.text(
                        f"Processing {i + 1}/{len(target_units)}: unit {unit['unit_id']}"
                    )
                    labels_text = render_labels_text(
                        unit["parent_label_ids"], unit["children_label_ids"],
                    )

                    drafts: Dict[str, str] = {}
                    failed = False
                    for mode in MODES:
                        req = GenerationRequest(
                            user_prompt=f"{mode} user, {labels_text}",
                            max_new_tokens=256,
                            temperature=0.7,
                        )
                        result = _run_async(
                            orchestrator.generate_inference,
                            provider="hf_local",
                            model_id=model_id,
                            request_data=req,
                        )
                        if result.get("status") != "success":
                            st.error(
                                f"unit {unit['unit_id']} {mode} failed: "
                                f"{result.get('message', 'Unknown error')}"
                            )
                            failed = True
                            break
                        drafts[mode] = result.get("result", "")

                    if failed:
                        errors += 1
                    else:
                        save_training_unit_fn(
                            unit, drafts["critique"], drafts["guide"], output_path,
                        )
                        existing_units[unit["unit_id"]] = {
                            "unit_id": unit["unit_id"],
                            "kind": unit["kind"],
                            "parent_label_ids": unit["parent_label_ids"],
                            "children_label_ids": unit["children_label_ids"],
                            "completion_critique": drafts["critique"],
                            "completion_guide": drafts["guide"],
                        }
                        success += 1

                    progress.progress((i + 1) / len(target_units))

                st.success(
                    f"Batch processing complete. "
                    f"Saved {success} units. Errors: {errors}"
                )
        except Exception as e:
            import traceback
            st.error(f"Error during AI generation: {e}")
            st.code(traceback.format_exc())
