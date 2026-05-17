"""Local-VLM lap-section annotation expander (manual.py).

Drives the shared ``annotation_root`` Agent — the same single agent the
detailed-annotation flow uses. The only difference between the two flows
is the input: this wrapper passes a lap-flavoured planner prompt, the
``lap_annotation_skill`` block, a lap-shaped output schema, and parses
the agent's raw response into a ``LapAnnotationResult``.

Mirrors ``manual_lap_agent_claude.py``'s control surface so the user can
swap backends without re-orienting.
"""

from __future__ import annotations

import streamlit as st

from ._lap_agent_shared import (
    KEY_LAP_RANGE, KEY_LAP_CIRCUIT,
    execute_lap_agent_run,
)


def render_lap_agent_local(df, session_id, selected_annotation_key, circuit_id, head):
    """Render the local-VLM lap-section excerpter expander.

    ``head`` is the current head segment from the shared lap panel.
    """
    with st.expander("🖥️ Local VLM — Lap-to-Segment Excerpter"):
        st.markdown(
            "Annotates **one rough-split section per click** using the "
            "shared **Planner → describe_graphs → Label Verifier → "
            "Synthesizer** harness over a local Qwen-VL GGUF model on "
            "llama-server. Same agent as detailed annotation — different "
            "prompts and the lap-annotation skill."
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            max_iterations = st.number_input(
                "Max iterations",
                min_value=1, max_value=10, value=3,
                key="lap_local_max_iter",
            )
        with col_s2:
            temperature = st.slider(
                "Temperature", min_value=0.1, max_value=1.5,
                value=0.3, step=0.1, key="lap_local_temp",
            )

        from app.services.llm.annotation_agent_llm_service import (
            QWEN25_VL_MODELS,
        )
        model_options = list(QWEN25_VL_MODELS.keys())
        default_idx = model_options.index("Qwen/Qwen2.5-VL-72B-Instruct")
        selected_model = st.selectbox(
            "VLM model", options=model_options,
            format_func=lambda x: QWEN25_VL_MODELS[x]["label"],
            index=default_idx,
            help="Model is downloaded from HuggingFace and converted to GGUF locally.",
            key="lap_local_model",
        )
        model_spec = QWEN25_VL_MODELS[selected_model]
        model_max_context = model_spec["max_context"]
        model_max_new_tokens = model_spec["max_new_tokens"]

        if "lap_local_ctx" not in st.session_state:
            st.session_state["lap_local_ctx"] = min(32768, model_max_context)
        else:
            st.session_state["lap_local_ctx"] = min(
                st.session_state["lap_local_ctx"], model_max_context,
            )
        if "lap_local_max_new_tokens" not in st.session_state:
            st.session_state["lap_local_max_new_tokens"] = min(
                1024, model_max_new_tokens,
            )
        else:
            st.session_state["lap_local_max_new_tokens"] = min(
                st.session_state["lap_local_max_new_tokens"], model_max_new_tokens,
            )

        quant_options = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        quantization_type = st.selectbox(
            "Quantization", options=quant_options, index=0,
            help="Lower = smaller & faster, higher = better quality.",
            key="lap_local_quant",
        )

        with st.expander("⚙️ Advanced model settings", expanded=False):
            gguf_path = st.text_input(
                "GGUF model path (override)", value="",
                help="Leave empty to auto-detect / auto-convert.",
                key="lap_local_gguf",
            )
            mmproj_path = st.text_input(
                "mmproj path (override)", value="",
                key="lap_local_mmproj",
            )
            context_size = st.slider(
                "Context size", min_value=2048, max_value=model_max_context,
                step=1024, key="lap_local_ctx",
            )
            n_gpu_layers = st.number_input(
                "GPU layers (-1 = all)", min_value=-1, max_value=200,
                value=-1, key="lap_local_ngl",
            )
            max_new_tokens = st.slider(
                "Max new tokens (per VLM call)", min_value=128,
                max_value=model_max_new_tokens, step=128,
                key="lap_local_max_new_tokens",
            )

        if head is None:
            st.caption(
                "Pick a valid lap range above — the splitter fills the "
                "array automatically."
            )
            return

        existing = _collect_existing_lap_annotations()

        if st.button(
            "▶ Run Local VLM on current section",
            key="lap_local_run", type="primary",
        ):
            try:
                from app.services.llm.annotation_agent_pipeline import (
                    AnnotationPipelineConfig,
                    run_lap_annotation_pipeline,
                )
            except ImportError as e:
                st.error(
                    f"Missing dependency: {e}\n\n"
                    "Install with: `pip install langgraph langchain-core`"
                )
                return

            pipeline_cfg = AnnotationPipelineConfig(
                max_iterations=int(max_iterations),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                backend="local",
                gguf_path=gguf_path or None,
                mmproj_path=mmproj_path or None,
                context_size=int(context_size),
                n_gpu_layers=int(n_gpu_layers),
                hf_repo=selected_model,
                quantization_type=quantization_type,
            )

            lap_start, lap_end = st.session_state[KEY_LAP_RANGE]
            execute_lap_agent_run(
                run_fn=run_lap_annotation_pipeline,
                df=df,
                lap_start=int(lap_start),
                lap_end=int(lap_end),
                head_segment=head,
                circuit_id=st.session_state[KEY_LAP_CIRCUIT],
                existing=existing,
                extra_kwargs={"config": pipeline_cfg},
            )


def _collect_existing_lap_annotations():
    """Existing annotations overlapping the picked lap range."""
    lap_range = st.session_state.get(KEY_LAP_RANGE)
    if not lap_range:
        return []
    lap_start, lap_end = int(lap_range[0]), int(lap_range[1])
    annotations = st.session_state.get("current_annotations", [])
    out = []
    for ann in annotations:
        s = int(getattr(ann, "start_index", 0))
        e = int(getattr(ann, "end_index", 0))
        if e <= lap_start or s >= lap_end:
            continue
        out.append({
            "start_index": s,
            "end_index": e,
            "labels": list(getattr(ann, "labels", [])),
        })
    return out
