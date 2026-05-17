"""Local-VLM sub-segment discovery (llama-server / Qwen-VL GGUF).

Renders the **🖥️ Local VLM** expander below the analysis section. Runs the
same Planner → Step Solver → Label Verifier → Proposal Synthesizer cycle
as the Claude variant but routes each VLM call through a locally hosted
llama-server backed by a Qwen-VL GGUF model.

Shares the streaming UI, result panel, and staged-review panel with the
Claude component via ``_agent_annotation_shared.py``.
"""

import streamlit as st

from ._agent_annotation_shared import (
    collect_parent_info,
    execute_pipeline_run,
)


def render_agent_annotation_local(
    df,
    form_start,
    form_end,
    form_labels,
    session_id,
    selected_annotation_key,
):
    """Render the local-VLM sub-segment discovery expander."""
    with st.expander("🖥️ Local VLM Sub-Segment Discovery"):
        st.markdown(
            "Run a **Planner → Step Solver (per step) → Label Verifier → "
            "Proposal Synthesizer** cycle using a **local Qwen-VL GGUF** "
            "model on llama-server. The planner picks a solver agent for "
            "each step; today the only solver is `describe_graphs`."
        )

        # --- Pipeline settings ---
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            max_iterations = st.number_input(
                "Max iterations",
                min_value=1,
                max_value=10,
                value=3,
                key="agent_annot_local_max_iter",
            )
        with col_s2:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.5,
                value=0.7,
                step=0.1,
                key="agent_annot_local_temp",
            )

        from app.services.llm.annotation_agent_llm_service import (
            QWEN25_VL_MODELS,
        )

        model_options = list(QWEN25_VL_MODELS.keys())
        default_idx = model_options.index("Qwen/Qwen2.5-VL-72B-Instruct")

        selected_model = st.selectbox(
            "VLM model",
            options=model_options,
            format_func=lambda x: QWEN25_VL_MODELS[x]["label"],
            index=default_idx,
            help="Model is downloaded from HuggingFace and converted to GGUF locally.",
            key="agent_annot_local_model",
        )

        model_spec = QWEN25_VL_MODELS[selected_model]
        model_max_context = model_spec["max_context"]
        model_max_new_tokens = model_spec["max_new_tokens"]

        # Seed defaults / clamp persisted slider values so switching to a
        # smaller-cap model doesn't raise StreamlitAPIException, and so we
        # can omit `value=` (passing both `value` and a key already in
        # session state triggers a Streamlit warning).
        if "agent_annot_local_ctx" not in st.session_state:
            st.session_state["agent_annot_local_ctx"] = min(32768, model_max_context)
        else:
            st.session_state["agent_annot_local_ctx"] = min(
                st.session_state["agent_annot_local_ctx"], model_max_context,
            )
        if "agent_annot_local_max_new_tokens" not in st.session_state:
            st.session_state["agent_annot_local_max_new_tokens"] = min(
                512, model_max_new_tokens,
            )
        else:
            st.session_state["agent_annot_local_max_new_tokens"] = min(
                st.session_state["agent_annot_local_max_new_tokens"],
                model_max_new_tokens,
            )

        quant_options = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        quantization_type = st.selectbox(
            "Quantization",
            options=quant_options,
            index=0,
            help="Lower = smaller & faster, higher = better quality.",
            key="agent_annot_local_quant",
        )

        with st.expander("⚙️ Advanced model settings", expanded=False):
            gguf_path = st.text_input(
                "GGUF model path (override)",
                value="",
                help="Leave empty to auto-detect / auto-convert. Only set to skip conversion.",
                key="agent_annot_local_gguf",
            )
            mmproj_path = st.text_input(
                "mmproj path (override)",
                value="",
                help="Leave empty to auto-detect / auto-convert.",
                key="agent_annot_local_mmproj",
            )
            context_size = st.slider(
                "Context size",
                min_value=2048,
                max_value=model_max_context,
                step=1024,
                help=f"Maximum for {model_spec['label']}: {model_max_context:,} tokens.",
                key="agent_annot_local_ctx",
            )
            n_gpu_layers = st.number_input(
                "GPU layers (-1 = all)",
                min_value=-1,
                max_value=200,
                value=-1,
                key="agent_annot_local_ngl",
            )
            max_new_tokens = st.slider(
                "Max new tokens (per VLM call)",
                min_value=128,
                max_value=model_max_new_tokens,
                step=128,
                help=(
                    f"Maximum for {model_spec['label']}: {model_max_new_tokens:,} tokens. "
                    "Bump higher for reasoning models (e.g. Qwen3-VL-Thinking) — "
                    "they spend most of their budget in the thinking phase "
                    "before emitting the final answer."
                ),
                key="agent_annot_local_max_new_tokens",
            )

        # --- Collect parent info ---
        parent_id, parent_main_label_ids, existing_children = collect_parent_info(
            form_labels,
        )

        if existing_children:
            st.caption(
                f"ℹ️ {len(existing_children)} existing child sub-segment(s) "
                "will be provided to the VLM to avoid duplicates."
            )

        # --- Run button ---
        if st.button(
            "▶ Run Local Sub-Segment Discovery",
            key="agent_annot_local_run",
            type="primary",
        ):
            try:
                from app.services.llm.annotation_agent_pipeline import (
                    AnnotationPipelineConfig,
                )
            except ImportError as e:
                st.error(
                    f"Missing dependency: {e}\n\n"
                    "Install with: `pip install langgraph langchain-core`"
                )
                return

            config = AnnotationPipelineConfig(
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

            execute_pipeline_run(
                df=df,
                form_start=form_start,
                form_end=form_end,
                session_id=session_id,
                parent_main_label_ids=parent_main_label_ids,
                existing_children=existing_children,
                config=config,
            )
