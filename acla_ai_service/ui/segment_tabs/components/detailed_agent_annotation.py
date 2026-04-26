"""
Streamlit UI component for the LangGraph multi-agent sub-segment discovery pipeline.

Renders below the analysis section in the annotation manager and
lets users run the Planner → Tool Executor → Step Solver → Evaluator
cycle to discover a new sub-segment within the currently selected parent segment.
The VLM analyses both telemetry statistics and **graph images**.
"""

import io
import uuid
import streamlit as st
import traceback

from PIL import Image

from ..shared import (
    LABEL_MAPPING,
    LABEL_NAME_TO_ID,
    LABEL_CATEGORIES,
    MAIN_LABEL_GUIDELINES,
    get_display_labels,
    save_annotations,
)

from app.models.segment_models import AnnotatedSegment


def render_agent_annotation(df, form_start, form_end, form_labels, session_id, selected_annotation_key):
    """Render the VLM sub-segment discovery expander.

    Parameters
    ----------
    df : pd.DataFrame
        Full session telemetry.
    form_start : int
        Parent segment start index.
    form_end : int
        Parent segment end index.
    form_labels : list[str]
        Display-name labels currently selected on the parent segment.
    session_id : str
        Current session identifier.
    selected_annotation_key : str
        Annotation key for persistence.
    """
    with st.expander("🔍 VLM Sub-Segment Discovery"):
        st.markdown(
            "Run a **Planner → Tool Executor → Step Solver → Evaluator** cycle "
            "using the **Vision Language Model** "
            "to discover a new sub-segment within this parent segment."
        )

        # --- Pipeline settings ---
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            max_iterations = st.number_input(
                "Max iterations",
                min_value=1,
                max_value=10,
                value=3,
                key="agent_annot_max_iter",
            )
        with col_s2:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.5,
                value=0.7,
                step=0.1,
                key="agent_annot_temp",
            )

        # --- Model selection ---
        from app.services.llm.annotation_agent_llm_service import QWEN25_VL_MODELS

        model_options = list(QWEN25_VL_MODELS.keys())
        model_labels = [QWEN25_VL_MODELS[k] for k in model_options]
        default_idx = model_options.index("Qwen/Qwen2.5-VL-72B-Instruct")

        selected_model = st.selectbox(
            "VLM model",
            options=model_options,
            format_func=lambda x: QWEN25_VL_MODELS[x],
            index=default_idx,
            help="Model is downloaded from HuggingFace and converted to GGUF locally.",
            key="agent_annot_model",
        )

        quant_options = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        quantization_type = st.selectbox(
            "Quantization",
            options=quant_options,
            index=0,
            help="Lower = smaller & faster, higher = better quality.",
            key="agent_annot_quant",
        )

        # --- Advanced model settings ---
        with st.expander("⚙️ Advanced model settings", expanded=False):
            gguf_path = st.text_input(
                "GGUF model path (override)",
                value="",
                help="Leave empty to auto-detect / auto-convert. Only set to skip conversion.",
                key="agent_annot_gguf",
            )
            mmproj_path = st.text_input(
                "mmproj path (override)",
                value="",
                help="Leave empty to auto-detect / auto-convert.",
                key="agent_annot_mmproj",
            )
            context_size = st.number_input(
                "Context size",
                min_value=2048,
                max_value=131072,
                value=16384,
                step=1024,
                key="agent_annot_ctx",
            )
            n_gpu_layers = st.number_input(
                "GPU layers (-1 = all)",
                min_value=-1,
                max_value=200,
                value=-1,
                key="agent_annot_ngl",
            )

        # --- Collect parent info ---
        # Only pass Main Labels as hints
        main_label_set = set(LABEL_CATEGORIES.get("Main Labels", []))
        parent_main_label_ids = [
            LABEL_NAME_TO_ID[l] for l in form_labels
            if l in LABEL_NAME_TO_ID and LABEL_NAME_TO_ID[l] in main_label_set
        ]

        # Collect existing children from current annotations
        annotations = st.session_state.get("current_annotations", [])
        selected_option = st.session_state.get("detailed_annotation_selector")
        parent_ann = annotations[selected_option] if (
            selected_option is not None and selected_option < len(annotations)
        ) else None
        parent_id = parent_ann.id if parent_ann else None

        existing_children = []
        if parent_id:
            for ann in annotations:
                if getattr(ann, "parent_id", None) == parent_id:
                    existing_children.append({
                        "start_index": ann.start_index,
                        "end_index": ann.end_index,
                        "labels": list(ann.labels),
                    })

        if existing_children:
            st.caption(
                f"ℹ️ {len(existing_children)} existing child sub-segment(s) "
                "will be provided to the VLM to avoid duplicates."
            )

        # --- Run button ---
        if st.button("▶ Run Sub-Segment Discovery", key="agent_annot_run", type="primary"):
            try:
                from app.services.llm.annotation_agent_pipeline import (
                    AnnotationPipelineConfig,
                    run_annotation_pipeline,
                )

                config = AnnotationPipelineConfig(
                    max_iterations=int(max_iterations),
                    temperature=float(temperature),
                    gguf_path=gguf_path or None,
                    mmproj_path=mmproj_path or None,
                    context_size=int(context_size),
                    n_gpu_layers=int(n_gpu_layers),
                    hf_repo=selected_model,
                    quantization_type=quantization_type,
                )

                # Progress display
                progress_area = st.container()
                status_text = progress_area.empty()
                progress_bar = progress_area.progress(0)
                step_log = progress_area.empty()
                log_entries = []

                def on_progress(node_name: str, iteration: int, detail: str):
                    """Callback to update Streamlit UI with progress."""
                    # Map node names to progress fraction
                    node_order = ["planner", "step_executor", "step_reasoner",
                                  "label_shortlister", "label_verifier",
                                  "proposal_synthesizer", "evaluator"]
                    idx = node_order.index(node_name) if node_name in node_order else 0
                    progress = min((idx + 1) / len(node_order), 0.99)
                    message = f"[Iter {iteration}] {node_name}: {detail}"

                    progress_bar.progress(progress)
                    status_text.markdown(f"**Status:** _{message}_")

                with st.spinner("Running sub-segment discovery pipeline..."):
                    result = run_annotation_pipeline(
                        df=df,
                        start_index=int(form_start),
                        end_index=int(form_end),
                        session_id=session_id,
                        parent_main_labels=parent_main_label_ids,
                        existing_children=existing_children,
                        config=config,
                        progress_callback=on_progress,
                    )

                progress_bar.progress(1.0)

                # --- Display results ---
                if result.accepted:
                    st.success(
                        f"✅ Accepted after {result.iterations} iteration(s)."
                    )
                else:
                    st.warning(
                        f"⚠️ Max iterations reached ({result.iterations}). "
                        "Showing best proposal."
                    )

                # Proposed sub-segment range
                st.markdown("##### Proposed Sub-Segment")
                st.markdown(
                    f"**Range:** [{result.sub_start}, {result.sub_end}]  "
                    f"(parent: [{form_start}, {form_end}])"
                )

                # Proposed labels
                st.markdown("##### Proposed Labels")
                proposed_display = [
                    LABEL_MAPPING.get(l, l) for l in result.final_labels
                ]
                for label in proposed_display:
                    st.markdown(f"- {label}")

                # Reasoning
                st.markdown("##### Reasoning")
                st.markdown(result.final_reasoning)

                # Telemetry graphs generated by tools
                if result.graph_images:
                    with st.expander(
                        f"📊 Telemetry Graphs ({len(result.graph_images)} images)",
                        expanded=False,
                    ):
                        cols = st.columns(min(len(result.graph_images), 3))
                        for idx, img_bytes in enumerate(result.graph_images):
                            with cols[idx % len(cols)]:
                                try:
                                    img = Image.open(io.BytesIO(img_bytes))
                                    if img.width == 0 or img.height == 0:
                                        st.warning(f"Graph {idx + 1}: image has zero dimensions and cannot be displayed.")
                                    else:
                                        st.image(img, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Graph {idx + 1}: failed to load image — {e}")

                # Store result for staged review
                st.session_state["agent_annot_result"] = result

                # Conversation log
                with st.expander("Full agent conversation log"):
                    for msg in result.messages:
                        role = msg.get("role", "unknown").replace("_", " ").title()
                        iteration = msg.get("iteration", "?")
                        content = msg.get("content", "")
                        verdict = msg.get("verdict", "")
                        header = f"**[Iter {iteration}] {role}**"
                        if verdict:
                            header += f" — _{verdict.upper()}_"
                        st.markdown(header)
                        st.text(content[:2000])
                        st.markdown("---")

            except ImportError as e:
                st.error(
                    f"Missing dependency: {e}\n\n"
                    "Install with: `pip install langgraph langchain-core`"
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.code(traceback.format_exc())

        # --- Staged review (visible when result exists) ---
        if "agent_annot_result" in st.session_state:
            result = st.session_state["agent_annot_result"]
            proposed_display = [
                LABEL_MAPPING.get(l, l) for l in result.final_labels
            ]
            st.info(
                f"Pending sub-segment: [{result.sub_start}, {result.sub_end}] — "
                f"{', '.join(proposed_display)} "
                f"({result.iterations} iter, "
                f"{'accepted' if result.accepted else 'max-iter'})"
            )

            st.markdown("##### Review & Edit Before Saving")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                staged_start = st.number_input(
                    "Sub-segment start",
                    min_value=int(form_start),
                    max_value=int(form_end),
                    value=int(result.sub_start) if result.sub_start is not None else int(form_start),
                    key="agent_staged_start",
                )
            with col_r2:
                staged_end = st.number_input(
                    "Sub-segment end",
                    min_value=int(form_start),
                    max_value=int(form_end),
                    value=int(result.sub_end) if result.sub_end is not None else int(form_end),
                    key="agent_staged_end",
                )

            # Editable label multiselect
            all_label_options = sorted(LABEL_MAPPING.values())
            staged_labels = st.multiselect(
                "Sub-segment labels",
                options=all_label_options,
                default=proposed_display,
                key="agent_staged_labels",
            )

            staged_notes = st.text_input(
                "Notes (optional)",
                value="",
                key="agent_staged_notes",
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button(
                    "✅ Confirm & Save Sub-Segment",
                    key="agent_annot_confirm",
                    type="primary",
                ):
                    _stage_subsegment(
                        staged_start=int(staged_start),
                        staged_end=int(staged_end),
                        staged_label_names=staged_labels,
                        staged_notes=staged_notes,
                        parent_id=parent_id,
                        session_id=session_id,
                        selected_annotation_key=selected_annotation_key,
                    )
            with col_btn2:
                if st.button(
                    "❌ Discard Proposal",
                    key="agent_annot_discard",
                ):
                    del st.session_state["agent_annot_result"]
                    st.rerun()


def _stage_subsegment(
    staged_start: int,
    staged_end: int,
    staged_label_names: list,
    staged_notes: str,
    parent_id: str | None,
    session_id: str,
    selected_annotation_key: str,
):
    """Create a child AnnotatedSegment and persist it."""
    if staged_start >= staged_end:
        st.error("Start index must be less than end index.")
        return

    label_ids = [
        LABEL_NAME_TO_ID[n] for n in staged_label_names if n in LABEL_NAME_TO_ID
    ]
    if not label_ids:
        st.error("Select at least one label.")
        return

    child = AnnotatedSegment(
        id=str(uuid.uuid4()),
        labels=label_ids,
        segment_length=staged_end - staged_start,
        start_index=staged_start,
        end_index=staged_end,
        notes=staged_notes,
        parent_id=parent_id,
    )

    annotations = list(st.session_state.get("current_annotations", []))
    annotations.append(child)
    st.session_state["current_annotations"] = annotations

    save_annotations(session_id, annotations, selected_annotation_key)

    # Clean up
    if "agent_annot_result" in st.session_state:
        del st.session_state["agent_annot_result"]

    st.success(
        f"Sub-segment [{staged_start}, {staged_end}] saved with "
        f"{len(label_ids)} label(s)."
    )
    st.rerun()
