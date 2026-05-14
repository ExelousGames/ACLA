"""
Streamlit UI component for the LangGraph multi-agent sub-segment discovery pipeline.

Renders below the analysis section in the annotation manager and lets users
run the full annotation cycle on the currently selected parent segment:

    planner → step_solver (dispatches each step to its declared solver
              agent — currently `describe_graphs`, repeated per plan step)
        → label_verifier → proposal_synthesizer

The VLM receives rendered graph images at each step, replicating the visual
evidence a human annotator would use. Each LLM-producing node runs its own
evaluator suite internally before writing to state — there is no separate
evaluator node or retry loop.
"""

import io
import time
import uuid
import streamlit as st
import traceback

from PIL import Image

from ..shared import (
    LABEL_MAPPING,
    LABEL_NAME_TO_ID,
    LABEL_CATEGORIES,
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
            "Run a **Planner → Step Solver (per step) → Label Verifier → "
            "Proposal Synthesizer** cycle using the **Vision Language Model** "
            "to discover a new sub-segment within this parent segment. "
            "The planner picks a solver agent for each step; today the only "
            "solver is `describe_graphs`."
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
        default_idx = model_options.index("Qwen/Qwen2.5-VL-72B-Instruct")

        selected_model = st.selectbox(
            "VLM model",
            options=model_options,
            format_func=lambda x: QWEN25_VL_MODELS[x]["label"],
            index=default_idx,
            help="Model is downloaded from HuggingFace and converted to GGUF locally.",
            key="agent_annot_model",
        )

        model_spec = QWEN25_VL_MODELS[selected_model]
        model_max_context = model_spec["max_context"]
        model_max_new_tokens = model_spec["max_new_tokens"]

        # Clamp persisted slider values so switching to a smaller-cap model
        # doesn't raise StreamlitAPIException on re-render.
        if "agent_annot_ctx" in st.session_state:
            st.session_state["agent_annot_ctx"] = min(
                st.session_state["agent_annot_ctx"], model_max_context,
            )
        if "agent_annot_max_new_tokens" in st.session_state:
            st.session_state["agent_annot_max_new_tokens"] = min(
                st.session_state["agent_annot_max_new_tokens"], model_max_new_tokens,
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
            context_size = st.slider(
                "Context size",
                min_value=2048,
                max_value=model_max_context,
                value=min(32768, model_max_context),
                step=1024,
                help=f"Maximum for {model_spec['label']}: {model_max_context:,} tokens.",
                key="agent_annot_ctx",
            )
            n_gpu_layers = st.number_input(
                "GPU layers (-1 = all)",
                min_value=-1,
                max_value=200,
                value=-1,
                key="agent_annot_ngl",
            )
            max_new_tokens = st.slider(
                "Max new tokens (per VLM call)",
                min_value=128,
                max_value=model_max_new_tokens,
                value=min(512, model_max_new_tokens),
                step=128,
                help=(
                    f"Maximum for {model_spec['label']}: {model_max_new_tokens:,} tokens. "
                    "Bump higher for reasoning models (e.g. Qwen3-VL-Thinking) — "
                    "they spend most of their budget in the thinking phase "
                    "before emitting the final answer."
                ),
                key="agent_annot_max_new_tokens",
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
                    max_new_tokens=int(max_new_tokens),
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

                # ----------------------------------------------------------
                # Live VLM output — collapsible steps
                # Each completed VLM call collapses into an expander with a
                # summary header (icon + node name + detail + duration).
                # The active / streaming call stays expanded at the bottom.
                # ----------------------------------------------------------
                st.markdown("**Live VLM Output**")
                # Two separate placeholders so the streaming "active" block
                # can be cleared independently of the completed expanders.
                # Mixing st.expander widgets with bare st.markdown inside a
                # single placeholder leaves the markdown elements stuck in
                # the DOM when the active state empties out at end-of-run.
                completed_area = st.empty()
                active_area = st.empty()

                # Each section: {meta: {node_name, phase, iteration, total},
                #                prompt, reasoning, text, duration}
                completed_sections: list[dict] = []
                vlm_buffer: list[str] = []        # response chunks for active call
                reasoning_buffer: list[str] = []  # chain-of-thought chunks (thinking models)
                active_prompt: list[str] = [""]
                active_meta: dict = {}            # set by on_vlm_prompt from pipeline stage
                step_start_time: list[float] = [0.0]

                _NODE_ICONS = {
                    "planner": "🧠",
                    "step_solver": "🛠️",
                    "describe_graphs": "🔍",
                    "zoom": "🔬",
                    "label_verifier": "✅",
                    "proposal_synthesizer": "📝",
                }

                _ATTACHMENT_ICONS = {
                    "text": "📎",
                    "structured": "📋",
                    "image_set": "🖼️",
                }

                def _render_attachments(meta: dict) -> None:
                    """Show a one-line chip row of attachments fed to this VLM call."""
                    atts = meta.get("attachments") or []
                    if not atts:
                        st.caption(
                            "_📎 (no upstream attachments — built from raw run inputs)_"
                        )
                        return
                    chips: list[str] = []
                    for att in atts:
                        icon = _ATTACHMENT_ICONS.get(att.get("kind", ""), "📎")
                        label = att.get("label") or att.get("name", "?")
                        count = att.get("count")
                        chip = f"{icon} {label}"
                        if count is not None:
                            chip = f"{chip} ({count})"
                        chips.append(chip)
                    st.caption(" · ".join(chips))

                def _format_header(meta: dict, duration: float | None = None) -> str:
                    """Build a section header like '🧠 Planner (1/2) — main (3.2s)'."""
                    node = meta.get("node_name") or "Processing"
                    icon = _NODE_ICONS.get(node, "●")
                    title = node.replace("_", " ").title()
                    graphs = meta.get("graphs") or []
                    graph_tag = f" [{', '.join(graphs)}]" if graphs else ""
                    iteration = meta.get("iteration")
                    total = meta.get("total")
                    iter_tag = f" ({iteration}/{total})" if iteration and total else ""
                    phase = meta.get("phase") or ""
                    phase_tag = f" — {phase}" if phase else ""
                    dur_tag = f"  ({duration:.1f}s)" if duration else ""
                    return f"{icon} {title}{graph_tag}{iter_tag}{phase_tag}{dur_tag}"

                def _finalize_active_section() -> None:
                    """Move the active VLM section to completed_sections."""
                    if active_prompt[0] or vlm_buffer or reasoning_buffer:
                        elapsed = (
                            time.time() - step_start_time[0]
                            if step_start_time[0] else 0
                        )
                        completed_sections.append({
                            "meta": dict(active_meta),
                            "prompt": active_prompt[0],
                            "reasoning": "".join(reasoning_buffer),
                            "text": "".join(vlm_buffer),
                            "duration": elapsed,
                        })
                        vlm_buffer.clear()
                        reasoning_buffer.clear()
                        active_prompt[0] = ""
                        active_meta.clear()
                        step_start_time[0] = 0.0

                def _render_completed() -> None:
                    """Re-render the collapsed expanders for completed steps."""
                    with completed_area.container():
                        for s in completed_sections:
                            header = _format_header(s["meta"], s.get("duration", 0))
                            with st.expander(header, expanded=False):
                                _render_attachments(s["meta"])
                                has_response = bool(s.get("text"))
                                st.markdown(
                                    "**Prompt:**" if has_response else "**Summary:**"
                                )
                                with st.container(border=True):
                                    st.markdown(s["prompt"])
                                if s.get("reasoning"):
                                    st.markdown(
                                        f"**💭 Thinking:**\n\n{s['reasoning']}"
                                    )
                                if has_response:
                                    st.markdown(f"**Response:**\n\n{s['text']}")

                def _render_active() -> None:
                    """Re-render the live streaming section, or clear it."""
                    if not (active_prompt[0] or vlm_buffer or reasoning_buffer):
                        active_area.empty()
                        return
                    with active_area.container():
                        elapsed = (
                            time.time() - step_start_time[0]
                            if step_start_time[0] else 0
                        )
                        st.markdown(
                            f"**{_format_header(active_meta)}** "
                            f"_{elapsed:.0f}s …_"
                        )
                        _render_attachments(active_meta)
                        st.markdown("*Prompt:*")
                        with st.container(border=True):
                            st.markdown(active_prompt[0])
                        if reasoning_buffer:
                            st.markdown(
                                f"*💭 Thinking (streaming…)*\n\n"
                                f"{''.join(reasoning_buffer)}"
                            )
                        st.markdown(
                            f"*Response (streaming…)*\n\n"
                            f"{''.join(vlm_buffer)}"
                        )

                def _render_vlm_output() -> None:
                    """Re-render both sections.

                    Completed steps → collapsed ``st.expander``
                    Active step    → shown open at the bottom (own placeholder)
                    """
                    _render_completed()
                    _render_active()

                # ---- Callbacks ----

                def on_vlm_prompt(prompt: str, stage: dict) -> None:
                    # Finalize any prior active call before starting the next.
                    _finalize_active_section()
                    active_prompt[0] = prompt
                    active_meta.clear()
                    active_meta.update(stage)
                    step_start_time[0] = time.time()
                    _render_vlm_output()

                def on_vlm_stream(chunk: str) -> None:
                    vlm_buffer.append(chunk)
                    _render_vlm_output()

                def on_vlm_reasoning(chunk: str) -> None:
                    reasoning_buffer.append(chunk)
                    _render_vlm_output()

                def on_step_event(summary: str, stage: dict) -> None:
                    # Non-VLM event (e.g. zoom rendering). Finalize any
                    # in-flight active section, then append a response-less
                    # completed section directly.
                    _finalize_active_section()
                    completed_sections.append({
                        "meta": dict(stage),
                        "prompt": summary,
                        "reasoning": "",
                        "text": "",
                        "duration": 0,
                    })
                    _render_vlm_output()

                def on_progress(node_name: str, detail: str) -> None:
                    # Rough progress: each completed VLM section nudges the bar.
                    pct = min((len(completed_sections) + 1) / 12, 0.99)
                    progress_bar.progress(pct)
                    status_text.markdown(
                        f"**Status:** _{node_name}: {detail[:200]}_"
                    )

                with st.spinner("Running sub-segment discovery pipeline…"):
                    result = run_annotation_pipeline(
                        df=df,
                        start_index=int(form_start),
                        end_index=int(form_end),
                        session_id=session_id,
                        parent_main_labels=parent_main_label_ids,
                        existing_children=existing_children,
                        config=config,
                        progress_callback=on_progress,
                        vlm_stream_callback=on_vlm_stream,
                        vlm_prompt_callback=on_vlm_prompt,
                        vlm_reasoning_callback=on_vlm_reasoning,
                        step_event_callback=on_step_event,
                    )

                # Finalise any still-active call and force a clean re-render
                # so the streaming placeholder is emptied even if the if
                # check above would otherwise skip it.
                _finalize_active_section()
                _render_vlm_output()

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
                                        st.image(img, width='stretch')
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
