import streamlit as st
import time
import traceback
import pandas as pd
import copy

from .shared import (
    load_session_data, load_annotations, save_annotations,
    get_available_sessions, build_segment,
    LABEL_MAPPING, LABEL_NAME_TO_ID,
    LABEL_CATEGORIES, MAIN_LABEL_GUIDELINES
)

def render_rule_based_annotation(df, selected_annotation_key):
    """
    Renders the Rule-Based Annotation section.
    Allows user to select a feature, a value, and a label.
    If any point in the segment matches the feature value, the label is applied.
    """
    st.header("Rule-Based Annotation")
    st.write("Automatically apply a label to segments where a selected feature matches a specific value.")
    
    # Ensure annotations exist
    if not st.session_state.get("current_annotations"):
        st.info("No segments available to process.")
        return

    # 1. Select Feature
    if df.empty:
        st.warning("Dataframe is empty.")
        return
        
    features = sorted(df.columns.tolist())
    selected_feature = st.selectbox("Select Feature", features, key="rule_feature_select")
    
    # 2. Input Value
    col_dtype = df[selected_feature].dtype
    
    # Heuristic to detect if "object" column is actually numeric
    is_numeric_col = pd.api.types.is_numeric_dtype(col_dtype)
    if not is_numeric_col and col_dtype == 'object':
        try:
            # Check first non-null value
            first_valid = df[selected_feature].dropna().iloc[0]
            if isinstance(first_valid, (int, float, complex)) or (isinstance(first_valid, str) and first_valid.replace('.','',1).isdigit()):
                 is_numeric_col = True
        except:
            pass

    # Unified Input Logic: Offer suggestions (selectbox) for ANY column type if low cardinality
    target_value_str = ""
    try:
         # Limit unique values for performance
         unique_raw = df[selected_feature].dropna().unique()
         type_info = f"{col_dtype} - treated as Numeric" if is_numeric_col else str(col_dtype)

         if len(unique_raw) < 1000:
             # Convert all to string for display/selection
             # Sort carefully if numeric to ensure logical ordering in dropdown
             if is_numeric_col:
                 # Sort numerically first, then stringify
                 try:
                    # Preserve numeric sort order!
                    unique_vals = [str(v) for v in sorted(unique_raw)]
                 except:
                    # Fallback to string sort
                    unique_vals = sorted([str(v) for v in unique_raw])
             else:
                 unique_vals = sorted([str(v) for v in unique_raw])
             
             # Use selectbox for "closest matching" feel via search
             # index=None makes it start with no selection (placeholder)
             target_value_str = st.selectbox(
                 f"Select Target Value (Type: {type_info})", 
                 unique_vals, 
                 index=None,
                 placeholder="Type to search...",
                 key="rule_value_select"
            )
             
             # If user hasn't selected anything yet, default to empty string so logic downstream handles it gracefully
             if target_value_str is None:
                 target_value_str = ""
         else:
             target_value_str = st.text_input(f"Target Value (Type: {type_info})", key="rule_value_input")
    except Exception:
         # Fallback if error getting unique values
         target_value_str = st.text_input(f"Target Value (Type: {col_dtype})", key="rule_value_input_fallback")
    
    # 3. Select Label
    # LABEL_NAME_TO_ID maps "Label Name" -> "Label ID"
    label_names = sorted(list(LABEL_NAME_TO_ID.keys()))
    selected_label_name = st.selectbox("Select Label", label_names, key="rule_label_select")
    
    # 4. Apply Rule Logic
    col_rule_1, col_rule_2 = st.columns([1, 4])
    with col_rule_1:
        apply_clicked = st.button("Apply Rule", key="rule_apply_btn")
    
    with col_rule_2:
        if st.session_state.get("last_rule_snapshot"):
            if st.button("Undo Last Change", key="rule_undo_btn"):
                st.session_state.current_annotations = st.session_state.last_rule_snapshot
                st.session_state.last_rule_snapshot = None # Clear after undo
                st.success("Reverted last rule application.")
                
                # Persist reversion
                if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                    save_annotations(
                        st.session_state.last_session_id,
                        st.session_state.current_annotations,
                        st.session_state.last_annotation_key,
                        silent=False
                    )
                time.sleep(1)
                st.rerun()

    if apply_clicked:
        if target_value_str == "":
            st.warning("Please enter a target value.")
            return
            
        # target_val preparation
        target_val = target_value_str
        try:
            if is_numeric_col:
                target_val = float(target_value_str)
            elif pd.api.types.is_bool_dtype(col_dtype):
                 if target_value_str.lower() in ['true', 't', '1', 'yes']:
                     target_val = True
                 elif target_value_str.lower() in ['false', 'f', '0', 'no']:
                     target_val = False
        except ValueError:
            st.error(f"Could not convert '{target_value_str}' for column type.")
            return

        selected_label_id = LABEL_NAME_TO_ID[selected_label_name]
        
        # Save snapshot for undo
        st.session_state.last_rule_snapshot = copy.deepcopy(st.session_state.current_annotations)
        
        count_updated = 0
        count_matching_value = 0
        total_segments = len(st.session_state.current_annotations)
        progress_bar = st.progress(0)
        
        for i, ann in enumerate(st.session_state.current_annotations):
            # Check bounds
            start = ann.start_index if ann.start_index is not None else 0
            end = ann.end_index if ann.end_index is not None else len(df) - 1
            
            if start < 0 or end >= len(df) or start > end:
                continue
                
            segment_data = df.iloc[start : end + 1]
            segment_series = segment_data[selected_feature]
            
            # If we are treating as numeric but column is object, try to convert segment series
            if is_numeric_col and segment_series.dtype == 'object':
                 segment_series = pd.to_numeric(segment_series, errors='coerce')

            # Comparison
            try:
                # Basic any() check
                # For numeric, direct equality can be tricky with floats, but fits "== input value" request.
                if (segment_series == target_val).any():
                    count_matching_value += 1
                    current_labels = set(ann.labels)
                    if selected_label_id not in current_labels:
                        ann.labels.append(selected_label_id)
                        count_updated += 1
            except Exception:
                continue
            
            progress_bar.progress((i + 1) / total_segments)
            
        if count_updated > 0:
            st.success(f"Updated {count_updated} segments with label '{selected_label_name}'.")
            # Save functionality
            if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                save_annotations(
                    st.session_state.last_session_id,
                    st.session_state.current_annotations,
                    st.session_state.last_annotation_key,
                    silent=False
                )
                time.sleep(1) # Give time for user to see success
                st.rerun()
        elif count_matching_value > 0:
            st.warning(f"Found {count_matching_value} matching segments, but they already have the label '{selected_label_name}'.")
        else:
            st.info(f"No segments matched the value '{target_value_str}' for feature '{selected_feature}'.")


def _render_local_vlm_config():
    """Render Local VLM settings (mirrors detailed_agent_annotation_local).

    Returns a callable that, when invoked, builds an ``AnnotationPipelineConfig``
    from the current widget values — deferring the import so the run path
    surfaces a clean error if the LangGraph deps are missing.
    """
    from app.agents.backends.local_vlm import QWEN25_VL_MODELS

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        max_iterations = st.number_input(
            "Max iterations",
            min_value=1, max_value=10, value=3,
            key="batch_agent_local_max_iter",
        )
    with col_s2:
        temperature = st.slider(
            "Temperature",
            min_value=0.1, max_value=1.5, value=0.7, step=0.1,
            key="batch_agent_local_temp",
        )

    model_options = list(QWEN25_VL_MODELS.keys())
    default_idx = model_options.index("Qwen/Qwen2.5-VL-72B-Instruct")
    selected_model = st.selectbox(
        "VLM model",
        options=model_options,
        format_func=lambda x: QWEN25_VL_MODELS[x]["label"],
        index=default_idx,
        key="batch_agent_local_model",
    )
    model_spec = QWEN25_VL_MODELS[selected_model]
    model_max_context = model_spec["max_context"]
    model_max_new_tokens = model_spec["max_new_tokens"]

    if "batch_agent_local_ctx" not in st.session_state:
        st.session_state["batch_agent_local_ctx"] = min(32768, model_max_context)
    else:
        st.session_state["batch_agent_local_ctx"] = min(
            st.session_state["batch_agent_local_ctx"], model_max_context,
        )
    if "batch_agent_local_max_new_tokens" not in st.session_state:
        st.session_state["batch_agent_local_max_new_tokens"] = min(512, model_max_new_tokens)
    else:
        st.session_state["batch_agent_local_max_new_tokens"] = min(
            st.session_state["batch_agent_local_max_new_tokens"], model_max_new_tokens,
        )

    quantization_type = st.selectbox(
        "Quantization",
        options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        index=0,
        key="batch_agent_local_quant",
    )

    with st.expander("⚙️ Advanced model settings", expanded=False):
        gguf_path = st.text_input(
            "GGUF model path (override)", value="",
            help="Leave empty to auto-detect / auto-convert.",
            key="batch_agent_local_gguf",
        )
        mmproj_path = st.text_input(
            "mmproj path (override)", value="",
            help="Leave empty to auto-detect / auto-convert.",
            key="batch_agent_local_mmproj",
        )
        context_size = st.slider(
            "Context size",
            min_value=2048, max_value=model_max_context, step=1024,
            key="batch_agent_local_ctx",
        )
        n_gpu_layers = st.number_input(
            "GPU layers (-1 = all)",
            min_value=-1, max_value=200, value=-1,
            key="batch_agent_local_ngl",
        )
        max_new_tokens = st.slider(
            "Max new tokens (per VLM call)",
            min_value=128, max_value=model_max_new_tokens, step=128,
            key="batch_agent_local_max_new_tokens",
        )

    def build_config():
        from app.pipelines.annotation import AnnotationPipelineConfig
        return AnnotationPipelineConfig(
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
    return build_config


def _render_claude_config():
    """Render Claude settings (mirrors detailed_agent_annotation_claude)."""
    from app.agents.backends.claude_sdk import CLAUDE_VLM_MODELS

    max_iterations = st.number_input(
        "Tool-call budget (×10)",
        min_value=1, max_value=10, value=3,
        help="Caps the agent loop at this many tool calls × 10 per parent segment.",
        key="batch_agent_claude_max_iter",
    )
    claude_model = st.selectbox(
        "Claude model",
        options=list(CLAUDE_VLM_MODELS.keys()),
        format_func=lambda x: CLAUDE_VLM_MODELS[x]["label"],
        index=0,
        key="batch_agent_claude_model",
    )
    claude_use_thinking = st.checkbox(
        "Use extended thinking",
        value=False,
        key="batch_agent_claude_thinking",
    )
    st.caption(
        "ℹ️ Routed through `claude-agent-sdk` → your local `claude` CLI login. "
        "Subject to Max-plan rate limits — heavy batches may stall."
    )

    def build_config():
        from app.pipelines.annotation import AnnotationPipelineConfig
        return AnnotationPipelineConfig(
            max_iterations=int(max_iterations),
            backend="claude",
            claude_model=claude_model,
            claude_use_thinking=bool(claude_use_thinking),
        )
    return build_config


def _persist_children_for_parent(parent, result, session_id, selected_annotation_key, df):
    """Auto-save AI-discovered children under ``parent``.

    Replaces any prior children of this parent atomically: removes every
    annotation whose ``parent_id == parent.id`` from ``current_annotations``,
    then appends the new children. The caller's skip toggle decides whether
    we reach this function at all when the parent already has children —
    if we get here, the user has opted into overwriting.

    Returns ``(saved_count, replaced_count)``. If the pipeline produced no
    usable proposals we keep the old children intact rather than wiping
    them out for a no-op.
    """
    from .components._agent_annotation_shared import group_proposals_by_range

    grouped = group_proposals_by_range(result)

    new_children = []
    for (gs, ge), anns in grouped:
        if gs >= ge:
            continue
        label_ids = [a["label_id"] for a in anns if a.get("label_id") in LABEL_MAPPING]
        if not label_ids:
            continue
        notes = "; ".join(a.get("reasoning", "") for a in anns if a.get("reasoning"))[:500]
        new_children.append(build_segment(
            df,
            start=int(gs),
            end=int(ge),
            label_ids=list(dict.fromkeys(label_ids)),
            notes=notes,
            parent_id=parent.id,
        ))

    if not new_children:
        return 0, 0

    annotations = list(st.session_state.get("current_annotations", []))
    before = len(annotations)
    annotations = [
        a for a in annotations
        if getattr(a, "parent_id", None) != parent.id
    ]
    replaced = before - len(annotations)
    annotations.extend(new_children)
    st.session_state["current_annotations"] = annotations
    save_annotations(session_id, annotations, selected_annotation_key, silent=True)
    return len(new_children), replaced


def render_batch_auto_annotation(df, selected_annotation_key):
    """Batch sub-segment discovery powered by Local VLM or Claude.

    For each parent segment in the selected range, runs the same agent
    pipeline used by the Detailed view's 🖥️ / ☁️ expanders and auto-saves
    the discovered children under that parent. No staged review per parent.
    """
    st.header("Batch Auto-Annotation (Sub-Segment Discovery)")
    st.write(
        "For each segment in the selected range, run the **Sub-Segment Discovery** "
        "agent and auto-save discovered children. Choose Local VLM or Claude."
    )

    if not st.session_state.get("current_annotations"):
        st.info("No segments available to process. Please create segments first.")
        return

    annotations = st.session_state.current_annotations
    total_segments = len(annotations)
    if total_segments > 1:
        batch_range = st.slider("Select Segment Range", 0, total_segments - 1,
                                (0, total_segments - 1), step=1)
    else:
        batch_range = (0, 0)
        st.write("1 segment available.")
    process_indices = list(range(batch_range[0], batch_range[1] + 1))
    st.write(f"Selected {len(process_indices)} parent segment(s) for analysis.")

    # --- Backend selector ---
    backend_label = st.radio(
        "Discovery backend",
        options=["🖥️ Local VLM", "☁️ Claude"],
        index=0,
        horizontal=True,
        key="batch_agent_backend",
    )
    backend_kind = "local" if backend_label.startswith("🖥️") else "claude"

    st.markdown("---")
    if backend_kind == "local":
        build_config = _render_local_vlm_config()
    else:
        build_config = _render_claude_config()

    st.markdown("---")
    skip_with_children = st.checkbox(
        "Skip parents that already have child sub-segments",
        value=True,
        key="batch_agent_skip_with_children",
        help=(
            "Recommended — avoids re-running discovery on parents you've already annotated. "
            "When unchecked, existing children of each parent are DELETED and replaced with "
            "the new AI-discovered ones (clean re-run; old proposals are not used as hints)."
        ),
    )

    session_id = st.session_state.get("last_session_id")

    if "batch_agent_stop" not in st.session_state:
        st.session_state["batch_agent_stop"] = False
    if "batch_agent_logs" not in st.session_state:
        st.session_state["batch_agent_logs"] = []

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_clicked = st.button(
            "▶ Run Batch Sub-Segment Discovery",
            key="batch_agent_run", type="primary",
        )
    with col_clear:
        if st.button("Clear log", key="batch_agent_clear_log"):
            st.session_state["batch_agent_logs"] = []
            st.rerun()

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    log_area = st.empty()

    def _flush_log():
        if st.session_state["batch_agent_logs"]:
            log_area.code(
                "\n".join(st.session_state["batch_agent_logs"]),
                language="text", line_numbers=True,
            )

    _flush_log()

    if not run_clicked:
        return

    # Resolve pipeline entrypoint (one unified entry handles both backends).
    try:
        from app.pipelines.annotation import run_annotation
    except ImportError as e:
        st.error(
            f"Missing dependency: {e}\n\n"
            "Install with: `pip install langgraph langchain-core` "
            "(or `pip install claude-agent-sdk` for the Claude backend)."
        )
        return

    config = build_config()

    # Pre-compute existing-children lookup keyed by parent.id so we can
    # both decide whether to skip and pass dup-avoidance hints to the agent.
    children_by_parent: dict[str, list[dict]] = {}
    for ann in annotations:
        pid = getattr(ann, "parent_id", None)
        if not pid:
            continue
        children_by_parent.setdefault(pid, []).append({
            "start_index": ann.start_index,
            "end_index": ann.end_index,
            "labels": list(ann.labels),
        })

    main_label_set = set(LABEL_CATEGORIES.get("Main Labels", []))
    st.session_state["batch_agent_stop"] = False
    logs = st.session_state["batch_agent_logs"]
    total = len(process_indices)
    success_parents = 0
    total_children = 0
    error_parents = 0

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        logs.append(f"[{ts}] {msg}")
        if len(logs) > 1000:
            del logs[: len(logs) - 1000]
        _flush_log()

    log(f"Starting batch sub-segment discovery: {total} parent(s), backend={backend_kind}")

    for i, idx in enumerate(process_indices):
        if st.session_state["batch_agent_stop"]:
            log("Stopped by user.")
            break

        if idx < 0 or idx >= len(annotations):
            log(f"Skipping invalid index {idx}.")
            continue
        parent = annotations[idx]

        existing = children_by_parent.get(parent.id, [])
        if skip_with_children and existing:
            log(f"Parent #{idx}: skipped ({len(existing)} existing children).")
            progress_bar.progress((i + 1) / total)
            continue

        # In replace mode we want the agent to re-explore from scratch — don't
        # bias it with the prior boundaries we're about to delete.
        existing_for_agent = [] if (existing and not skip_with_children) else existing

        parent_main_labels = [l for l in parent.labels if l in main_label_set]
        p_start = int(parent.start_index) if parent.start_index is not None else 0
        p_end = int(parent.end_index) if parent.end_index is not None else len(df) - 1

        status_text.markdown(
            f"**Parent #{idx}** _({i + 1}/{total})_ — [{p_start}, {p_end}], "
            f"running {backend_kind} pipeline…"
        )
        log(f"Parent #{idx}: running [{p_start}, {p_end}] "
            f"main_labels={parent_main_labels or '∅'}")

        try:
            result = run_annotation(
                flow="detailed",
                df=df,
                start_index=p_start,
                end_index=p_end,
                session_id=session_id,
                parent_main_labels=parent_main_labels,
                existing_children=existing_for_agent,
                config=config,
            )
        except Exception as e:
            error_parents += 1
            log(f"Parent #{idx}: ERROR — {e}")
            log(traceback.format_exc().splitlines()[-1])
            progress_bar.progress((i + 1) / total)
            continue

        try:
            n_children, replaced = _persist_children_for_parent(
                parent, result, session_id, selected_annotation_key, df,
            )
        except Exception as e:
            error_parents += 1
            log(f"Parent #{idx}: persistence ERROR — {e}")
            progress_bar.progress((i + 1) / total)
            continue

        if n_children > 0:
            success_parents += 1
            total_children += n_children
            if replaced:
                log(f"Parent #{idx}: replaced {replaced} existing child(ren) with "
                    f"{n_children} new sub-segment(s).")
            else:
                log(f"Parent #{idx}: saved {n_children} child sub-segment(s).")
        else:
            log(f"Parent #{idx}: pipeline produced no usable proposals "
                f"(existing children left untouched).")

        progress_bar.progress((i + 1) / total)

    progress_bar.progress(1.0)
    status_text.markdown(
        f"**Done.** Parents updated: {success_parents}, "
        f"new children: {total_children}, errors: {error_parents}."
    )
    log(f"Finished. {success_parents}/{total} parents updated, "
        f"{total_children} children created, {error_parents} error(s).")

    # Discovery is per-parent and doesn't leave staged-review state for the
    # shared panel; clear any stale follow-up chat context from prior detailed
    # runs so the next page render doesn't dangle.
    st.session_state.pop("agent_annot_result", None)
    st.session_state.pop("agent_annot_followup_ctx", None)
    st.session_state.pop("agent_annot_followup_chat", None)


def render_batch_lap_agent_claude(df, session_id, selected_annotation_key):
    """Batch Claude Lap-to-Segment Excerpter.

    Picks a lap range, rough-splits it via ``split_lap_by_circuit_sections``,
    then runs the Claude `flow="lap"` pipeline on every section in order.
    Each result is auto-saved as a new ``AnnotatedSegment``. When the agent
    revises a boundary, the downstream tail is re-split from the new end
    so the next iteration sees the shifted partition.
    """
    from .components._lap_agent_shared import (
        track_name_to_circuit_id, run_split, rebuild_remaining_segments,
    )
    st.header("Batch Lap-to-Segment Excerpter (☁️ Claude)")
    st.write(
        "Pick a lap range; the deterministic splitter partitions it into "
        "per-`circuit_section` sub-ranges, then Claude annotates **every** "
        "section automatically and auto-saves each result as a new segment."
    )

    track_name = (
        df["Static_track"].iloc[0]
        if "Static_track" in df.columns and not df.empty else None
    )
    circuit_id = track_name_to_circuit_id(track_name)
    if not circuit_id:
        st.warning(
            "Cannot detect the circuit from `Static_track`. The lap "
            "excerpter needs a recognised circuit. Skipping."
        )
        return
    st.caption(f"Detected circuit: `{circuit_id}`")

    col1, col2 = st.columns(2)
    with col1:
        lap_start = st.number_input(
            "Lap start index", min_value=0, max_value=max(len(df) - 1, 0),
            value=0, key="batch_lap_claude_start",
        )
    with col2:
        lap_end = st.number_input(
            "Lap end index", min_value=1, max_value=len(df),
            value=min(len(df), 5000), key="batch_lap_claude_end",
        )

    if lap_end - lap_start < 3:
        st.warning(f"Lap range too short — pick at least 3 ilocs (currently {lap_end - lap_start}).")
        return

    coverage_slot = st.empty()
    _render_lap_coverage_bar(coverage_slot, int(lap_start), int(lap_end))

    st.markdown("---")
    max_iterations = st.number_input(
        "Tool-call budget (×10)", min_value=1, max_value=10, value=3,
        help="Caps the agent loop at this many tool calls × 10 per section.",
        key="batch_lap_claude_max_iter",
    )
    from app.agents.backends.claude_sdk import CLAUDE_VLM_MODELS
    claude_model = st.selectbox(
        "Claude model", options=list(CLAUDE_VLM_MODELS.keys()),
        format_func=lambda x: CLAUDE_VLM_MODELS[x]["label"],
        index=0, key="batch_lap_claude_model",
    )
    use_thinking = st.checkbox(
        "Use extended thinking", value=False, key="batch_lap_claude_thinking",
    )
    st.caption(
        "ℹ️ Routed through `claude-agent-sdk` → your local `claude` CLI login. "
        "Subject to Max-plan rate limits — heavy batches may stall."
    )

    st.markdown("---")
    skip_overlap = st.checkbox(
        "Skip sections that overlap existing annotations",
        value=True, key="batch_lap_claude_skip_overlap",
        help=(
            "Recommended — avoids re-annotating sections you've already "
            "labelled. When unchecked, the agent runs on every section "
            "and creates duplicate annotations on top of existing ones."
        ),
    )

    if "batch_lap_claude_logs" not in st.session_state:
        st.session_state["batch_lap_claude_logs"] = []

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_clicked = st.button(
            "▶ Run Batch Lap Excerpter",
            key="batch_lap_claude_run", type="primary",
        )
    with col_clear:
        if st.button("Clear log", key="batch_lap_claude_clear_log"):
            st.session_state["batch_lap_claude_logs"] = []
            st.rerun()

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    log_area = st.empty()
    logs = st.session_state["batch_lap_claude_logs"]

    def _flush_log():
        if logs:
            log_area.code("\n".join(logs), language="text", line_numbers=True)

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        logs.append(f"[{ts}] {msg}")
        if len(logs) > 1000:
            del logs[: len(logs) - 1000]
        _flush_log()

    _flush_log()

    if not run_clicked:
        return

    try:
        from app.pipelines.annotation import (
            AnnotationPipelineConfig, run_annotation,
        )
    except ImportError as e:
        st.error(
            f"Missing dependency: {e}\n\nInstall with `pip install claude-agent-sdk`."
        )
        return

    config = AnnotationPipelineConfig(
        backend="claude",
        max_iterations=int(max_iterations),
        claude_model=claude_model,
        claude_use_thinking=bool(use_thinking),
    )

    segments = run_split(df, int(lap_start), int(lap_end), circuit_id)
    if not segments:
        st.info(
            "The splitter produced zero sections — the circuit's "
            "`normalized_position_range` values may not cover the picked range."
        )
        return

    log(f"Starting batch lap excerpter: {len(segments)} section(s), "
        f"lap=[{int(lap_start)}, {int(lap_end)}], circuit={circuit_id}")

    saved_count = 0
    skipped_count = 0
    error_count = 0
    i = 0
    while i < len(segments):
        seg = segments[i]
        sec_id = seg["circuit_section_id"]
        sec_start = int(seg["start_index"])
        sec_end = int(seg["end_index"])

        if skip_overlap and _section_overlaps_existing(sec_start, sec_end):
            log(f"Section #{i} `{sec_id}` [{sec_start}, {sec_end}]: skipped (overlaps existing annotation).")
            skipped_count += 1
            i += 1
            progress_bar.progress((i) / len(segments))
            continue

        existing = _collect_existing_lap_annotations(int(lap_start), int(lap_end))

        status_text.markdown(
            f"**Section #{i + 1}/{len(segments)}** `{sec_id}` — [{sec_start}, {sec_end}], running Claude…"
        )
        log(f"Section #{i} `{sec_id}`: running [{sec_start}, {sec_end}]")

        try:
            result = run_annotation(
                flow="lap",
                df=df,
                config=config,
                session_id=session_id,
                lap_start=int(lap_start),
                lap_end=int(lap_end),
                section_id=sec_id,
                section_start=sec_start,
                section_end=sec_end,
                circuit_id=circuit_id,
                existing_section_annotations=existing,
            )
        except Exception as e:
            error_count += 1
            log(f"Section #{i} `{sec_id}`: ERROR — {e}")
            log(traceback.format_exc().splitlines()[-1])
            i += 1
            progress_bar.progress(i / len(segments))
            continue

        label_ids = [l for l in result.label_ids if l in LABEL_MAPPING]
        if not label_ids:
            log(f"Section #{i} `{sec_id}`: no valid labels resolved — skipped.")
            error_count += 1
            i += 1
            progress_bar.progress(i / len(segments))
            continue

        new_ann = build_segment(
            df,
            start=int(result.start_index),
            end=int(result.end_index),
            label_ids=label_ids,
            notes=(result.reasoning or "")[:1500],
        )
        annotations = list(st.session_state.get("current_annotations", []))
        annotations.append(new_ann)
        st.session_state["current_annotations"] = annotations
        save_annotations(session_id, annotations, selected_annotation_key, silent=True)
        saved_count += 1

        if result.revised:
            tail = rebuild_remaining_segments(
                df, int(lap_start), int(lap_end), circuit_id, int(result.end_index),
            )
            segments = segments[: i + 1] + tail
            segments[i] = {
                **segments[i],
                "start_index": int(result.start_index),
                "end_index": int(result.end_index),
                "circuit_section_id": result.section_id,
            }
            log(f"Section #{i} `{sec_id}`: saved (revised → "
                f"[{result.start_index}, {result.end_index}]); tail rebuilt → "
                f"{len(tail)} downstream section(s).")
        else:
            log(f"Section #{i} `{sec_id}`: saved [{result.start_index}, {result.end_index}] "
                f"with {len(label_ids)} label(s).")

        i += 1
        progress_bar.progress(i / len(segments))

    progress_bar.progress(1.0)
    status_text.markdown(
        f"**Done.** Saved: {saved_count}, skipped: {skipped_count}, errors: {error_count}."
    )
    log(f"Finished. {saved_count} saved, {skipped_count} skipped, {error_count} error(s).")

    _render_lap_coverage_bar(coverage_slot, int(lap_start), int(lap_end))


def _section_overlaps_existing(sec_start: int, sec_end: int) -> bool:
    """True if any current annotation overlaps the section range."""
    for ann in st.session_state.get("current_annotations", []) or []:
        s = int(getattr(ann, "start_index", 0) or 0)
        e = int(getattr(ann, "end_index", 0) or 0)
        if e > sec_start and s < sec_end:
            return True
    return False


def _compute_lap_coverage(lap_start: int, lap_end: int):
    """Merge saved annotations overlapping the lap range into (covered, gaps)."""
    raw = []
    for ann in st.session_state.get("current_annotations", []) or []:
        s = int(getattr(ann, "start_index", 0) or 0)
        e = int(getattr(ann, "end_index", 0) or 0)
        if e <= lap_start or s >= lap_end:
            continue
        raw.append((max(s, lap_start), min(e, lap_end)))
    raw.sort()
    covered: list[tuple[int, int]] = []
    for s, e in raw:
        if covered and s <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(covered[-1][1], e))
        else:
            covered.append((s, e))
    gaps: list[tuple[int, int]] = []
    pos = lap_start
    for s, e in covered:
        if s > pos:
            gaps.append((pos, s))
        pos = e
    if pos < lap_end:
        gaps.append((pos, lap_end))
    return covered, gaps


def _render_lap_coverage_bar(slot, lap_start: int, lap_end: int) -> None:
    """Horizontal coverage strip over the lap range — red = uncovered, green = annotated."""
    import plotly.graph_objects as go

    if lap_end <= lap_start:
        slot.empty()
        return

    covered, gaps = _compute_lap_coverage(lap_start, lap_end)
    total = lap_end - lap_start
    covered_len = sum(e - s for s, e in covered)
    gap_len = total - covered_len
    coverage_pct = (covered_len / total * 100) if total else 0.0
    longest_gap = max((e - s for s, e in gaps), default=0)

    fig = go.Figure()
    fig.add_shape(
        type="rect", x0=lap_start, x1=lap_end, y0=0, y1=1,
        fillcolor="rgba(220, 53, 69, 0.75)", line=dict(width=0), layer="below",
    )
    for s, e in covered:
        fig.add_shape(
            type="rect", x0=s, x1=e, y0=0, y1=1,
            fillcolor="rgba(40, 167, 69, 0.9)", line=dict(width=0), layer="below",
        )
    # Invisible hover trace per gap so the user can read exact gap ranges/lengths.
    if gaps:
        fig.add_trace(go.Scatter(
            x=[(s + e) / 2 for s, e in gaps],
            y=[0.5] * len(gaps),
            mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)"),
            hovertext=[f"gap [{s}, {e}] · {e - s} iloc(s)" for s, e in gaps],
            hoverinfo="text",
            showlegend=False,
        ))
    fig.update_layout(
        height=110,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(range=[lap_start, lap_end], title="iloc index", fixedrange=True),
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        showlegend=False,
    )

    with slot.container():
        st.caption(
            f"**Annotation coverage:** {coverage_pct:.1f}% covered · "
            f"{len(gaps)} gap(s) · {gap_len} iloc(s) uncovered · "
            f"longest gap {longest_gap} iloc(s)  "
            "(🟩 annotated · 🟥 not yet reached)"
        )
        st.plotly_chart(fig, use_container_width=True)


def _collect_existing_lap_annotations(lap_start: int, lap_end: int):
    """Annotations overlapping the lap range — passed as dup-avoidance hints."""
    out = []
    for ann in st.session_state.get("current_annotations", []) or []:
        s = int(getattr(ann, "start_index", 0) or 0)
        e = int(getattr(ann, "end_index", 0) or 0)
        if e <= lap_start or s >= lap_end:
            continue
        out.append({
            "start_index": s,
            "end_index": e,
            "labels": list(getattr(ann, "labels", [])),
        })
    return out


def render_bulk_label_utils(selected_annotation_key):
    """
    Renders bulk utilities like removing a specific label from all segments.
    """
    st.header("Bulk Label Management")
    st.write("Perform operations on all segments in the current session.")
    
    # Ensure current annotations exist
    if not st.session_state.get("current_annotations"):
        st.info("No segments available.")
        return

    # --- Remove Specific Label ---
    st.subheader("Remove Label from All Segments")
    st.caption("Select a label to remove from every segment where it exists.")
    
    label_names = sorted(list(LABEL_NAME_TO_ID.keys()))
    if not label_names:
        st.info("No labels configured in LABEL_NAME_TO_ID.")
        return

    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_label_name = st.selectbox("Select Label to Remove", label_names, key="bulk_remove_label_select")
    
    if st.button("Remove Label from All Segments", key="bulk_remove_label_btn", type="primary"):
        selected_label_id = LABEL_NAME_TO_ID[selected_label_name]
        count_removed = 0
        
        for ann in st.session_state.current_annotations:
            if selected_label_id in ann.labels:
                # Remove all instances of the label (just in case of duplicates)
                ann.labels = [lid for lid in ann.labels if lid != selected_label_id]
                count_removed += 1
        
        if count_removed > 0:
            st.success(f"Removed label '{selected_label_name}' from {count_removed} segments.")
            # Save functionality
            if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                save_annotations(
                    st.session_state.last_session_id,
                    st.session_state.current_annotations,
                    st.session_state.last_annotation_key,
                    silent=False
                )
                time.sleep(1) # Give time for user to see success
                st.rerun()
        else:
            st.info(f"Label '{selected_label_name}' was not found in any segment.")


def render_classifier_auto_annotation(df, selected_annotation_key):
    """
    Renders the Segment Classifier Auto-Annotation section.
    Allows user to scan a range of telemetry data using the trained LSTM model.
    """
    st.header("Classifier Auto-Annotation")
    st.write("Automatically identify segments and apply main labels using trained LSTM Classifier.")
    st.warning("⚠️ Warning: Any existing segments within the selected range will be removed and replaced by the newly identified segments.")

    try:
        from app.ml.segment_classifier.service import segment_classifier
    except ImportError:
        try:
            from app.ml.segment_classifier.service import segment_classifier
        except ImportError:
            segment_classifier = None

    if not segment_classifier:
        st.error("SegmentClassifierService could not be imported.")
        return

    total_rows = len(df)
    if total_rows > 1:
        range_slider = st.slider(
            "Select Data Range to Scan (Row Indices)", 
            0, total_rows-1, (0, total_rows-1), step=1, 
            key="classifier_range_slider"
        )
    else:
        st.write("Not enough data to scan.")
        return

    if st.button("Identify Segments with Classifier", type="primary", key="classifier_scan_btn"):
        with st.spinner("Loading model and scanning telemetry data..."):
            try:
                if not segment_classifier.load_model():
                    st.error("Could not load trained model. Ensure it is trained first.")
                    return
                
                scan_df = df.iloc[range_slider[0]:range_slider[1] + 1].copy()
                found_segments = segment_classifier.scan_telemetry_data(scan_df)
                
                main_labels_set = set(LABEL_CATEGORIES.get("Main Labels", []))
                
                new_annotations = []
                count_added = 0
                for seg in found_segments:
                    start_idx = seg.start_index + range_slider[0]
                    end_idx = seg.end_index + range_slider[0]
                    
                    filtered_labels = [lbl for lbl in seg.labels if str(lbl) in main_labels_set]
                    
                    if filtered_labels:
                        from app.domain.segment import AnnotatedSegment
                        # Classifier emits inclusive-end indices; the slice is
                        # df.iloc[start_idx : end_idx + 1] to match. Stored
                        # end_index stays inclusive to align with the rule-based
                        # reader, which already adds +1 when slicing by it.
                        seg_rows = df.iloc[start_idx:end_idx + 1].to_dict(orient="records")
                        new_ann = AnnotatedSegment(
                            labels=filtered_labels,
                            segment_length=end_idx - start_idx + 1,
                            start_index=start_idx,
                            end_index=end_idx,
                            notes="Auto-identified by Segment Classifier",
                            telemetry_data=seg_rows,
                        )
                        new_annotations.append(new_ann)
                        count_added += 1
                
                if count_added > 0:
                    if "current_annotations" not in st.session_state or st.session_state.current_annotations is None:
                        st.session_state.current_annotations = []
                    else:
                        filtered_annotations = []
                        removed_count = 0
                        for ann in st.session_state.current_annotations:
                            start = ann.start_index if ann.start_index is not None else 0
                            end = ann.end_index if ann.end_index is not None else len(df) - 1
                            
                            if start <= range_slider[1] and end >= range_slider[0]:
                                removed_count += 1
                            else:
                                filtered_annotations.append(ann)
                                
                        st.session_state.current_annotations = filtered_annotations
                        if removed_count > 0:
                            st.info(f"Removed {removed_count} existing segments in the selected range.")
                        
                    st.session_state.current_annotations.extend(new_annotations)
                    st.session_state.current_annotations.sort(key=lambda x: (x.start_index if x.start_index is not None else 0))
                    
                    st.success(f"Successfully identified and added {count_added} segments with main labels.")
                    
                    if "last_session_id" in st.session_state and "last_annotation_key" in st.session_state:
                         save_annotations(
                             st.session_state.last_session_id,
                             st.session_state.current_annotations,
                             st.session_state.last_annotation_key,
                             silent=False
                         )
                         time.sleep(1)
                         st.rerun()
                else:
                    st.info("No new segments with main labels were identified in the selected range.")
                    
            except Exception as e:
                st.error(f"Error classifying segments: {str(e)}")


def _load_batch_session(selected_annotation_key, selected_session_key, available_sessions):
    """Shared session selector + dataframe load for every batch page.

    Returns ``(df, session_id)`` or ``(None, None)`` if the page should
    short-circuit (no data / nothing selected).
    """
    annotated_sessions = set(get_available_sessions(selected_annotation_key))

    def format_session_option(s):
        status = "✅" if s in annotated_sessions else "⭕"
        return f"{status} {s}"

    current_session = st.session_state.get("detailed_session_selector")
    index = 0
    if current_session and current_session in available_sessions:
        index = available_sessions.index(current_session)

    col_sel1, _ = st.columns([1, 3])
    with col_sel1:
        session_id = st.selectbox(
            "Select Session for Batch Analysis",
            options=available_sessions,
            format_func=format_session_option,
            index=index,
            key="batch_session_selector",
        )

    with st.spinner(f"Loading session {session_id}..."):
        df = load_session_data(selected_session_key, session_id)
        if ("last_session_id" not in st.session_state or
            st.session_state.last_session_id != session_id or
            "last_annotation_key" not in st.session_state or
            st.session_state.last_annotation_key != selected_annotation_key):
            st.session_state.current_annotations = load_annotations(session_id, selected_annotation_key)
            st.session_state.last_session_id = session_id
            st.session_state.last_annotation_key = selected_annotation_key

    if df.empty:
        st.warning("Selected session has no data.")
        return None, None

    if "Static_track" in df.columns:
        track_name = df["Static_track"].iloc[0]
        st.markdown(f"**Track:** {track_name}")

    return df, session_id


def render_batch_bulk_label(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_bulk_label_utils(selected_annotation_key)


def render_batch_rule_based(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_rule_based_annotation(df, selected_annotation_key)


def render_batch_classifier(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_classifier_auto_annotation(df, selected_annotation_key)


def render_batch_subseg(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_batch_auto_annotation(df, selected_annotation_key)


def render_batch_lap(selected_annotation_key, selected_session_key, available_sessions):
    df, session_id = _load_batch_session(
        selected_annotation_key, selected_session_key, available_sessions,
    )
    if df is None:
        return
    render_batch_lap_agent_claude(df, session_id, selected_annotation_key)
