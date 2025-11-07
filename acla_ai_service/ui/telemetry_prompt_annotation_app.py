"""Streamlit UI for annotating telemetry prompt datasets.

This tool lets race engineers review telemetry windows, visualise samples,
and attach concise coaching explanations before fine-tuning the local LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService
from app.services.local_llm_service import GenerationRequest

# Default directory where datasets are written by the ML service
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "models" / "llm_datasets"
ANNOTATION_SUFFIX = ".annotations.jsonl"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _run_async(func, *args, **kwargs):
    """Execute an async function from a synchronous context."""

    try:
        return asyncio.run(func(*args, **kwargs))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@st.cache_resource(show_spinner=False)
def get_ml_service() -> Full_dataset_TelemetryMLService:
    """Return a cached instance of the telemetry ML service."""

    return Full_dataset_TelemetryMLService()


@st.cache_resource(show_spinner=False)
def load_guidance_model() -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """Load the active LLM guidance model, if available."""

    service = get_ml_service()
    try:
        model, metadata = _run_async(service.get_llm_guidance_model)
        if model is None:
            error_msg = metadata.get("error") if isinstance(metadata, dict) else "No active model found"
            return None, metadata, error_msg
        return model, metadata, None
    except Exception as error:  # pragma: no cover - guarded for runtime issues
        return None, None, str(error)


def _extract_system_prompt(entry: Dict[str, Any]) -> str:
    """Return the system prompt stored with a dataset entry."""

    raw_record = entry.get("raw", {})
    messages = raw_record.get("messages", []) if isinstance(raw_record, dict) else []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _parse_cli_args() -> argparse.Namespace:
    """Parse optional CLI arguments passed via `streamlit run ... -- ...`."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--server-address", type=str, default="")
    parser.add_argument("--server-port", type=int, default=0)
    parser.add_argument("--browser-address", type=str, default="")
    parser.add_argument("--browser-port", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


def _resolve_network_config(args: argparse.Namespace) -> Tuple[str, int, Optional[str], Optional[int], str]:
    """Resolve the host/port pairing used for Streamlit when running in Docker."""

    env = os.getenv

    server_address = args.server_address or env("ANNOTATION_UI_SERVER_ADDRESS") or "0.0.0.0"
    server_port = args.server_port or int(env("ANNOTATION_UI_SERVER_PORT", "8501"))

    browser_address = args.browser_address or env("ANNOTATION_UI_BROWSER_ADDRESS") or ""
    browser_port_raw: Optional[str] = ""
    if args.browser_port:
        browser_port_raw = str(args.browser_port)
    elif env("ANNOTATION_UI_BROWSER_PORT"):
        browser_port_raw = env("ANNOTATION_UI_BROWSER_PORT")

    browser_port = int(browser_port_raw) if browser_port_raw else None

    if not browser_address and Path("/.dockerenv").exists():
        browser_address = env("ANNOTATION_UI_HOST", "localhost")

    effective_host = browser_address or server_address
    effective_port = browser_port or server_port
    access_url = f"http://{effective_host}:{effective_port}"

    return server_address, server_port, browser_address or None, browser_port, access_url


# ---------------------------------------------------------------------------
# Dataset loading / persistence helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> List[Dict[str, Any]]:
    """Load dataset records from disk into memory for quick annotation."""

    dataset_path = Path(path_str)
    entries: List[Dict[str, Any]] = []

    if not dataset_path.exists():
        return entries

    with dataset_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            metadata = record.get("metadata", {}) or {}
            annotation = metadata.get("annotation", {}) or {}
            window = record.get("window", {}) or {}
            coaching_explanation = annotation.get("coaching_explanation", "")

            entries.append(
                {
                    "window_id": metadata.get("window_id"),
                    "annotation_complete": bool(metadata.get("annotation_complete")),
                    "coaching_explanation": coaching_explanation,
                    "window": window,
                    "config": metadata.get("config", {}),
                    "metadata": metadata,
                    "messages": record.get("messages", []),
                    "raw": record,
                }
            )

    return entries


@st.cache_data(show_spinner=False)
def load_annotation_store(path_str: str) -> Dict[str, Dict[str, Any]]:
    """Load existing annotations stored alongside the dataset."""

    annotation_path = Path(path_str)
    store: Dict[str, Dict[str, Any]] = {}

    if not annotation_path.exists():
        return store

    with annotation_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            window_id = payload.get("window_id")
            if window_id:
                store[window_id] = {
                    "window_id": window_id,
                    "coaching_explanation": payload.get("coaching_explanation", ""),
                    "updated_at": payload.get("updated_at"),
                }

    return store


def _build_user_prompt(
    window: Dict[str, Any],
    config: Dict[str, Any],
    segment_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Compose the LLM user prompt for segment-purpose explanations."""

    context_payload = window.get("context", []) or []
    future_payload = window.get("future", []) or []
    context_steps = config.get("context_steps", len(context_payload))
    prediction_steps = config.get("prediction_steps", len(future_payload))

    metadata_payload: Dict[str, Any] = {
        "context_steps": context_steps,
        "future_steps": prediction_steps,
    }
    if segment_metadata:
        metadata_payload.update(segment_metadata)

    return (
        "Task: Provide a concise coaching explanation that captures the purpose of this telemetry segment.\n"
        f"Segment metadata:\n{json.dumps(metadata_payload, indent=2, ensure_ascii=False)}\n\n"
        "Telemetry context (ordered timesteps):\n"
        f"{json.dumps(context_payload, indent=2, ensure_ascii=False)}\n\n"
        "Continuation of the segment (if available):\n"
        f"{json.dumps(future_payload, indent=2, ensure_ascii=False)}\n\n"
        "Respond with a JSON object containing:\n"
        "- `coaching_summary`: 2-3 sentences describing the segment's focus or coaching insight.\n"
        "- Optional `key_focus`: list of short bullet strings highlighting the key observations."
    )


def _save_annotation(
    dataset_path: Path,
    annotation_path: Path,
    window_id: str,
    coaching_explanation: str,
) -> Tuple[int, int]:
    """Persist the updated annotation to both dataset and annotation log."""

    dataset_records: List[Dict[str, Any]] = []
    total_examples = 0
    annotated_examples = 0

    with dataset_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            metadata = record.setdefault("metadata", {})
            window_meta = metadata.get("window_id")
            window = record.get("window", {}) or {}
            config = metadata.get("config", {}) or {}
            segment_meta = metadata.get("segment_metadata")

            if window_meta == window_id:
                annotation_block = metadata.setdefault("annotation", {})
                annotation_block.pop("driver_note", None)
                annotation_block["coaching_explanation"] = coaching_explanation
                annotation_block["updated_at"] = datetime.utcnow().isoformat()
                metadata["annotation_complete"] = bool(coaching_explanation)

                user_prompt = _build_user_prompt(window, config, segment_meta)
                assistant_payload = {
                    "coaching_summary": coaching_explanation,
                }

                messages = record.setdefault("messages", [])
                if len(messages) >= 2:
                    messages[1]["content"] = user_prompt
                if len(messages) >= 3:
                    messages[2]["content"] = json.dumps(assistant_payload, ensure_ascii=False)

            if metadata.get("annotation_complete"):
                annotated_examples += 1
            total_examples += 1
            dataset_records.append(record)

    with dataset_path.open("w", encoding="utf-8") as jsonl_file:
        for record in dataset_records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Update annotation store
    store = load_annotation_store(annotation_path.as_posix())
    store[window_id] = {
        "window_id": window_id,
        "coaching_explanation": coaching_explanation,
        "updated_at": datetime.utcnow().isoformat(),
    }
    with annotation_path.open("w", encoding="utf-8") as jsonl_file:
        for payload in store.values():
            jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return total_examples, annotated_examples


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _prepare_plot_data(window: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    """Flatten context and future windows for plotting."""

    context_records = window.get("context", [])
    future_records = window.get("future", [])

    if not context_records and not future_records:
        return pd.DataFrame()

    context_df = pd.DataFrame(context_records)
    context_df["phase"] = "context"
    future_df = pd.DataFrame(future_records)
    future_df["phase"] = "future"

    if "relative_index" in context_df.columns:
        start_offset = context_df["relative_index"].max() + 1
        future_df["relative_index"] = future_df.get("relative_index", pd.Series(range(len(future_df)))) + start_offset

    combined = pd.concat([context_df, future_df], ignore_index=True, sort=False)
    melted = combined.melt(
        id_vars=[col for col in ["relative_index", "phase"] if col in combined.columns],
        value_vars=[feature for feature in feature_names if feature in combined.columns],
        var_name="feature",
        value_name="value",
    )
    return melted


def _render_plot(data: pd.DataFrame) -> None:
    """Render telemetry lines using Plotly."""

    if data.empty:
        st.info("No telemetry data available for plotting.")
        return

    fig = px.line(
        data,
        x="relative_index" if "relative_index" in data.columns else data.index,
        y="value",
        color="feature",
        line_dash="phase" if "phase" in data.columns else None,
        markers=True,
    )
    fig.update_layout(height=360, legend_orientation="h", legend_y=-0.2)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def _get_annotation_sample(
    dataset_identifier: str,
    window_ids: List[str],
    sample_size: int,
    *,
    force_refresh: bool = False,
) -> List[str]:
    """Return a stable random sample of window IDs for annotation."""

    valid_ids = [window_id for window_id in window_ids if window_id]
    if not valid_ids:
        return []

    session_key = f"annotation_sample::{dataset_identifier}"
    size_key = f"{session_key}::size"

    if force_refresh:
        st.session_state.pop(session_key, None)
        st.session_state.pop(size_key, None)

    stored_sample: Optional[List[str]] = st.session_state.get(session_key)
    stored_size = st.session_state.get(size_key)

    if stored_sample:
        stored_sample = [window_id for window_id in stored_sample if window_id in valid_ids]

    if not stored_sample or stored_size != sample_size:
        effective_size = max(1, min(sample_size, len(valid_ids)))
        if effective_size >= len(valid_ids):
            sampled_ids = list(valid_ids)
        else:
            sampled_ids = random.sample(valid_ids, effective_size)

        st.session_state[session_key] = sampled_ids
        st.session_state[size_key] = sample_size
        return sampled_ids

    return stored_sample


def main() -> None:
    args = _parse_cli_args()
    server_address, server_port, browser_address, browser_port, access_url = _resolve_network_config(args)

    if os.environ.get("ANNOTATION_UI_URL_PRINTED") != access_url:
        print(f"Telemetry annotation UI will be reachable at {access_url}", flush=True)
        os.environ["ANNOTATION_UI_URL_PRINTED"] = access_url

    st.set_page_config(page_title="Telemetry Prompt Annotation", layout="wide")
    st.title("Telemetry Prompt Annotation")
    st.sidebar.info(f"Open {access_url} from the host browser")
    st.sidebar.caption("Override via --server-port/--server-address or ANNOTATION_UI_* environment variables.")

    ml_service = get_ml_service()
    llm_model, llm_metadata, llm_error = load_guidance_model()
    llm_available = llm_model is not None

    if llm_available:
        st.sidebar.success("LLM guidance model loaded")
        if llm_metadata and isinstance(llm_metadata, dict):
            adapter_name = llm_metadata.get("backend_metadata", {}).get("adapter_directory")
            if adapter_name:
                st.sidebar.caption(f"Adapter: {adapter_name}")
    else:
        st.sidebar.info("LLM auto-coaching disabled. Train or import a guidance model to enable suggestions.")
        if llm_error:
            st.sidebar.caption(f"LLM load issue: {llm_error}")

    if llm_error:
        st.warning(f"Auto-coaching is currently unavailable: {llm_error}")

    dataset_dir = Path(args.dataset_dir or DEFAULT_DATASET_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = sorted(dataset_dir.glob("*.jsonl"))
    if not dataset_files:
        st.warning(f"No dataset files found in {dataset_dir}. Generate a dataset first.")
        return

    dataset_labels = [file.name for file in dataset_files]

    default_index = 0
    if args.dataset:
        try:
            default_index = dataset_labels.index(Path(args.dataset).name)
        except ValueError:
            default_index = 0

    selected_label = st.sidebar.selectbox("Dataset file", dataset_labels, index=default_index)
    dataset_path = dataset_dir / selected_label
    annotation_path = dataset_path.with_suffix(dataset_path.suffix + ANNOTATION_SUFFIX)

    entries = load_dataset(dataset_path.as_posix())
    if not entries:
        st.warning("Dataset is empty or could not be parsed.")
        return

    store = load_annotation_store(annotation_path.as_posix())

    total_examples = len(entries)
    annotated_examples = sum(1 for entry in entries if entry["annotation_complete"])

    active_dataset_key = "_active_annotation_dataset"
    dataset_identifier = dataset_path.as_posix()
    if st.session_state.get(active_dataset_key) != dataset_identifier:
        st.session_state[active_dataset_key] = dataset_identifier
        st.session_state.pop(f"annotation_sample::{dataset_identifier}", None)
        st.session_state.pop(f"annotation_sample::{dataset_identifier}::size", None)

    default_sample_size = min(50, total_examples) if total_examples else 1
    sample_size = st.sidebar.number_input(
        "Sample size for annotation",
        min_value=1,
        max_value=max(1, total_examples),
        value=default_sample_size,
        step=1,
    )
    reshuffle_requested = st.sidebar.button("Reshuffle sample", use_container_width=True)

    sampled_window_ids = _get_annotation_sample(
        dataset_identifier,
        [entry["window_id"] for entry in entries],
        int(sample_size),
        force_refresh=reshuffle_requested,
    )

    st.sidebar.caption(
        "Only sampled windows appear below; remaining windows stay reserved for training datasets."
    )

    st.sidebar.metric("Annotated", f"{annotated_examples}/{total_examples}",
                      "{:.0%}".format(annotated_examples / total_examples) if total_examples else "0%")
    show_pending_only = st.sidebar.checkbox("Show only pending annotations", value=False)

    entries_to_view = [entry for entry in entries if entry["window_id"] in sampled_window_ids]
    if not entries_to_view:
        entries_to_view = entries
    if show_pending_only:
        pending_entries = [entry for entry in entries_to_view if not entry["annotation_complete"]]
        if pending_entries:
            entries_to_view = pending_entries
        else:
            st.success("All sampled windows are annotated! Reshuffle the sample to review different windows.")

    summary_df = pd.DataFrame(
        [
            {
                "Window ID": entry["window_id"],
                "Annotated": "Yes" if entry["annotation_complete"] else "No",
                "Existing Explanation": (entry["coaching_explanation"][:50] + "…") if entry["coaching_explanation"] else "",
            }
            for entry in entries_to_view
        ]
    )
    st.dataframe(summary_df, use_container_width=True, height=260)

    window_ids = [entry["window_id"] for entry in entries_to_view]
    if not window_ids:
        st.error("No windows available for annotation.")
        return

    selected_window_id = st.selectbox("Select window", window_ids)
    selected_entry = next((entry for entry in entries if entry["window_id"] == selected_window_id), None)
    if not selected_entry:
        st.error("Unable to locate the selected window in the dataset.")
        return

    config = selected_entry.get("config", {})
    feature_options = config.get("telemetry_features", [])
    selected_features = st.multiselect(
        "Telemetry features to display",
        feature_options,
        default=feature_options[: min(5, len(feature_options))],
    )

    if selected_features:
        plot_data = _prepare_plot_data(selected_entry.get("window", {}), selected_features)
        _render_plot(plot_data)
    else:
        st.info("Select at least one telemetry feature to visualise.")

    existing_annotation = store.get(selected_window_id, {})
    coaching_note_default = existing_annotation.get("coaching_explanation") or selected_entry.get("coaching_explanation", "")

    st.subheader("Annotation")

    active_window_key = "_active_annotation_window"
    coaching_key = f"coaching_note_{selected_window_id}"
    key_focus_key = f"{coaching_key}_key_focus"
    raw_output_key = f"{coaching_key}_raw_output"

    if st.session_state.get(active_window_key) != selected_window_id:
        st.session_state[active_window_key] = selected_window_id
        st.session_state[coaching_key] = coaching_note_default
        st.session_state.pop(key_focus_key, None)
        st.session_state.pop(raw_output_key, None)
    else:
        st.session_state.setdefault(coaching_key, coaching_note_default)

    coaching_explanation = st.text_area("Coaching explanation", key=coaching_key, height=160)

    actions_col1, actions_col2, stats_col = st.columns([1, 1, 1])

    if llm_error and st.session_state.get(raw_output_key):
        st.session_state.pop(raw_output_key, None)
        st.session_state.pop(key_focus_key, None)

    with actions_col1:
        generate_disabled = bool(llm_error) or not llm_available
        generate_help = None if llm_available and not llm_error else (llm_error or "Train or import a guidance model to enable auto-suggestions.")
        if st.button(
            "Generate explanation",
            type="secondary",
            disabled=generate_disabled,
            help=generate_help,
            use_container_width=True,
        ):
            if llm_error:
                st.error(f"Cannot generate coaching explanation right now: {llm_error}")
            elif not llm_available:
                st.warning("No guidance model available for inference.")
            else:
                with st.spinner("Generating coaching suggestion..."):
                    system_prompt = _extract_system_prompt(selected_entry)
                    metadata = selected_entry.get("metadata", {}) or {}
                    user_prompt = _build_user_prompt(
                        selected_entry.get("window", {}),
                        config,
                        metadata.get("segment_metadata"),
                    )
                    request = GenerationRequest(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_new_tokens=ml_service.llm_config.generation_max_new_tokens,
                        temperature=ml_service.llm_config.generation_temperature,
                        top_p=ml_service.llm_config.generation_top_p,
                        do_sample=ml_service.llm_config.generation_do_sample,
                    )

                    try:
                        raw_output = llm_model.generate(request)
                        commentary = ml_service._parse_llm_output(raw_output)
                    except Exception as inference_error:  # pragma: no cover - inference safeguards
                        st.error(f"LLM generation failed: {inference_error}")
                    else:
                        summary_text = commentary.get("coaching_summary") or commentary.get("summary") or raw_output
                        st.session_state[coaching_key] = summary_text.strip()
                        key_focus = commentary.get("key_focus")
                        if isinstance(key_focus, list) and key_focus:
                            st.session_state[key_focus_key] = key_focus
                        else:
                            st.session_state.pop(key_focus_key, None)
                        st.session_state[raw_output_key] = raw_output
                        st.success("Coaching explanation generated. Review and edit before saving.")

    with actions_col2:
        if st.button("Save annotation", type="primary", use_container_width=True):
            total, annotated = _save_annotation(
                dataset_path=dataset_path,
                annotation_path=annotation_path,
                window_id=selected_window_id,
                coaching_explanation=coaching_explanation.strip(),
            )
            load_dataset.clear()
            load_annotation_store.clear()
            st.success(f"Annotation saved for {selected_window_id}")
            st.experimental_rerun()

    with stats_col:
        st.metric(
            "Annotation progress",
            f"{annotated_examples}/{total_examples}",
            "{:.0%}".format(annotated_examples / total_examples) if total_examples else "0%",
        )

    suggested_focus = st.session_state.get(key_focus_key)
    if suggested_focus:
        st.markdown("**Suggested focus areas**")
        for item in suggested_focus:
            st.markdown(f"- {item}")

    raw_output = st.session_state.get(raw_output_key)
    if raw_output:
        with st.expander("Raw LLM output", expanded=False):
            st.code(raw_output, language="json")

    st.caption(
        "Annotations are written back to the dataset file and a companion annotations log."
        " Training can proceed once the necessary windows have notes and explanations."
    )


if __name__ == "__main__":
    main()
