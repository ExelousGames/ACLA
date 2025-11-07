"""Streamlit UI for annotating telemetry prompt datasets.

This tool lets race engineers review telemetry windows, capture driver notes
alongside the raw telemetry, visualise samples, and attach coaching
explanations before fine-tuning the local LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

# Default directory where datasets are written by the ML service
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "models" / "llm_datasets"
ANNOTATION_SUFFIX = ".annotations.jsonl"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


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


def _compute_window_signature(window: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """Generate a deterministic signature for a window/config pair."""

    try:
        context = window.get("context") or []
        future = window.get("future") or []
        if not context and not future:
            return None

        features = config.get("telemetry_features") or []
        payload = {
            "context": context,
            "future": future,
            "features": list(features),
            "context_steps": config.get("context_steps"),
            "prediction_steps": config.get("prediction_steps"),
        }
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    except Exception:
        return None


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

            metadata = record.get("metadata", {})
            annotation = metadata.get("annotation", {})
            window = record.get("window", {})
            driver_note = annotation.get("driver_note", "")
            coaching_explanation = annotation.get("coaching_explanation", "")

            config = metadata.get("config", {})
            signature = (
                metadata.get("annotation", {}).get("window_signature")
                or _compute_window_signature(window, config)
            )

            entries.append(
                {
                    "window_id": metadata.get("window_id"),
                    "annotation_complete": bool(metadata.get("annotation_complete")),
                    "driver_note": driver_note,
                    "coaching_explanation": coaching_explanation,
                    "window": window,
                    "config": config,
                    "messages": record.get("messages", []),
                    "signature": signature,
                    "raw": record,
                }
            )

    return entries


@st.cache_data(show_spinner=False)
def load_annotation_store(path_str: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load existing annotations stored alongside the dataset."""

    annotation_path = Path(path_str)
    store_by_window: Dict[str, Dict[str, Any]] = {}
    store_by_signature: Dict[str, Dict[str, Any]] = {}

    if not annotation_path.exists():
        return store_by_window, store_by_signature

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
            window_signature = payload.get("window_signature")

            if window_id:
                store_by_window[window_id] = payload
            if window_signature:
                store_by_signature[window_signature] = payload

    return store_by_window, store_by_signature


def _build_user_prompt(window: Dict[str, Any], config: Dict[str, Any], driver_note: str) -> str:
    """Recreate the user prompt string with the supplied driver note."""

    context_steps = config.get("context_steps", len(window.get("context", [])))
    prediction_steps = config.get("prediction_steps", len(window.get("future", [])))
    context_payload = window.get("context", [])

    return (
        "Task: Forecast telemetry and continue the coaching explanation.\n"
        f"Driver note (partial): {driver_note}\n\n"
        f"Recent telemetry window (last {context_steps} steps):\n"
        f"{json.dumps(context_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Provide the next {prediction_steps} telemetry steps in JSON "
        "alongside a clear coaching explanation extending the driver note."
    )


def _save_annotation(
    dataset_path: Path,
    annotation_path: Path,
    window_id: str,
    driver_note: str,
    coaching_explanation: str,
) -> Tuple[int, int]:
    """Persist the updated annotation to both dataset and annotation log."""

    dataset_records: List[Dict[str, Any]] = []
    records_info: List[Dict[str, Any]] = []
    target_signature: Optional[str] = None
    timestamp_iso = datetime.utcnow().isoformat()

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
            window = record.get("window", {})
            config = metadata.get("config", {})
            signature = _compute_window_signature(window, config)

            if window_meta == window_id:
                annotation_block = metadata.setdefault("annotation", {})
                annotation_block["driver_note"] = driver_note
                annotation_block["coaching_explanation"] = coaching_explanation
                annotation_block["updated_at"] = timestamp_iso
                if signature:
                    annotation_block["window_signature"] = signature
                metadata["annotation_complete"] = bool(driver_note and coaching_explanation)

                user_prompt = _build_user_prompt(window, config, driver_note)
                assistant_payload = {
                    "future_telemetry": window.get("future", []),
                    "coaching_explanation": coaching_explanation,
                }

                messages = record.setdefault("messages", [])
                if len(messages) >= 2:
                    messages[1]["content"] = user_prompt
                if len(messages) >= 3:
                    messages[2]["content"] = json.dumps(assistant_payload, ensure_ascii=False)

                target_signature = signature
            else:
                annotation_block = metadata.setdefault("annotation", {})
                if signature and "window_signature" not in annotation_block:
                    annotation_block["window_signature"] = signature

            dataset_records.append(record)
            records_info.append(
                {
                    "record": record,
                    "metadata": metadata,
                    "window_id": window_meta,
                    "signature": signature,
                    "window": window,
                    "config": config,
                }
            )

    if target_signature:
        for info in records_info:
            if info["window_id"] == window_id:
                continue
            if not info["signature"] or info["signature"] != target_signature:
                continue

            metadata = info["metadata"]
            annotation_block = metadata.setdefault("annotation", {})

            note_missing = not annotation_block.get("driver_note") and driver_note
            explanation_missing = not annotation_block.get("coaching_explanation") and coaching_explanation

            if not (note_missing or explanation_missing):
                continue

            if note_missing:
                annotation_block["driver_note"] = driver_note
            if explanation_missing:
                annotation_block["coaching_explanation"] = coaching_explanation
            annotation_block["updated_at"] = timestamp_iso
            if info["signature"]:
                annotation_block["window_signature"] = info["signature"]

            metadata["annotation_complete"] = bool(driver_note and coaching_explanation)

            user_prompt = _build_user_prompt(info["window"], info["config"], driver_note)
            assistant_payload = {
                "future_telemetry": info["window"].get("future", []),
                "coaching_explanation": coaching_explanation,
            }

            messages = info["record"].setdefault("messages", [])
            if len(messages) >= 2:
                messages[1]["content"] = user_prompt
            if len(messages) >= 3:
                messages[2]["content"] = json.dumps(assistant_payload, ensure_ascii=False)

    with dataset_path.open("w", encoding="utf-8") as jsonl_file:
        for record in dataset_records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Update annotation store
    store_by_window, _ = load_annotation_store(annotation_path.as_posix())
    store_by_window[window_id] = {
        "window_id": window_id,
        "driver_note": driver_note,
        "coaching_explanation": coaching_explanation,
        "updated_at": timestamp_iso,
        "window_signature": target_signature,
    }

    if target_signature:
        for info in records_info:
            if info["window_id"] == window_id:
                continue
            if info["signature"] != target_signature:
                continue
            metadata = info["metadata"]
            annotation_block = metadata.get("annotation", {})
            if not annotation_block.get("driver_note") or not annotation_block.get("coaching_explanation"):
                continue

            store_by_window[info["window_id"]] = {
                "window_id": info["window_id"],
                "driver_note": annotation_block.get("driver_note", ""),
                "coaching_explanation": annotation_block.get("coaching_explanation", ""),
                "updated_at": annotation_block.get("updated_at", timestamp_iso),
                "window_signature": target_signature,
            }

    with annotation_path.open("w", encoding="utf-8") as jsonl_file:
        for payload in store_by_window.values():
            jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    total_examples = len(records_info)
    annotated_examples = sum(1 for info in records_info if info["metadata"].get("annotation_complete"))

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

    store_by_window, store_by_signature = load_annotation_store(annotation_path.as_posix())

    total_examples = len(entries)
    annotated_examples = sum(1 for entry in entries if entry["annotation_complete"])

    st.sidebar.metric("Annotated", f"{annotated_examples}/{total_examples}",
                      "{:.0%}".format(annotated_examples / total_examples) if total_examples else "0%")
    show_pending_only = st.sidebar.checkbox("Show only pending annotations", value=False)

    entries_to_view = entries
    if show_pending_only:
        entries_to_view = [entry for entry in entries if not entry["annotation_complete"]]
        if not entries_to_view:
            st.success("All windows are annotated! Showing complete list instead.")
            entries_to_view = entries

    summary_df = pd.DataFrame(
        [
            {
                "Window ID": entry["window_id"],
                "Annotated": "Yes" if entry["annotation_complete"] else "No",
                "Existing Driver Note": (entry["driver_note"][:50] + "…") if entry["driver_note"] else "",
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

    signature = selected_entry.get("signature")
    existing_annotation = store_by_window.get(selected_window_id, {})
    if not existing_annotation and signature:
        existing_annotation = store_by_signature.get(signature, {})
    driver_note_default = existing_annotation.get("driver_note") or selected_entry.get("driver_note", "")
    coaching_note_default = existing_annotation.get("coaching_explanation") or selected_entry.get("coaching_explanation", "")

    st.subheader("Annotation")
    driver_note = st.text_area("Driver note", value=driver_note_default, height=120)
    coaching_explanation = st.text_area("Coaching explanation", value=coaching_note_default, height=160)

    save_col, stats_col = st.columns([1, 1])

    with save_col:
        if st.button("Save annotation", type="primary"):
            total, annotated = _save_annotation(
                dataset_path=dataset_path,
                annotation_path=annotation_path,
                window_id=selected_window_id,
                driver_note=driver_note.strip(),
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

    st.caption(
        "Annotations are written back to the dataset file and a companion annotations log."
        " Training can proceed once the necessary windows have notes and explanations."
    )


if __name__ == "__main__":
    main()
