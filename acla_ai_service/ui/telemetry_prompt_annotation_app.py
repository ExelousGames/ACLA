"""Streamlit UI for annotating telemetry prompt datasets.

This tool lets race engineers review telemetry windows, add human coaching notes,
visualise telemetry samples, and record assistant explanations before
fine-tuning the local LLM.
"""

from __future__ import annotations

import argparse
import json
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
    args, _ = parser.parse_known_args()
    return args


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

            metadata = record.get("metadata", {})
            annotation = metadata.get("annotation", {})
            window = record.get("window", {})
            entries.append(
                {
                    "window_id": metadata.get("window_id"),
                    "annotation_complete": bool(metadata.get("annotation_complete")),
                    "human_note": annotation.get("human_note", ""),
                    "assistant_explanation": annotation.get("assistant_explanation", ""),
                    "window": window,
                    "config": metadata.get("config", {}),
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
                store[window_id] = payload

    return store


def _build_user_prompt(window: Dict[str, Any], config: Dict[str, Any], human_note: str) -> str:
    """Recreate the user prompt string with the supplied human note."""

    context_steps = config.get("context_steps", len(window.get("context", [])))
    prediction_steps = config.get("prediction_steps", len(window.get("future", [])))
    context_payload = window.get("context", [])

    return (
        "Task: Forecast telemetry and continue the coaching note.\n"
        f"Human note (partial): {human_note}\n\n"
        f"Recent telemetry window (last {context_steps} steps):\n"
        f"{json.dumps(context_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Provide the next {prediction_steps} telemetry steps in JSON "
        "alongside a clear explanation extending the human note."
    )


def _save_annotation(
    dataset_path: Path,
    annotation_path: Path,
    window_id: str,
    human_note: str,
    assistant_explanation: str,
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
            window = record.get("window", {})
            config = metadata.get("config", {})

            if window_meta == window_id:
                annotation_block = metadata.setdefault("annotation", {})
                annotation_block["human_note"] = human_note
                annotation_block["assistant_explanation"] = assistant_explanation
                annotation_block["updated_at"] = datetime.utcnow().isoformat()
                metadata["annotation_complete"] = bool(human_note and assistant_explanation)

                user_prompt = _build_user_prompt(window, config, human_note)
                assistant_payload = {
                    "future_telemetry": window.get("future", []),
                    "explanation": assistant_explanation,
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
        "human_note": human_note,
        "assistant_explanation": assistant_explanation,
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


def main() -> None:
    st.set_page_config(page_title="Telemetry Prompt Annotation", layout="wide")
    st.title("Telemetry Prompt Annotation")

    args = _parse_cli_args()
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
                "Existing Note": (entry["human_note"][:50] + "…") if entry["human_note"] else "",
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
    human_note_default = existing_annotation.get("human_note") or selected_entry.get("human_note", "")
    assistant_note_default = existing_annotation.get("assistant_explanation") or selected_entry.get("assistant_explanation", "")

    st.subheader("Annotation")
    human_note = st.text_area("Human coaching note", value=human_note_default, height=120)
    assistant_explanation = st.text_area("Assistant explanation", value=assistant_note_default, height=160)

    save_col, stats_col = st.columns([1, 1])

    with save_col:
        if st.button("Save annotation", type="primary"):
            total, annotated = _save_annotation(
                dataset_path=dataset_path,
                annotation_path=annotation_path,
                window_id=selected_window_id,
                human_note=human_note.strip(),
                assistant_explanation=assistant_explanation.strip(),
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
