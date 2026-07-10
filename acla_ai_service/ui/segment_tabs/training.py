"""Training tab: kick off and monitor classifier / transformer / LLM training.

Each card runs the corresponding training pipeline entrypoint as a background
subprocess via :mod:`segment_tabs._training_runner`. Logs persist on disk and
survive browser refreshes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

from app.pipelines.training.config import TrainingPipelineConfig

from segment_tabs._training_runner import render_card, spawn
from segment_tabs.shared import get_available_sessions


_AI_SERVICE_DIR = Path(__file__).resolve().parents[2]
_TRAINING_ENTRYPOINTS = _AI_SERVICE_DIR / "app" / "pipelines" / "training" / "entrypoints"
TRAINING_ROUTES = frozenset({"classifier", "transformer", "opportunity_forecaster"})


def render_training(active_view: str, annotation_key: Optional[str]) -> None:
    routed_node = st.session_state.pop("pipeline_training_node", None)

    cfg = TrainingPipelineConfig()
    default_ann_key = annotation_key or cfg.annotation_cache_key

    if active_view == "classifier":
        st.header("🏋️ Segment classifier (LSTM)")
        if routed_node:
            st.info(f"Configuring training node `{routed_node}` from the active pipeline.")
        _show_input_location("Input annotation dataset", default_ann_key)
        render_card(
            "classifier",
            title="1️⃣ Segment classifier (LSTM)",
            description="Trains on the currently-selected annotation dataset.",
            render_start_form=lambda: _classifier_form(default_ann_key),
        )
        return

    if active_view == "transformer":
        st.header("🏋️ Transformer guidance")
        if routed_node:
            st.info(f"Configuring training node `{routed_node}` from the active pipeline.")
        _show_input_location("Input annotation dataset", default_ann_key)
        render_card(
            "transformer",
            title="2️⃣ Transformer guidance",
            description="Trains on EA/RM-labelled segments from the annotation dataset.",
            render_start_form=lambda: _transformer_form(default_ann_key),
        )
        return

    if active_view == "opportunity_forecaster":
        st.header("Opportunity forecaster")
        if routed_node:
            st.info(f"Configuring training node `{routed_node}` from the active pipeline.")
        _show_input_location("Input annotation dataset", default_ann_key)
        render_card(
            "opportunity_forecaster",
            title="3. Opportunity forecaster",
            description=(
                "Trains future successful overtake / defense probabilities "
                "from O/OD annotated segments."
            ),
            render_start_form=lambda: _opportunity_forecaster_form(default_ann_key),
        )
        return

    st.error(f"Unknown training view: `{active_view}`")


def _show_input_location(label: str, value: Optional[str]) -> None:
    if value:
        st.info(f"📂 {label}: `{value}`")
    else:
        st.warning(
            f"No {label.lower()} configured — set it from the Pipeline view "
            "by picking an annotation output on this training node."
        )


# ---------------------------------------------------------------------------
# Per-card start forms — each calls spawn(...) + st.rerun() on submit.
# ---------------------------------------------------------------------------

def _classifier_form(default_ann_key: str) -> None:
    ann_key = st.text_input("Annotation key", value=default_ann_key)
    ann_key = ann_key.strip()
    available_sessions = get_available_sessions(ann_key) if ann_key else []

    selection_key = "classifier_selected_sessions"
    selection_source_key = "classifier_selected_sessions_source"
    if st.session_state.get(selection_source_key) != ann_key:
        st.session_state[selection_source_key] = ann_key
        st.session_state[selection_key] = available_sessions
    elif selection_key not in st.session_state:
        st.session_state[selection_key] = available_sessions
    else:
        available_set = set(available_sessions)
        st.session_state[selection_key] = [
            session_id
            for session_id in st.session_state[selection_key]
            if session_id in available_set
        ]

    selected_sessions = st.multiselect(
        "Training sessions",
        options=available_sessions,
        key=selection_key,
        help="Selected sessions are pooled into one classifier training run.",
    )
    if not available_sessions:
        st.warning("The annotation dataset has no session chunks available for training.")
    elif not selected_sessions:
        st.warning("Select at least one session before starting training.")

    with st.form("classifier_form"):
        c1, c2, c3, c4 = st.columns(4)
        epochs = c1.number_input("Epochs", min_value=1, max_value=500, value=10)
        batch_size = c2.number_input("Batch size", min_value=1, max_value=2048, value=32)
        lr = c3.number_input(
            "Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f",
        )
        val_split = c4.slider("Val split", 0.0, 0.5, 0.1, 0.05)
        if st.form_submit_button(
            "🚀 Start",
            width="stretch",
            disabled=not available_sessions or not selected_sessions,
        ):
            cmd = [
                sys.executable, "-u", str(_TRAINING_ENTRYPOINTS / "train_segment_classifier.py"),
                "--epochs", str(int(epochs)),
                "--batch-size", str(int(batch_size)),
                "--lr", str(float(lr)),
                "--val-split", str(float(val_split)),
                "--annotation-key", ann_key,
            ]
            for session_id in selected_sessions:
                cmd.extend(["--session-id", session_id])
            spawn("classifier", cmd)
            st.rerun()


def _transformer_form(default_ann_key: str) -> None:
    with st.form("transformer_form"):
        ann_key = st.text_input("Annotation key", value=default_ann_key)
        if st.form_submit_button("🚀 Start", width="stretch"):
            cmd = [
                sys.executable, "-u", str(_TRAINING_ENTRYPOINTS / "train_transformer_guidance.py"),
                "--annotation-key", ann_key,
            ]
            spawn("transformer", cmd)
            st.rerun()


def _opportunity_forecaster_form(default_ann_key: str) -> None:
    with st.form("opportunity_forecaster_form"):
        ann_key = st.text_input("Annotation key", value=default_ann_key)
        c1, c2 = st.columns(2)
        input_fraction = c1.slider(
            "Input fraction of labeled segment",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help=(
                "Use the early part of each labeled O/OD segment as the "
                "recent telemetry window. Lower values make the model "
                "predict earlier."
            ),
        )
        max_negatives = c2.number_input(
            "Max negative examples",
            min_value=0,
            max_value=100000,
            value=5000,
            help="Caps NO_OPPORTUNITY examples so positives are not drowned out.",
        )
        if st.form_submit_button("Start", width="stretch"):
            cmd = [
                sys.executable, "-u", str(_TRAINING_ENTRYPOINTS / "train_opportunity_forecaster.py"),
                "--annotation-key", ann_key,
                "--input-fraction", str(float(input_fraction)),
                "--max-negatives", str(int(max_negatives)),
            ]
            spawn("opportunity_forecaster", cmd)
            st.rerun()
