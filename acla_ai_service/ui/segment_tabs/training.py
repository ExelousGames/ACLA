"""Training tab: kick off and monitor classifier / transformer / LLM training.

Each card runs the corresponding CLI script in `scripts/` as a background
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


_AI_SERVICE_DIR = Path(__file__).resolve().parents[2]
_SCRIPTS = _AI_SERVICE_DIR / "scripts"


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
            render_start_form=_classifier_form,
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

def _classifier_form() -> None:
    with st.form("classifier_form"):
        c1, c2, c3, c4 = st.columns(4)
        epochs = c1.number_input("Epochs", min_value=1, max_value=500, value=10)
        batch_size = c2.number_input("Batch size", min_value=1, max_value=2048, value=32)
        lr = c3.number_input(
            "Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f",
        )
        val_split = c4.slider("Val split", 0.0, 0.5, 0.1, 0.05)
        if st.form_submit_button("🚀 Start", use_container_width=True):
            cmd = [
                sys.executable, "-u", str(_SCRIPTS / "train_segment_classifier.py"),
                "--epochs", str(int(epochs)),
                "--batch-size", str(int(batch_size)),
                "--lr", str(float(lr)),
                "--val-split", str(float(val_split)),
            ]
            spawn("classifier", cmd)
            st.rerun()


def _transformer_form(default_ann_key: str) -> None:
    with st.form("transformer_form"):
        ann_key = st.text_input("Annotation key", value=default_ann_key)
        if st.form_submit_button("🚀 Start", use_container_width=True):
            cmd = [
                sys.executable, "-u", str(_SCRIPTS / "train_transformer_guidance.py"),
                "--annotation-key", ann_key,
            ]
            spawn("transformer", cmd)
            st.rerun()


