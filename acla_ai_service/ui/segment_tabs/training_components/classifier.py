"""Segment-classifier training component."""

from __future__ import annotations

import sys

import streamlit as st

from .._training_runner import render_card, spawn
from ..shared import get_available_sessions
from ._shared import (
    TRAINING_ENTRYPOINTS,
    resolve_input_key,
    show_input_location,
    show_node_context,
)
from .contract import TrainingComponent, TrainingComponentContext


class ClassifierTrainingComponent(TrainingComponent):
    def open(self, context: TrainingComponentContext) -> None:
        default_ann_key = resolve_input_key(context)

        st.header("🏋️ Segment classifier (LSTM)")
        show_node_context(context)
        show_input_location("Input annotation dataset", default_ann_key)
        render_card(
            "classifier",
            title="1️⃣ Segment classifier (LSTM)",
            description="Trains on the currently-selected annotation dataset.",
            render_start_form=lambda: self._render_form(default_ann_key),
        )

    @staticmethod
    def _render_form(ann_key: str) -> None:
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
            st.warning(
                "The annotation dataset has no session chunks available for training."
            )
        elif not selected_sessions:
            st.warning("Select at least one session before starting training.")

        with st.form("classifier_form"):
            c1, c2, c3, c4 = st.columns(4)
            epochs = c1.number_input("Epochs", min_value=1, max_value=500, value=10)
            batch_size = c2.number_input(
                "Batch size", min_value=1, max_value=2048, value=32
            )
            lr = c3.number_input(
                "Learning rate",
                min_value=1e-6,
                max_value=1.0,
                value=1e-3,
                format="%.6f",
            )
            val_split = c4.slider("Val split", 0.0, 0.5, 0.1, 0.05)
            if st.form_submit_button(
                "🚀 Start",
                width="stretch",
                disabled=(
                    not ann_key or not available_sessions or not selected_sessions
                ),
            ) and ann_key:
                cmd = [
                    sys.executable,
                    "-u",
                    str(TRAINING_ENTRYPOINTS / "train_segment_classifier.py"),
                    "--epochs",
                    str(int(epochs)),
                    "--batch-size",
                    str(int(batch_size)),
                    "--lr",
                    str(float(lr)),
                    "--val-split",
                    str(float(val_split)),
                    "--annotation-key",
                    ann_key,
                ]
                for session_id in selected_sessions:
                    cmd.extend(["--session-id", session_id])
                spawn("classifier", cmd)
                st.rerun()


__all__ = ["ClassifierTrainingComponent"]
