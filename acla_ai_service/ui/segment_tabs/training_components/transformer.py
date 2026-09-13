"""Transformer-guidance training component."""

from __future__ import annotations

import sys

import streamlit as st

from .._training_runner import render_card, spawn
from ._shared import (
    TRAINING_ENTRYPOINTS,
    resolve_input_key,
    show_input_location,
    show_node_context,
)
from .contract import TrainingComponent, TrainingComponentContext


class TransformerTrainingComponent(TrainingComponent):
    def open(self, context: TrainingComponentContext) -> None:
        default_ann_key = resolve_input_key(context)

        st.header("🏋️ Transformer guidance")
        show_node_context(context)
        show_input_location("Input annotation dataset", default_ann_key)
        render_card(
            "transformer",
            title="2️⃣ Transformer guidance",
            description="Trains on EA/RM-labelled segments from the annotation dataset.",
            render_start_form=lambda: self._render_form(default_ann_key),
        )

    @staticmethod
    def _render_form(ann_key: str) -> None:
        with st.form("transformer_form"):
            if st.form_submit_button(
                "🚀 Start",
                width="stretch",
                disabled=not bool(ann_key),
            ) and ann_key:
                cmd = [
                    sys.executable,
                    "-u",
                    str(TRAINING_ENTRYPOINTS / "train_transformer_guidance.py"),
                    "--annotation-key",
                    ann_key,
                ]
                spawn("transformer", cmd)
                st.rerun()


__all__ = ["TransformerTrainingComponent"]
