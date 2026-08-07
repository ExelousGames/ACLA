"""Segment-cropper training component."""

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


class SegmentCropperTrainingComponent(TrainingComponent):
    def open(self, context: TrainingComponentContext) -> None:
        default_ann_key = resolve_input_key(context)

        st.header("Segment cropper (Boundary TCN)")
        show_node_context(context)
        show_input_location("Complete session dataset", default_ann_key)
        render_card(
            "segment_cropper",
            title="Segment cropper (Boundary TCN)",
            description=(
                "Trains class-agnostic boundaries on complete session chunks. "
                "The session split is fixed at 90% training and 10% validation."
            ),
            render_start_form=lambda: self._render_form(default_ann_key),
        )

    @staticmethod
    def _render_form(default_ann_key: str) -> None:
        with st.form("segment_cropper_form"):
            c1, c2, c3 = st.columns(3)
            epochs = c1.number_input("Epochs", min_value=1, max_value=500, value=10)
            batch_size = c2.number_input(
                "Batch size", min_value=1, max_value=256, value=8
            )
            learning_rate = c3.number_input(
                "Learning rate",
                min_value=1e-6,
                max_value=1.0,
                value=1e-3,
                format="%.6f",
            )
            if st.form_submit_button(
                "Start",
                width="stretch",
                disabled=not bool(default_ann_key),
            ) and default_ann_key:
                cmd = [
                    sys.executable,
                    "-u",
                    str(TRAINING_ENTRYPOINTS / "train_segment_cropper.py"),
                    "--epochs",
                    str(int(epochs)),
                    "--batch-size",
                    str(int(batch_size)),
                    "--lr",
                    str(float(learning_rate)),
                    "--annotation-key",
                    default_ann_key,
                ]
                spawn("segment_cropper", cmd)
                st.rerun()


__all__ = ["SegmentCropperTrainingComponent"]
