"""Opportunity-forecaster training component."""

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


class OpportunityForecasterTrainingComponent(TrainingComponent):
    def open(self, context: TrainingComponentContext) -> None:
        default_ann_key = resolve_input_key(context)

        st.header("Opportunity forecaster")
        show_node_context(context)
        show_input_location("Input annotation dataset", default_ann_key)
        render_card(
            "opportunity_forecaster",
            title="3. Opportunity forecaster",
            description=(
                "Trains future successful overtake / defense probabilities "
                "from O/OD annotated segments."
            ),
            render_start_form=lambda: self._render_form(default_ann_key),
        )

    @staticmethod
    def _render_form(ann_key: str) -> None:
        with st.form("opportunity_forecaster_form"):
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
            if st.form_submit_button(
                "Start",
                width="stretch",
                disabled=not bool(ann_key),
            ) and ann_key:
                cmd = [
                    sys.executable,
                    "-u",
                    str(TRAINING_ENTRYPOINTS / "train_opportunity_forecaster.py"),
                    "--annotation-key",
                    ann_key,
                    "--input-fraction",
                    str(float(input_fraction)),
                    "--max-negatives",
                    str(int(max_negatives)),
                ]
                spawn("opportunity_forecaster", cmd)
                st.rerun()


__all__ = ["OpportunityForecasterTrainingComponent"]
