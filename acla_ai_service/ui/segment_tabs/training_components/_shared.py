"""Small rendering helpers shared by training components."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .contract import TrainingComponentContext


_AI_SERVICE_DIR = Path(__file__).resolve().parents[3]
TRAINING_ENTRYPOINTS = (
    _AI_SERVICE_DIR / "app" / "pipelines" / "training" / "entrypoints"
)


def resolve_input_key(context: TrainingComponentContext) -> str:
    return context.input_key or ""


def show_node_context(context: TrainingComponentContext) -> None:
    if context.node is not None:
        st.info(
            f"Configuring training node `{context.node.id}` from the active pipeline."
        )


def show_input_location(label: str, value: str | None) -> None:
    if value:
        st.info(f"📂 {label}: `{value}`")
    else:
        st.warning(
            f"No {label.lower()} configured — set it from the Pipeline view "
            "by picking an annotation output on this training node."
        )
