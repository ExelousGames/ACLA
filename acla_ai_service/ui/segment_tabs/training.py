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

from app.infra.config.pipeline import PipelineConfig
from app.pipelines.training.llm_trainer import DEFAULT_MODEL as _DEFAULT_LLM_MODEL

from segment_tabs._training_runner import render_card, spawn


_AI_SERVICE_DIR = Path(__file__).resolve().parents[2]
_SCRIPTS = _AI_SERVICE_DIR / "scripts"
_DEFAULT_CHAT_JSONL = (
    _AI_SERVICE_DIR / "models" / "llm_datasets" / "telemetry_descriptions_v1.chat.jsonl"
)

_HF_MODELS = [
    _DEFAULT_LLM_MODEL,
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]


def render_training(annotation_key: Optional[str]) -> None:
    st.header("🏋️ Training")
    st.caption(
        "Train the segment classifier, transformer guidance, and LLM. Each runs "
        "as a background subprocess; logs persist across browser refreshes."
    )

    cfg = PipelineConfig()
    default_ann_key = annotation_key or cfg.annotation_cache_key
    default_proc_key = cfg.processed_session_data_cache_key

    # ── Run all (sequential) ───────────────────────────────────────────────
    render_card(
        "runall",
        title="⛓ Run all (sequential)",
        description="Classifier → Transformer → LLM, one after the other in a single subprocess.",
        render_start_form=lambda: _runall_form(default_ann_key, default_proc_key),
    )

    st.divider()

    # ── Segment classifier ─────────────────────────────────────────────────
    render_card(
        "classifier",
        title="1️⃣ Segment classifier (LSTM)",
        description="Trains on the currently-selected annotation dataset.",
        render_start_form=_classifier_form,
    )

    st.divider()

    # ── Transformer guidance ───────────────────────────────────────────────
    render_card(
        "transformer",
        title="2️⃣ Transformer guidance",
        description="Trains on EA/RM-labelled segments from the annotation dataset.",
        render_start_form=lambda: _transformer_form(default_ann_key, default_proc_key),
    )

    st.divider()

    # ── LLM fine-tune ──────────────────────────────────────────────────────
    render_card(
        "llm",
        title="3️⃣ LLM fine-tune",
        description=(
            "Fine-tunes a HuggingFace base model on the chat-format JSONL "
            "exported from the LLM Pipeline tab."
        ),
        render_start_form=_llm_form,
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


def _transformer_form(default_ann_key: str, default_proc_key: str) -> None:
    with st.form("transformer_form"):
        ann_key = st.text_input("Annotation key", value=default_ann_key)
        proc_key = st.text_input("Processed-session key", value=default_proc_key)
        max_seg = st.number_input(
            "Max segment length", min_value=1, max_value=512, value=20,
        )
        if st.form_submit_button("🚀 Start", use_container_width=True):
            cmd = [
                sys.executable, "-u", str(_SCRIPTS / "train_transformer_guidance.py"),
                "--annotation-key", ann_key,
                "--processed-key", proc_key,
                "--max-segment-length", str(int(max_seg)),
            ]
            spawn("transformer", cmd)
            st.rerun()


def _llm_form() -> None:
    chat_path_default = st.session_state.get("llm_chat_path", str(_DEFAULT_CHAT_JSONL))
    with st.form("llm_form"):
        chat_path = st.text_input("Chat-format JSONL", value=chat_path_default)
        model = st.selectbox("Base model", _HF_MODELS, index=0)
        custom = st.text_input(
            "…or custom HF model id (overrides the selection above)",
            value="",
            placeholder="e.g. meta-llama/Llama-3.2-1B-Instruct",
        )
        if st.form_submit_button("🚀 Start", use_container_width=True):
            chosen_model = custom.strip() or model
            if not Path(chat_path).exists():
                st.error(f"Chat JSONL not found: {chat_path}")
                return
            cmd = [
                sys.executable, "-u", str(_SCRIPTS / "train_telemetry_llm.py"),
                "--dataset", chat_path,
                "--model", chosen_model,
            ]
            spawn("llm", cmd)
            st.rerun()


def _runall_form(default_ann_key: str, default_proc_key: str) -> None:
    chat_path_default = st.session_state.get("llm_chat_path", str(_DEFAULT_CHAT_JSONL))
    with st.form("runall_form"):
        ann_key = st.text_input("Annotation key (transformer)", value=default_ann_key)
        proc_key = st.text_input("Processed-session key (transformer)", value=default_proc_key)
        max_seg = st.number_input(
            "Max segment length", min_value=1, max_value=512, value=20,
        )
        chat_path = st.text_input("LLM chat-format JSONL", value=chat_path_default)
        model = st.selectbox("LLM base model", _HF_MODELS, index=0)
        custom = st.text_input("…or custom LLM HF model id", value="")
        if st.form_submit_button("🚀 Start all", use_container_width=True):
            chosen_model = custom.strip() or model
            if not Path(chat_path).exists():
                st.error(f"Chat JSONL not found: {chat_path}")
                return
            cmd = [
                sys.executable, "-u", str(_SCRIPTS / "run_all_trainings.py"),
                "--annotation-key", ann_key,
                "--processed-key", proc_key,
                "--max-segment-length", str(int(max_seg)),
                "--chat-dataset", chat_path,
                "--llm-model", chosen_model,
            ]
            spawn(
                "runall", cmd,
                extra_info={"jobs": ["classifier", "transformer", "llm"]},
            )
            st.rerun()
