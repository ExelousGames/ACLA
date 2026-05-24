"""Shared runner for fine-tuning the telemetry LLM from a chat-format JSONL.

Used by:
- `scripts/train_telemetry_llm.py` (CLI)
- `ui/segment_tabs/llm_pipeline.py` (Streamlit, via a subprocess that re-invokes
  the CLI and tails its log)

Progress is communicated with plain ``print()`` so the subprocess log captures it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.llm.local_llm import LocalLLMConfig
from app.pipelines.chat.orchestrator import TelemetryLLMOrchestrator


DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"


@dataclass
class LLMTrainingResult:
    success: bool
    adapter_directory: Optional[str] = None
    train_path: Optional[Path] = None
    eval_path: Optional[Path] = None
    train_examples: int = 0
    eval_examples: int = 0
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _split_dataset(
    dataset_path: Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Path, Path, int, int]:
    with dataset_path.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    random.Random(seed).shuffle(lines)

    split_idx = max(1, int(len(lines) * train_ratio))
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]

    train_path = dataset_path.with_name(dataset_path.stem + "_train.jsonl")
    eval_path = dataset_path.with_name(dataset_path.stem + "_eval.jsonl")

    train_path.write_text(
        "\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8",
    )
    eval_path.write_text(
        "\n".join(eval_lines) + ("\n" if eval_lines else ""), encoding="utf-8",
    )

    return train_path, eval_path, len(train_lines), len(eval_lines)


def _build_orchestrator(model: str, project_root: Path) -> TelemetryLLMOrchestrator:
    cfg = LocalLLMConfig()
    cfg.model.base_model = model
    cfg.model.tokenizer_name = model
    cfg.model.load_in_4bit = True
    cfg.lora.use_lora = True
    cfg.training.use_gradient_checkpointing = True

    return TelemetryLLMOrchestrator(
        llm_config=cfg,
        adapter_directory=project_root / "models" / "llm_adapters",
        dataset_directory=project_root / "models" / "llm_datasets",
    )


async def run_llm_training(
    dataset_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    project_root: Optional[Path] = None,
) -> LLMTrainingResult:
    """Split → configure → train. All progress is printed for log capture."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return LLMTrainingResult(success=False, error=f"Dataset not found: {dataset_path}")

    if project_root is None:
        # app/pipelines/training/llm_trainer.py → parents[3] = acla_ai_service/
        project_root = Path(__file__).resolve().parents[3]

    print(f"[INFO] Initializing orchestrator with model: {model}")
    train_path, eval_path, n_train, n_eval = _split_dataset(dataset_path)
    print(f"[INFO] Split dataset: {n_train} train, {n_eval} eval")
    print(f"[INFO]   train → {train_path}")
    print(f"[INFO]   eval  → {eval_path}")

    orchestrator = _build_orchestrator(model, project_root)

    print("[INFO] Starting training (this may take a while)...")
    result = await orchestrator.train_from_dataset(
        dataset_path=train_path,
        eval_dataset_path=eval_path,
        cleanup_dataset_file=False,
    )

    if not result.get("success"):
        return LLMTrainingResult(
            success=False,
            train_path=train_path,
            eval_path=eval_path,
            train_examples=n_train,
            eval_examples=n_eval,
            error=str(result),
        )

    adapter_dir = result.get("adapter_directory")
    print(f"[INFO] Training complete. Adapter directory: {adapter_dir}")
    return LLMTrainingResult(
        success=True,
        adapter_directory=adapter_dir,
        train_path=train_path,
        eval_path=eval_path,
        train_examples=n_train,
        eval_examples=n_eval,
        metrics=result.get("training_metrics"),
    )
