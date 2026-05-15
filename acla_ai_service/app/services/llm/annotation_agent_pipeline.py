"""
Multi-agent annotation pipeline runner.

The pipeline is the ``annotation_root`` Agent in ``AGENT_REGISTRY`` — a
uniform planner → step_solvers → synthesizer → evaluator subgraph defined
in ``app.services.llm.agents.annotation_root``. This module just wires up
the VLM/LLM callables and invokes that compiled graph.

Public API (kept stable for UI consumers):
    AnnotationPipelineConfig
    AnnotationResult
    run_annotation_pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.models.segment_models import LABEL_CATEGORIES
from app.services.llm.agent_framework import AGENT_REGISTRY
from app.services.llm.step_evaluator_agents import (
    get_active_stage,
    set_eval_llm,
    set_step_event_callback,
)

# Side-effect import: registers all agents (including annotation_root).
import app.services.llm.agents  # noqa: F401

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnnotationPipelineConfig:
    """Configuration for the annotation pipeline."""

    max_iterations: int = 3
    max_new_tokens: int = 1500
    temperature: float = 0.7

    # llama-cpp VLM settings
    gguf_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    context_size: int = 32768
    n_gpu_layers: int = -1

    # HuggingFace source repo + conversion
    hf_repo: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    quantization_type: str = "Q4_K_M"


@dataclass
class AnnotationResult:
    """Result returned by the pipeline."""

    final_labels: List[str]
    final_reasoning: str
    accepted: bool
    iterations: int
    messages: List[dict]
    graph_images: List[bytes] = field(default_factory=list)  # PNG bytes
    sub_start: Optional[int] = None
    sub_end: Optional[int] = None
    # Per-label proposals from the synthesizer. Each entry:
    #   {label_id, start_index, end_index, reasoning}
    # The UI groups these by (start_index, end_index) to materialise one
    # sub-segment per AI-discovered range, rather than collapsing them all
    # into the union span [sub_start, sub_end].
    label_annotations: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _prepare_segment_data(
    session_id: str,
    start_index: int,
    end_index: int,
) -> dict:
    """Return a minimal segment metadata dict (no statistics computed)."""
    return {"session_id": session_id, "start_index": start_index, "end_index": end_index}


def _default_root_state(
    df,
    start_index: int,
    end_index: int,
    session_id: str,
    parent_main_labels: List[str],
    existing_children: List[dict],
) -> Dict[str, Any]:
    """Seed state for an annotation_root invocation."""
    return {
        "segment_data": _prepare_segment_data(session_id, start_index, end_index),
        "df_ref": df,
        "parent_main_labels": parent_main_labels,
        "existing_children": existing_children,
        "parent_start": start_index,
        "parent_end": end_index,
        "available_labels": LABEL_CATEGORIES,
        "plan": "",
        "plan_steps": [],
        "current_step_index": 0,
        "step_results": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
        "attachment_pool": {},
        "verified_labels": [],
        "verified_label_reasoning": {},
        "evaluation": "",
        "final_labels": [],
        "final_label_annotations": [],
        "final_reasoning": "",
        "final_sub_start": 0,
        "final_sub_end": 0,
        "messages": [],
        "depth": 0,
        "call_stack": [],
        "spawn_log": [],
        "total_spawns": 0,
    }


def run_annotation_pipeline(
    df,
    start_index: int,
    end_index: int,
    session_id: str,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]] = None,
    config: Optional[AnnotationPipelineConfig] = None,
    progress_callback=None,
    vlm_stream_callback: Optional[Callable] = None,
    vlm_prompt_callback: Optional[Callable] = None,
    vlm_reasoning_callback: Optional[Callable] = None,
    step_event_callback: Optional[Callable] = None,
) -> AnnotationResult:
    """Execute the annotation pipeline by invoking the annotation_root Agent.

    Parameters
    ----------
    df : pandas.DataFrame
        Full session telemetry DataFrame.
    start_index, end_index : int
        Parent segment boundaries.
    session_id : str
        Current session identifier.
    parent_main_labels : list[str]
        Main-label IDs from the parent segment (used as hints).
    existing_children : list[dict], optional
        Already-existing child sub-segments ``[{start_index, end_index, labels}, ...]``.
    config : AnnotationPipelineConfig, optional
        VLM and pipeline parameters.
    progress_callback : callable, optional
        ``fn(node_name: str, detail: str)`` called after each top-level
        annotation_root node completes.
    """
    config = config or AnnotationPipelineConfig()
    existing_children = existing_children or []

    # ------------------------------------------------------------------
    # Build VLM generate function (llama-cpp backed)
    # ------------------------------------------------------------------
    from app.services.llm.annotation_agent_llm_service import (
        get_or_start_service,
        AnnotationAgentLLMConfig,
    )

    agent_llm_config = AnnotationAgentLLMConfig(
        gguf_path=config.gguf_path,
        mmproj_path=config.mmproj_path,
        context_size=config.context_size,
        n_gpu_layers=config.n_gpu_layers,
        hf_repo=config.hf_repo,
        quantization_type=config.quantization_type,
    )
    vlm_service = get_or_start_service(agent_llm_config)

    def vlm_generate(
        prompt: str, images: Optional[List[bytes]] = None,
    ) -> str:
        """Send prompt (with optional images) to the llama-cpp VLM."""
        if vlm_prompt_callback:
            vlm_prompt_callback(prompt, get_active_stage())
        return vlm_service.generate(
            prompt,
            images=images,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            stream_callback=vlm_stream_callback,
            reasoning_callback=vlm_reasoning_callback,
        )

    def llm_generate(prompt: str) -> str:
        """Text-only call to the same server (no images, lower temperature)."""
        if vlm_prompt_callback:
            vlm_prompt_callback(prompt, get_active_stage())
        return vlm_service.generate(
            prompt,
            images=None,
            max_tokens=config.max_new_tokens,
            temperature=0.1,
            stream_callback=vlm_stream_callback,
            reasoning_callback=vlm_reasoning_callback,
        )

    set_eval_llm(vlm_generate, llm_generate)
    set_step_event_callback(step_event_callback)

    graph = AGENT_REGISTRY["annotation_root"]
    initial_state = _default_root_state(
        df, start_index, end_index, session_id,
        parent_main_labels, existing_children,
    )

    final_state: Dict[str, Any] = dict(initial_state)
    for event in graph.stream(initial_state, config={"recursion_limit": 100}):
        for node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue
            final_state.update(node_output)
            if progress_callback:
                detail = _progress_detail(node_name, final_state)
                progress_callback(node_name, detail)

    set_eval_llm(None, None)

    accepted = final_state.get("evaluation") == "pass"

    return AnnotationResult(
        sub_start=final_state.get("final_sub_start"),
        sub_end=final_state.get("final_sub_end"),
        final_labels=final_state.get("final_labels", []),
        final_reasoning=final_state.get("final_reasoning", ""),
        accepted=accepted,
        iterations=1,
        messages=final_state.get("messages", []),
        graph_images=final_state.get("all_graph_images", []),
        label_annotations=final_state.get("final_label_annotations", []),
    )


def _progress_detail(node_name: str, state: Dict[str, Any]) -> str:
    """Compose a progress-display string for a freshly completed node."""
    from app.models.segment_models import LABEL_MAPPING

    if node_name == "planner":
        n_steps = len(state.get("plan_steps", []))
        return f"Planned {n_steps} step(s)"
    if node_name == "executor":
        idx = state.get("current_step_index", 0)
        total = len(state.get("plan_steps", []))
        plan_steps = state.get("plan_steps", [])
        just_done = idx - 1
        if 0 <= just_done < len(plan_steps):
            step = plan_steps[just_done]
            return (
                f"Ran step {idx}/{total} via agent '{step.get('agent', '?')}'"
            )
        return f"Ran step {idx}/{total}"
    if node_name == "synthesizer":
        labels = state.get("final_labels", [])
        ss = state.get("final_sub_start", "?")
        se = state.get("final_sub_end", "?")
        verdict = state.get("evaluation", "")
        return (
            f"Range [{ss}, {se}], "
            f"labels: {', '.join(LABEL_MAPPING.get(l, l) for l in labels)}, "
            f"eval: {verdict}"
        )
    if node_name == "evaluator":
        return f"Verdict: {state.get('evaluation', '?')}"
    return ""
