"""
Evaluator suite — format + evidence checks for any step's output.

The suite is domain-free. Each evaluator runs on the parent step's own
prompt/output/inputs only, so it judges the step against the evidence
the step actually had. The default chain runs both evaluators; each one
auto-skips when not applicable (format_evaluator skips non-JSON output;
evidence_evaluator skips when no evidence is attached). Callers wanting
a specific chain pass ``evaluators=[...]`` explicitly.

Also owns the shared attachment plumbing — ``PipelineAttachment``, the
named pool, structured-content formatters, and the
``set_eval_llm`` / ``set_active_stage`` callback holders that the box's
runners and sub-agents share during a run.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM function holder — registered by the runner at the start of a run
# ---------------------------------------------------------------------------

_eval_llm_holder: Dict[str, Any] = {
    "vlm": None,
    "llm": None,
    "vlm_chat_with_tools": None,
    "stage_node": "",
    "stage_phase": "",
    "stage_iter": None,
    "stage_total": None,
    "stage_attachments": [],
    "stage_graphs": None,
    "step_event_callback": None,
}
_eval_llm_lock = threading.Lock()


def set_eval_llm(vlm: Optional[Callable], llm: Optional[Callable]) -> None:
    """Register VLM/LLM callables for evaluator + sub-agent use."""
    with _eval_llm_lock:
        _eval_llm_holder["vlm"] = vlm
        _eval_llm_holder["llm"] = llm


def set_vlm_chat_with_tools(fn: Optional[Callable]) -> None:
    """Register the chat-with-tools closure sub-agents call for VLM tool use."""
    with _eval_llm_lock:
        _eval_llm_holder["vlm_chat_with_tools"] = fn


def set_step_event_callback(cb: Optional[Callable]) -> None:
    """Register a callback for non-VLM step events (e.g. zoom rendering)."""
    with _eval_llm_lock:
        _eval_llm_holder["step_event_callback"] = cb


def emit_step_event(
    node_name: str,
    phase: str,
    summary: str,
    iteration: Optional[int] = None,
    total: Optional[int] = None,
    attachments: Optional[List["PipelineAttachment"]] = None,
    graphs: Optional[List[str]] = None,
) -> None:
    """Surface a non-VLM event to the UI.

    Used by render-only sub-agents (zoom) that don't go through
    ``vlm_generate`` but still want a section in the live transcript.
    Builds the stage payload locally so it doesn't disturb the global
    stage state used by the next real VLM call.
    """
    cb = _eval_llm_holder.get("step_event_callback")
    if cb is None:
        return
    att_summaries: List[Dict[str, Any]] = []
    for att in attachments or []:
        count = (
            len(att.content)
            if att.kind == "image_set" and isinstance(att.content, list)
            else None
        )
        att_summaries.append({
            "name": att.name,
            "label": att.label,
            "kind": att.kind,
            "count": count,
        })
    stage = {
        "node_name": node_name,
        "phase": phase,
        "iteration": iteration,
        "total": total,
        "attachments": att_summaries,
        "graphs": list(graphs) if graphs else None,
    }
    cb(summary, stage)


def set_active_stage(
    node_name: str,
    phase: str,
    graphs: Optional[List[str]] = None,
) -> None:
    """Record which (node, phase) is about to make a VLM call."""
    with _eval_llm_lock:
        if node_name != _eval_llm_holder["stage_node"]:
            _eval_llm_holder["stage_iter"] = None
            _eval_llm_holder["stage_total"] = None
            _eval_llm_holder["stage_attachments"] = []
            _eval_llm_holder["stage_graphs"] = None
        _eval_llm_holder["stage_node"] = node_name
        _eval_llm_holder["stage_phase"] = phase
        if graphs is not None:
            _eval_llm_holder["stage_graphs"] = list(graphs)


def set_active_iteration(iteration: Optional[int], total: Optional[int]) -> None:
    """Set the iteration tag for the current node (e.g. describe_graphs k/N)."""
    with _eval_llm_lock:
        _eval_llm_holder["stage_iter"] = iteration
        _eval_llm_holder["stage_total"] = total


def set_active_attachments(inputs: Optional[List["PipelineAttachment"]]) -> None:
    """Record the attachments the current agent is feeding to its VLM call.

    Evaluator sub-phases inherit the parent agent's attachments —
    ``run_evaluator_suite`` re-registers the same list with each evaluator
    before its VLM call.
    """
    summaries: List[Dict[str, Any]] = []
    for att in inputs or []:
        count = (
            len(att.content)
            if att.kind == "image_set" and isinstance(att.content, list)
            else None
        )
        summaries.append({
            "name": att.name,
            "label": att.label,
            "kind": att.kind,
            "count": count,
        })
    with _eval_llm_lock:
        _eval_llm_holder["stage_attachments"] = summaries


def get_active_stage() -> Dict[str, Any]:
    """Snapshot the current stage tags for callback dispatch."""
    with _eval_llm_lock:
        graphs = _eval_llm_holder["stage_graphs"]
        return {
            "node_name": _eval_llm_holder["stage_node"],
            "phase": _eval_llm_holder["stage_phase"],
            "iteration": _eval_llm_holder["stage_iter"],
            "total": _eval_llm_holder["stage_total"],
            "attachments": list(_eval_llm_holder["stage_attachments"]),
            "graphs": list(graphs) if graphs else None,
        }


def _call_eval_vlm(
    prompt: str,
    images: Optional[List[bytes]] = None,
) -> str:
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""
    if images:
        return vlm_fn(prompt, images)
    return vlm_fn(prompt)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Output from any single evaluator agent."""

    evaluator_name: str
    verdict: Literal["pass", "fail"]
    feedback: str
    revised_result: Optional[str] = None  # edited output, or None on pass
    edit_count: int = 0


class EvalPipelineResult(BaseModel):
    """Aggregated result from the full evaluator suite."""

    final_verdict: Literal["pass", "fail"]
    final_result: str
    evaluator_results: List[EvalResult]
    total_edits: int


# ---------------------------------------------------------------------------
# Pipeline attachments — named, typed artifacts in a shared pool
# ---------------------------------------------------------------------------


class PipelineAttachment(BaseModel):
    """A typed, named artifact shared between pipeline agents.

    Agents declare ``consumes`` / ``produces`` lists of attachment names.
    The system keeps a pool keyed by ``name`` and serves attachments to
    agents (and to their child evaluators) by name.
    """

    name: str
    kind: Literal["text", "image_set", "structured"]
    label: str
    content_schema: str = ""
    content: Any


# ---------------------------------------------------------------------------
# Formatters for structured attachments — keyed by content_schema
# ---------------------------------------------------------------------------


def _format_step_observation(content: Any) -> str:
    if not isinstance(content, dict):
        return str(content)
    parts: list[str] = []
    descs = content.get("graph_descriptions") or []
    if descs:
        parts.append("Graphs: " + ", ".join(descs))
    obs = content.get("graph_observations")
    if obs:
        parts.append(str(obs))
    return "\n".join(parts)


def _format_graph_descriptions(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(f"- {c}" for c in content)
    return str(content)


_STRUCTURED_FORMATTERS: Dict[str, Callable[[Any], str]] = {
    "step_observation": _format_step_observation,
    "graph_descriptions": _format_graph_descriptions,
}


def register_structured_formatter(
    schema: str, formatter: Callable[[Any], str],
) -> None:
    """Register a formatter for a custom ``content_schema``.

    Callers outside the box (e.g. the annotation package) use this to
    teach ``render_inputs_for_prompt`` how to render their own structured
    attachments without the box knowing the schema in advance.
    """
    _STRUCTURED_FORMATTERS[schema] = formatter


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


AttachmentPool = Dict[str, PipelineAttachment]


def merge_pool(
    left: Optional[AttachmentPool],
    right: Optional[AttachmentPool],
) -> AttachmentPool:
    """LangGraph state reducer — later writes win on the same name."""
    return {**(left or {}), **(right or {})}


def pool_put(pool: AttachmentPool, attachment: PipelineAttachment) -> AttachmentPool:
    return {**pool, attachment.name: attachment}


def pool_get_many(pool: AttachmentPool, names: List[str]) -> List[PipelineAttachment]:
    return [pool[n] for n in names if n in pool]


def collect_image_bytes(inputs: List[PipelineAttachment]) -> List[bytes]:
    out: List[bytes] = []
    for att in inputs:
        if att.kind == "image_set" and isinstance(att.content, list):
            out.extend(att.content)
    return out


def render_inputs_for_prompt(inputs: List[PipelineAttachment]) -> str:
    """Render attachments as ``#### <label>`` markdown sections for a VLM prompt."""
    if not inputs:
        return ""
    blocks: list[str] = []
    for att in inputs:
        if att.kind == "image_set":
            n = len(att.content) if isinstance(att.content, list) else 0
            blocks.append(
                f"#### {att.label}\n[{n} image(s) attached to this message]"
            )
            continue
        if att.kind == "structured":
            formatter = _STRUCTURED_FORMATTERS.get(
                att.content_schema,
                lambda c: json.dumps(c, indent=2) if c is not None else "",
            )
            body = formatter(att.content)
            blocks.append(f"#### {att.label}\n{body}")
            continue
        body = att.content if isinstance(att.content, str) else str(att.content)
        blocks.append(f"#### {att.label}\n{body}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def result_editor(
    original_result: str,
    failing_sections: List[str],
    revisions: List[str],
) -> str:
    """Targeted string replacement on the step output."""
    revised = original_result
    for old, new in zip(failing_sections, revisions):
        revised = revised.replace(old, new, 1)
    return revised


def json_reformatter(malformed_json: str) -> str:
    """Best-effort parse + re-serialize broken JSON."""
    text = malformed_json.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.dumps(json.loads(text), indent=2)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.dumps(json.loads(fixed), indent=2)
    except json.JSONDecodeError:
        pass

    fixed2 = fixed.replace("'", '"')
    try:
        return json.dumps(json.loads(fixed2), indent=2)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r'\{[\s\S]*\}', fixed)
    if brace_match:
        try:
            return json.dumps(json.loads(brace_match.group()), indent=2)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not reformat JSON: {malformed_json[:200]}...")


# ---------------------------------------------------------------------------
# Evaluator response parsing
# ---------------------------------------------------------------------------


def _parse_eval_response(raw: str) -> dict:
    """Parse a structured evaluator VLM response (VERDICT/FEEDBACK/EDITS)."""
    result = {"verdict": "fail", "feedback": "", "edits": None}

    lines = raw.strip().split("\n")
    feedback_lines: list[str] = []
    edits_block: list[str] = []
    in_edits = False

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.upper().startswith("VERDICT:"):
            verdict_text = line_stripped[len("VERDICT:"):].strip().lower()
            result["verdict"] = "pass" if "pass" in verdict_text else "fail"
            continue

        if line_stripped.upper().startswith("EDITS:"):
            in_edits = True
            remainder = line_stripped[len("EDITS:"):].strip()
            if remainder:
                edits_block.append(remainder)
            continue

        if in_edits:
            edits_block.append(line)
        else:
            if line_stripped.upper().startswith("FEEDBACK:"):
                feedback_lines.append(line_stripped[len("FEEDBACK:"):].strip())
            else:
                feedback_lines.append(line)

    result["feedback"] = "\n".join(feedback_lines).strip()

    if edits_block:
        edits_text = "\n".join(edits_block).strip()
        if edits_text.startswith("```"):
            edits_text = re.sub(r'^```\w*\n?', '', edits_text)
            edits_text = re.sub(r'\n?```$', '', edits_text)
        try:
            result["edits"] = json.loads(edits_text)
        except json.JSONDecodeError:
            result["feedback"] += f"\n[Unparseable edit suggestion: {edits_text[:200]}]"

    return result


def _replace_matching_node(node: Any, old: Any, new: Any) -> bool:
    if isinstance(node, list):
        for i, item in enumerate(node):
            if item == old:
                node[i] = new
                return True
            if _replace_matching_node(item, old, new):
                return True
        return False
    if isinstance(node, dict):
        for k, v in node.items():
            if v == old:
                node[k] = new
                return True
            if _replace_matching_node(v, old, new):
                return True
    return False


def _apply_eval_replacement(
    current_result: str,
    old: Any,
    new: Any,
) -> tuple[str, bool]:
    if isinstance(old, str) and isinstance(new, str):
        if old and old in current_result:
            return current_result.replace(old, new, 1), True
        return current_result, False

    if isinstance(old, (dict, list)):
        try:
            parsed = json.loads(current_result)
        except (json.JSONDecodeError, TypeError):
            return current_result, False
        if parsed == old:
            return json.dumps(new, indent=2), True
        if _replace_matching_node(parsed, old, new):
            return json.dumps(parsed, indent=2), True

    return current_result, False


# ---------------------------------------------------------------------------
# Individual evaluator runners
# ---------------------------------------------------------------------------


def _run_format_evaluator(
    current_result: str,
    context: Dict[str, Any],
    max_edit_rounds: int = 2,
) -> EvalResult:
    """Check JSON structural correctness of the parent agent's output."""
    if not _looks_like_json(current_result):
        return EvalResult(
            evaluator_name="format_evaluator",
            verdict="pass",
            feedback="Non-JSON output — format check skipped.",
        )

    parent_prompt = context.get("parent_prompt", "")
    step_name = context.get("step_name", "")
    edit_count = 0
    parsed: Dict[str, Any] = {"verdict": "fail", "feedback": "", "edits": None}

    for attempt in range(1 + max_edit_rounds):
        eval_prompt = (
            "You evaluate whether a pipeline step's JSON output is well-formed.\n"
            "Compared to the original request, does the response fulfil the "
            "requirement *within MY narrow concern* (JSON structure only)?\n\n"
            "Check:\n"
            "- The output parses as JSON.\n"
            "- The top-level keys requested by the original prompt are present.\n"
            "- Value types match what was requested (string, integer, list, object).\n"
            "- Nesting is correct and the JSON is not truncated.\n\n"
            "Do NOT judge whether the values are factually correct or whether the "
            "indices/labels are sensible — only whether the JSON structure matches "
            "what the original prompt asked for.\n\n"
            f"#### Parent Prompt (what was requested)\n{parent_prompt}\n\n"
            f"#### Step Name\n{step_name}\n\n"
            f"#### Output to Evaluate\n{current_result}\n\n"
            "#### Instructions\n"
            "Respond in this exact format:\n"
            "VERDICT: pass (if JSON structure is valid) or fail (if structure issues exist)\n"
            "FEEDBACK: Describe any structural issues found, or confirm the structure is valid.\n"
            "EDITS: (only if VERDICT is fail) A JSON object:\n"
            '  {"replacements": [{"old": "<unique snippet from the output>", "new": "<replacement>"}]}\n'
            "  Each `old` MUST be a verbatim substring of the output that uniquely "
            "identifies the location to change (include enough surrounding context "
            "if the literal text occurs more than once). To rewrite the whole "
            "output, supply the entire current output as `old` and the corrected "
            "version as `new`.\n"
        )

        raw_response = _call_eval_vlm(eval_prompt)
        if not raw_response:
            return EvalResult(
                evaluator_name="format_evaluator",
                verdict="pass",
                feedback="VLM unavailable — skipping format check.",
            )

        parsed = _parse_eval_response(raw_response)

        if parsed["verdict"] == "pass":
            return EvalResult(
                evaluator_name="format_evaluator",
                verdict="pass",
                feedback=parsed["feedback"],
                revised_result=current_result if edit_count > 0 else None,
                edit_count=edit_count,
            )

        if parsed["edits"] and attempt < max_edit_rounds:
            edits = parsed["edits"]
            if isinstance(edits, dict) and "replacements" in edits:
                for repl in edits["replacements"]:
                    old = repl.get("old", "")
                    new = repl.get("new", "")
                    current_result, applied = _apply_eval_replacement(
                        current_result, old, new
                    )
                    if applied:
                        edit_count += 1
            try:
                current_result = json_reformatter(current_result)
                edit_count += 1
            except ValueError:
                pass
        else:
            break

    return EvalResult(
        evaluator_name="format_evaluator",
        verdict="fail",
        feedback=parsed["feedback"],
        revised_result=current_result if edit_count > 0 else None,
        edit_count=edit_count,
    )


def _run_evidence_evaluator(
    current_result: str,
    context: Dict[str, Any],
    max_edit_rounds: int = 2,
) -> EvalResult:
    """Check that the response's claims are grounded in the parent's inputs."""
    parent_prompt = context.get("parent_prompt", "")
    parent_inputs: List[PipelineAttachment] = context.get("parent_inputs", [])

    rendered_inputs = render_inputs_for_prompt(parent_inputs)
    images = collect_image_bytes(parent_inputs)
    edit_count = 0
    parsed: Dict[str, Any] = {"verdict": "fail", "feedback": "", "edits": None}

    has_evidence = bool(rendered_inputs) or bool(parent_prompt) or bool(images)
    if not has_evidence:
        return EvalResult(
            evaluator_name="evidence_evaluator",
            verdict="pass",
            feedback="No prompt, inputs, or images available — skipping evidence check.",
        )

    for attempt in range(1 + max_edit_rounds):
        inputs_section = (
            f"#### Parent Inputs (the evidence the parent agent consumed)\n"
            f"{rendered_inputs}\n\n"
            if rendered_inputs else ""
        )
        eval_prompt = (
            "You check whether a pipeline step's response is supported by the "
            "evidence the step itself had access to: the parent prompt and the "
            "parent inputs (named attachments and image sets the parent "
            "consumed).\n"
            "Compared to the original request, does the response fulfil the "
            "requirement *within MY narrow concern* (logic and evidence)?\n\n"
            "Check:\n"
            "- Are claims grounded in something the parent prompt, parent "
            "inputs, or images actually contain?\n"
            "- Does the reasoning follow logically from that evidence (no "
            "contradictions, no leaps)?\n"
            "- Are cited indices, labels, observations, or trends actually "
            "present in the evidence — or are some invented?\n"
            "- Are any ranges / index pairs in the output picked correctly "
            "per the parent request? Each `[start, end]` (or equivalent) "
            "must satisfy whatever the parent prompt asks of it and reflect "
            "a feature actually visible in the evidence rather than an "
            "arbitrary slice.\n"
            "- For visual claims (graph shapes, trace positions, colours, "
            "comparisons), do they match what is visible in the attached "
            "images? Treat the images as primary evidence for visual claims.\n\n"
            "Do NOT judge JSON well-formedness — focus on whether the "
            "response's content is grounded in the parent's evidence.\n\n"
            f"#### Parent Prompt (what was requested)\n{parent_prompt}\n\n"
            f"{inputs_section}"
            f"#### Output to Evaluate\n{current_result}\n\n"
            "#### Instructions\n"
            "Respond in this exact format:\n"
            "VERDICT: pass or fail\n"
            "FEEDBACK: Briefly state why entries / claims fail evidence (one or two lines).\n"
            "EDITS: (only if fail) A JSON object:\n"
            '  {"replacements": [{"old": "<unique snippet from the output>", "new": "<replacement>"}]}\n'
            "  Each `old` MUST be a verbatim substring of the output that "
            "uniquely identifies the location to change — include enough "
            "surrounding context (e.g. the full enclosing JSON entry) so the "
            "match cannot occur elsewhere.  To drop an entry, set `old` to the "
            "entry's exact JSON text (with its trailing comma if any) and "
            "`new` to the empty string.  To rewrite the whole output, supply "
            "the entire current output as `old` and the corrected version "
            "as `new`.\n"
        )

        raw_response = _call_eval_vlm(eval_prompt, images=images or None)
        if not raw_response:
            return EvalResult(
                evaluator_name="evidence_evaluator",
                verdict="pass",
                feedback="VLM unavailable — skipping evidence check.",
            )

        parsed = _parse_eval_response(raw_response)

        if parsed["verdict"] == "pass":
            return EvalResult(
                evaluator_name="evidence_evaluator",
                verdict="pass",
                feedback=parsed["feedback"],
                revised_result=current_result if edit_count > 0 else None,
                edit_count=edit_count,
            )

        if parsed["edits"] and attempt < max_edit_rounds:
            edits = parsed["edits"]
            if isinstance(edits, dict) and "replacements" in edits:
                for repl in edits["replacements"]:
                    old = repl.get("old", "")
                    new = repl.get("new", "")
                    current_result, applied = _apply_eval_replacement(
                        current_result, old, new
                    )
                    if applied:
                        edit_count += 1
        else:
            break

    return EvalResult(
        evaluator_name="evidence_evaluator",
        verdict="fail",
        feedback=parsed["feedback"],
        revised_result=current_result if edit_count > 0 else None,
        edit_count=edit_count,
    )


# ---------------------------------------------------------------------------
# Evaluator registry — no step-name profiles; default runs the full chain
# ---------------------------------------------------------------------------


_EVALUATOR_REGISTRY: Dict[str, Callable] = {
    "format_evaluator": _run_format_evaluator,
    "evidence_evaluator": _run_evidence_evaluator,
}

EVALUATOR_CHAIN: List[str] = [
    "format_evaluator",
    "evidence_evaluator",
]


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    if "```json" in stripped:
        return True
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_evaluator_suite(
    parent_prompt: str,
    parent_output_text: str,
    parent_inputs: List[PipelineAttachment],
    step_name: str,
    parent_start: int = 0,
    parent_end: int = 0,
    evaluators: Optional[List[str]] = None,
    max_edits_per_evaluator: int = 2,
) -> EvalPipelineResult:
    """Run the evaluator suite on a parent step's output.

    Default chain: all evaluators in ``EVALUATOR_CHAIN``. Each evaluator
    auto-skips when not applicable (format_evaluator skips non-JSON,
    evidence_evaluator skips when no evidence is attached), so the
    default is safe across all step types. Pass ``evaluators=[...]`` to
    override (e.g. ``["format_evaluator"]`` for a planner that only needs
    structural validation).

    ``step_name`` is forwarded to evaluator prompts purely as a label —
    the box does not branch on it.
    """
    inputs_list: List[PipelineAttachment] = list(parent_inputs or [])

    chain = (
        [name for name in EVALUATOR_CHAIN if name in evaluators]
        if evaluators
        else list(EVALUATOR_CHAIN)
    )

    context = {
        "parent_prompt": parent_prompt,
        "step_name": step_name,
        "parent_start": parent_start,
        "parent_end": parent_end,
        "parent_inputs": inputs_list,
    }

    current_result = parent_output_text
    all_eval_results: List[EvalResult] = []
    total_edits = 0

    for evaluator_name in chain:
        runner = _EVALUATOR_REGISTRY.get(evaluator_name)
        if runner is None:
            LOGGER.warning("Unknown evaluator '%s' — skipping.", evaluator_name)
            continue

        LOGGER.info(
            "Running %s on step '%s' output (%d chars)...",
            evaluator_name, step_name, len(current_result),
        )

        set_active_stage(step_name, evaluator_name)
        set_active_attachments(inputs_list)
        eval_result: EvalResult = runner(
            current_result=current_result,
            context=context,
            max_edit_rounds=max_edits_per_evaluator,
        )
        all_eval_results.append(eval_result)
        total_edits += eval_result.edit_count

        if eval_result.revised_result is not None:
            current_result = eval_result.revised_result

        LOGGER.info(
            "  %s: %s (edits=%d) — %s",
            evaluator_name,
            eval_result.verdict,
            eval_result.edit_count,
            eval_result.feedback[:120],
        )

    final_verdict: Literal["pass", "fail"] = "pass"
    for r in all_eval_results:
        if r.verdict == "fail":
            final_verdict = "fail"
            break

    return EvalPipelineResult(
        final_verdict=final_verdict,
        final_result=current_result,
        evaluator_results=all_eval_results,
        total_edits=total_edits,
    )
