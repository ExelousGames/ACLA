"""
Suite of specialized evaluator agents for the annotation pipeline.

Each evaluator handles ONE evaluation concern, can edit the result
on failure, and re-evaluates its own edit.  They run in sequence —
each one's output feeds into the next — so the result is progressively
refined.

Architecture: the pipeline maintains a named ``attachment_pool`` of
``PipelineAttachment`` objects.  Every step agent declares which named
attachments it ``consumes`` and which it ``produces``.  Evaluators are
sub-agents of a parent step: they see ONLY the parent's ``parent_inputs``
(the exact set of attachments the parent agent consumed) and the parent's
emitted ``parent_output`` text.  This bounds the evaluator's judgement to
"did the parent agent do its job given what it had access to?" — it
cannot drift into evidence the parent never saw.

Evaluator chain order (each focuses on one concern only):
    1. format_evaluator    — JSON structure of parent_output (skipped for prose)
    2. range_evaluator     — numeric range/index bounds against parent_segment
    3. evidence_evaluator  — parent_output's claims vs. parent_inputs
                             (text + structured + image_set attachments)
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
# LLM function holder — shared with annotation_agent_pipeline
# ---------------------------------------------------------------------------

_eval_llm_holder: Dict[str, Any] = {
    "vlm": None,
    "llm": None,
    "stage_node": "",
    "stage_phase": "",
    "stage_iter": None,
    "stage_total": None,
}
_eval_llm_lock = threading.Lock()


def set_eval_llm(vlm: Optional[Callable], llm: Optional[Callable]) -> None:
    """Register VLM/LLM callables for evaluator use."""
    with _eval_llm_lock:
        _eval_llm_holder["vlm"] = vlm
        _eval_llm_holder["llm"] = llm


def set_active_stage(node_name: str, phase: str) -> None:
    """Record which (node, phase) is about to make a VLM call.

    Read by the pipeline's vlm_generate wrapper to tag prompt-callback
    events with the correct source.  Iteration counter is reset when
    the node changes, and preserved otherwise so evaluator phases
    inherit the iteration set by the pipeline node.
    """
    with _eval_llm_lock:
        if node_name != _eval_llm_holder["stage_node"]:
            _eval_llm_holder["stage_iter"] = None
            _eval_llm_holder["stage_total"] = None
        _eval_llm_holder["stage_node"] = node_name
        _eval_llm_holder["stage_phase"] = phase


def set_active_iteration(iteration: Optional[int], total: Optional[int]) -> None:
    """Set the iteration tag for the current node (e.g. step_describer k/N)."""
    with _eval_llm_lock:
        _eval_llm_holder["stage_iter"] = iteration
        _eval_llm_holder["stage_total"] = total


def get_active_stage() -> Dict[str, Any]:
    """Snapshot the current stage tags for callback dispatch."""
    with _eval_llm_lock:
        return {
            "node_name": _eval_llm_holder["stage_node"],
            "phase": _eval_llm_holder["stage_phase"],
            "iteration": _eval_llm_holder["stage_iter"],
            "total": _eval_llm_holder["stage_total"],
        }


def _call_eval_vlm(
    prompt: str,
    images: Optional[List[bytes]] = None,
) -> str:
    """Dispatch a prompt to the VLM, optionally with images."""
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""
    if images:
        return vlm_fn(prompt, images)
    return vlm_fn(prompt)


# ---------------------------------------------------------------------------
# Shared result models
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
    final_result: str  # the result after all evaluators
    evaluator_results: List[EvalResult]  # per-evaluator detail
    total_edits: int


# ---------------------------------------------------------------------------
# Pipeline attachments — named, typed artifacts in a shared pool
# ---------------------------------------------------------------------------


class PipelineAttachment(BaseModel):
    """A typed, named artifact shared between pipeline agents.

    Agents declare ``consumes`` / ``produces`` lists of attachment names.
    The system keeps a pool keyed by ``name`` and serves attachments to
    agents (and to their child evaluators) by name.

    Fields:
        name           Stable, namespaced handle (e.g. ``"step_describer.3.observations"``).
        kind           ``"text"``, ``"image_set"``, or ``"structured"``.
                       Determines how the attachment is rendered into a
                       prompt and whether its bytes are forwarded to a VLM.
        label          Human-readable display label used as the section
                       header when the attachment is rendered into a prompt.
        content_schema Optional discriminator for ``structured`` content
                       (e.g. ``"step_observation"``, ``"verified_labels"``,
                       ``"parent_segment"``).  Looks up a formatter in
                       ``_STRUCTURED_FORMATTERS``.
        content        ``str`` for text, ``list[bytes]`` for image_set,
                       ``dict`` / ``list`` for structured.
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
    step_id = content.get("step_id")
    description = content.get("description")
    if step_id is not None or description:
        header = f"Step {step_id}" if step_id is not None else "Step"
        if description:
            header = f"{header} — {description}"
        parts.append(header)
    descs = content.get("graph_descriptions") or []
    if descs:
        parts.append("Graphs: " + ", ".join(descs))
    obs = content.get("graph_observations")
    if obs:
        parts.append(str(obs))
    return "\n".join(parts)


def _format_verified_labels(content: Any) -> str:
    if not isinstance(content, list):
        return str(content)
    if not content:
        return "(no labels passed verification)"
    lines: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            lines.append(str(entry))
            continue
        lid = entry.get("label_id", "?")
        name = entry.get("name", "")
        sim = entry.get("similarity")
        desc = entry.get("description", "")
        sim_part = f" | sim={sim:.3f}" if isinstance(sim, (int, float)) else ""
        line = f"- {lid} | {name}{sim_part}"
        if desc:
            line = f"{line} — {desc}"
        lines.append(line)
    return "\n".join(lines)


def _format_parent_segment(content: Any) -> str:
    if not isinstance(content, dict):
        return str(content)
    parts: list[str] = []
    ps = content.get("parent_start")
    pe = content.get("parent_end")
    if ps is not None and pe is not None:
        parts.append(f"Range: [{ps}, {pe}] (length {pe - ps})")
    main_labels = content.get("main_labels") or []
    if main_labels:
        parts.append(f"Main labels: {', '.join(main_labels)}")
    children = content.get("existing_children") or []
    if children:
        child_lines = ["Existing children (avoid overlap):"]
        for c in children:
            cs = c.get("start_index")
            ce = c.get("end_index")
            cls = c.get("labels") or []
            child_lines.append(f"  - [{cs}, {ce}] labels={', '.join(cls)}")
        parts.append("\n".join(child_lines))
    return "\n".join(parts)


def _format_graph_descriptions(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(f"- {c}" for c in content)
    return str(content)


_STRUCTURED_FORMATTERS: Dict[str, Callable[[Any], str]] = {
    "step_observation": _format_step_observation,
    "verified_labels": _format_verified_labels,
    "parent_segment": _format_parent_segment,
    "graph_descriptions": _format_graph_descriptions,
}


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


AttachmentPool = Dict[str, PipelineAttachment]


def merge_pool(
    left: Optional[AttachmentPool],
    right: Optional[AttachmentPool],
) -> AttachmentPool:
    """Reducer for the LangGraph state — later writes win on the same name."""
    return {**(left or {}), **(right or {})}


def pool_put(pool: AttachmentPool, attachment: PipelineAttachment) -> AttachmentPool:
    """Return a new pool with ``attachment`` inserted under its ``name``."""
    return {**pool, attachment.name: attachment}


def pool_get_many(pool: AttachmentPool, names: List[str]) -> List[PipelineAttachment]:
    """Fetch attachments by name, in the requested order, skipping misses."""
    return [pool[n] for n in names if n in pool]


def find_by_schema(
    inputs: List[PipelineAttachment],
    content_schema: str,
) -> List[PipelineAttachment]:
    """Filter attachments by ``content_schema`` (for structured kinds)."""
    return [a for a in inputs if a.content_schema == content_schema]


def collect_image_bytes(inputs: List[PipelineAttachment]) -> List[bytes]:
    """Flatten all image_set attachments' bytes into a single list."""
    out: List[bytes] = []
    for att in inputs:
        if att.kind == "image_set" and isinstance(att.content, list):
            out.extend(att.content)
    return out


def render_inputs_for_prompt(inputs: List[PipelineAttachment]) -> str:
    """Render attachments as ``=== <label> ===`` sections for a VLM prompt.

    - ``text`` attachments are inlined as their string content.
    - ``structured`` attachments use the formatter registered for their
      ``content_schema``, falling back to JSON dump.
    - ``image_set`` attachments emit a placeholder note — their bytes are
      forwarded separately via ``collect_image_bytes``.
    """
    if not inputs:
        return ""
    blocks: list[str] = []
    for att in inputs:
        if att.kind == "image_set":
            n = len(att.content) if isinstance(att.content, list) else 0
            blocks.append(
                f"=== {att.label} ===\n[{n} image(s) attached — see below]"
            )
            continue
        if att.kind == "structured":
            formatter = _STRUCTURED_FORMATTERS.get(
                att.content_schema,
                lambda c: json.dumps(c, indent=2) if c is not None else "",
            )
            body = formatter(att.content)
            blocks.append(f"=== {att.label} ===\n{body}")
            continue
        # text
        body = att.content if isinstance(att.content, str) else str(att.content)
        blocks.append(f"=== {att.label} ===\n{body}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Shared tools
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
    """Attempt to parse and re-serialize broken JSON.

    Handles common VLM mistakes: trailing commas, unquoted keys, etc.
    """
    text = malformed_json.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        parsed = json.loads(fixed)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass

    fixed2 = fixed.replace("'", '"')
    try:
        parsed = json.loads(fixed2)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r'\{[\s\S]*\}', fixed)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not reformat JSON: {malformed_json[:200]}...")


def range_clamp(value: int, min_bound: int, max_bound: int) -> int:
    """Clamp an index to valid parent bounds."""
    return max(min_bound, min(value, max_bound))


# ---------------------------------------------------------------------------
# Evaluator response parsing
# ---------------------------------------------------------------------------


def _parse_eval_response(raw: str) -> dict:
    """Parse a structured evaluator VLM response.

    Expected format from the evaluator VLM call:
        VERDICT: pass/fail
        FEEDBACK: ...
        EDITS: (optional JSON block with edits)
    """
    result = {
        "verdict": "fail",
        "feedback": "",
        "edits": None,
    }

    lines = raw.strip().split("\n")
    feedback_lines: list[str] = []
    edits_block: list[str] = []
    in_edits = False

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.upper().startswith("VERDICT:"):
            verdict_text = line_stripped[len("VERDICT:"):].strip().lower()
            if "pass" in verdict_text:
                result["verdict"] = "pass"
            else:
                result["verdict"] = "fail"
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
    """Recursively walk node; replace the first child equal to old with new."""
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
    """Apply a single old→new replacement from an evaluator's EDITS block.

    Handles two shapes the VLM may emit:
      - both strings: a verbatim text replacement on ``current_result``
      - structured dicts/lists: parse ``current_result`` as JSON, find a
        sub-node deeply equal to ``old`` and swap it for ``new``

    Returns ``(possibly-modified-result, edit_applied)``.  Mismatched or
    unsupported types are skipped silently.
    """
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
    """Check JSON structural correctness of the parent agent's output.

    Reads only ``parent_prompt`` and ``current_result`` — no attachments,
    since structure is judged against the prompt's schema demands.
    """
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
            f"=== Parent Prompt (what was requested) ===\n{parent_prompt[:2000]}\n\n"
            f"=== Step Name ===\n{step_name}\n\n"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass (if JSON structure is valid) or fail (if structure issues exist)\n"
            "FEEDBACK: Describe any structural issues found, or confirm the structure is valid.\n"
            "EDITS: (only if VERDICT is fail) A JSON object with keys:\n"
            '  {"replacements": [{"old": "text to find", "new": "replacement text"}]}\n'
            "  OR if the entire output needs reformatting:\n"
            '  {"full_replacement": "the corrected output"}\n'
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
            if isinstance(edits, dict):
                if "full_replacement" in edits:
                    current_result = edits["full_replacement"]
                    edit_count += 1
                elif "replacements" in edits:
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


def _run_range_evaluator(
    current_result: str,
    context: Dict[str, Any],
    max_edit_rounds: int = 2,
) -> EvalResult:
    """Validate that indices/ranges in the response are within bounds.

    Reads ``parent_segment`` from ``parent_inputs`` (matched by
    ``content_schema``) and falls back to ``parent_start``/``parent_end``
    in context.  Other parent inputs are ignored — bounds are this
    evaluator's only concern.
    """
    parent_prompt = context.get("parent_prompt", "")
    parent_inputs: List[PipelineAttachment] = context.get("parent_inputs", [])

    parent_start = context.get("parent_start", 0)
    parent_end = context.get("parent_end", 0)
    parent_atts = find_by_schema(parent_inputs, "parent_segment")
    if parent_atts:
        ps_content = parent_atts[0].content
        if isinstance(ps_content, dict):
            parent_start = ps_content.get("parent_start", parent_start)
            parent_end = ps_content.get("parent_end", parent_end)

    edit_count = 0
    parsed: Dict[str, Any] = {"verdict": "fail", "feedback": "", "edits": None}

    if _looks_like_json(current_result):
        current_result, deterministic_edits = _deterministic_range_fix(
            current_result, parent_start, parent_end
        )
        edit_count += deterministic_edits

    for attempt in range(1 + max_edit_rounds):
        eval_prompt = (
            "You validate numeric range/index constraints in a pipeline step's output.\n"
            "Compared to the original request, does the response fulfil the "
            "requirement *within MY narrow concern* (index bounds only)?\n\n"
            "Check:\n"
            "- start_index < end_index for every entry\n"
            f"- All indices within [{parent_start}, {parent_end}]\n"
            "- Non-trivial segment lengths (not zero-length)\n\n"
            "Do NOT judge label IDs, label hierarchy, or whether the chosen "
            "ranges semantically match the evidence — focus only on numeric "
            "bounds. On failure, decide the most appropriate fix: clamp, swap, "
            "or remove the offending entry.\n\n"
            f"=== Parent Prompt (what was requested) ===\n{parent_prompt[:2000]}\n\n"
            f"=== Parent Range ===\n[{parent_start}, {parent_end}]\n\n"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass or fail\n"
            "FEEDBACK: Describe any range/index violations found.\n"
            "EDITS: (only if fail) A JSON object with:\n"
            '  {"replacements": [{"old": "...", "new": "..."}]}\n'
        )

        raw_response = _call_eval_vlm(eval_prompt)
        if not raw_response:
            return EvalResult(
                evaluator_name="range_evaluator",
                verdict="pass",
                feedback="VLM unavailable — deterministic checks applied.",
                revised_result=current_result if edit_count > 0 else None,
                edit_count=edit_count,
            )

        parsed = _parse_eval_response(raw_response)

        if parsed["verdict"] == "pass":
            return EvalResult(
                evaluator_name="range_evaluator",
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
        evaluator_name="range_evaluator",
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
    """Check that the response's reasoning is grounded in the parent's inputs.

    The "evidence" is exactly what the parent agent consumed: the
    ``parent_inputs`` attachments (text + structured + image_set) plus
    the ``parent_prompt`` that was sent to it.  This evaluator must NOT
    look at attachments outside the parent's input set — that would let
    it judge claims against evidence the parent never saw.
    """
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
            f"=== Parent Inputs (the evidence the parent agent consumed) ===\n"
            f"{rendered_inputs}\n\n"
            if rendered_inputs else ""
        )
        images_note = (
            f"=== Images ===\n"
            f"{len(images)} image(s) from the parent agent's image_set inputs are "
            f"attached below.  Treat these as primary evidence — the response's "
            f"claims about graphs, traces, indices, or visual features must be "
            f"supported by what is actually visible in these images.\n\n"
            if images else ""
        )
        eval_prompt = (
            "You check whether a pipeline step's response is supported by the "
            "evidence the step itself had access to: the parent prompt and the "
            "parent inputs (named attachments and image sets the parent consumed).\n"
            "Compared to the original request, does the response fulfil the "
            "requirement *within MY narrow concern* (logic and evidence only)?\n\n"
            "Check:\n"
            "- Are claims in the response grounded in something the parent prompt, "
            "parent inputs, or images actually contain?\n"
            "- Does the response's reasoning follow logically from that evidence "
            "(no contradictions, no leaps)?\n"
            "- Are cited indices, labels, observations, or trends actually present "
            "in the evidence — or are some invented?\n"
            "- For visual claims (graph shapes, trace positions, colours, "
            "comparisons), do they match what is visible in the attached images?\n\n"
            "Do NOT judge JSON structure or numeric range bounds — focus only on "
            "whether the response's logic is grounded in the parent's evidence.\n\n"
            f"=== Parent Prompt (what was requested) ===\n{parent_prompt[:2000]}\n\n"
            f"{inputs_section}"
            f"{images_note}"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass or fail\n"
            "FEEDBACK: Describe which claims are supported or unsupported.\n"
            "EDITS: (only if fail) A JSON object with:\n"
            '  {"replacements": [{"old": "unsupported claim", "new": "grounded claim"}]}\n'
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
# Evaluator registry & profiles
# ---------------------------------------------------------------------------

_EVALUATOR_REGISTRY: Dict[str, Callable] = {
    "format_evaluator": _run_format_evaluator,
    "range_evaluator": _run_range_evaluator,
    "evidence_evaluator": _run_evidence_evaluator,
}

EVALUATOR_CHAIN: List[str] = [
    "format_evaluator",
    "range_evaluator",
    "evidence_evaluator",
]

STEP_EVALUATOR_PROFILES: Dict[str, List[str]] = {
    "planner": [
        "format_evaluator",
    ],
    "step_describer": [
        "evidence_evaluator",
    ],
    "proposal_synthesizer": [
        "format_evaluator",
        "range_evaluator",
        "evidence_evaluator",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_json(text: str) -> bool:
    """Heuristic: does the text look like it contains JSON?"""
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    if "```json" in stripped:
        return True
    return False


def _deterministic_range_fix(
    text: str,
    parent_start: int,
    parent_end: int,
) -> tuple[str, int]:
    """Apply deterministic range fixes to JSON output without VLM calls.

    Range-only: clamps indices to ``[parent_start, parent_end]``, swaps
    inverted ranges, and ensures non-zero length.  Does NOT validate
    label IDs — that is not this evaluator's concern.

    Returns (fixed_text, number_of_edits).
    """
    edit_count = 0

    try:
        json_text = text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif json_text.startswith("```"):
            json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]

        parsed = json.loads(json_text.strip())
    except (json.JSONDecodeError, IndexError, ValueError):
        return text, 0

    if not isinstance(parsed, dict):
        return text, 0

    labels_arr = parsed.get("labels", [])
    if not isinstance(labels_arr, list):
        return text, 0

    modified = False
    cleaned_labels = []

    for entry in labels_arr:
        if not isinstance(entry, dict):
            continue

        si = entry.get("start_index")
        ei = entry.get("end_index")

        if isinstance(si, (int, float)) and isinstance(ei, (int, float)):
            si_int, ei_int = int(si), int(ei)

            if si_int >= ei_int:
                si_int, ei_int = ei_int, si_int
                if si_int == ei_int:
                    ei_int = min(si_int + 10, parent_end)
                entry["start_index"] = si_int
                entry["end_index"] = ei_int
                modified = True
                edit_count += 1

            clamped_start = range_clamp(si_int, parent_start, parent_end)
            clamped_end = range_clamp(ei_int, parent_start, parent_end)
            if clamped_start != si_int or clamped_end != ei_int:
                entry["start_index"] = clamped_start
                entry["end_index"] = clamped_end
                modified = True
                edit_count += 1

            if entry["start_index"] >= entry["end_index"]:
                entry["end_index"] = min(entry["start_index"] + 10, parent_end)
                modified = True
                edit_count += 1

        cleaned_labels.append(entry)

    if modified:
        parsed["labels"] = cleaned_labels
        new_json = json.dumps(parsed, indent=2)
        if "```json" in text:
            return f"```json\n{new_json}\n```", edit_count
        return new_json, edit_count

    return text, 0


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
    """Run the evaluator suite for a parent agent.

    Evaluators are sub-agents of the parent step.  They see ONLY the
    parent's own context — ``parent_inputs`` (the exact set of attachments
    the parent agent consumed), ``parent_prompt`` (the prompt the parent
    sent to its LLM/VLM), and ``parent_output_text`` (the parent's raw
    text output).  This bounds each evaluator to judging the parent
    against the same evidence the parent had.

    Args:
        parent_prompt: The prompt the parent agent sent to its LLM/VLM.
        parent_output_text: The parent agent's raw text output.
        parent_inputs: Named attachments the parent declared as ``consumes``.
                       Includes any image_set attachments the parent saw —
                       evaluators will forward those bytes to the eval VLM.
        step_name: The parent step's name (e.g. ``"planner"``,
                   ``"step_describer"``, ``"proposal_synthesizer"``).
                   Used to select the evaluator profile.
        parent_start, parent_end: Fallback parent bounds when no
                       ``parent_segment`` attachment is in ``parent_inputs``.
        evaluators: Optional explicit evaluator-name list (overrides the
                    profile for this step).
        max_edits_per_evaluator: Maximum edit rounds per evaluator (default 2).

    Returns:
        EvalPipelineResult with the final (possibly corrected) output text.
    """
    inputs_list: List[PipelineAttachment] = list(parent_inputs or [])

    if evaluators:
        chain = [name for name in EVALUATOR_CHAIN if name in evaluators]
    else:
        chain = STEP_EVALUATOR_PROFILES.get(step_name, EVALUATOR_CHAIN)

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
