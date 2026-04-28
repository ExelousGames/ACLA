"""
Suite of specialized evaluator agents for the annotation pipeline.

Each evaluator handles one evaluation concern, can edit the result
on failure, and re-evaluates its own edit.  They run in sequence —
each one's output feeds into the next — so the result is progressively
refined.

IMPORTANT: This suite is called after EVERY pipeline step, with a
step-appropriate subset of evaluators.  This ensures errors are caught
at the point of origin rather than compounding downstream.

Evaluator chain order:
    1. format_evaluator   — fix broken JSON / missing sections
    2. range_evaluator    — fix constraint violations
    3. intent_evaluator   — ensure prompt compliance
    4. evidence_evaluator — cross-check against graph images
    5. consistency_evaluator — final coherence pass
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel

from app.models.segment_models import LABEL_MAPPING, LABEL_CATEGORIES

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM function holder — shared with annotation_agent_pipeline
# ---------------------------------------------------------------------------

_eval_llm_holder: Dict[str, Optional[Callable]] = {"vlm": None, "llm": None}
_eval_llm_lock = threading.Lock()


def set_eval_llm(vlm: Optional[Callable], llm: Optional[Callable]) -> None:
    """Register VLM/LLM callables for evaluator use."""
    with _eval_llm_lock:
        _eval_llm_holder["vlm"] = vlm
        _eval_llm_holder["llm"] = llm


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
    # Strip markdown fences if present
    text = malformed_json.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        parsed = json.loads(fixed)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass

    # Fix single quotes → double quotes
    fixed2 = fixed.replace("'", '"')
    try:
        parsed = json.loads(fixed2)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass

    # Last resort: find { ... } block
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

        # Check for verdict line
        if line_stripped.upper().startswith("VERDICT:"):
            verdict_text = line_stripped[len("VERDICT:"):].strip().lower()
            if "pass" in verdict_text:
                result["verdict"] = "pass"
            else:
                result["verdict"] = "fail"
            continue

        # Check for edits section start
        if line_stripped.upper().startswith("EDITS:"):
            in_edits = True
            remainder = line_stripped[len("EDITS:"):].strip()
            if remainder:
                edits_block.append(remainder)
            continue

        if in_edits:
            edits_block.append(line)
        else:
            # Check for FEEDBACK: prefix
            if line_stripped.upper().startswith("FEEDBACK:"):
                feedback_lines.append(line_stripped[len("FEEDBACK:"):].strip())
            else:
                feedback_lines.append(line)

    result["feedback"] = "\n".join(feedback_lines).strip()

    # Try to parse edits block as JSON
    if edits_block:
        edits_text = "\n".join(edits_block).strip()
        # Strip markdown fences
        if edits_text.startswith("```"):
            edits_text = re.sub(r'^```\w*\n?', '', edits_text)
            edits_text = re.sub(r'\n?```$', '', edits_text)
        try:
            result["edits"] = json.loads(edits_text)
        except json.JSONDecodeError:
            # Edits were not parseable — include in feedback
            result["feedback"] += f"\n[Unparseable edit suggestion: {edits_text[:200]}]"

    return result


# ---------------------------------------------------------------------------
# Individual evaluator runners
# ---------------------------------------------------------------------------


def _run_format_evaluator(
    current_result: str,
    context: Dict[str, Any],
    max_edit_rounds: int = 2,
) -> EvalResult:
    """Check structural/format correctness of the step output.

    Validates JSON parsability, required keys, correct value types,
    proper nesting, no truncation.  For prose output, checks that all
    requested sections are present.
    """
    original_prompt = context.get("original_prompt", "")
    step_name = context.get("step_name", "")
    edit_count = 0

    for attempt in range(1 + max_edit_rounds):
        eval_prompt = (
            "You evaluate whether a pipeline step's output is well-formed structurally.\n\n"
            "Check:\n"
            "- If JSON was requested: parsability, required keys present, correct value types, "
            "proper nesting, no truncation.\n"
            "- If prose was requested: all requested sections are present and complete.\n\n"
            "Do NOT judge whether values are correct — only whether the format is valid.\n\n"
            f"=== Original Prompt (what was requested) ===\n{original_prompt[:2000]}\n\n"
            f"=== Step Name ===\n{step_name}\n\n"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass (if format is valid) or fail (if format issues exist)\n"
            "FEEDBACK: Describe any format issues found, or confirm the format is valid.\n"
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

        # Try to apply edits
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
                        if old and old in current_result:
                            current_result = current_result.replace(old, new, 1)
                            edit_count += 1
            # Also try json_reformatter for JSON outputs
            if _looks_like_json(current_result):
                try:
                    current_result = json_reformatter(current_result)
                    edit_count += 1
                except ValueError:
                    pass
        else:
            # No edits or max rounds reached
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
    """Validate numeric constraints: indices within parent bounds,
    start < end, valid label IDs, correct hierarchy."""
    parent_start = context.get("parent_start", 0)
    parent_end = context.get("parent_end", 0)
    label_mapping = context.get("label_mapping", {})
    edit_count = 0

    # Quick deterministic pre-check for JSON outputs
    if _looks_like_json(current_result):
        current_result, deterministic_edits = _deterministic_range_fix(
            current_result, parent_start, parent_end, label_mapping
        )
        edit_count += deterministic_edits

    for attempt in range(1 + max_edit_rounds):
        eval_prompt = (
            "You validate numeric constraints in a pipeline step's output.\n\n"
            "Check:\n"
            "- start_index < end_index for every entry\n"
            f"- All indices within [{parent_start}, {parent_end}]\n"
            "- Non-trivial segment lengths (not zero-length)\n"
            "- Valid label IDs from the known label mapping\n"
            "- Correct label hierarchy (sub-labels have parents included)\n\n"
            "On failure, decide the most appropriate fix: clamp, swap, or remove.\n\n"
            f"=== Parent Range ===\n[{parent_start}, {parent_end}]\n\n"
            f"=== Known Label IDs ===\n{json.dumps(list(label_mapping.keys())[:50])}\n\n"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass or fail\n"
            "FEEDBACK: Describe any constraint violations found.\n"
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
                    if old and old in current_result:
                        current_result = current_result.replace(old, new, 1)
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


def _run_intent_evaluator(
    current_result: str,
    context: Dict[str, Any],
    max_edit_rounds: int = 2,
) -> EvalResult:
    """Check whether the output faithfully follows the original prompt's
    instructions: all requested elements present, nothing prohibited included,
    correct structure."""
    original_prompt = context.get("original_prompt", "")
    step_name = context.get("step_name", "")
    edit_count = 0

    for attempt in range(1 + max_edit_rounds):
        eval_prompt = (
            "You check whether a step's output faithfully follows the original prompt's "
            "instructions. Compare instruction by instruction:\n"
            "- Did the output do everything the prompt asked?\n"
            "- Did it avoid everything the prompt prohibited?\n"
            "- Does the output's structure match what was requested?\n\n"
            "Do NOT judge factual accuracy — only whether the output addresses the prompt.\n\n"
            f"=== Original Prompt ===\n{original_prompt[:3000]}\n\n"
            f"=== Step: {step_name} ===\n\n"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass or fail\n"
            "FEEDBACK: Which instructions were followed and which were missed.\n"
            "EDITS: (only if fail) A JSON object with:\n"
            '  {"replacements": [{"old": "...", "new": "..."}]}\n'
            "  OR\n"
            '  {"full_replacement": "corrected output"}\n'
        )

        raw_response = _call_eval_vlm(eval_prompt)
        if not raw_response:
            return EvalResult(
                evaluator_name="intent_evaluator",
                verdict="pass",
                feedback="VLM unavailable — skipping intent check.",
            )

        parsed = _parse_eval_response(raw_response)

        if parsed["verdict"] == "pass":
            return EvalResult(
                evaluator_name="intent_evaluator",
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
                        if old and old in current_result:
                            current_result = current_result.replace(old, new, 1)
                            edit_count += 1
        else:
            break

    return EvalResult(
        evaluator_name="intent_evaluator",
        verdict="fail",
        feedback=parsed["feedback"],
        revised_result=current_result if edit_count > 0 else None,
        edit_count=edit_count,
    )


def _run_evidence_evaluator(
    current_result: str,
    context: Dict[str, Any],
    graph_images: Optional[List[bytes]] = None,
    max_edit_rounds: int = 2,
) -> EvalResult:
    """Cross-reference the output's claims against graph images.

    This is the only evaluator that receives graph images.  It checks
    whether numerical claims (indices, values, magnitudes) match what
    the graphs show.
    """
    original_prompt = context.get("original_prompt", "")
    parent_start = context.get("parent_start", 0)
    parent_end = context.get("parent_end", 0)
    edit_count = 0

    if not graph_images:
        return EvalResult(
            evaluator_name="evidence_evaluator",
            verdict="pass",
            feedback="No graph images available — skipping evidence check.",
        )

    for attempt in range(1 + max_edit_rounds):
        eval_prompt = (
            "You cross-reference a step's output against graph images and observations.\n\n"
            "Check:\n"
            "- Do numerical claims (indices, values, magnitudes) match what the graphs show?\n"
            "- Do descriptions of trends (rises, drops, plateaus) match the visual evidence?\n"
            "- Are cited graph features actually visible in the images?\n\n"
            "You will receive graph images — examine them carefully.\n\n"
            f"=== Parent Range ===\n[{parent_start}, {parent_end}]\n\n"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass or fail\n"
            "FEEDBACK: Describe which claims match or contradict the visual evidence.\n"
            "EDITS: (only if fail) A JSON object with:\n"
            '  {"replacements": [{"old": "incorrect claim", "new": "corrected claim"}]}\n'
        )

        raw_response = _call_eval_vlm(eval_prompt, images=graph_images)
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
                    if old and old in current_result:
                        current_result = current_result.replace(old, new, 1)
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


def _run_consistency_evaluator(
    current_result: str,
    context: Dict[str, Any],
    max_edit_rounds: int = 2,
) -> EvalResult:
    """Check internal consistency: no contradictions between prose and JSON,
    no conflicting claims across sections, label-reasoning alignment."""
    step_name = context.get("step_name", "")
    edit_count = 0

    for attempt in range(1 + max_edit_rounds):
        eval_prompt = (
            "You check internal consistency of a step's output. Look for:\n"
            "- Contradictions between prose descriptions and JSON values\n"
            "- Conflicting claims across different sections\n"
            "- Label-reasoning mismatches (e.g., reasoning says 'braking' but label says 'oversteer')\n"
            "- Overlapping ranges with conflicting descriptions\n\n"
            "Do NOT judge external accuracy — only whether the output agrees with itself.\n\n"
            f"=== Step: {step_name} ===\n\n"
            f"=== Output to Evaluate ===\n{current_result}\n\n"
            "=== Instructions ===\n"
            "Respond in this exact format:\n"
            "VERDICT: pass or fail\n"
            "FEEDBACK: Describe any internal contradictions or confirm coherence.\n"
            "EDITS: (only if fail) A JSON object with:\n"
            '  {"replacements": [{"old": "contradictory text", "new": "consistent text"}]}\n'
        )

        raw_response = _call_eval_vlm(eval_prompt)
        if not raw_response:
            return EvalResult(
                evaluator_name="consistency_evaluator",
                verdict="pass",
                feedback="VLM unavailable — skipping consistency check.",
            )

        parsed = _parse_eval_response(raw_response)

        if parsed["verdict"] == "pass":
            return EvalResult(
                evaluator_name="consistency_evaluator",
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
                    if old and old in current_result:
                        current_result = current_result.replace(old, new, 1)
                        edit_count += 1
        else:
            break

    return EvalResult(
        evaluator_name="consistency_evaluator",
        verdict="fail",
        feedback=parsed["feedback"],
        revised_result=current_result if edit_count > 0 else None,
        edit_count=edit_count,
    )


# ---------------------------------------------------------------------------
# Evaluator registry & profiles
# ---------------------------------------------------------------------------

# Evaluator name → runner function
_EVALUATOR_REGISTRY: Dict[str, Callable] = {
    "format_evaluator": _run_format_evaluator,
    "range_evaluator": _run_range_evaluator,
    "intent_evaluator": _run_intent_evaluator,
    "evidence_evaluator": _run_evidence_evaluator,
    "consistency_evaluator": _run_consistency_evaluator,
}

# Ordered chain — defines the sequence evaluators run in
EVALUATOR_CHAIN: List[str] = [
    "format_evaluator",       # 1. fix structure first
    "range_evaluator",        # 2. fix constraint violations
    "intent_evaluator",       # 3. ensure prompt compliance
    "evidence_evaluator",     # 4. verify against graphs (needs images)
    "consistency_evaluator",  # 5. final coherence check
]

# Per-step evaluator profiles: which evaluators run for which step
STEP_EVALUATOR_PROFILES: Dict[str, List[str]] = {
    "planner": [
        "format_evaluator",
        "intent_evaluator",
    ],
    "step_reasoner": [
        "format_evaluator",
        "intent_evaluator",
        "evidence_evaluator",
    ],
    "label_verifier": [
        "format_evaluator",
        "range_evaluator",
        "intent_evaluator",
        "consistency_evaluator",
    ],
    "proposal_synthesizer": [
        "format_evaluator",
        "range_evaluator",
        "intent_evaluator",
        "evidence_evaluator",
        "consistency_evaluator",
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
    label_mapping: dict,
) -> tuple[str, int]:
    """Apply deterministic range fixes to JSON output without VLM calls.

    Returns (fixed_text, number_of_edits).
    """
    edit_count = 0

    # Try to extract and parse JSON
    try:
        # Strip markdown fences
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

        lid = entry.get("label_id")
        # Remove entries with unknown label IDs
        if lid and lid not in label_mapping:
            modified = True
            edit_count += 1
            continue

        si = entry.get("start_index")
        ei = entry.get("end_index")

        if isinstance(si, (int, float)) and isinstance(ei, (int, float)):
            si_int, ei_int = int(si), int(ei)

            # Swap inverted ranges
            if si_int >= ei_int:
                si_int, ei_int = ei_int, si_int
                # Ensure non-zero length after swap
                if si_int == ei_int:
                    ei_int = min(si_int + 10, parent_end)
                entry["start_index"] = si_int
                entry["end_index"] = ei_int
                modified = True
                edit_count += 1

            # Clamp to parent bounds
            clamped_start = range_clamp(si_int, parent_start, parent_end)
            clamped_end = range_clamp(ei_int, parent_start, parent_end)
            if clamped_start != si_int or clamped_end != ei_int:
                entry["start_index"] = clamped_start
                entry["end_index"] = clamped_end
                modified = True
                edit_count += 1

            # Ensure start < end after clamping
            if entry["start_index"] >= entry["end_index"]:
                entry["end_index"] = min(entry["start_index"] + 10, parent_end)
                modified = True
                edit_count += 1

        cleaned_labels.append(entry)

    if modified:
        parsed["labels"] = cleaned_labels
        # Reconstruct the text preserving markdown fences if present
        new_json = json.dumps(parsed, indent=2)
        if "```json" in text:
            return f"```json\n{new_json}\n```", edit_count
        return new_json, edit_count

    return text, 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_evaluator_suite(
    original_prompt: str,
    step_output: str,
    step_name: str,
    parent_start: int,
    parent_end: int,
    graph_images: Optional[List[bytes]] = None,
    label_mapping: Optional[dict] = None,
    evaluators: Optional[List[str]] = None,
    max_edits_per_evaluator: int = 2,
) -> EvalPipelineResult:
    """Run the evaluator suite appropriate for the given step.

    This function is called after EVERY pipeline step.  It selects the
    correct evaluator subset based on step_name (or uses an explicit
    evaluators list if provided), runs them in sequence, and returns
    the progressively refined result.

    Args:
        original_prompt: The prompt that was sent to the LLM for this step.
        step_output: The raw LLM output to evaluate.
        step_name: The pipeline step that produced this output
                   (e.g., "planner", "step_reasoner", "proposal_synthesizer").
                   Used to select the appropriate evaluator subset from
                   STEP_EVALUATOR_PROFILES.
        parent_start: Parent segment start index.
        parent_end: Parent segment end index.
        graph_images: Optional list of PNG image bytes for evidence evaluation.
        label_mapping: Optional label ID → name mapping for validation.
        evaluators: Optional explicit list of evaluator names to override
                    the default profile for this step.
        max_edits_per_evaluator: Maximum edit rounds per evaluator (default 2).

    Returns:
        EvalPipelineResult with the final (possibly corrected) output and
        per-evaluator details.
    """
    effective_label_mapping = label_mapping or LABEL_MAPPING

    # Select evaluator subset for this step
    if evaluators:
        chain = [name for name in EVALUATOR_CHAIN if name in evaluators]
    else:
        chain = STEP_EVALUATOR_PROFILES.get(step_name, EVALUATOR_CHAIN)

    context = {
        "original_prompt": original_prompt,
        "step_name": step_name,
        "parent_start": parent_start,
        "parent_end": parent_end,
        "label_mapping": effective_label_mapping,
    }

    current_result = step_output
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

        # Build kwargs for this evaluator
        kwargs: Dict[str, Any] = {
            "current_result": current_result,
            "context": context,
            "max_edit_rounds": max_edits_per_evaluator,
        }

        # Evidence evaluator gets graph images
        if evaluator_name == "evidence_evaluator":
            kwargs["graph_images"] = graph_images

        eval_result: EvalResult = runner(**kwargs)
        all_eval_results.append(eval_result)
        total_edits += eval_result.edit_count

        # Apply revised result if the evaluator edited it
        if eval_result.revised_result is not None:
            current_result = eval_result.revised_result

        LOGGER.info(
            "  %s: %s (edits=%d) — %s",
            evaluator_name,
            eval_result.verdict,
            eval_result.edit_count,
            eval_result.feedback[:120],
        )

    # Final verdict: fail if any evaluator still says fail after its edits
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
