# Evaluator Agent Suite — Implementation Plan

## Overview

Instead of one monolithic evaluator, the pipeline uses a **set of specialized evaluator agents**, each responsible for exactly one evaluation concern. They live in a new file `step_evaluator_agents.py` and are built on LangChain DeepAgents. Each agent receives the step's output, evaluates it from its specific angle, and edits the result directly on failure. The agents run in sequence — each one's edited output feeds into the next — so the result is progressively refined.

**Critical design principle: the evaluator suite runs after EVERY pipeline step, not just at the end.** Each node in the graph (planner, step_reasoner, label_verifier, proposal_synthesizer) calls `run_evaluator_suite()` on its own output before writing to state. This means errors are caught and corrected immediately at the point of origin, rather than accumulating and compounding through downstream steps. A different subset of evaluators is selected per step based on what that step produces.

## The Evaluator Agents

### 1. `format_evaluator` — Structural / Format Correction

**Concern:** Is the output well-formed for what was requested?

This agent checks whether the output matches the structural format the original prompt asked for. If the prompt requested JSON with a `"labels"` array, it checks parsability, required keys (`label_id`, `start_index`, `end_index`, `reasoning`), correct value types (strings vs. integers), and proper nesting. If the prompt requested prose, it checks for completeness (all requested sections present, no truncation).

**When it edits:** Malformed JSON (missing quotes, trailing commas, unclosed braces), missing required keys, wrong value types (string `"1400"` instead of integer `1400`), truncated output. These are mechanical fixes — the agent rewrites the broken syntax while preserving the semantic content.

**Tools:** `result_editor` (string replacement for targeted fixes), `json_reformatter` (parses broken JSON, fixes structural issues, re-serializes).

**Does NOT check:** Whether the values themselves are correct — that's for other evaluators.

---

### 2. `range_evaluator` — Boundary & Constraint Validation

**Concern:** Do all numeric values respect the hard constraints?

Checks that `start_index < end_index` for every label entry, all indices fall within `[parent_start, parent_end]`, and segment lengths are non-trivial (not zero-length or single-point). Also validates that label IDs exist in the `LABEL_MAPPING` and that the label hierarchy is satisfied (sub-labels have their parents included).

**When it edits:** Clamps out-of-range indices to parent bounds, swaps start/end when inverted, removes entries with unknown label IDs, auto-inserts missing parent labels. These are deterministic corrections — no LLM judgment needed for the fix itself, but the DeepAgent's LLM decides *which* fix is most appropriate when multiple options exist (e.g., should an out-of-range end_index be clamped to parent_end, or should the whole entry be removed?).

**Tools:** `result_editor`, `range_clamp` (adjusts indices to valid bounds).

**Does NOT check:** Whether the boundaries are *accurate* relative to the evidence — that's for the evidence evaluator.

---

### 3. `intent_evaluator` — Prompt Intent Faithfulness

**Concern:** Does the output actually do what the original prompt asked?

Compares the output against the original prompt instruction by instruction. If the prompt said "determine separate start_index and end_index for every label," did the output do that or did it give a single range for all labels? If the prompt said "reference the graph observations from the step reasoners," does the reasoning actually cite specific observations? If the prompt said "find a DIFFERENT region" (because existing children were listed), does the proposal overlap with an existing child?

**When it edits:** Adds missing sections the prompt required, removes content the prompt explicitly prohibited, restructures the output to match the requested format. For example, if the prompt asked for per-label boundaries but the output gave one global range, the intent evaluator splits it into per-label entries using the reasoning context.

**Tools:** `result_editor`.

**Does NOT check:** Whether the *content* is factually accurate — only whether the output's structure and coverage matches what was asked for.

---

### 4. `evidence_evaluator` — Graph & Data Consistency

**Concern:** Do the claims in the output match the actual graph images and observations?

This is the only evaluator that receives **graph images**. It cross-references the output's claims against the visual evidence: if the output says "braking begins at index 1450," does the brake graph actually show onset at that point? If the output describes "a sharp speed drop from 120 to 45 km/h," does the speed graph show that magnitude? If the step reasoner's prose says "the throttle trace is flat," is it actually flat in the image?

**When it edits:** Corrects specific numerical claims (indices, values, magnitudes) to match what's visible in the graphs. Removes or rewrites descriptions that contradict the visual evidence. This is the most judgment-heavy evaluator — it's doing visual QA.

**Tools:** `result_editor`.

**Requires:** Graph images passed to the DeepAgent so the VLM can see them.

---

### 5. `consistency_evaluator` — Internal Logic & Coherence

**Concern:** Is the output internally consistent and logically sound?

Checks that the reasoning doesn't contradict itself. If one paragraph says "the anomaly starts at index 1400" but the JSON entry says `start_index: 1500`, that's an internal contradiction. If the reasoning says "this is a braking event" but the proposed label is `oversteer`, that's a logic error. If two label entries have overlapping ranges with conflicting descriptions, that's incoherent.

**When it edits:** Resolves contradictions by choosing the version that has more supporting detail, aligns JSON values with prose descriptions (or vice versa), removes logically impossible combinations.

**Tools:** `result_editor`.

**Does NOT check:** Whether the reasoning is *correct* relative to external evidence — only whether the output agrees with itself.

---

## Architecture

### File: `step_evaluator_agents.py`

```python
"""
Suite of specialized DeepAgents evaluators.

Each evaluator handles one evaluation concern, can edit the result
on failure, and re-evaluates its own edit. They run in sequence —
each one's output feeds into the next.

IMPORTANT: This suite is called after EVERY pipeline step, with a
step-appropriate subset of evaluators. This ensures errors are caught
at the point of origin rather than compounding downstream.
"""

from deepagents import create_deep_agent, SubAgent
from pydantic import BaseModel
from typing import List, Literal, Optional


# ── Shared result model ──────────────────────────────────────────

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
    final_result: str                     # the result after all evaluators
    evaluator_results: List[EvalResult]   # per-evaluator detail
    total_edits: int


# ── Shared tools ─────────────────────────────────────────────────

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
    """Attempt to parse and re-serialize broken JSON."""
    import json, re
    # Try common fixes: trailing commas, unquoted keys, etc.
    fixed = re.sub(r',\s*([}\]])', r'\1', malformed_json)
    parsed = json.loads(fixed)
    return json.dumps(parsed, indent=2)


def range_clamp(
    value: int,
    min_bound: int,
    max_bound: int,
) -> int:
    """Clamp an index to valid parent bounds."""
    return max(min_bound, min(value, max_bound))


# ── Evaluator SubAgent definitions ───────────────────────────────

FORMAT_EVALUATOR: SubAgent = {
    "name": "format_evaluator",
    "system_prompt": (
        "You evaluate whether a pipeline step's output is well-formed structurally. "
        "Check: JSON parsability, required keys present, correct value types, "
        "proper nesting, no truncation. If the prompt asked for prose, check "
        "all requested sections are present.\n\n"
        "Do NOT judge whether values are correct — only whether the format is valid.\n\n"
        "On failure, use result_editor or json_reformatter to fix structural issues. "
        "Then re-evaluate. Change only what is broken."
    ),
    "tools": [result_editor, json_reformatter],
    "output_schema": EvalResult,
    "max_tool_calls": 3,
}

RANGE_EVALUATOR: SubAgent = {
    "name": "range_evaluator",
    "system_prompt": (
        "You validate numeric constraints in a pipeline step's output. "
        "Check: start_index < end_index for every entry, all indices within "
        "[parent_start, parent_end], non-trivial segment lengths, valid label IDs "
        "from LABEL_MAPPING, correct label hierarchy (sub-labels have parents).\n\n"
        "On failure, use result_editor and range_clamp to fix constraint violations. "
        "Decide the most appropriate fix: clamp, swap, or remove the entry. "
        "Then re-evaluate."
    ),
    "tools": [result_editor, range_clamp],
    "output_schema": EvalResult,
    "max_tool_calls": 3,
}

INTENT_EVALUATOR: SubAgent = {
    "name": "intent_evaluator",
    "system_prompt": (
        "You check whether a step's output faithfully follows the original prompt's "
        "instructions. Compare instruction by instruction: did the output do everything "
        "the prompt asked? Did it avoid everything the prompt prohibited? Does the "
        "output's structure match what was requested?\n\n"
        "Do NOT judge factual accuracy — only whether the output addresses the prompt.\n\n"
        "On failure, use result_editor to add missing sections, remove prohibited "
        "content, or restructure to match the requested format. Then re-evaluate."
    ),
    "tools": [result_editor],
    "output_schema": EvalResult,
    "max_tool_calls": 3,
}

EVIDENCE_EVALUATOR: SubAgent = {
    "name": "evidence_evaluator",
    "system_prompt": (
        "You cross-reference a step's output against graph images and observations. "
        "Check: do numerical claims (indices, values, magnitudes) match what the graphs "
        "show? Do descriptions of trends (rises, drops, plateaus) match the visual? "
        "Are cited graph features actually visible?\n\n"
        "You will receive graph images — examine them carefully.\n\n"
        "On failure, use result_editor to correct specific numerical claims, rewrite "
        "inaccurate descriptions, or flag unsupported assertions. Then re-evaluate."
    ),
    "tools": [result_editor],
    "output_schema": EvalResult,
    "max_tool_calls": 4,  # needs more room for visual verification
}

CONSISTENCY_EVALUATOR: SubAgent = {
    "name": "consistency_evaluator",
    "system_prompt": (
        "You check internal consistency of a step's output. Look for: "
        "contradictions between prose and JSON values, conflicting claims across "
        "sections, label-reasoning mismatches (e.g., reasoning says 'braking' but "
        "label says 'oversteer'), overlapping ranges with conflicting descriptions.\n\n"
        "Do NOT judge external accuracy — only whether the output agrees with itself.\n\n"
        "On failure, resolve contradictions by keeping the version with more supporting "
        "detail. Use result_editor to align conflicting sections. Then re-evaluate."
    ),
    "tools": [result_editor],
    "output_schema": EvalResult,
    "max_tool_calls": 3,
}


# ── Evaluator ordering ──────────────────────────────────────────

EVALUATOR_CHAIN = [
    FORMAT_EVALUATOR,       # 1. fix structure first
    RANGE_EVALUATOR,        # 2. fix constraint violations
    INTENT_EVALUATOR,       # 3. ensure prompt compliance
    EVIDENCE_EVALUATOR,     # 4. verify against graphs (needs images)
    CONSISTENCY_EVALUATOR,  # 5. final coherence check
]


# ── Per-step evaluator profiles ─────────────────────────────────
# Each pipeline step uses a specific subset of evaluators appropriate
# to the type of output it produces.

STEP_EVALUATOR_PROFILES = {
    "planner": [
        FORMAT_EVALUATOR,
        INTENT_EVALUATOR,
    ],
    "step_reasoner": [
        FORMAT_EVALUATOR,
        INTENT_EVALUATOR,
        EVIDENCE_EVALUATOR,
    ],
    "label_verifier": [
        FORMAT_EVALUATOR,
        RANGE_EVALUATOR,
        INTENT_EVALUATOR,
        CONSISTENCY_EVALUATOR,
    ],
    "proposal_synthesizer": [
        FORMAT_EVALUATOR,
        RANGE_EVALUATOR,
        INTENT_EVALUATOR,
        EVIDENCE_EVALUATOR,
        CONSISTENCY_EVALUATOR,
    ],
}


# ── Runner ───────────────────────────────────────────────────────

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

    This function is called after EVERY pipeline step. It selects the
    correct evaluator subset based on step_name (or uses an explicit
    evaluators list if provided), runs them in sequence, and returns
    the progressively refined result.

    Args:
        step_name: The pipeline step that produced this output
                   (e.g., "planner", "step_reasoner", "proposal_synthesizer").
                   Used to select the appropriate evaluator subset from
                   STEP_EVALUATOR_PROFILES.
        evaluators: Optional explicit list of evaluator names to override
                    the default profile for this step.
    """
    # Select evaluator subset for this step
    if evaluators:
        chain = [e for e in EVALUATOR_CHAIN if e["name"] in evaluators]
    else:
        chain = STEP_EVALUATOR_PROFILES.get(step_name, EVALUATOR_CHAIN)

    current_result = step_output
    all_eval_results: List[EvalResult] = []
    total_edits = 0

    for subagent_def in chain:
        agent = create_deep_agent(
            subagent_def,
            # pass images only to evidence_evaluator
            images=graph_images if subagent_def["name"] == "evidence_evaluator" else None,
            # pass constraints as context
            context={
                "original_prompt": original_prompt,
                "step_name": step_name,
                "parent_start": parent_start,
                "parent_end": parent_end,
                "label_mapping": label_mapping or {},
            },
        )

        eval_result: EvalResult = agent.run(current_result)

        all_eval_results.append(eval_result)
        total_edits += eval_result.edit_count

        if eval_result.revised_result is not None:
            current_result = eval_result.revised_result

    # Final verdict: fail if any evaluator still says fail after its edits
    final_verdict = "pass"
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
```

## Evaluator Chain — Why This Order

The evaluators run in a specific sequence because each one's fixes create preconditions for the next:

```
step output
    │
    ▼
┌─────────────────┐
│ format_evaluator │  Fix broken JSON / missing sections so downstream
│                  │  evaluators can actually parse the content.
└────────┬────────┘
         ▼
┌─────────────────┐
│ range_evaluator  │  Fix constraint violations (out-of-bounds, inverted
│                  │  ranges) so intent/evidence checks aren't confused
└────────┬────────┘  by impossible values.
         ▼
┌─────────────────┐
│ intent_evaluator │  Ensure the output covers what the prompt asked for.
│                  │  Now that format and ranges are valid, this can focus
└────────┬────────┘  purely on prompt compliance.
         ▼
┌─────────────────────┐
│ evidence_evaluator   │  Cross-check claims against graph images.
│ (receives images)    │  Now that the output is well-formed and complete,
└────────┬────────────┘  this can trust the structure and focus on accuracy.
         ▼
┌───────────────────────┐
│ consistency_evaluator  │  Final coherence pass — catch any contradictions
│                        │  introduced by earlier edits.
└────────┬──────────────┘
         ▼
    evaluated result
```

If an earlier evaluator's edit accidentally introduces a new issue (e.g., range_evaluator clamps an index, creating a contradiction with the prose), consistency_evaluator catches it at the end.

## Pipeline Flow — Evaluator After Every Step

The evaluator suite is invoked after **every** pipeline step. Each step's node function calls `run_evaluator_suite()` on its own output before writing to state. This is the core architectural invariant:

```
                    ┌──────────────────────────────────────────────────┐
                    │  EVERY step follows the same pattern:            │
                    │  1. Generate output (LLM call)                   │
                    │  2. Run evaluator suite (step-appropriate subset)│
                    │  3. Write evaluated result to state              │
                    └──────────────────────────────────────────────────┘

 ┌─────────┐     ┌──────────────────┐     ┌───────────────┐     ┌────────────────────────┐
 │ planner  │ ──▶ │ steps_data_fetcher│ ──▶ │ step_reasoner │ ──▶ │ label_verifier          │
 │          │     │ (no LLM output,  │     │               │     │                        │
 │ eval:    │     │  no evaluator)   │     │ eval:         │     │ eval:                  │
 │ format,  │     │                  │     │ format,       │     │ format, range,         │
 │ intent   │     │                  │     │ intent,       │     │ intent, consistency    │
 └─────────┘     └──────────────────┘     │ evidence      │     └────────────────────────┘
                                           └───────────────┘                │
                                                                            ▼
                                                                 ┌────────────────────────┐
                                                                 │ proposal_synthesizer    │
                                                                 │                        │
                                                                 │ eval: ALL FIVE          │
                                                                 │ (format, range, intent, │
                                                                 │  evidence, consistency) │
                                                                 └────────────────────────┘
                                                                            │
                                                                            ▼
                                                                           END
```

**Why after every step?**

1. **Early error correction:** A malformed planner output would previously cascade into confusing step_reasoner prompts. Now the planner's output is validated and fixed before anything downstream sees it.
2. **Simpler debugging:** When something goes wrong, you know exactly which step produced the bad output — the evaluator feedback is attached per-step.
3. **No retry loop needed:** Because each step's output is corrected in-place, the old `evaluator → planner` retry loop is eliminated. The pipeline runs forward-only.
4. **Cheaper than retries:** Catching a format error at the planner step costs 2 evaluator calls. Letting it cascade and retrying the whole pipeline costs 15+ VLM calls.

## What Changes in the Codebase

### New file: `step_evaluator_agents.py`

Contains:
- `EvalResult` and `EvalPipelineResult` Pydantic models
- Shared tools: `result_editor`, `json_reformatter`, `range_clamp`
- Five SubAgent definitions: `FORMAT_EVALUATOR`, `RANGE_EVALUATOR`, `INTENT_EVALUATOR`, `EVIDENCE_EVALUATOR`, `CONSISTENCY_EVALUATOR`
- `STEP_EVALUATOR_PROFILES` mapping (which evaluators run for which step)
- `EVALUATOR_CHAIN` ordering list
- `run_evaluator_suite()` runner function (accepts `step_name` to auto-select profile)

### `annotation_agent_pipeline.py`

1. **Every node that produces LLM output** now calls `run_evaluator_suite()` internally before returning its state update. This is the key change — evaluation is embedded in each node, not a separate graph node.

2. **Remove the standalone `evaluator` node** from the graph. Evaluation is no longer a graph node — it's an internal quality gate within each step's node function.

3. **Remove:** `_validate_solver_json_format()` (replaced by `format_evaluator` + `range_evaluator`), `_extract_verdict()` (each evaluator returns structured verdicts), `should_retry()` and the `evaluator → planner` conditional edge.

4. **`build_annotation_graph`** — simplified linear flow:
   ```python
   graph.add_node("planner", planner_node)
   graph.add_node("steps_data_fetcher", steps_data_fetcher_node)
   graph.add_node("step_reasoner", step_reasoner_node)
   graph.add_node("label_verifier", label_verifier_node)
   graph.add_node("proposal_synthesizer", proposal_synthesizer_node)
   # No separate evaluator node — evaluation happens inside each node

   graph.set_entry_point("planner")
   graph.add_edge("planner", "steps_data_fetcher")
   graph.add_edge("steps_data_fetcher", "step_reasoner")
   graph.add_conditional_edges(
       "step_reasoner",
       step_router,
       {
           "steps_data_fetcher": "steps_data_fetcher",
           "label_verifier": "label_verifier",
       },
   )
   graph.add_edge("label_verifier", "proposal_synthesizer")
   graph.add_edge("proposal_synthesizer", END)
   # No retry edge — evaluators fix in-place within each node
   ```

5. **`planner_node`** — remove feedback/retry sections. Runs once, calls evaluator on its own output.

### `requirements.common.txt`

Add `deepagents`.

## Usage Pattern — Every Node Follows This Template

Each node that produces LLM output follows the same pattern:

```python
from .step_evaluator_agents import run_evaluator_suite, EvalPipelineResult

def planner_node(state: AnnotationState) -> dict:
    # 1. Generate output (existing LLM call)
    raw_output = _call_llm(planner_prompt, ...)

    # 2. Run evaluator suite for THIS step
    suite_result: EvalPipelineResult = run_evaluator_suite(
        original_prompt=planner_prompt,
        step_output=raw_output,
        step_name="planner",  # selects [format_evaluator, intent_evaluator]
        parent_start=state["parent_start"],
        parent_end=state["parent_end"],
        label_mapping=LABEL_MAPPING,
    )

    # 3. Use the evaluated (possibly corrected) result
    evaluated_output = suite_result.final_result
    parsed = _parse_planner_response(evaluated_output)

    return {
        "plan_steps": parsed,
        "eval_feedback_planner": suite_result,  # stored for debugging
        "messages": messages,
    }


def step_reasoner_node(state: AnnotationState) -> dict:
    # 1. Generate output
    raw_output = _call_vlm(reasoner_prompt, images=graph_images, ...)

    # 2. Run evaluator suite for THIS step (includes evidence_evaluator)
    suite_result: EvalPipelineResult = run_evaluator_suite(
        original_prompt=reasoner_prompt,
        step_output=raw_output,
        step_name="step_reasoner",  # selects [format, intent, evidence]
        parent_start=state["parent_start"],
        parent_end=state["parent_end"],
        graph_images=state.get("all_graph_images"),
        label_mapping=LABEL_MAPPING,
    )

    # 3. Use evaluated result
    evaluated_output = suite_result.final_result
    ...


def proposal_synthesizer_node(state: AnnotationState) -> dict:
    # 1. Generate output
    raw_output = _call_vlm(synthesizer_prompt, images=graph_images, ...)

    # 2. Run FULL evaluator suite (all 5 evaluators)
    suite_result: EvalPipelineResult = run_evaluator_suite(
        original_prompt=synthesizer_prompt,
        step_output=raw_output,
        step_name="proposal_synthesizer",  # selects ALL evaluators
        parent_start=state["parent_start"],
        parent_end=state["parent_end"],
        graph_images=state.get("all_graph_images"),
        label_mapping=LABEL_MAPPING,
    )

    # 3. Parse the fully evaluated final result
    parsed = _parse_json_response(suite_result.final_result)

    return {
        "evaluation": suite_result.final_verdict,
        "evaluation_feedback": "\n".join(
            f"[{r.evaluator_name}] {r.verdict}: {r.feedback}"
            for r in suite_result.evaluator_results
        ),
        "final_labels": ...,
        "final_sub_start": ...,
        "final_sub_end": ...,
        "messages": messages,
    }
```

## Per-Step Evaluator Profiles

| Pipeline Step | Evaluators Used | Rationale |
|---|---|---|
| `planner` | format, intent | JSON plan steps — no graph data, no numeric ranges to validate |
| `step_reasoner` | format, intent, evidence | Prose graph descriptions — must match visual evidence, no label ranges |
| `label_verifier` | format, range, intent, consistency | JSON with label proposals — needs range checks but no graph re-verification (already done at step_reasoner) |
| `proposal_synthesizer` | ALL FIVE | Final JSON output — full validation including re-checking evidence against the synthesized boundaries |

**Note:** `steps_data_fetcher` does not produce LLM output (it fetches data), so it does not call the evaluator suite.

## Cost

**Per step:** Each evaluator is 1 VLM call on pass, up to 3 on fail (1 eval + 2 edit rounds).

**Per pipeline run (typical case):**
- planner: 2 evaluators × 1 call = 2 calls (rarely fails)
- step_reasoner: 3 evaluators × ~1.3 calls = ~4 calls (evidence sometimes needs edits)
- label_verifier: 4 evaluators × ~1.2 calls = ~5 calls
- proposal_synthesizer: 5 evaluators × ~1.5 calls = ~8 calls

**Typical total: ~19 evaluator calls across the full pipeline.**

**Worst case:** All evaluators fail and need max edits at every step: (2×3) + (3×3) + (4×3) + (5×3) = 42 calls.

**Comparison to old approach:** The old retry loop was 5+ VLM calls per retry × up to 3 retries = 15+ calls for just the final evaluation, plus full pipeline re-execution costs. The new approach costs more evaluator calls but eliminates retries entirely — the pipeline always runs forward, and total wall-clock time is lower because no work is thrown away and redone.
