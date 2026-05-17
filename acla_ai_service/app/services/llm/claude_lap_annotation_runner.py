"""
Agentic Claude lap-section annotation runner — one Claude session per click.

The manual.py lap-to-segment excerpter rough-splits a user-picked lap range
into per-`circuit_section` sub-ranges (via the deterministic
``split_lap_by_circuit_sections`` tool). This runner annotates ONE of those
sections at a time: the agent inspects telemetry, optionally shrinks /
extends the boundary by calling ``revise_segment_range``, then submits a
single label proposal via ``submit_lap_proposal``.

Differences from ``claude_annotation_runner.ClaudeAnnotationRunner``:
  - Parent context is one circuit section, not a free-form parent segment.
  - The system prompt injects the lap_annotation_skill block for the
    selected circuit (section order, shrink/extend rules, ST1–ST6 hints).
  - Tool surface adds ``revise_segment_range`` (the agent's escape hatch
    when the rough boundary disagrees with the player telemetry) and
    swaps ``submit_proposal`` for ``submit_lap_proposal`` (one annotation,
    parent labels only — the downstream detailed flow handles sub-labels).

Public API:
    run_claude_lap_annotation(...) -> LapAnnotationResult
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.models.graph_analysis_skill import get_graph_skill
from app.models.label_catalog import get_label_catalog
from app.models.lap_annotation_skill import get_lap_skill
from app.models.segment_models import LABEL_MAPPING

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class LapAnnotationResult:
    """One agent-decided lap-section annotation.

    ``label_ids`` is the flat list of parent labels the agent picked
    (circuit + circuit_section + segment_type + optional main). The UI
    persists this as a single AnnotatedSegment whose range is
    ``[start_index, end_index]`` — potentially revised from the rough
    splitter boundary when ``revised`` is True.
    """
    section_id: str
    start_index: int
    end_index: int
    label_ids: List[str]
    reasoning: str
    revised: bool                            # True when the agent shrunk / extended
    submitted: bool
    rough_start: int                         # original boundary the runner was handed
    rough_end: int
    rejected_proposals: List[Dict[str, Any]] = field(default_factory=list)
    rendered_images: List[bytes] = field(default_factory=list)
    transcript: str = ""
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# Internal capture
# ---------------------------------------------------------------------------


@dataclass
class _LapRunCapture:
    """Mutable state captured during one Claude session."""
    cur_start: int = 0
    cur_end: int = 0
    revised: bool = False
    submitted: bool = False
    label_ids: List[str] = field(default_factory=list)
    reasoning: str = ""
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    rendered_images: List[bytes] = field(default_factory=list)
    text_chunks: List[str] = field(default_factory=list)
    tool_calls: int = 0


_CLAUDE_NODE = "claude_lap_analyst"


def _stage(phase: str, **extra) -> Dict[str, Any]:
    return {"node_name": _CLAUDE_NODE, "phase": phase, **extra}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ClaudeLapAnnotationRunner:
    """One agentic Claude session that annotates ONE circuit section."""

    def __init__(
        self,
        *,
        df,
        lap_start: int,
        lap_end: int,
        section_id: str,
        section_start: int,
        section_end: int,
        circuit_id: str,
        existing_section_annotations: List[Dict[str, Any]],
        claude_model: str,
        use_thinking: bool,
        max_turns: int,
        progress_callback: Optional[Callable] = None,
        vlm_prompt_callback: Optional[Callable] = None,
        vlm_stream_callback: Optional[Callable] = None,
        vlm_reasoning_callback: Optional[Callable] = None,
        step_event_callback: Optional[Callable] = None,
    ) -> None:
        self.df = df
        self.lap_start = int(lap_start)
        self.lap_end = int(lap_end)
        self.section_id = section_id
        self.rough_start = int(section_start)
        self.rough_end = int(section_end)
        self.circuit_id = circuit_id
        self.existing = list(existing_section_annotations)

        self.claude_model = claude_model
        self.use_thinking = use_thinking
        self.max_turns = max_turns

        self.on_progress = progress_callback
        self.on_prompt = vlm_prompt_callback
        self.on_stream = vlm_stream_callback
        self.on_reasoning = vlm_reasoning_callback
        self.on_step_event = step_event_callback

        self.capture = _LapRunCapture(
            cur_start=self.rough_start,
            cur_end=self.rough_end,
        )

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _emit_tool_event(self, name: str, inp: Dict[str, Any], summary: str) -> None:
        if not self.on_step_event:
            return
        inp_str = json.dumps(inp, default=str)
        if len(inp_str) > 400:
            inp_str = inp_str[:400] + "…"
        if len(summary) > 600:
            summary = summary[:600] + "…"
        msg = (
            f"**Tool:** `{name}`\n\n"
            f"**Input:** `{inp_str}`\n\n"
            f"**Result:** {summary}"
        )
        self.on_step_event(msg, _stage(f"tool:{name}"))

    def _clamp_to_lap(self, s: int, e: int) -> tuple[int, int]:
        s = max(self.lap_start, int(s))
        e = min(self.lap_end, int(e))
        if e <= s:
            e = min(self.lap_end, s + 1)
        return s, e

    # -- standard observation tools (mirror ClaudeAnnotationRunner) -----

    def tool_list_graphs(self) -> str:
        from app.services.llm.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS
        out = [
            {"id": g["id"], "title": g["title"], "description": g["description"]}
            for g in AGENT_GRAPH_DEFINITIONS
        ]
        return json.dumps({"graphs": out}, indent=2)

    def tool_get_graph_guidance(self, graph_ids: List[str]) -> str:
        skill = get_graph_skill()
        text = skill.build_graph_prompt(list(graph_ids))
        return text or "(no guidance available for the requested graph(s))"

    def tool_render_graph(self, graph_id: str, start: int, end: int) -> Dict[str, Any]:
        from app.services.llm.annotation_agent_tools import (
            build_graph, render_graph_builds,
        )
        s, e = self._clamp_to_lap(start, end)
        table = build_graph(graph_id, self.df)
        if table is None or table.empty:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"Cannot render `{graph_id}` over [{s}, {e}]: the "
                        f"underlying telemetry columns are not present."
                    ),
                }],
                "is_error": True,
            }
        rendered = render_graph_builds({graph_id: table}, s, e)
        if not rendered:
            return {
                "content": [{
                    "type": "text",
                    "text": f"`{graph_id}` produced no image for [{s}, {e}].",
                }],
                "is_error": True,
            }
        img, desc = rendered[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        self.capture.rendered_images.append(png_bytes)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return {
            "content": [
                {"type": "image", "data": encoded, "mimeType": "image/png"},
                {"type": "text", "text": f"{desc} (rendered over [{s}, {e}])"},
            ],
        }

    def tool_query_telemetry(self, query_id: str, params_json: str) -> str:
        from app.services.llm.annotation_agent_tools import run_pipeline_query
        try:
            params = json.loads(params_json) if params_json else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"params_json was not valid JSON: {e}"})
        if not isinstance(params, dict):
            return json.dumps({"error": "params_json must decode to a JSON object."})
        payload, err = run_pipeline_query(self.df, query_id, params)
        result = {"query": query_id, "params": params, "result": payload}
        if err:
            result["error"] = err
        return json.dumps(result, default=str)

    def tool_compute_expert_phases(self, start: int, end: int) -> str:
        from app.services.llm.annotation_agent_tools import compute_expert_phases
        s, e = self._clamp_to_lap(start, end)
        att = compute_expert_phases(self.df, s, e)
        return json.dumps({"phases_range": [s, e], "data": att.content}, default=str)

    def tool_locate_circuit_section(self, start: int, end: int) -> str:
        from app.services.llm.annotation_agent_tools import locate_circuit_section
        s, e = self._clamp_to_lap(start, end)
        att = locate_circuit_section(self.df, s, e)
        return json.dumps({"range": [s, e], "data": att.content}, default=str)

    # -- lap-specific tools --------------------------------------------

    def tool_revise_segment_range(self, new_start: int, new_end: int) -> str:
        """Shrink / extend the current section's boundary.

        Records the new range in the capture state so the UI can rebuild
        the rough-split array for the remaining sections after the run.
        Validates the new range lies inside the lap and is at least 3
        ilocs long (per the global skill rule).
        """
        s, e = int(new_start), int(new_end)
        if not (self.lap_start <= s < e <= self.lap_end):
            return json.dumps({
                "ok": False,
                "error": (
                    f"new range [{s}, {e}] must satisfy "
                    f"{self.lap_start} <= start < end <= {self.lap_end}"
                ),
            })
        if (e - s) < 3:
            return json.dumps({
                "ok": False,
                "error": (
                    f"new range too short ({e - s} ilocs) — minimum 3. "
                    "Drop the section instead by submitting an empty proposal."
                ),
            })
        self.capture.cur_start = s
        self.capture.cur_end = e
        self.capture.revised = True
        return json.dumps({
            "ok": True,
            "new_range": [s, e],
            "note": (
                "Range revised. After you submit_lap_proposal, the UI will "
                "rebuild the rough-split array using this new end as the "
                "starting point for the next section."
            ),
        })

    def tool_submit_lap_proposal(
        self, label_ids_json: str, reasoning: str,
    ) -> str:
        """Capture Claude's final label set for this section.

        ``label_ids_json`` must be a JSON array of label_id strings:
        ``["brands_hatch", "brands_hatch3", "ST1"]``. Empty array is a
        valid "drop this section" signal (e.g. when the section was
        traversed but should not be annotated).
        """
        try:
            parsed = json.loads(label_ids_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "ok": False,
                "error": (
                    f"label_ids_json was not valid JSON: {e}. "
                    "Pass a JSON array like '[\"brands_hatch\", \"brands_hatch3\", \"ST1\"]'."
                ),
            })
        if not isinstance(parsed, list):
            return json.dumps({"ok": False, "error": "label_ids_json must be a JSON array."})

        cleaned: List[str] = []
        rejected: List[Dict[str, Any]] = []
        for i, raw in enumerate(parsed):
            if not isinstance(raw, str):
                rejected.append({"index": i, "value": raw, "reason": "must be a string label_id"})
                continue
            if raw not in LABEL_MAPPING:
                rejected.append({"index": i, "value": raw, "reason": f"unknown label_id '{raw}'"})
                continue
            if raw in cleaned:
                continue
            cleaned.append(raw)

        self.capture.label_ids = cleaned
        self.capture.reasoning = str(reasoning or "")
        self.capture.rejected = rejected
        self.capture.submitted = True
        return json.dumps({
            "ok": True,
            "accepted_label_ids": cleaned,
            "rejected": rejected,
            "final_range": [self.capture.cur_start, self.capture.cur_end],
            "note": "Proposal captured. Session can end now.",
        })

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        from app.services.llm.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS

        catalog = get_label_catalog()
        lap_skill_block = get_lap_skill().build_prompt(self.circuit_id)

        section_entry = catalog.get_label(self.section_id)
        section_name = section_entry.name if section_entry else self.section_id
        section_desc = section_entry.description if section_entry else "(no description)"
        section_rng = (
            section_entry.normalized_position_range if section_entry else None
        )
        section_rng_str = (
            f"[{section_rng[0]:.3f}, {section_rng[1]:.3f}]"
            if section_rng is not None else "[null, null]"
        )

        # Eligible labels: the circuit + every circuit_section under it +
        # ST1–ST6 + main labels. The agent picks a subset.
        eligible_lines: List[str] = []
        circuit_entry = catalog.get_label(self.circuit_id)
        if circuit_entry is not None:
            eligible_lines.append(
                f"  - `{circuit_entry.id}` ({circuit_entry.name}): {circuit_entry.description.strip()}"
            )
        for entry in catalog.entries_by_type("circuit_section"):
            if entry.parent != self.circuit_id:
                continue
            rng = entry.normalized_position_range
            rng_str = (
                f"[{rng[0]:.3f}, {rng[1]:.3f}]" if rng is not None else "[null, null]"
            )
            eligible_lines.append(
                f"  - `{entry.id}` ({entry.name}): "
                f"{(entry.description or '').strip()} — range {rng_str}"
            )
        for entry in catalog.get_segment_types():
            eligible_lines.append(
                f"  - `{entry.id}` ({entry.name}): {(entry.description or '').strip()}"
            )
        for entry in catalog.get_main_labels():
            # Main labels are optional — only include EA / MS / RM / Pit / Overtaking / Missing.
            eligible_lines.append(
                f"  - `{entry.id}` ({entry.name}): {(entry.description or '').strip()}"
            )

        existing_block = ""
        if self.existing:
            lines = []
            for c in self.existing:
                names = ", ".join(LABEL_MAPPING.get(l, l) for l in c.get("labels", []))
                lines.append(
                    f"  - [{c['start_index']}, {c['end_index']}] — {names}"
                )
            existing_block = (
                "\n### Sections already annotated on this lap "
                "(reference, do NOT re-annotate)\n"
                + "\n".join(lines) + "\n"
            )

        graph_list = "\n".join(
            f"  - `{g['id']}` — {g['title']}: {g['description']}"
            for g in AGENT_GRAPH_DEFINITIONS
        )

        thinking_clause = (
            "\nThink step-by-step before each tool call: name the candidate "
            "label whose `characteristics` block you are testing, pick the "
            "most direct tool, read the result before deciding the next step.\n"
            if self.use_thinking else ""
        )

        return (
            "You are a racing telemetry data analyst annotating ONE circuit "
            "section of a lap. The deterministic splitter has handed you a "
            "rough iloc boundary; your job is to pick the parent labels for "
            "the section by matching the section's telemetry against each "
            "candidate label's `characteristics` block in the skill below.\n"
            "\n"
            "### Lap context\n"
            f"- Circuit: {self.circuit_id}\n"
            f"- Lap range: [{self.lap_start}, {self.lap_end}] "
            f"(length {self.lap_end - self.lap_start})\n"
            "\n"
            "### Section under review\n"
            f"- section_id: `{self.section_id}` ({section_name})\n"
            f"- description: {section_desc}\n"
            f"- normalized_position_range: {section_rng_str}\n"
            f"- rough iloc boundary: [{self.rough_start}, {self.rough_end}] "
            f"(length {self.rough_end - self.rough_start})\n"
            f"{existing_block}"
            "\n"
            f"{lap_skill_block}"
            "\n"
            "### Eligible label IDs\n"
            "Only IDs from this list are accepted by `submit_lap_proposal`. "
            "Always include the circuit and the section. The ST1–ST6 pick "
            "is OPTIONAL — include one only when the trajectory shape is "
            "unambiguous. Main labels (EA / MS / RM / PS / OV / MD) follow "
            "the skill's `characteristics` blocks.\n"
            + "\n".join(eligible_lines)
            + "\n\n"
            "### Available telemetry graphs\n"
            + graph_list
            + "\n\n"
            "### How to work\n"
            "Follow the numbered procedure in the skill's `global_rules` "
            "block above. Call `revise_segment_range` only if "
            "`locate_circuit_section` shows the rough range straddles two "
            "catalog sections — boundary mechanics are not the goal of this "
            "flow. After `submit_lap_proposal` returns `ok: true`, stop.\n"
            "\n"
            "### Hard rules\n"
            f"- Final range must satisfy {self.lap_start} <= start < end <= {self.lap_end} and be ≥ 3 ilocs.\n"
            "- Do not invent label IDs.\n"
            "- One proposal per session — do NOT annotate downstream sections you can see in the array.\n"
            "- An empty `label_ids_json` (`[]`) is a valid 'drop this section' signal.\n"
            "- Budget tool calls: a typical section needs 3-6 calls total.\n"
            f"{thinking_clause}"
        )

    def _initial_user_prompt(self) -> str:
        return (
            f"Annotate the `{self.section_id}` section over rough range "
            f"[{self.rough_start}, {self.rough_end}]. Confirm the boundary "
            "against player telemetry, revise it if a shrink/extend rule "
            "fires, then submit the parent labels."
        )

    # ------------------------------------------------------------------
    # MCP tool registration + session
    # ------------------------------------------------------------------

    def _build_tool_set(self):
        from claude_agent_sdk import tool, create_sdk_mcp_server

        runner = self

        @tool(
            "list_graphs",
            "List every available telemetry graph as a compact catalog of `id` + `title` + `description`.",
            {},
        )
        async def list_graphs(args):  # noqa: ARG001
            text = runner.tool_list_graphs()
            runner._emit_tool_event("list_graphs", {}, f"{text.count(chr(10))}-line catalog")
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "get_graph_guidance",
            "Return the per-graph `how_to_analyze` block + glossary for the given graph IDs.",
            {"graph_ids": list},
        )
        async def get_graph_guidance(args):
            ids = args.get("graph_ids") or []
            if not isinstance(ids, list):
                ids = [str(ids)]
            ids = [str(x) for x in ids]
            text = runner.tool_get_graph_guidance(ids)
            runner._emit_tool_event(
                "get_graph_guidance", {"graph_ids": ids},
                f"{text.count(chr(10))} line(s) of guidance",
            )
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "render_graph",
            "Render one telemetry graph over an iloc window and return the PNG image + description.",
            {"graph_id": str, "start": int, "end": int},
        )
        async def render_graph(args):
            result = runner.tool_render_graph(
                str(args["graph_id"]), int(args["start"]), int(args["end"]),
            )
            runner._emit_tool_event(
                "render_graph",
                {"graph_id": args.get("graph_id"), "start": args.get("start"), "end": args.get("end")},
                "image returned" if not result.get("is_error") else "render failed",
            )
            return result

        @tool(
            "query_telemetry",
            "Run a deterministic numeric query on the raw telemetry DataFrame. "
            "`query_id` is one of: find_extremum, find_first_match, "
            "read_values_at_indices, compute_slope, find_dips_on_main_slope, "
            "find_threshold_crossing. `params_json` is a JSON-encoded object "
            "that must include `range: [start_iloc, end_iloc]` plus the "
            "per-query fields.",
            {"query_id": str, "params_json": str},
        )
        async def query_telemetry(args):
            params_json = str(args.get("params_json", ""))
            text = runner.tool_query_telemetry(str(args["query_id"]), params_json)
            runner._emit_tool_event(
                "query_telemetry",
                {"query_id": args.get("query_id"), "params_json": params_json},
                text,
            )
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "compute_expert_phases",
            "Detect expert-anchored corner phases (per-arc entry / apex / exit ilocs).",
            {"start": int, "end": int},
        )
        async def compute_expert_phases(args):
            text = runner.tool_compute_expert_phases(int(args["start"]), int(args["end"]))
            runner._emit_tool_event(
                "compute_expert_phases",
                {"start": args.get("start"), "end": args.get("end")},
                text,
            )
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "locate_circuit_section",
            "Identify which named circuit_section the iloc range overlaps. "
            "Useful after `revise_segment_range` when the new boundary may "
            "have crossed into a neighbouring section's catalog range.",
            {"start": int, "end": int},
        )
        async def locate_circuit_section(args):
            text = runner.tool_locate_circuit_section(int(args["start"]), int(args["end"]))
            runner._emit_tool_event(
                "locate_circuit_section",
                {"start": args.get("start"), "end": args.get("end")},
                text,
            )
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "revise_segment_range",
            "Shrink or extend the current section's iloc boundary BEFORE "
            "submitting. Call this when a shrink/extend rule from the lap "
            "skill fires (e.g. brake initiation onset lies upstream of the "
            "section's catalog start). After you submit_lap_proposal, the UI "
            "will use this new end as the starting point for the next "
            "section's rough split. The new range must lie inside the lap "
            "and be ≥ 3 ilocs long.",
            {"new_start": int, "new_end": int},
        )
        async def revise_segment_range(args):
            text = runner.tool_revise_segment_range(
                int(args["new_start"]), int(args["new_end"]),
            )
            runner._emit_tool_event(
                "revise_segment_range",
                {"new_start": args.get("new_start"), "new_end": args.get("new_end")},
                text,
            )
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "submit_lap_proposal",
            "Submit the final parent labels for this circuit section and end "
            "the session. `label_ids_json` is a JSON array of label IDs "
            "(circuit + circuit_section + ST1–ST6 + optional main). An empty "
            "array `[]` is a valid 'drop this section' signal. `reasoning` is "
            "a 1-3 sentence justification citing ilocs / values.",
            {"label_ids_json": str, "reasoning": str},
        )
        async def submit_lap_proposal(args):
            text = runner.tool_submit_lap_proposal(
                str(args.get("label_ids_json", "")),
                str(args.get("reasoning", "")),
            )
            parsed = json.loads(text)
            summary_line = (
                f"submitted {len(parsed.get('accepted_label_ids', []))} label(s)"
                if parsed.get("ok") else f"REJECTED: {parsed.get('error', '?')}"
            )
            runner._emit_tool_event(
                "submit_lap_proposal", {"reasoning": args.get("reasoning")},
                summary_line,
            )
            return {"content": [{"type": "text", "text": text}]}

        tools_list = [
            list_graphs, get_graph_guidance, render_graph, query_telemetry,
            compute_expert_phases, locate_circuit_section,
            revise_segment_range, submit_lap_proposal,
        ]
        tool_names = [
            "mcp__telemetry__list_graphs",
            "mcp__telemetry__get_graph_guidance",
            "mcp__telemetry__render_graph",
            "mcp__telemetry__query_telemetry",
            "mcp__telemetry__compute_expert_phases",
            "mcp__telemetry__locate_circuit_section",
            "mcp__telemetry__revise_segment_range",
            "mcp__telemetry__submit_lap_proposal",
        ]

        server = create_sdk_mcp_server(
            name="telemetry",
            version="1.0.0",
            tools=tools_list,
        )
        return server, tool_names

    @staticmethod
    def _import_sdk_types():
        from types import SimpleNamespace
        try:
            from claude_agent_sdk import (
                query, ClaudeAgentOptions,
                AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock,
            )
        except ImportError as exc:
            raise RuntimeError(
                "claude-agent-sdk is not installed. Install with "
                "`pip install claude-agent-sdk` and ensure the `claude` CLI "
                "is logged in."
            ) from exc
        try:
            from claude_agent_sdk.types import ThinkingBlock  # type: ignore
        except ImportError:
            ThinkingBlock = None  # type: ignore
        return SimpleNamespace(
            query=query, ClaudeAgentOptions=ClaudeAgentOptions,
            AssistantMessage=AssistantMessage, TextBlock=TextBlock,
            ToolUseBlock=ToolUseBlock, ToolResultBlock=ToolResultBlock,
            ThinkingBlock=ThinkingBlock,
        )

    async def run_async(self) -> None:
        sdk = self._import_sdk_types()
        server, tool_names = self._build_tool_set()

        options = sdk.ClaudeAgentOptions(
            model=self.claude_model,
            mcp_servers={"telemetry": server},
            allowed_tools=tool_names,
            system_prompt=self._build_system_prompt(),
            max_turns=self.max_turns,
        )

        if self.on_prompt:
            self.on_prompt(self._initial_user_prompt(), _stage("main"))
        if self.on_progress:
            self.on_progress("claude_lap_analyst", "session starting")

        async for message in sdk.query(
            prompt=self._initial_user_prompt(), options=options,
        ):
            self._handle_message(
                message, sdk.AssistantMessage, sdk.TextBlock,
                sdk.ToolUseBlock, sdk.ToolResultBlock, sdk.ThinkingBlock,
            )

        if self.on_progress:
            self.on_progress(
                "claude_lap_analyst",
                f"done — {self.capture.tool_calls} tool call(s), "
                f"submitted={self.capture.submitted}, revised={self.capture.revised}",
            )

    def _handle_message(
        self, message, AssistantMessage, TextBlock, ToolUseBlock,
        ToolResultBlock, ThinkingBlock,
    ) -> None:
        if not isinstance(message, AssistantMessage):
            return
        for block in getattr(message, "content", None) or []:
            if isinstance(block, TextBlock):
                text = getattr(block, "text", "") or ""
                if text:
                    self.capture.text_chunks.append(text)
                    if self.on_stream:
                        self.on_stream(text)
            elif ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                thinking = getattr(block, "thinking", "") or ""
                if thinking and self.on_reasoning:
                    self.on_reasoning(thinking)
            elif isinstance(block, ToolUseBlock):
                self.capture.tool_calls += 1
                if self.on_progress:
                    self.on_progress(
                        "claude_lap_analyst",
                        f"tool {self.capture.tool_calls}: {block.name}",
                    )

    def run(self) -> None:
        asyncio.run(self.run_async())


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_claude_lap_annotation(
    *,
    df,
    lap_start: int,
    lap_end: int,
    section_id: str,
    section_start: int,
    section_end: int,
    circuit_id: str,
    existing_section_annotations: Optional[List[Dict[str, Any]]] = None,
    claude_model: str = "claude-sonnet-4-6",
    use_thinking: bool = False,
    max_turns: int = 30,
    progress_callback: Optional[Callable] = None,
    vlm_stream_callback: Optional[Callable] = None,
    vlm_prompt_callback: Optional[Callable] = None,
    vlm_reasoning_callback: Optional[Callable] = None,
    step_event_callback: Optional[Callable] = None,
) -> LapAnnotationResult:
    """Run one Claude lap-section annotation session.

    The caller (manual_lap_agent_claude.py UI) is responsible for:
      - Pre-running `split_lap_by_circuit_sections` to populate the array.
      - Picking the head of the array (one call per click).
      - Applying ``result.revised`` to rebuild the array via the splitter
        when ``revised`` is True.
      - Persisting ``label_ids`` as an AnnotatedSegment over
        ``[result.start_index, result.end_index]``.
    """
    runner = ClaudeLapAnnotationRunner(
        df=df,
        lap_start=int(lap_start),
        lap_end=int(lap_end),
        section_id=section_id,
        section_start=int(section_start),
        section_end=int(section_end),
        circuit_id=circuit_id,
        existing_section_annotations=list(existing_section_annotations or []),
        claude_model=claude_model,
        use_thinking=bool(use_thinking),
        max_turns=int(max_turns),
        progress_callback=progress_callback,
        vlm_prompt_callback=vlm_prompt_callback,
        vlm_stream_callback=vlm_stream_callback,
        vlm_reasoning_callback=vlm_reasoning_callback,
        step_event_callback=step_event_callback,
    )

    started = time.time()
    runner.run()
    elapsed = time.time() - started
    LOGGER.info(
        "Claude lap annotation: section=%s tool_calls=%d submitted=%s revised=%s elapsed=%.1fs",
        section_id, runner.capture.tool_calls,
        runner.capture.submitted, runner.capture.revised, elapsed,
    )

    cap = runner.capture
    transcript = "".join(cap.text_chunks).strip()

    return LapAnnotationResult(
        section_id=section_id,
        start_index=cap.cur_start,
        end_index=cap.cur_end,
        label_ids=list(cap.label_ids),
        reasoning=cap.reasoning or transcript or "(no reasoning)",
        revised=cap.revised,
        submitted=cap.submitted,
        rough_start=int(section_start),
        rough_end=int(section_end),
        rejected_proposals=list(cap.rejected),
        rendered_images=list(cap.rendered_images),
        transcript=transcript,
        tool_calls=cap.tool_calls,
    )
