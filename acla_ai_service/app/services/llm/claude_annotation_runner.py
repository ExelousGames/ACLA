"""
Agentic Claude annotation runner — one Claude session per annotation.

Why a separate runner: the local-VLM ``annotation_root`` LangGraph pipeline
chains 10-30 stateless VLM calls (planner → describe_graphs → zoom planner →
zoom synth → label_verifier → proposal_synth). Routed through
``claude-agent-sdk`` each chain step spawns a fresh ``claude`` subprocess —
seconds of startup tax per call. For Claude we instead expose the underlying
telemetry primitives as in-process MCP tools and let ONE Claude session
iterate: render → look → query → reason → submit. Single subprocess start,
multi-turn reasoning in one context.

Public API:

    run_claude_annotation(
        df, start_index, end_index, session_id, parent_main_labels,
        existing_children, config, progress_callback=..., vlm_*_callback=...,
        step_event_callback=...,
    ) -> AnnotationResult

The result shape matches ``annotation_agent_pipeline.AnnotationResult`` so
the shared UI (``_agent_annotation_shared.render_pipeline_result`` /
``render_staged_review``) consumes both backends identically.
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
from app.models.segment_models import LABEL_MAPPING

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result shape — kept structurally compatible with AnnotationResult.
# Imported lazily where needed to avoid circular import with the pipeline.
# ---------------------------------------------------------------------------

@dataclass
class _ClaudeRunCapture:
    """State captured during a single Claude annotation session."""
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    final_reasoning: str = ""
    rendered_images: List[bytes] = field(default_factory=list)
    submitted: bool = False
    text_chunks: List[str] = field(default_factory=list)
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# Stage label helpers for the live UI
# ---------------------------------------------------------------------------

_CLAUDE_NODE = "claude_analyst"


def _stage(phase: str, **extra) -> Dict[str, Any]:
    return {"node_name": _CLAUDE_NODE, "phase": phase, **extra}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ClaudeAnnotationRunner:
    """One agentic Claude session that discovers sub-segments via tools."""

    def __init__(
        self,
        *,
        df,
        start_index: int,
        end_index: int,
        parent_main_labels: List[str],
        existing_children: List[Dict[str, Any]],
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
        self.parent_start = int(start_index)
        self.parent_end = int(end_index)
        self.parent_main_labels = list(parent_main_labels)
        self.existing_children = list(existing_children)
        self.claude_model = claude_model
        self.use_thinking = use_thinking
        self.max_turns = max_turns

        self.on_progress = progress_callback
        self.on_prompt = vlm_prompt_callback
        self.on_stream = vlm_stream_callback
        self.on_reasoning = vlm_reasoning_callback
        self.on_step_event = step_event_callback

        self.capture = _ClaudeRunCapture()

    # -- tool implementations --------------------------------------------

    def _emit_tool_event(self, name: str, inp: Dict[str, Any], result_summary: str) -> None:
        if not self.on_step_event:
            return
        # Trim huge inputs / outputs so the UI stays readable.
        inp_str = json.dumps(inp, default=str)
        if len(inp_str) > 400:
            inp_str = inp_str[:400] + "…"
        if len(result_summary) > 600:
            result_summary = result_summary[:600] + "…"
        summary = (
            f"**Tool:** `{name}`\n\n"
            f"**Input:** `{inp_str}`\n\n"
            f"**Result:** {result_summary}"
        )
        self.on_step_event(summary, _stage(f"tool:{name}"))

    def tool_list_graphs(self) -> str:
        from app.services.llm.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS
        out = [
            {"id": g["id"], "title": g["title"], "description": g["description"]}
            for g in AGENT_GRAPH_DEFINITIONS
        ]
        return json.dumps({"graphs": out}, indent=2)

    def tool_get_graph_guidance(self, graph_ids: List[str]) -> str:
        """Return the skill's `how_to_analyze` + glossary block for the given graphs."""
        skill = get_graph_skill()
        text = skill.build_graph_prompt(list(graph_ids))
        return text or "(no guidance available for the requested graph(s))"

    def _clamp_range(self, start: int, end: int) -> tuple[int, int]:
        s = max(self.parent_start, int(start))
        e = min(self.parent_end, int(end))
        if e <= s:
            e = min(self.parent_end, s + 1)
        return s, e

    def tool_render_graph(self, graph_id: str, start: int, end: int) -> Dict[str, Any]:
        """Render one graph over [start, end] and return image + descriptor."""
        from app.services.llm.annotation_agent_tools import (
            build_graph, render_graph_builds,
        )
        s, e = self._clamp_range(start, end)
        table = build_graph(graph_id, self.df)
        if table is None or table.empty:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"Cannot render `{graph_id}` over [{s}, {e}]: the "
                        f"underlying telemetry columns are not present in this "
                        f"session."
                    ),
                }],
                "is_error": True,
            }
        rendered = render_graph_builds({graph_id: table}, s, e)
        if not rendered:
            return {
                "content": [{
                    "type": "text",
                    "text": f"`{graph_id}` produced no image for [{s}, {e}] (empty slice).",
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
                {"type": "text", "text": f"{desc} (rendered over indices [{s}, {e}])"},
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
        s, e = self._clamp_range(start, end)
        attachment = compute_expert_phases(self.df, s, e)
        # PipelineAttachment.content is already JSON-friendly.
        return json.dumps({"phases_range": [s, e], "data": attachment.content}, default=str)

    def tool_locate_circuit_section(self, start: int, end: int) -> str:
        from app.services.llm.annotation_agent_tools import locate_circuit_section
        s, e = self._clamp_range(start, end)
        attachment = locate_circuit_section(self.df, s, e)
        return json.dumps({"range": [s, e], "data": attachment.content}, default=str)

    def tool_submit_proposal(self, proposals_json: str, summary: str) -> str:
        """Capture Claude's final sub-segment proposals."""
        try:
            parsed = json.loads(proposals_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "ok": False,
                "error": f"proposals_json was not valid JSON: {e}. "
                         f"Pass a JSON array string like "
                         f'[{{"label_id": "...", "start_index": N, "end_index": M, "reasoning": "..."}}]',
            })
        if not isinstance(parsed, list):
            return json.dumps({"ok": False, "error": "proposals_json must be a JSON array."})

        cleaned: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for i, p in enumerate(parsed):
            if not isinstance(p, dict):
                rejected.append({"index": i, "reason": "entry is not an object"})
                continue
            lid = p.get("label_id")
            if lid not in LABEL_MAPPING:
                rejected.append({"index": i, "reason": f"unknown label_id '{lid}'"})
                continue
            try:
                s = int(p.get("start_index"))
                e = int(p.get("end_index"))
            except (TypeError, ValueError):
                rejected.append({"index": i, "reason": "start_index/end_index must be ints"})
                continue
            if not (self.parent_start <= s < e <= self.parent_end):
                rejected.append({
                    "index": i,
                    "reason": (
                        f"range [{s}, {e}] outside parent "
                        f"[{self.parent_start}, {self.parent_end}] or s>=e"
                    ),
                })
                continue
            cleaned.append({
                "label_id": lid,
                "start_index": s,
                "end_index": e,
                "reasoning": str(p.get("reasoning", "")),
            })

        if not cleaned and rejected:
            return json.dumps({
                "ok": False,
                "error": "no valid proposals — fix the issues below and call submit_proposal again",
                "rejected": rejected,
            })

        self.capture.proposals = cleaned
        self.capture.final_reasoning = str(summary or "")
        self.capture.submitted = True
        return json.dumps({
            "ok": True,
            "accepted_count": len(cleaned),
            "rejected": rejected,
            "note": "Proposals captured. Session can end now.",
        })

    # -- prompt building --------------------------------------------------

    def _build_system_prompt(self) -> str:
        from app.services.llm.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS

        catalog = get_label_catalog()

        parent_label_blocks: List[str] = []
        for pid in self.parent_main_labels:
            entry = catalog.get_label(pid)
            if entry is None:
                # Fall back to the raw id / display name when the catalog has no record.
                parent_label_blocks.append(f"  - `{pid}` ({LABEL_MAPPING.get(pid, pid)})")
                continue
            desc = entry.description or "(no description)"
            guideline = (
                f"\n      guideline: {entry.annotation_guideline}"
                if entry.annotation_guideline else ""
            )
            parent_label_blocks.append(
                f"  - `{entry.id}` ({entry.name}): {desc}{guideline}"
            )

        sub_label_blocks: List[str] = []
        seen: set = set()
        for pid in self.parent_main_labels:
            for entry in catalog.get_sublabels(pid):
                if entry.id in seen:
                    continue
                seen.add(entry.id)
                desc = entry.description or "(no description)"
                guideline = (
                    f"\n    guideline: {entry.annotation_guideline}"
                    if entry.annotation_guideline else ""
                )
                sub_label_blocks.append(
                    f"  - `{entry.id}` ({entry.name}): {desc}{guideline}"
                )
        for entry in catalog.get_segment_types():
            if entry.id in seen:
                continue
            seen.add(entry.id)
            desc = entry.description or "(no description)"
            sub_label_blocks.append(
                f"  - `{entry.id}` ({entry.name}): {desc}"
            )

        graph_list = "\n".join(
            f"  - `{g['id']}` — {g['title']}: {g['description']}"
            for g in AGENT_GRAPH_DEFINITIONS
        )

        existing_block = ""
        if self.existing_children:
            lines = []
            for c in self.existing_children:
                names = ", ".join(LABEL_MAPPING.get(l, l) for l in c.get("labels", []))
                lines.append(
                    f"  - [{c['start_index']}, {c['end_index']}] — {names}"
                )
            existing_block = (
                "\n### Already discovered sub-segments (do NOT re-propose)\n"
                + "\n".join(lines) + "\n"
            )

        thinking_clause = (
            "\nThink step-by-step before each tool call: state what you "
            "need to confirm, pick the most direct tool, then read the "
            "result before deciding the next step.\n"
            if self.use_thinking else ""
        )

        return (
            "You are a racing telemetry data analyst. Your job is to discover "
            f"the most notable sub-segment(s) within a parent segment of a "
            f"session, then submit them as label proposals via the "
            f"`submit_proposal` tool.\n"
            "\n"
            "### Parent segment\n"
            f"- index range: [{self.parent_start}, {self.parent_end}] "
            f"(length {self.parent_end - self.parent_start})\n"
            "- parent main label(s):\n"
            + ("\n".join(parent_label_blocks) or "  (none)")
            + "\n"
            f"{existing_block}"
            "\n"
            "### Candidate labels you may propose\n"
            "Only label_ids from this list are accepted by `submit_proposal`. "
            "Each label's description is the definition — match the predicate "
            "and the qualifiers before proposing it.\n"
            + ("\n".join(sub_label_blocks) or "  (no candidates — investigate but expect no submissions)")
            + "\n\n"
            "### Available telemetry graphs\n"
            + graph_list
            + "\n\n"
            "### How to work\n"
            "1. **Start from the parent label guideline above** — it tells you what aspect of the segment matters and which signals to look at. Use it to pick the 3-5 graphs you actually need.\n"
            "2. Call `list_graphs` once for the compact catalog (id / title / description) so you know which graph IDs exist.\n"
            "3. Call `get_graph_guidance(graph_ids)` on JUST the subset you chose in step 1 to pull the per-graph `how_to_analyze` procedure + glossary. Do NOT request all 11 graphs.\n"
            "4. Use `render_graph(graph_id, start, end)` to inspect signals. You can see images natively. Render at the parent range first, then zoom into interesting regions.\n"
            "5. Use `query_telemetry(query_id, params_json)` for exact ilocs / values — never estimate numbers from images when a deterministic query exists. The query catalog: `find_extremum`, `find_first_match`, `read_values_at_indices`, `compute_slope`, `find_dips_on_main_slope`, `find_threshold_crossing`. `params_json` is a JSON-encoded object and every call requires `\"range\": [start_iloc, end_iloc]` plus the per-query fields. See `get_graph_guidance` output for column names.\n"
            "6. Use `compute_expert_phases(start, end)` once if you reason about corner entry/apex/exit — phases derive from EXPERT telemetry (the player may stop mid-corner).\n"
            "6a. To pick a `circuit_section` label (named corner / straight), call `locate_circuit_section(start, end)` — it matches the segment's normalized lap position against each section's range and returns ranked candidates. Trust `best_match` when its `overlap_fraction` dominates; cite the `top_matches` list otherwise. Never infer the named section from trajectory shape alone.\n"
            "7. When evidence is sufficient, call `submit_proposal(proposals_json, summary)` with a JSON array of `{label_id, start_index, end_index, reasoning}`. Cite specific ilocs and values in `reasoning`.\n"
            "\n"
            "### Hard rules\n"
            f"- Every proposed range must satisfy {self.parent_start} <= start_index < end_index <= {self.parent_end}.\n"
            "- Do not invent label IDs. Only IDs from the candidate list above are accepted.\n"
            "- Do not propose ranges that exactly match an already-discovered sub-segment.\n"
            "- Budget your tool calls — prefer the smallest set of renders/queries that proves a label. You have a limited turn count.\n"
            "- After `submit_proposal` returns `ok: true`, stop. Do not call more tools.\n"
            f"{thinking_clause}"
        )

    def _initial_user_prompt(self) -> str:
        return (
            "Discover notable sub-segment(s) in the parent segment and submit "
            "them via `submit_proposal`. Start from the parent label "
            "guideline above to decide which graphs to look at, then pull "
            "their `how_to_analyze` blocks via `get_graph_guidance` before "
            "rendering."
        )

    # -- session ---------------------------------------------------------

    def _build_tool_set(self, *, include_submit_proposal: bool):
        """Define MCP tools and return ``(server, tool_names)``.

        ``include_submit_proposal=False`` is used by the follow-up chat
        session: no new annotations are submitted there, so the tool is
        omitted entirely.
        """
        from claude_agent_sdk import tool, create_sdk_mcp_server

        runner = self  # for tool closures

        @tool(
            "list_graphs",
            "List every available telemetry graph as a compact catalog of `id` + `title` + `description`. No arguments. Call this once early to see what graphs exist; then call `get_graph_guidance` for the subset you actually want to inspect.",
            {},
        )
        async def list_graphs(args):  # noqa: ARG001
            text = runner.tool_list_graphs()
            runner._emit_tool_event("list_graphs", {}, f"{text.count(chr(10))} line catalog")
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "get_graph_guidance",
            "Return the per-graph `how_to_analyze` skill block + glossary for ONE OR MORE graph IDs (and any cross-graph guidelines that apply to that set). Use this AFTER `list_graphs` on just the 3-5 graphs the parent label guideline points to — do NOT request all 11 at once.",
            {"graph_ids": list},
        )
        async def get_graph_guidance(args):
            ids = args.get("graph_ids") or []
            if not isinstance(ids, list):
                ids = [str(ids)]
            ids = [str(x) for x in ids]
            text = runner.tool_get_graph_guidance(ids)
            runner._emit_tool_event(
                "get_graph_guidance",
                {"graph_ids": ids},
                f"{text.count(chr(10))} line(s) of guidance",
            )
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "render_graph",
            "Render ONE telemetry graph over an index window and return the PNG image (which you can see) plus a one-line description. Use this to inspect signals. Render the parent range first, then zoom into interesting regions.",
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
            "Run a deterministic numeric query on the raw telemetry DataFrame. Returns exact ilocs / values — use this instead of estimating from rendered images. `query_id` is one of: find_extremum, find_first_match, read_values_at_indices, compute_slope, find_dips_on_main_slope, find_threshold_crossing. `params_json` is a JSON-encoded object that must include `range: [start_iloc, end_iloc]` plus the per-query fields. Example: `{\"range\": [120, 180], \"column\": \"Physics_brake\", \"kind\": \"max\"}`. See list_graphs output for column names.",
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
            "Detect expert-anchored corner phases (per-arc entry / apex / exit ilocs) over an index window. Phases derive from the EXPERT trace (the player may stop mid-corner). Empty list on straights.",
            {"start": int, "end": int},
        )
        async def compute_expert_phases(args):
            text = runner.tool_compute_expert_phases(int(args["start"]), int(args["end"]))
            runner._emit_tool_event("compute_expert_phases", {"start": args.get("start"), "end": args.get("end")}, text)
            return {"content": [{"type": "text", "text": text}]}

        @tool(
            "locate_circuit_section",
            "Identify which named circuit_section (corner / straight) the segment overlaps by matching `Graphics_normalized_car_position` against each section's `normalized_position_range`. Returns ranked candidates with overlap fractions and a `best_match` suggestion. Call this whenever you need a circuit_section label — never guess from telemetry shape alone.",
            {"start": int, "end": int},
        )
        async def locate_circuit_section(args):
            text = runner.tool_locate_circuit_section(int(args["start"]), int(args["end"]))
            runner._emit_tool_event("locate_circuit_section", {"start": args.get("start"), "end": args.get("end")}, text)
            return {"content": [{"type": "text", "text": text}]}

        tools_list = [
            list_graphs, get_graph_guidance, render_graph, query_telemetry,
            compute_expert_phases, locate_circuit_section,
        ]
        tool_names = [
            "mcp__telemetry__list_graphs",
            "mcp__telemetry__get_graph_guidance",
            "mcp__telemetry__render_graph",
            "mcp__telemetry__query_telemetry",
            "mcp__telemetry__compute_expert_phases",
            "mcp__telemetry__locate_circuit_section",
        ]

        if include_submit_proposal:
            @tool(
                "submit_proposal",
                "Submit your final sub-segment proposals and end the session. `proposals_json` must be a JSON-encoded array of objects: [{label_id, start_index, end_index, reasoning}]. `summary` is a short overall paragraph for the human reviewer. Returns `ok: true` on success; on failure it returns the validation errors so you can fix and call again.",
                {"proposals_json": str, "summary": str},
            )
            async def submit_proposal(args):
                text = runner.tool_submit_proposal(
                    str(args.get("proposals_json", "")),
                    str(args.get("summary", "")),
                )
                parsed = json.loads(text)
                summary_line = (
                    f"submitted {parsed.get('accepted_count', 0)} proposal(s)"
                    if parsed.get("ok") else f"REJECTED: {parsed.get('error', '?')}"
                )
                runner._emit_tool_event("submit_proposal", {"summary": args.get("summary")}, summary_line)
                return {"content": [{"type": "text", "text": text}]}

            tools_list.append(submit_proposal)
            tool_names.append("mcp__telemetry__submit_proposal")

        server = create_sdk_mcp_server(
            name="telemetry",
            version="1.0.0",
            tools=tools_list,
        )
        return server, tool_names

    @staticmethod
    def _import_sdk_types():
        """Import claude-agent-sdk runtime + message types in one place.

        Returns a ``SimpleNamespace`` so callers attribute-access only the
        fields they need (linters then can't complain about unused names).
        """
        from types import SimpleNamespace
        try:
            from claude_agent_sdk import (
                query, ClaudeAgentOptions,
                AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock,
            )
        except ImportError as exc:
            raise RuntimeError(
                "claude-agent-sdk is not installed. Install with "
                "`pip install claude-agent-sdk` and ensure the `claude` CLI is "
                "installed and logged in."
            ) from exc
        try:
            from claude_agent_sdk.types import ThinkingBlock  # type: ignore
        except ImportError:
            ThinkingBlock = None  # type: ignore
        return SimpleNamespace(
            query=query,
            ClaudeAgentOptions=ClaudeAgentOptions,
            AssistantMessage=AssistantMessage,
            TextBlock=TextBlock,
            ToolUseBlock=ToolUseBlock,
            ToolResultBlock=ToolResultBlock,
            ThinkingBlock=ThinkingBlock,
        )

    async def run_async(self) -> None:
        sdk = self._import_sdk_types()

        server, tool_names = self._build_tool_set(include_submit_proposal=True)

        options = sdk.ClaudeAgentOptions(
            model=self.claude_model,
            mcp_servers={"telemetry": server},
            allowed_tools=tool_names,
            system_prompt=self._build_system_prompt(),
            max_turns=self.max_turns,
        )

        # Announce the session to the live UI as one "active" call.
        if self.on_prompt:
            self.on_prompt(self._initial_user_prompt(), _stage("main"))
        if self.on_progress:
            self.on_progress("claude_analyst", "session starting")

        async for message in sdk.query(prompt=self._initial_user_prompt(), options=options):
            self._handle_message(
                message, sdk.AssistantMessage, sdk.TextBlock, sdk.ToolUseBlock,
                sdk.ToolResultBlock, sdk.ThinkingBlock,
            )

        if self.on_progress:
            self.on_progress(
                "claude_analyst",
                f"done — {self.capture.tool_calls} tool call(s), "
                f"{len(self.capture.proposals)} proposal(s)",
            )

    # -- follow-up chat --------------------------------------------------

    def _build_followup_system_prompt(self, prior_result) -> str:
        """System prompt for a follow-up Q&A turn.

        Same parent-segment / candidate-labels / graph-list context as the
        annotation prompt, but reframed: the user is interrogating prior
        proposals so they can refine the skill text. Tools are still
        available for re-investigation; ``submit_proposal`` is NOT.
        """
        # Reuse the annotation system prompt and then redirect the framing.
        # The annotation prompt has the parent context, the candidate label
        # block, the graph list, and the rules — all of that still applies.
        base = self._build_system_prompt()

        proposals_block = self._format_prior_proposals(prior_result)
        return (
            base
            + "\n\n---\n\n"
            "### THIS IS A FOLLOW-UP CHAT SESSION\n"
            "The annotation pass above already ran. Your job NOW is to help "
            "the user understand the prior proposals so they can edit the "
            "skill YAMLs (label catalog descriptions / annotation guidelines / "
            "per-graph `how_to_analyze` blocks). You are NOT producing new "
            "proposals — `submit_proposal` is not available this session.\n\n"
            "How to answer:\n"
            "- Ground every claim in telemetry evidence. Cite ilocs and values. "
            "Use `render_graph` / `query_telemetry` / `compute_expert_phases` "
            "to re-inspect when the question demands fresh evidence.\n"
            "- When asked 'why didn't label X fit?', quote the relevant text "
            "from the label's description / guideline above, then say which "
            "predicate failed against the data.\n"
            "- If the prior proposal was wrong, say so directly. Don't defend "
            "a bad call.\n"
            "- When the user is debugging the skill text, suggest concrete "
            "edits — the specific wording that was ambiguous or missing.\n"
            "- Keep replies tight. Bullets > paragraphs.\n\n"
            f"### Prior session output\n{proposals_block}"
        )

    @staticmethod
    def _format_prior_proposals(prior_result) -> str:
        proposals = list(getattr(prior_result, "label_annotations", None) or [])
        if not proposals:
            body = "  (no proposals were submitted)\n"
        else:
            lines = []
            for p in proposals:
                lid = p.get("label_id", "?")
                name = LABEL_MAPPING.get(lid, lid)
                lines.append(
                    f"  - `{lid}` ({name}) over [{p.get('start_index')}, "
                    f"{p.get('end_index')}]\n"
                    f"    reasoning: {p.get('reasoning', '') or '(none)'}"
                )
            body = "\n".join(lines) + "\n"
        summary = getattr(prior_result, "final_reasoning", "") or "(none)"
        return (
            f"Proposals submitted:\n{body}\n"
            f"Overall summary: {summary}\n"
        )

    def _build_followup_initial_prompt(
        self,
        chat_history: List[Dict[str, str]],
        user_question: str,
    ) -> str:
        if not chat_history:
            history_block = "  (this is the first follow-up question)"
        else:
            lines = []
            for turn in chat_history:
                role = turn.get("role", "user")
                content = (turn.get("content", "") or "").strip()
                tag = "User" if role == "user" else "You"
                lines.append(f"- {tag}: {content}")
            history_block = "\n".join(lines)

        return (
            "Earlier conversation in this follow-up chat:\n"
            f"{history_block}\n\n"
            f"Latest user question:\n{user_question.strip()}\n\n"
            "Answer concisely, cite ilocs / values, and use the telemetry "
            "tools if you need fresh evidence."
        )

    async def run_followup_async(
        self,
        prior_result,
        chat_history: List[Dict[str, str]],
        user_question: str,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        """Run one follow-up Q&A turn. Returns the assistant's final text."""
        sdk = self._import_sdk_types()

        server, tool_names = self._build_tool_set(include_submit_proposal=False)

        options = sdk.ClaudeAgentOptions(
            model=self.claude_model,
            mcp_servers={"telemetry": server},
            allowed_tools=tool_names,
            system_prompt=self._build_followup_system_prompt(prior_result),
            max_turns=self.max_turns,
        )

        prompt = self._build_followup_initial_prompt(
            chat_history, user_question,
        )
        response_chunks: List[str] = []

        async for message in sdk.query(prompt=prompt, options=options):
            if not isinstance(message, sdk.AssistantMessage):
                continue
            for block in getattr(message, "content", None) or []:
                if isinstance(block, sdk.TextBlock):
                    text = getattr(block, "text", "") or ""
                    if text:
                        response_chunks.append(text)
                        if on_text_chunk is not None:
                            on_text_chunk(text)
                elif isinstance(block, sdk.ToolUseBlock):
                    self.capture.tool_calls += 1
                    if on_tool_event is not None:
                        on_tool_event(
                            getattr(block, "name", "tool"),
                            getattr(block, "input", {}) or {},
                        )

        return "".join(response_chunks).strip()

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
                        "claude_analyst",
                        f"tool {self.capture.tool_calls}: {block.name}",
                    )

    def run(self) -> None:
        asyncio.run(self.run_async())


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_claude_annotation(
    df,
    start_index: int,
    end_index: int,
    session_id: str,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]] = None,
    config=None,
    progress_callback: Optional[Callable] = None,
    vlm_stream_callback: Optional[Callable] = None,
    vlm_prompt_callback: Optional[Callable] = None,
    vlm_reasoning_callback: Optional[Callable] = None,
    step_event_callback: Optional[Callable] = None,
):
    """Run one agentic Claude annotation session, return an ``AnnotationResult``.

    Mirrors the signature of ``run_annotation_pipeline`` so the UI can
    swap backends transparently.
    """
    from app.services.llm.annotation_agent_pipeline import (
        AnnotationPipelineConfig, AnnotationResult,
    )
    config = config or AnnotationPipelineConfig(backend="claude")

    runner = ClaudeAnnotationRunner(
        df=df,
        start_index=int(start_index),
        end_index=int(end_index),
        parent_main_labels=list(parent_main_labels),
        existing_children=list(existing_children or []),
        claude_model=getattr(config, "claude_model", "claude-sonnet-4-6"),
        use_thinking=bool(getattr(config, "claude_use_thinking", False)),
        max_turns=int(getattr(config, "max_iterations", 3)) * 10,
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
        "Claude annotation session: %d tool calls, %d proposals, %.1fs",
        runner.capture.tool_calls, len(runner.capture.proposals), elapsed,
    )

    proposals = runner.capture.proposals
    label_ids: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    for p in proposals:
        if p["label_id"] not in label_ids:
            label_ids.append(p["label_id"])
        starts.append(p["start_index"])
        ends.append(p["end_index"])

    sub_start = min(starts) if starts else int(start_index)
    sub_end = max(ends) if ends else int(end_index)

    transcript = "".join(runner.capture.text_chunks).strip()
    messages = [{
        "role": "claude_analyst",
        "iteration": 1,
        "content": transcript or "(no text output)",
        "verdict": "submitted" if runner.capture.submitted else "no_submission",
    }]

    return AnnotationResult(
        final_labels=label_ids,
        final_reasoning=runner.capture.final_reasoning or transcript or "(no reasoning)",
        accepted=runner.capture.submitted and len(proposals) > 0,
        iterations=1,
        messages=messages,
        graph_images=runner.capture.rendered_images,
        sub_start=sub_start,
        sub_end=sub_end,
        label_annotations=proposals,
    )


# ---------------------------------------------------------------------------
# Follow-up chat entrypoint
# ---------------------------------------------------------------------------


def run_claude_followup(
    *,
    df,
    start_index: int,
    end_index: int,
    parent_main_labels: List[str],
    existing_children: Optional[List[dict]],
    claude_model: str,
    use_thinking: bool,
    max_turns: int,
    prior_result,
    chat_history: List[Dict[str, str]],
    user_question: str,
    on_text_chunk: Optional[Callable[[str], None]] = None,
    on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> str:
    """Run one follow-up Q&A turn against the just-finished annotation.

    Returns the assistant's text reply. Uses the same parent context and
    telemetry tools as the original annotation session, minus
    ``submit_proposal`` — this session never creates new proposals.
    """
    runner = ClaudeAnnotationRunner(
        df=df,
        start_index=int(start_index),
        end_index=int(end_index),
        parent_main_labels=list(parent_main_labels),
        existing_children=list(existing_children or []),
        claude_model=claude_model,
        use_thinking=bool(use_thinking),
        max_turns=int(max_turns),
    )
    return asyncio.run(
        runner.run_followup_async(
            prior_result=prior_result,
            chat_history=list(chat_history),
            user_question=user_question,
            on_text_chunk=on_text_chunk,
            on_tool_event=on_tool_event,
        )
    )
