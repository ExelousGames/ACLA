"""Shared tool-agent surface for annotation providers.

Claude CLI and OpenAI-compatible annotation providers both use this module:
the runner owns the transport, while this surface owns telemetry tools,
range revision, and submit-result capture.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from app.shared.contracts import AgentRequest, StepEvent


def tool_agent_stage(node_name: str, phase: str, **extra) -> Dict[str, Any]:
    return {"node_name": node_name, "phase": phase, **extra}


@dataclass
class ToolAgentCapture:
    node_name: str = "annotation_agent"
    cur_start: int = 0
    cur_end: int = 0
    revised: bool = False
    submit_payload: str = ""
    submit_summary: str = ""
    submitted: bool = False
    rendered_images: List[bytes] = field(default_factory=list)
    text_chunks: List[str] = field(default_factory=list)
    step_events: List[StepEvent] = field(default_factory=list)
    tool_calls: int = 0


def _object_schema(properties: Dict[str, Dict[str, Any]], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _openai_tool_schema(defn: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": defn["name"],
            "description": defn["description"],
            "parameters": _object_schema(defn["openai_properties"], defn["required"]),
        },
    }


ANNOTATION_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "list_graphs",
        "description": "List available telemetry graphs.",
        "params_schema": {},
        "openai_properties": {},
        "required": [],
    },
    {
        "name": "get_circuit_id",
        "description": "Return canonical circuit id from Static_track.",
        "params_schema": {},
        "openai_properties": {},
        "required": [],
    },
    {
        "name": "get_graph_guidance",
        "description": "Return graph analysis guidance.",
        "params_schema": {"graph_ids": list},
        "openai_properties": {
            "graph_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["graph_ids"],
    },
    {
        "name": "render_graph",
        "description": "Render one telemetry graph over an iloc window.",
        "params_schema": {"graph_id": str, "start": int, "end": int},
        "openai_properties": {
            "graph_id": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["graph_id", "start", "end"],
    },
    {
        "name": "peek_graph",
        "description": "Render one graph for lap-context outside the working section.",
        "params_schema": {"graph_id": str, "start": int, "end": int},
        "openai_properties": {
            "graph_id": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["graph_id", "start", "end"],
    },
    {
        "name": "query_telemetry",
        "description": "Run a deterministic telemetry query.",
        "params_schema": {"query_id": str, "params_json": str},
        "openai_properties": {
            "query_id": {"type": "string"},
            "params_json": {"type": "string"},
        },
        "required": ["query_id", "params_json"],
    },
    {
        "name": "compute_expert_phases",
        "description": "Detect expert corner phases.",
        "params_schema": {"start": int, "end": int},
        "openai_properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["start", "end"],
    },
    {
        "name": "measure_segment_shape",
        "description": "Measure segment shape and altitude trends.",
        "params_schema": {"start": int, "end": int},
        "openai_properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["start", "end"],
    },
    {
        "name": "locate_circuit_section",
        "description": "Identify named circuit section overlap.",
        "params_schema": {"start": int, "end": int},
        "openai_properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["start", "end"],
    },
    {
        "name": "find_nearest_opponent",
        "description": "Rank nearest opponent cars.",
        "params_schema": {"start": int, "end": int},
        "openai_properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["start", "end"],
    },
    {
        "name": "classify_opponent_interaction",
        "description": "Classify opponent-relative pattern.",
        "params_schema": {"start": int, "end": int},
        "openai_properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["start", "end"],
    },
    {
        "name": "query_opponent_trajectory",
        "description": "Sample one opponent slot trajectory.",
        "params_schema": {"start": int, "end": int, "slot": int, "n_samples": int},
        "openai_properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "slot": {"type": "integer"},
            "n_samples": {"type": "integer"},
        },
        "required": ["start", "end", "slot", "n_samples"],
    },
    {
        "name": "revise_range",
        "description": "Revise the working iloc range before submitting.",
        "params_schema": {"new_start": int, "new_end": int},
        "openai_properties": {
            "new_start": {"type": "integer"},
            "new_end": {"type": "integer"},
        },
        "required": ["new_start", "new_end"],
    },
    {
        "name": "submit_result",
        "description": "Submit final structured JSON result.",
        "params_schema": {"payload_json": str, "summary": str},
        "openai_properties": {
            "payload_json": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["payload_json", "summary"],
    },
]


def annotation_tool_names() -> List[str]:
    return [str(defn["name"]) for defn in ANNOTATION_TOOL_DEFINITIONS]


def annotation_openai_tool_schemas() -> List[Dict[str, Any]]:
    return [_openai_tool_schema(defn) for defn in ANNOTATION_TOOL_DEFINITIONS]


def tool_agent_extra_tools(request: AgentRequest) -> List[Dict[str, Any]]:
    return request.extra_state.get("tool_agent_extra_tools") or []


def _schema_type_for_python_type(typ: Any) -> Dict[str, Any]:
    if typ is int:
        return {"type": "integer"}
    if typ is float:
        return {"type": "number"}
    if typ is bool:
        return {"type": "boolean"}
    if typ is list:
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string"}


def annotation_openai_extra_tool_schemas(request: AgentRequest) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for spec in tool_agent_extra_tools(request):
        params = spec.get("params_schema") or {}
        properties = {str(key): _schema_type_for_python_type(typ) for key, typ in params.items()}
        schemas.append(_openai_tool_schema({
            "name": str(spec["name"]),
            "description": str(spec.get("description") or spec["name"]),
            "openai_properties": properties,
            "required": list(properties),
        }))
    return schemas


def _text_from_tool_result(result: Any) -> Tuple[str, List[str]]:
    if isinstance(result, str):
        return result, []
    if not isinstance(result, dict):
        return json.dumps(result, default=str), []
    texts: List[str] = []
    images: List[str] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            texts.append(str(item.get("text") or ""))
        elif item.get("type") == "image" and item.get("data"):
            images.append(str(item["data"]))
    if not texts:
        texts.append(json.dumps(result, default=str))
    return "\n".join(texts), images


class AnnotationToolSurface:
    """Thin object holding df + capture; runners call its methods as tools."""

    def __init__(self, request: AgentRequest, capture: ToolAgentCapture) -> None:
        self.df = request.df_ref
        self.request = request
        self.capture = capture

    def _clamp_to_window(self, s: int, e: int) -> tuple[int, int]:
        lo = min(self.capture.cur_start, self.request.parent_start)
        hi = max(self.capture.cur_end, self.request.parent_end)
        s2 = max(lo, int(s))
        e2 = min(hi, int(e))
        if e2 <= s2:
            e2 = min(hi, s2 + 1)
        return s2, e2

    def _clamp_to_lap(self, s: int, e: int) -> tuple[int, int]:
        n = len(self.df)
        s2 = max(0, int(s))
        e2 = min(n, int(e))
        if e2 <= s2:
            e2 = min(n, s2 + 1)
        return s2, e2

    def _emit_tool_event(self, name: str, inp: Dict[str, Any], summary: str) -> None:
        inp_str = json.dumps(inp, default=str)
        if len(inp_str) > 400:
            inp_str = inp_str[:400] + "..."
        if len(summary) > 600:
            summary = summary[:600] + "..."
        msg = (
            f"**Tool:** `{name}`\n\n"
            f"**Input:** `{inp_str}`\n\n"
            f"**Result:** {summary}"
        )
        stage = tool_agent_stage(self.capture.node_name, f"tool:{name}")
        self.capture.step_events.append(StepEvent(
            stage=stage["node_name"], summary=msg, detail=stage,
        ))
        cb = self.request.callbacks
        if cb.step_event:
            cb.step_event(msg, stage)

    def list_graphs(self) -> str:
        from app.shared.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS
        out = [
            {"id": g["id"], "title": g["title"], "description": g["description"]}
            for g in AGENT_GRAPH_DEFINITIONS
        ]
        return json.dumps({"graphs": out}, indent=2)

    def get_circuit_id(self) -> str:
        from app.shared.annotation_agent_tools import get_circuit_id
        att = get_circuit_id(self.df)
        return json.dumps(att.content, default=str)

    def get_graph_guidance(self, graph_ids: List[str]) -> str:
        from app.shared.annotation_agent_tools import graph_analysis_prompt
        text = graph_analysis_prompt(graph_ids=list(graph_ids))
        return text or "(no guidance available for the requested graph(s))"

    def render_graph(self, graph_id: str, start: int, end: int) -> Dict[str, Any]:
        from app.shared.annotation_agent_tools import build_graph, render_graph_builds
        s, e = self._clamp_to_window(start, end)
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

    def peek_graph(self, graph_id: str, start: int, end: int) -> Dict[str, Any]:
        from app.shared.annotation_agent_tools import build_graph, render_graph_builds
        s, e = self._clamp_to_lap(start, end)
        table = build_graph(graph_id, self.df)
        if table is None or table.empty:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"Cannot peek `{graph_id}` over [{s}, {e}]: the "
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
                {"type": "text", "text": (
                    f"{desc} (peek - context only, working range unchanged, "
                    f"rendered over [{s}, {e}])"
                )},
            ],
        }

    def query_telemetry(self, query_id: str, params_json: str) -> str:
        from app.shared.annotation_agent_tools import run_pipeline_query
        try:
            params = json.loads(params_json) if params_json else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"params_json was not valid JSON: {exc}"})
        if not isinstance(params, dict):
            return json.dumps({"error": "params_json must decode to a JSON object."})
        payload, err = run_pipeline_query(self.df, query_id, params)
        out = {"query": query_id, "params": params, "result": payload}
        if err:
            out["error"] = err
        return json.dumps(out, default=str)

    def compute_expert_phases(self, start: int, end: int) -> str:
        from app.shared.annotation_agent_tools import compute_expert_phases
        s, e = self._clamp_to_window(start, end)
        att = compute_expert_phases(self.df, s, e)
        return json.dumps({"phases_range": [s, e], "data": att.content}, default=str)

    def measure_segment_shape(self, start: int, end: int) -> str:
        from app.shared.annotation_agent_tools import measure_segment_shape
        s, e = self._clamp_to_window(start, end)
        att = measure_segment_shape(self.df, s, e)
        return json.dumps({"range": [s, e], "data": att.content}, default=str)

    def locate_circuit_section(self, start: int, end: int) -> str:
        from app.shared.annotation_agent_tools import locate_circuit_section
        s, e = self._clamp_to_window(start, end)
        att = locate_circuit_section(self.df, s, e)
        return json.dumps({"range": [s, e], "data": att.content}, default=str)

    def find_nearest_opponent(self, start: int, end: int) -> str:
        from app.shared.annotation_agent_tools import find_nearest_opponent
        s, e = self._clamp_to_window(start, end)
        att = find_nearest_opponent(self.df, s, e)
        return json.dumps({"range": [s, e], "data": att.content}, default=str)

    def classify_opponent_interaction(self, start: int, end: int) -> str:
        from app.shared.annotation_agent_tools import classify_opponent_interaction
        s, e = self._clamp_to_window(start, end)
        att = classify_opponent_interaction(self.df, s, e)
        return json.dumps({"range": [s, e], "data": att.content}, default=str)

    def query_opponent_trajectory(self, start: int, end: int, slot: int, n_samples: int) -> str:
        from app.shared.annotation_agent_tools import query_opponent_trajectory
        s, e = self._clamp_to_window(start, end)
        att = query_opponent_trajectory(self.df, s, e, slot=int(slot), n_samples=int(n_samples))
        return json.dumps({"range": [s, e], "data": att.content}, default=str)

    def revise_range(self, new_start: int, new_end: int) -> str:
        s, e = int(new_start), int(new_end)
        if e <= s:
            return json.dumps({"ok": False, "error": f"new range [{s}, {e}] requires start < end"})
        if (e - s) < 5:
            return json.dumps({"ok": False, "error": f"new range too short ({e - s} ilocs) - minimum 5 required"})
        self.capture.cur_start = s
        self.capture.cur_end = e
        self.capture.revised = True
        return json.dumps({
            "ok": True,
            "new_range": [s, e],
            "note": "Working range updated. Tool calls now operate against this range.",
        })

    def submit_result(self, payload_json: str, summary: str) -> str:
        try:
            json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return json.dumps({
                "ok": False,
                "error": f"payload_json was not valid JSON: {exc}. Re-emit with valid JSON.",
            })
        self.capture.submit_payload = payload_json
        self.capture.submit_summary = str(summary or "")
        self.capture.submitted = True
        return json.dumps({"ok": True, "note": "Result captured. Session can end now."})

    def _extra_tool_handler(self, name: str) -> Callable[["AnnotationToolSurface", Dict[str, Any]], Any] | None:
        for spec in tool_agent_extra_tools(self.request):
            if str(spec.get("name")) == name:
                return spec["handler"]
        return None

    def call_tool(self, name: str, args: Dict[str, Any]) -> Tuple[Any, str, List[str]]:
        if name == "list_graphs":
            result = self.list_graphs()
        elif name == "get_circuit_id":
            result = self.get_circuit_id()
        elif name == "get_graph_guidance":
            result = self.get_graph_guidance(list(args.get("graph_ids") or []))
        elif name == "render_graph":
            result = self.render_graph(str(args["graph_id"]), int(args["start"]), int(args["end"]))
        elif name == "peek_graph":
            result = self.peek_graph(str(args["graph_id"]), int(args["start"]), int(args["end"]))
        elif name == "query_telemetry":
            result = self.query_telemetry(str(args["query_id"]), str(args.get("params_json") or ""))
        elif name == "compute_expert_phases":
            result = self.compute_expert_phases(int(args["start"]), int(args["end"]))
        elif name == "measure_segment_shape":
            result = self.measure_segment_shape(int(args["start"]), int(args["end"]))
        elif name == "locate_circuit_section":
            result = self.locate_circuit_section(int(args["start"]), int(args["end"]))
        elif name == "find_nearest_opponent":
            result = self.find_nearest_opponent(int(args["start"]), int(args["end"]))
        elif name == "classify_opponent_interaction":
            result = self.classify_opponent_interaction(int(args["start"]), int(args["end"]))
        elif name == "query_opponent_trajectory":
            result = self.query_opponent_trajectory(
                int(args["start"]), int(args["end"]),
                int(args["slot"]), int(args.get("n_samples") or 5),
            )
        elif name == "revise_range":
            result = self.revise_range(int(args["new_start"]), int(args["new_end"]))
        elif name == "submit_result":
            result = self.submit_result(str(args.get("payload_json") or ""), str(args.get("summary") or ""))
        else:
            handler = self._extra_tool_handler(name)
            if handler is None:
                result = json.dumps({"error": f"unknown tool {name!r}"})
            else:
                result = handler(self, args)
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)

        text, images = _text_from_tool_result(result)
        self._emit_tool_event(name, args, text)
        return result, text, images


def build_tool_agent_system_prompt(request: AgentRequest) -> str:
    thinking_clause = (
        "\nThink step-by-step before each tool call: state what you need "
        "to confirm, pick the most direct tool, read the result before "
        "deciding the next step.\n"
        if bool(request.config.provider_options.get("use_thinking"))
        else ""
    )

    return (
        "You are an analyst with agentic access to a domain dataset via tools. "
        "Your task is described in the user message. Inspect the data, run "
        "queries, then submit a final structured result.\n\n"
        "Available tools include list_graphs, get_graph_guidance, render_graph, "
        "peek_graph, query_telemetry, compute_expert_phases, "
        "measure_segment_shape, locate_circuit_section, find_nearest_opponent, "
        "classify_opponent_interaction, query_opponent_trajectory, "
        "get_circuit_id, revise_range, and submit_result.\n\n"
        f"Initial range: [{request.parent_start}, {request.parent_end}]. "
        "Do not invent identifiers. Use only IDs, labels, and categories the "
        "user message authorizes. Budget tool calls. After submit_result "
        "returns ok: true, do not call more tools.\n"
        f"{thinking_clause}"
    )


def tool_agent_response(capture: ToolAgentCapture, request: AgentRequest):
    from app.shared.contracts import AgentResponse, Attachment

    transcript = "".join(capture.text_chunks).strip()
    attachments: Dict[str, Attachment] = {}
    if capture.submitted and capture.submit_summary:
        attachments["synthesizer.summary"] = Attachment(
            name="synthesizer.summary",
            kind="text",
            label="Provider Submission Summary",
            content=capture.submit_summary,
        )
    if transcript:
        attachments["tool_agent.transcript"] = Attachment(
            name="tool_agent.transcript",
            kind="text",
            label="Provider Transcript",
            content=transcript,
        )
    if capture.revised:
        attachments["tool_agent.revised_range"] = Attachment(
            name="tool_agent.revised_range",
            kind="structured",
            label="Revised Range",
            content={
                "start_index": capture.cur_start,
                "end_index": capture.cur_end,
                "revised_from": [request.parent_start, request.parent_end],
            },
        )

    messages = [{
        "role": capture.node_name,
        "content": transcript or "(no text output)",
        "verdict": "submitted" if capture.submitted else "no_submission",
    }]
    return AgentResponse(
        raw_response=capture.submit_payload,
        verdict="submitted" if capture.submitted else "no_submission",
        attachments=attachments,
        step_events=capture.step_events,
        graph_images=list(capture.rendered_images),
        plan_steps=[],
        messages=messages,
    )
