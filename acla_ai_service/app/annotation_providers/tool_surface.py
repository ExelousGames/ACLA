"""Shared tool-agent surface for annotation providers.

Claude CLI and OpenAI-compatible annotation providers both use this module:
the runner owns the transport, while this surface owns telemetry tools,
and submit-result capture.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Tuple

from app.shared.contracts import AgentRequest, StepEvent


def tool_agent_stage(node_name: str, phase: str, **extra) -> Dict[str, Any]:
    return {"node_name": node_name, "phase": phase, **extra}


@dataclass
class ToolAgentCapture:
    node_name: str = "annotation_agent"
    cur_start: int = 0
    cur_end: int = 0
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


def _tool_def(
    name: str,
    description: str,
    params_schema: Dict[str, Any],
    *,
    required: List[str] | None = None,
    category: str = "general",
) -> Dict[str, Any]:
    required_params = list(required if required is not None else params_schema.keys())
    return {
        "name": name,
        "description": description,
        "params_schema": params_schema,
        "openai_properties": {
            str(key): _schema_type_for_python_type(typ)
            for key, typ in params_schema.items()
        },
        "required": required_params,
        "category": category,
    }


ANNOTATION_TOOL_REGISTRY: List[Dict[str, Any]] = [
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
        "description": "Render one graph over the current working range without changing it.",
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
]


EXPOSED_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    _tool_def(
        "recommend_tools",
        (
            "Recommend annotation capability tool IDs for the stated intent. "
            "Call this before inspecting data; then execute returned IDs with "
            "run_annotation_tool."
        ),
        {"intent": str, "context_json": str},
        category="meta",
    ),
    _tool_def(
        "run_annotation_tool",
        (
            "Execute a recommended annotation capability by ID. args_json is "
            "a JSON object matching that capability's parameters."
        ),
        {"tool_id": str, "args_json": str},
        category="meta",
    ),
    _tool_def(
        "search_annotation_guidance",
        (
            "Search annotation skill guidance and workflow rules. query is "
            "plain language; scope may be a skill name or empty."
        ),
        {"query": str, "scope": str},
        category="knowledge",
    ),
    {
        "name": "submit_result",
        "description": "Submit final structured JSON result.",
        "params_schema": {"payload_json": str, "summary": str},
        "openai_properties": {
            "payload_json": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["payload_json", "summary"],
        "category": "control",
    },
]


def annotation_tool_names() -> List[str]:
    return [str(defn["name"]) for defn in EXPOSED_TOOL_DEFINITIONS]


def annotation_tool_registry() -> List[Dict[str, Any]]:
    return [dict(defn) for defn in ANNOTATION_TOOL_REGISTRY]


def _normalise_extra_tool_def(spec: Dict[str, Any]) -> Dict[str, Any]:
    params = spec.get("params_schema") or {}
    properties = {
        str(key): _schema_type_for_python_type(typ)
        for key, typ in params.items()
    }
    return {
        **spec,
        "name": str(spec["name"]),
        "description": str(spec.get("description") or spec["name"]),
        "openai_properties": properties,
        "required": list(properties),
        "category": str(spec.get("category") or "domain"),
    }


def annotation_tool_definitions(request: AgentRequest | None = None) -> List[Dict[str, Any]]:
    return [
        *EXPOSED_TOOL_DEFINITIONS,
        *(
            _normalise_extra_tool_def(spec)
            for spec in (tool_agent_extra_tools(request) if request is not None else [])
        ),
    ]


def annotation_openai_tool_schemas(request: AgentRequest | None = None) -> List[Dict[str, Any]]:
    return [_openai_tool_schema(defn) for defn in annotation_tool_definitions(request)]


def tool_agent_extra_tools(request: AgentRequest) -> List[Dict[str, Any]]:
    return request.extra_state.get("tool_agent_extra_tools") or []


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


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9_]+", text.lower())
        if len(token) >= 3
    }


def _safe_json_object(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_search_text(defn: Dict[str, Any]) -> str:
    fields: List[str] = [
        str(defn.get("name") or ""),
        str(defn.get("description") or ""),
        str(defn.get("category") or ""),
    ]
    fields.extend(str(key) for key in (defn.get("params_schema") or {}).keys())
    return " ".join(fields)


def _capability_by_name() -> Dict[str, Dict[str, Any]]:
    return {str(defn["name"]): defn for defn in ANNOTATION_TOOL_REGISTRY}


def _iter_guidance_records() -> Iterable[Dict[str, str]]:
    from app.internal_knowledge_base._registry import get_registry

    def walk(value: Any, path: str, skill_name: str) -> Iterable[Dict[str, str]]:
        if isinstance(value, str):
            text = value.strip()
            if text:
                yield {"skill": skill_name, "path": path, "text": text}
        elif isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                yield from walk(child, child_path, skill_name)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                yield from walk(child, f"{path}[{idx}]", skill_name)

    for skill in get_registry().all_skills():
        yield from walk(skill.raw_body, skill.name, skill.name)


def _request_session_context(request: AgentRequest) -> str:
    context = str(request.extra_state.get("annotation_session_context") or "").strip()
    return context if context in {"practice", "racing"} else ""


def _request_eligible_behavior_label_ids(request: AgentRequest) -> List[str]:
    raw = request.extra_state.get("eligible_behavior_label_ids") or []
    return [str(label_id) for label_id in raw if str(label_id)]


def _label_path_allowed_for_request(request: AgentRequest, path: str) -> bool:
    eligible = set(_request_eligible_behavior_label_ids(request))
    if not eligible:
        return True
    required_parents = {"O", "OD", "PS", "RM", "MSP", "MSR"}

    lap_match = re.search(r"lap_annotation\.labels\.([A-Za-z0-9_]+)", path)
    if lap_match:
        return lap_match.group(1) in eligible

    sub_match = re.search(r"sub_label_annotation\.labels\.([A-Za-z0-9_]+)", path)
    if sub_match:
        try:
            from app.internal_knowledge_base.label_lookup import get_label

            doc = get_label(sub_match.group(1))
        except Exception:
            doc = None
        parent = str((doc or {}).get("parent") or "")
        if parent in required_parents:
            return parent in eligible
    return True


def _mode_specific_guidance_record(request: AgentRequest) -> Dict[str, str] | None:
    context = _request_session_context(request)
    eligible = _request_eligible_behavior_label_ids(request)
    if not context or not eligible:
        return None
    label_set = "{" + ", ".join(eligible) + "}"
    if context == "racing":
        text = (
            "Detected session mode: racing / opponent interaction. Only "
            f"behavior parent labels from {label_set} are eligible. Use "
            "O for a completed attack, OD for a held defense, or MSR for a "
            "failed attack / broken defense; use PS for pit-lane procedure "
            "when pit evidence fits the whole range. Gate O / OD / MSR with "
            "`classify_opponent_interaction(start, end)` over the full "
            "working range. Do not evaluate or attach practice-session "
            "behavior parents MSP / RM; submit [] if no racing or "
            "pit-stop label fits."
        )
    else:
        text = (
            "Detected session mode: practice / solo section. Only behavior "
            f"parent labels from {label_set} are eligible. Use MSP for "
            "technical driving mistakes, RM for recovery, or PS for pit-lane "
            "procedure. Do not "
            "evaluate or attach racing-session behavior parents O / OD / "
            "MSR."
        )
    return {
        "skill": "lap_annotation",
        "path": "lap_annotation.detected_session_rules",
        "text": text,
    }


class AnnotationToolSurface:
    """Thin object holding df + capture; runners call its methods as tools."""

    def __init__(self, request: AgentRequest, capture: ToolAgentCapture) -> None:
        self.df = request.df_ref
        self.request = request
        self.capture = capture

    def _current_window(self) -> tuple[int, int]:
        start = int(self.capture.cur_start)
        end = int(self.capture.cur_end)
        if end <= start:
            return int(self.request.parent_start), int(self.request.parent_end)
        return start, end

    def _clamp_to_window(self, s: int, e: int) -> tuple[int, int]:
        lo, hi = self._current_window()
        s2 = max(lo, int(s))
        e2 = min(hi, int(e))
        if e2 <= s2:
            e2 = min(hi, s2 + 1)
        return s2, e2

    def _clamp_query_params_to_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        raw_range = out.get("range")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            try:
                out["range"] = list(self._clamp_to_window(
                    int(raw_range[0]), int(raw_range[1]),
                ))
            except (TypeError, ValueError):
                pass
        return out

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

    def recommend_tools(self, intent: str, context_json: str) -> str:
        context = _safe_json_object(context_json)
        top_k = int(context.get("top_k") or 6)
        query_tokens = _tokens(f"{intent} {json.dumps(context, default=str)}")
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for defn in ANNOTATION_TOOL_REGISTRY:
            tool_tokens = _tokens(_tool_search_text(defn))
            score = len(query_tokens & tool_tokens)
            if score:
                scored.append((score, defn))
        scored.sort(key=lambda item: (-item[0], str(item[1]["name"])))

        recommendations = []
        for score, defn in scored[:max(1, top_k)]:
            recommendations.append({
                "tool_id": defn["name"],
                "description": defn["description"],
                "args_schema": defn.get("openai_properties") or {},
                "required": defn.get("required") or [],
                "match_score": score,
            })
        return json.dumps({
            "intent": intent,
            "recommendations": recommendations,
            "note": (
                "Use run_annotation_tool with one of these tool_id values. "
                "If recommendations is empty, call again with a more specific intent."
            ),
        }, default=str)

    def run_annotation_tool(self, tool_id: str, args_json: str) -> Any:
        tool_id = str(tool_id or "").strip()
        if tool_id not in _capability_by_name():
            return json.dumps({
                "error": f"unknown annotation capability {tool_id!r}",
                "known_tool_ids": sorted(_capability_by_name()),
            })
        args = _safe_json_object(args_json)
        return self._call_annotation_capability(tool_id, args)

    def search_annotation_guidance(self, query: str, scope: str) -> str:
        q = str(query or "").strip()
        if not q:
            return json.dumps({"error": "query is required"})
        scope = str(scope or "").strip()
        query_tokens = _tokens(q)
        matches: List[Tuple[int, Dict[str, str]]] = []
        for record in _iter_guidance_records():
            if scope and scope not in record["path"] and scope != record["skill"]:
                continue
            if not _label_path_allowed_for_request(self.request, record["path"]):
                continue
            text_tokens = _tokens(
                f"{record['skill']} {record['path']} {record['text']}"
            )
            score = len(query_tokens & text_tokens)
            if score:
                matches.append((score, record))
        matches.sort(key=lambda item: (-item[0], item[1]["path"]))
        return json.dumps({
            "query": q,
            "scope": scope,
            "results": [
                {
                    "skill": record["skill"],
                    "path": record["path"],
                    "text": record["text"],
                    "match_score": score,
                }
                for score, record in matches[:5]
            ],
        }, default=str)

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
        s, e = self._clamp_to_window(start, end)
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
        params = self._clamp_query_params_to_window(params)
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

    def _call_annotation_capability(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "list_graphs":
            return self.list_graphs()
        if name == "get_circuit_id":
            return self.get_circuit_id()
        if name == "get_graph_guidance":
            return self.get_graph_guidance(list(args.get("graph_ids") or []))
        if name == "render_graph":
            return self.render_graph(str(args["graph_id"]), int(args["start"]), int(args["end"]))
        if name == "peek_graph":
            return self.peek_graph(str(args["graph_id"]), int(args["start"]), int(args["end"]))
        if name == "query_telemetry":
            return self.query_telemetry(str(args["query_id"]), str(args.get("params_json") or ""))
        if name == "compute_expert_phases":
            return self.compute_expert_phases(int(args["start"]), int(args["end"]))
        if name == "measure_segment_shape":
            return self.measure_segment_shape(int(args["start"]), int(args["end"]))
        if name == "locate_circuit_section":
            return self.locate_circuit_section(int(args["start"]), int(args["end"]))
        if name == "find_nearest_opponent":
            return self.find_nearest_opponent(int(args["start"]), int(args["end"]))
        if name == "classify_opponent_interaction":
            return self.classify_opponent_interaction(int(args["start"]), int(args["end"]))
        if name == "query_opponent_trajectory":
            return self.query_opponent_trajectory(
                int(args["start"]), int(args["end"]),
                int(args["slot"]), int(args.get("n_samples") or 5),
            )
        return json.dumps({"error": f"unknown annotation capability {name!r}"})

    def call_tool(self, name: str, args: Dict[str, Any]) -> Tuple[Any, str, List[str]]:
        if name == "recommend_tools":
            result = self.recommend_tools(
                str(args.get("intent") or ""),
                str(args.get("context_json") or ""),
            )
        elif name == "run_annotation_tool":
            result = self.run_annotation_tool(
                str(args.get("tool_id") or ""),
                str(args.get("args_json") or ""),
            )
        elif name == "search_annotation_guidance":
            result = self.search_annotation_guidance(
                str(args.get("query") or ""),
                str(args.get("scope") or ""),
            )
        elif name in _capability_by_name():
            result = self._call_annotation_capability(name, args)
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

    context = _request_session_context(request)
    eligible = _request_eligible_behavior_label_ids(request)
    context_line = ""
    if context and eligible:
        context_line = (
            f"Detected annotation mode: {context}. Eligible behavior parent "
            f"labels: {{{', '.join(eligible)}}}. Do not inspect or submit "
            "behavior parents outside that set.\n"
        )

    return (
        "You are an analyst with agentic access to a domain dataset via tools. "
        "Your task is described in the user message. Inspect the data, run "
        "queries, then submit a final structured result.\n\n"
        "Use `recommend_tools` to discover the most relevant data-inspection "
        "capabilities, then execute chosen capability IDs with "
        "`run_annotation_tool`. Use `search_annotation_guidance` and "
        "`search_labels` to retrieve rules and label definitions instead of "
        "guessing from memory. Finish with `submit_result`.\n\n"
        "A label is valid only when its definition fits the whole range it "
        "will be attached to; if it fits only a smaller slice, omit that "
        "label.\n\n"
        f"Initial range: [{request.parent_start}, {request.parent_end}]. "
        f"{context_line}"
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
