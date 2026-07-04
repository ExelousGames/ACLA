"""Shared tool-agent surface for annotation providers.

Claude CLI and OpenAI-compatible annotation providers both use this module:
the runner owns the transport, while this surface owns telemetry tools,
and submit-result capture.
"""

from __future__ import annotations

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


ANNOTATION_TOOL_REGISTRY: List[Dict[str, Any]] = [
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
        "description": "Identify named circuit section overlap within a circuit.",
        "params_schema": {"circuit_id": str, "start": int, "end": int},
        "openai_properties": {
            "circuit_id": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
        "required": ["circuit_id", "start", "end"],
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
        *ANNOTATION_TOOL_REGISTRY,
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


def _capability_by_name() -> Dict[str, Dict[str, Any]]:
    return {str(defn["name"]): defn for defn in ANNOTATION_TOOL_REGISTRY}


def _request_session_context(request: AgentRequest) -> str:
    context = str(request.extra_state.get("annotation_session_context") or "").strip()
    return context if context in {"practice", "racing"} else ""


def _request_eligible_behavior_label_ids(request: AgentRequest) -> List[str]:
    raw = request.extra_state.get("eligible_behavior_label_ids") or []
    return [str(label_id) for label_id in raw if str(label_id)]


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

    def locate_circuit_section(self, circuit_id: str, start: int, end: int) -> str:
        from app.shared.annotation_agent_tools import locate_circuit_section
        s, e = self._clamp_to_window(start, end)
        att = locate_circuit_section(self.df, circuit_id, s, e)
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
        if name == "query_telemetry":
            return self.query_telemetry(str(args["query_id"]), str(args.get("params_json") or ""))
        if name == "compute_expert_phases":
            return self.compute_expert_phases(int(args["start"]), int(args["end"]))
        if name == "measure_segment_shape":
            return self.measure_segment_shape(int(args["start"]), int(args["end"]))
        if name == "locate_circuit_section":
            return self.locate_circuit_section(
                str(args["circuit_id"]),
                int(args["start"]),
                int(args["end"]),
            )
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
        if name in _capability_by_name():
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
        "You are an analyst with access to deterministic telemetry analysis "
        "tools and label search. "
        "Your task is described in the user message. "
        "Use the upfront preflight data included in that user message as the "
        "primary evidence package. Run deterministic analysis tools only when "
        "a concrete numeric check is needed, then submit a final structured "
        "result.\n\n"
        "The user message includes a Required Upfront Annotation Preflight "
        "block. Treat it as the primary analysis package: deterministic tool "
        "outputs, tool output tags, and semantic label candidates were already "
        "computed before this session. Use label search only for targeted "
        "semantic re-queries and deterministic analysis tools only for "
        "targeted numeric checks. Finish with "
        "`submit_result`.\n\n"
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
