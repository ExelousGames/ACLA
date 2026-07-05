"""Local annotation harness backed by the llama.cpp VLM service.

The local provider runs a small, explicit harness:

    planner -> task workers -> truth verifier -> finalizer -> final verifier

The planner must split the requester prompt into separate tasks. Workers run
those tasks independently and report claims with evidence. The verifier checks
whether each worker is telling the truth, asks follow-up questions when a claim
looks unsupported, and only verified task results reach the finalizer. The
finalizer returns to the requester through ``submit_result``.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.annotation_providers.tool_surface import (
    AnnotationToolSurface,
    ToolAgentCapture,
    annotation_openai_tool_schemas,
    build_tool_agent_system_prompt,
    tool_agent_stage,
)
from app.local_annotation_agent.backend import (
    LocalVLMConfig,
    get_or_start_service,
)
from app.shared.contracts import (
    AgentRequest,
    AgentResponse,
    Attachment,
    StepEvent,
)

LOGGER = logging.getLogger(__name__)

_LOCAL_NODE = "local_harness"
_MIN_PLANNED_TASKS = 2
_MAX_VERIFIER_CHALLENGES = 2


class PlannerFormatError(RuntimeError):
    """Raised when the local planner returns no valid task plan."""


@dataclass
class HarnessTask:
    task_id: str
    title: str
    instructions: str
    success_criteria: List[str] = field(default_factory=list)
    agent: str = "worker"


@dataclass
class WorkerRecord:
    task: HarnessTask
    result: str
    verifier_reports: List[Dict[str, Any]] = field(default_factory=list)
    verified: bool = False


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------


def _local_vlm_config_from_request(request: AgentRequest) -> LocalVLMConfig:
    opts = request.config.provider_options
    return LocalVLMConfig(
        gguf_path=opts.get("gguf_path") or None,
        mmproj_path=opts.get("mmproj_path") or None,
        context_size=int(opts.get("context_size") or 32768),
        n_gpu_layers=int(opts.get("n_gpu_layers") if opts.get("n_gpu_layers") is not None else -1),
        hf_repo=request.config.model or str(opts.get("hf_repo") or "Qwen/Qwen2.5-VL-72B-Instruct"),
        quantization_type=str(opts.get("quantization_type") or "Q4_K_M"),
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _loads_object(text: str) -> Dict[str, Any] | None:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed.strip())
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_object(raw: str) -> Dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None

    parsed = _loads_object(text)
    if parsed is not None:
        return parsed

    if "```json" in text:
        fenced = text.split("```json", 1)[1].split("```", 1)[0]
        parsed = _loads_object(fenced)
        if parsed is not None:
            return parsed
    elif "```" in text:
        fenced = text.split("```", 1)[1].split("```", 1)[0]
        parsed = _loads_object(fenced)
        if parsed is not None:
            return parsed

    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return _loads_object(brace_match.group())
    return None


def _parse_plan(raw: str) -> Tuple[List[HarnessTask], str]:
    parsed = _extract_json_object(raw)
    if not parsed:
        raise PlannerFormatError("Planner did not return a JSON object.")

    raw_tasks = parsed.get("tasks") or parsed.get("steps")
    if not isinstance(raw_tasks, list):
        raise PlannerFormatError("Planner JSON did not contain a tasks array.")

    tasks: List[HarnessTask] = []
    for idx, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or item.get("step_id") or f"task_{idx}")
        title = str(item.get("title") or item.get("name") or f"Task {idx}")
        instructions = str(
            item.get("instructions")
            or item.get("description")
            or item.get("prompt")
            or ""
        ).strip()
        if not instructions:
            continue
        criteria = item.get("success_criteria") or []
        if not isinstance(criteria, list):
            criteria = [str(criteria)]
        tasks.append(HarnessTask(
            task_id=task_id,
            title=title,
            instructions=instructions,
            success_criteria=[str(x) for x in criteria if str(x).strip()],
            agent=str(item.get("agent") or "worker"),
        ))

    if len(tasks) < _MIN_PLANNED_TASKS:
        raise PlannerFormatError(
            f"Planner returned {len(tasks)} task(s); expected at least "
            f"{_MIN_PLANNED_TASKS} separate tasks."
        )
    return tasks, str(parsed.get("final_instructions") or "")


def _parse_verifier(raw: str) -> Dict[str, Any]:
    parsed = _extract_json_object(raw) or {}
    verdict = str(parsed.get("verdict") or "").lower()
    if verdict not in {"pass", "challenge", "fail"}:
        return {
            "verdict": "challenge",
            "questions": ["Restate the answer with explicit evidence for each claim."],
            "reason": "Verifier response was not valid; asking worker for evidence.",
        }
    questions = parsed.get("questions") or []
    if not isinstance(questions, list):
        questions = [str(questions)]
    return {
        "verdict": verdict,
        "questions": [str(q) for q in questions if str(q).strip()],
        "reason": str(parsed.get("reason") or parsed.get("feedback") or ""),
    }


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _attachment_context(request: AgentRequest) -> str:
    if not request.initial_attachments:
        return "(no initial attachments)"

    blocks: List[str] = []
    for att in request.initial_attachments:
        content = att.content
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, default=str)
        blocks.append(
            f"### {att.name} - {att.label}\n"
            f"kind: {att.kind}\n"
            f"schema: {att.content_schema or '(none)'}\n"
            f"{str(content)[:6000]}"
        )
    return "\n\n".join(blocks)


def _plan_prompt(request: AgentRequest) -> str:
    return (
        "You are the planner for a local AI harness. Split the requester task "
        "into separate, checkable tasks for worker agents. Do not solve the "
        "task yourself and do not submit a final answer.\n\n"
        "Return JSON only with this shape:\n"
        "{\n"
        '  "tasks": [\n'
        "    {\n"
        '      "task_id": "task_1",\n'
        '      "title": "short name",\n'
        '      "instructions": "specific work for one worker",\n'
        '      "success_criteria": ["what would prove this task is done"]\n'
        "    }\n"
        "  ],\n"
        '  "final_instructions": "how the finalizer should combine verified work"\n'
        "}\n\n"
        f"Create at least {_MIN_PLANNED_TASKS} separate tasks. Each task must "
        "be small enough for a verifier to check against evidence.\n\n"
        "## Requester Prompt\n"
        f"{request.planner_prompt}\n\n"
        "## Initial Attachments\n"
        f"{_attachment_context(request)}"
    )


def _worker_prompt(
    request: AgentRequest,
    task: HarnessTask,
    verified_records: List[WorkerRecord],
    challenge: Optional[Dict[str, Any]] = None,
) -> str:
    prior = "\n\n".join(
        f"### {record.task.task_id}: {record.task.title}\n{record.result}"
        for record in verified_records
        if record.verified
    ) or "(none)"
    challenge_block = ""
    if challenge:
        questions = "\n".join(f"- {q}" for q in challenge.get("questions", []))
        challenge_block = (
            "\n\n## Verifier Questions\n"
            f"{questions}\n\n"
            "Answer these questions directly and revise the task result if "
            "needed. Do not ignore the verifier."
        )
    return (
        "You are a worker agent in a local AI harness. Complete only the task "
        "assigned below. Do not submit the requester-facing final answer.\n\n"
        "Report your result as JSON when possible:\n"
        "{\n"
        '  "task_id": "...",\n'
        '  "answer": "...",\n'
        '  "claims": [{"claim": "...", "evidence": "..."}],\n'
        '  "uncertainties": []\n'
        "}\n\n"
        "Every factual claim must cite evidence from the requester prompt, "
        "initial attachments, prior verified work, or deterministic tool "
        "output. Say when evidence is missing.\n\n"
        "## Requester Prompt\n"
        f"{request.planner_prompt}\n\n"
        "## Initial Attachments\n"
        f"{_attachment_context(request)}\n\n"
        "## Prior Verified Work\n"
        f"{prior}\n\n"
        "## Assigned Task\n"
        f"task_id: {task.task_id}\n"
        f"title: {task.title}\n"
        f"instructions: {task.instructions}\n"
        f"success_criteria: {json.dumps(task.success_criteria)}"
        f"{challenge_block}"
    )


def _verifier_prompt(
    request: AgentRequest,
    task: HarnessTask,
    worker_result: str,
) -> str:
    return (
        "You are the truth verifier in a local AI harness. Check whether the "
        "worker's answer is supported by the evidence available to the worker. "
        "You are not checking style; you are checking truthfulness, evidence, "
        "range/ID constraints, and whether the worker invented facts.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "verdict": "pass|challenge|fail",\n'
        '  "questions": ["question to ask the worker when verdict is challenge"],\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        "Use `challenge` when a worker could fix the issue by answering a "
        "specific question. Use `fail` only when the answer remains unsupported "
        "or contradicts evidence.\n\n"
        "## Requester Prompt\n"
        f"{request.planner_prompt}\n\n"
        "## Initial Attachments\n"
        f"{_attachment_context(request)}\n\n"
        "## Task\n"
        f"{json.dumps(task.__dict__, default=str)}\n\n"
        "## Worker Result\n"
        f"{worker_result}"
    )


def _finalizer_prompt(
    request: AgentRequest,
    records: List[WorkerRecord],
    final_instructions: str,
) -> str:
    verified = "\n\n".join(
        f"### {record.task.task_id}: {record.task.title}\n{record.result}"
        for record in records
        if record.verified
    ) or "(no verified worker results)"
    unverified = "\n\n".join(
        f"### {record.task.task_id}: {record.task.title}\n"
        f"verifier_reports={json.dumps(record.verifier_reports, default=str)}\n"
        f"{record.result}"
        for record in records
        if not record.verified
    ) or "(none)"
    return (
        "You are the finalizer for a local AI harness. Return the result to "
        "the requester by calling `submit_result(payload_json, summary)`. Do "
        "not invent evidence. Base the payload on verified worker results; if "
        "the evidence supports no positive annotation, submit the empty/negative "
        "payload shape requested by the requester prompt.\n\n"
        "## Requester Prompt\n"
        f"{request.planner_prompt}\n\n"
        "## Planner Final Instructions\n"
        f"{final_instructions or '(none)'}\n\n"
        "## Verified Worker Results\n"
        f"{verified}\n\n"
        "## Unverified Worker Results\n"
        f"{unverified}\n\n"
        "Call `submit_result` exactly once."
    )


def _final_verifier_prompt(
    request: AgentRequest,
    records: List[WorkerRecord],
    payload_json: str,
    summary: str,
) -> str:
    verified = "\n\n".join(
        f"### {record.task.task_id}: {record.task.title}\n{record.result}"
        for record in records
        if record.verified
    ) or "(no verified worker results)"
    return (
        "You are the final truth verifier. Check whether the final submitted "
        "payload is supported by the verified worker results and obeys the "
        "requester prompt. Return JSON only with verdict pass or fail.\n\n"
        "{\n"
        '  "verdict": "pass|fail",\n'
        '  "questions": [],\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        "## Requester Prompt\n"
        f"{request.planner_prompt}\n\n"
        "## Verified Worker Results\n"
        f"{verified}\n\n"
        "## Submitted Payload JSON\n"
        f"{payload_json}\n\n"
        "## Submitted Summary\n"
        f"{summary}"
    )


# ---------------------------------------------------------------------------
# Harness execution helpers
# ---------------------------------------------------------------------------


def _emit_event(
    request: AgentRequest,
    capture: ToolAgentCapture,
    summary: str,
    stage: Dict[str, Any],
) -> None:
    event = StepEvent(
        stage=str(stage.get("node_name") or ""),
        summary=summary,
        detail=dict(stage),
    )
    capture.step_events.append(event)
    if request.callbacks.step_event:
        request.callbacks.step_event(summary, stage)


def _generate(
    service: Any,
    request: AgentRequest,
    capture: ToolAgentCapture,
    prompt: str,
    stage: Dict[str, Any],
    *,
    temperature: Optional[float] = None,
) -> str:
    if request.callbacks.vlm_prompt:
        request.callbacks.vlm_prompt(prompt, stage)
    return service.generate(
        prompt,
        max_tokens=request.config.max_new_tokens,
        temperature=request.config.temperature if temperature is None else temperature,
        stream_callback=request.callbacks.vlm_stream,
        reasoning_callback=request.callbacks.vlm_reasoning,
    )


def _call_tool(
    request: AgentRequest,
    capture: ToolAgentCapture,
    surface: AnnotationToolSurface,
    name: str,
    args: Dict[str, Any],
    *,
    allow_submit: bool,
) -> str:
    if name == "submit_result" and not allow_submit:
        return json.dumps({
            "ok": False,
            "error": "submit_result is only available to the finalizer.",
        })

    capture.tool_calls += 1
    if request.callbacks.progress:
        request.callbacks.progress(_LOCAL_NODE, f"tool {capture.tool_calls}: {name}")
    _result, text, images = surface.call_tool(name, args)
    for img_b64 in images:
        try:
            capture.rendered_images.append(base64.b64decode(img_b64))
        except (ValueError, TypeError):
            LOGGER.warning("local harness: invalid image payload from %s", name)
    return text


def _chat_with_optional_tools(
    service: Any,
    request: AgentRequest,
    capture: ToolAgentCapture,
    surface: AnnotationToolSurface,
    prompt: str,
    tools: List[Dict[str, Any]],
    stage: Dict[str, Any],
    *,
    allow_submit: bool,
    system_prompt: str,
) -> str:
    if request.callbacks.vlm_prompt:
        request.callbacks.vlm_prompt(prompt, stage)
    if not tools:
        return service.generate(
            prompt,
            max_tokens=request.config.max_new_tokens,
            temperature=request.config.temperature,
            stream_callback=request.callbacks.vlm_stream,
            reasoning_callback=request.callbacks.vlm_reasoning,
        )

    def handler(name: str, args: Dict[str, Any]) -> str:
        return _call_tool(
            request,
            capture,
            surface,
            name,
            args,
            allow_submit=allow_submit,
        )

    return service.chat_with_tools(
        prompt,
        tools=tools,
        tool_handler=handler,
        max_tokens=request.config.max_new_tokens,
        temperature=request.config.temperature,
        max_rounds=int(request.config.provider_options.get("max_turns") or 15),
        stream_callback=request.callbacks.vlm_stream,
        reasoning_callback=request.callbacks.vlm_reasoning,
        system_prompt=system_prompt,
    )


def _run_planner(
    service: Any,
    request: AgentRequest,
    capture: ToolAgentCapture,
) -> Tuple[List[HarnessTask], str, str]:
    prompt = _plan_prompt(request)
    raw = _generate(
        service,
        request,
        capture,
        prompt,
        tool_agent_stage(_LOCAL_NODE, "planner"),
        temperature=0.1,
    )
    try:
        tasks, final_instructions = _parse_plan(raw)
        return tasks, final_instructions, raw
    except PlannerFormatError as exc:
        repair_prompt = (
            f"{prompt}\n\n"
            "Your previous response was invalid:\n"
            f"{exc}\n\n"
            "Return only valid planner JSON now."
        )
        raw = _generate(
            service,
            request,
            capture,
            repair_prompt,
            tool_agent_stage(_LOCAL_NODE, "planner_repair"),
            temperature=0.1,
        )
        tasks, final_instructions = _parse_plan(raw)
        return tasks, final_instructions, raw


def _run_worker(
    service: Any,
    request: AgentRequest,
    capture: ToolAgentCapture,
    surface: AnnotationToolSurface,
    task: HarnessTask,
    verified_records: List[WorkerRecord],
    challenge: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = _worker_prompt(request, task, verified_records, challenge=challenge)
    stage = tool_agent_stage(_LOCAL_NODE, "worker", task_id=task.task_id)
    tools = annotation_openai_tool_schemas(request, include_control=False)
    raw = _chat_with_optional_tools(
        service,
        request,
        capture,
        surface,
        prompt,
        tools,
        stage,
        allow_submit=False,
        system_prompt="You are a worker. Complete the assigned task only.",
    )
    return raw.strip()


def _run_verifier(
    service: Any,
    request: AgentRequest,
    capture: ToolAgentCapture,
    task: HarnessTask,
    worker_result: str,
) -> Dict[str, Any]:
    raw = _generate(
        service,
        request,
        capture,
        _verifier_prompt(request, task, worker_result),
        tool_agent_stage(_LOCAL_NODE, "verifier", task_id=task.task_id),
        temperature=0.1,
    )
    report = _parse_verifier(raw)
    report["raw"] = raw
    return report


def _run_finalizer(
    service: Any,
    request: AgentRequest,
    capture: ToolAgentCapture,
    surface: AnnotationToolSurface,
    records: List[WorkerRecord],
    final_instructions: str,
) -> str:
    prompt = _finalizer_prompt(request, records, final_instructions)
    return _chat_with_optional_tools(
        service,
        request,
        capture,
        surface,
        prompt,
        annotation_openai_tool_schemas(
            request,
            include_control=True,
            include_domain=False,
        ),
        tool_agent_stage(_LOCAL_NODE, "finalizer"),
        allow_submit=True,
        system_prompt=build_tool_agent_system_prompt(request),
    ).strip()


def _run_final_verifier(
    service: Any,
    request: AgentRequest,
    capture: ToolAgentCapture,
    records: List[WorkerRecord],
) -> Dict[str, Any]:
    raw = _generate(
        service,
        request,
        capture,
        _final_verifier_prompt(
            request,
            records,
            capture.submit_payload,
            capture.submit_summary,
        ),
        tool_agent_stage(_LOCAL_NODE, "final_verifier"),
        temperature=0.1,
    )
    report = _parse_verifier(raw)
    if report["verdict"] == "challenge":
        report["verdict"] = "fail"
    report["raw"] = raw
    return report


def _response(
    request: AgentRequest,
    capture: ToolAgentCapture,
    tasks: List[HarnessTask],
    plan_text: str,
    records: List[WorkerRecord],
    final_verifier: Optional[Dict[str, Any]],
    *,
    verdict: str,
) -> AgentResponse:
    transcript = "\n\n".join(capture.text_chunks).strip()
    attachments: Dict[str, Attachment] = {
        "planner.plan": Attachment(
            name="planner.plan",
            kind="text",
            label="Planner Plan",
            content=plan_text,
        )
    }
    for record in records:
        attachments[f"worker.{record.task.task_id}.result"] = Attachment(
            name=f"worker.{record.task.task_id}.result",
            kind="text",
            label=f"Worker Result: {record.task.title}",
            content=record.result,
        )
        attachments[f"verifier.{record.task.task_id}.report"] = Attachment(
            name=f"verifier.{record.task.task_id}.report",
            kind="structured",
            label=f"Verifier Report: {record.task.title}",
            content=record.verifier_reports,
            content_schema="local_harness_verifier_reports",
        )
    if final_verifier is not None:
        attachments["verifier.final.report"] = Attachment(
            name="verifier.final.report",
            kind="structured",
            label="Final Verifier Report",
            content=final_verifier,
            content_schema="local_harness_final_verifier",
        )
    if capture.submit_summary:
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

    messages: List[Dict[str, Any]] = []
    if plan_text:
        messages.append({"role": "planner", "content": plan_text})
    for record in records:
        messages.append({
            "role": f"worker:{record.task.task_id}",
            "content": record.result,
            "verified": record.verified,
        })
        messages.append({
            "role": f"verifier:{record.task.task_id}",
            "content": json.dumps(record.verifier_reports, default=str),
        })
    if final_verifier is not None:
        messages.append({
            "role": "final_verifier",
            "content": json.dumps(final_verifier, default=str),
        })

    return AgentResponse(
        raw_response=capture.submit_payload if verdict == "submitted" else "",
        verdict=verdict,
        attachments=attachments,
        step_events=capture.step_events,
        graph_images=[],
        plan_steps=[task.__dict__ for task in tasks],
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_local(request: AgentRequest) -> AgentResponse:
    """Execute one local harness run."""
    service = get_or_start_service(_local_vlm_config_from_request(request))
    capture = ToolAgentCapture(
        node_name=_LOCAL_NODE,
        cur_start=int(request.parent_start),
        cur_end=int(request.parent_end),
    )
    surface = AnnotationToolSurface(request, capture)

    if request.callbacks.progress:
        request.callbacks.progress(_LOCAL_NODE, "harness starting")

    tasks, final_instructions, plan_text = _run_planner(service, request, capture)
    capture.text_chunks.append(f"[planner]\n{plan_text}")
    _emit_event(
        request,
        capture,
        f"planned {len(tasks)} task(s)",
        tool_agent_stage(_LOCAL_NODE, "planner"),
    )

    records: List[WorkerRecord] = []
    for task in tasks:
        if request.callbacks.progress:
            request.callbacks.progress(_LOCAL_NODE, f"worker {task.task_id}: {task.title}")
        result = _run_worker(service, request, capture, surface, task, records)
        capture.text_chunks.append(f"[worker:{task.task_id}]\n{result}")
        record = WorkerRecord(task=task, result=result)

        for _attempt in range(_MAX_VERIFIER_CHALLENGES + 1):
            report = _run_verifier(service, request, capture, task, record.result)
            record.verifier_reports.append(report)
            capture.text_chunks.append(
                f"[verifier:{task.task_id}]\n{json.dumps(report, default=str)}"
            )
            verdict = report["verdict"]
            if verdict == "pass":
                record.verified = True
                break
            if verdict == "fail":
                break
            if _attempt >= _MAX_VERIFIER_CHALLENGES:
                record.verifier_reports.append({
                    "verdict": "fail",
                    "questions": [],
                    "reason": "Verifier challenge budget exhausted.",
                })
                break
            record.result = _run_worker(
                service,
                request,
                capture,
                surface,
                task,
                records,
                challenge=report,
            )
            capture.text_chunks.append(f"[worker:{task.task_id}:challenge]\n{record.result}")

        records.append(record)
        _emit_event(
            request,
            capture,
            f"worker {task.task_id} verified={record.verified}",
            tool_agent_stage(_LOCAL_NODE, "verifier", task_id=task.task_id),
        )

    final_text = _run_finalizer(
        service,
        request,
        capture,
        surface,
        records,
        final_instructions,
    )
    if final_text:
        capture.text_chunks.append(f"[finalizer]\n{final_text}")

    if not capture.submitted:
        if request.callbacks.progress:
            request.callbacks.progress(_LOCAL_NODE, "done - no submission")
        return _response(
            request,
            capture,
            tasks,
            plan_text,
            records,
            None,
            verdict="no_submission",
        )

    final_report = _run_final_verifier(service, request, capture, records)
    capture.text_chunks.append(
        f"[final_verifier]\n{json.dumps(final_report, default=str)}"
    )
    if final_report["verdict"] != "pass":
        if request.callbacks.progress:
            request.callbacks.progress(_LOCAL_NODE, "done - verification failed")
        return _response(
            request,
            capture,
            tasks,
            plan_text,
            records,
            final_report,
            verdict="verification_failed",
        )

    if request.callbacks.progress:
        request.callbacks.progress(
            _LOCAL_NODE,
            f"done - {capture.tool_calls} tool call(s), submitted=True",
        )
    return _response(
        request,
        capture,
        tasks,
        plan_text,
        records,
        final_report,
        verdict="submitted",
    )
