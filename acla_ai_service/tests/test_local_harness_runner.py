import json

from app.local_annotation_agent import runner
from app.shared.contracts import AgentRequest, NoopCallbacks, ProviderConfig


class _FakeHarnessService:
    def __init__(self, *, final_verdict="pass"):
        self.final_verdict = final_verdict
        self.worker_prompts = []
        self.verifier_calls = 0
        self.finalizer_tools = []

    def generate(self, prompt, **_kwargs):
        if "You are the planner" in prompt:
            return json.dumps({
                "tasks": [
                    {
                        "task_id": "task_1",
                        "title": "Check evidence",
                        "instructions": "Check the evidence package.",
                        "success_criteria": ["claims cite evidence"],
                    },
                    {
                        "task_id": "task_2",
                        "title": "Check payload",
                        "instructions": "Check the output shape.",
                        "success_criteria": ["payload shape is valid"],
                    },
                ],
                "final_instructions": "Combine verified task results.",
            })
        if "You are a worker agent" in prompt:
            self.worker_prompts.append(prompt)
            return json.dumps({
                "answer": f"worker answer {len(self.worker_prompts)}",
                "claims": [{"claim": "supported", "evidence": "requester prompt"}],
                "uncertainties": [],
            })
        if "You are the truth verifier" in prompt:
            self.verifier_calls += 1
            if self.verifier_calls == 1:
                return json.dumps({
                    "verdict": "challenge",
                    "questions": ["Where is the evidence?"],
                    "reason": "Needs explicit support.",
                })
            return json.dumps({
                "verdict": "pass",
                "questions": [],
                "reason": "Claims are supported.",
            })
        if "You are the final truth verifier" in prompt:
            return json.dumps({
                "verdict": self.final_verdict,
                "questions": [],
                "reason": "final check",
            })
        raise AssertionError(f"unexpected prompt: {prompt[:120]}")

    def chat_with_tools(self, prompt, tools, tool_handler, **_kwargs):
        self.finalizer_tools = [
            tool["function"]["name"]
            for tool in tools
        ]
        assert self.finalizer_tools == ["submit_result"]
        tool_handler(
            "submit_result",
            {
                "payload_json": json.dumps({"label_ids": ["MSP"], "reasoning": "ok"}),
                "summary": "done",
            },
        )
        return "submitted"


def _request():
    return AgentRequest(
        provider_id="local_vlm",
        config=ProviderConfig(
            provider_id="local_vlm",
            model="fake",
            max_new_tokens=500,
            temperature=0.0,
        ),
        planner_prompt="Return an annotation.",
        synth_prompt=lambda _state: ("", ""),
        df_ref=[],
        parent_start=0,
        parent_end=10,
        callbacks=NoopCallbacks(),
    )


def test_local_harness_runs_planned_tasks_and_challenge_loop(monkeypatch):
    service = _FakeHarnessService()
    monkeypatch.setattr(runner, "get_or_start_service", lambda _config: service)

    response = runner.run_local(_request())

    assert response.verdict == "submitted"
    assert json.loads(response.raw_response) == {
        "label_ids": ["MSP"],
        "reasoning": "ok",
    }
    assert len(response.plan_steps) == 2
    assert len(service.worker_prompts) == 3
    assert "Verifier Questions" in service.worker_prompts[1]
    assert service.finalizer_tools == ["submit_result"]
    assert "verifier.task_1.report" in response.attachments


def test_local_harness_blocks_failed_final_verification(monkeypatch):
    service = _FakeHarnessService(final_verdict="fail")
    monkeypatch.setattr(runner, "get_or_start_service", lambda _config: service)

    response = runner.run_local(_request())

    assert response.verdict == "verification_failed"
    assert response.raw_response == ""
    assert response.attachments["verifier.final.report"].content["verdict"] == "fail"
