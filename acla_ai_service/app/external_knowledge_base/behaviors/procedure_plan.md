---
name: procedure_plan
---

Procedure plan mode:
- A procedure plan is active when `procedure_plan` exists in frontend session
  context or an observation includes `goal`, `requests`, and `current_request`.
- The frontend owns visible plan state and frontend/subscriber execution.
  Use `advance_plan_step` to complete the active request and let the frontend
  execute subscribed requests or nested frontend `tool_call` requests.
- When an observation confirms the current request is complete, ready, or
  executable now, call `advance_plan_step` before speaking. Do not merely
  acknowledge the observation.
- If `advance_plan_step` returns AI-visible `tool_result` data, read it before
  deciding what to say or whether another plan step should advance.
- Do not skip, clear, or abandon an active plan unless the driver explicitly
  asks to cancel, clear, stop, skip, or opt out of the plan.
- Do not invent hidden plan steps. If a new plan is needed, call
  `set_procedure_plan` with explicit request objects.
