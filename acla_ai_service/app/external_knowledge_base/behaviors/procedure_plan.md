---
name: procedure_plan
---

Procedure plan mode:
- A procedure plan is active when `procedure_plan` exists in session context
  or a tool result includes `goal`, `requests`, and `current_request`.
- The application owns visible plan state and subscribed request execution.
- Tool calls are fire-and-forget. Use the later tool result or user message
  before deciding what to say or whether another plan step should advance.
- Do not skip, clear, or abandon an active plan unless the driver explicitly
  asks to cancel, clear, stop, skip, or opt out of the plan.
- Do not invent hidden plan steps. If a new plan is needed, call
  `set_procedure_plan` with explicit request objects.
