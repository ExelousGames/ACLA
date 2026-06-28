---
name: advance_plan_step
title: Advancing plan
description: >
  Report that the current visible procedure plan request is complete so the UI
  can move to the next request.
parameters:
  reason:
    description: Optional short reason the current plan request is complete.
---

## Usage notes

The frontend plan component self-advances when plan observations already mark
the active request complete. Use this tool only when the AI service itself
detects that the current visible request has been completed or is no longer the
best focus and needs to report that completion to the frontend.

If the user explicitly asks to skip, cancel, clear, stop, or opt out of the
visible procedure plan, acknowledge it briefly and do not call this tool for
that plan anymore. Only resume plan-request progression after a new procedure
plan is explicitly started.

Do not use this to create a plan. Create the plan with `set_procedure_plan`,
then advance one request at a time.
