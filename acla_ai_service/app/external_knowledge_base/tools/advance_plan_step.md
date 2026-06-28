---
name: advance_plan_step
title: Advancing plan
description: >
  Move the visible procedure plan UI to the next request.
parameters:
  reason:
    description: Optional short reason the current plan request is complete.
---

## Usage notes

Use this only after a procedure plan is visible and the current request has
been completed or is no longer the best focus.

If the user explicitly asks to skip, cancel, clear, stop, or opt out of the
visible procedure plan, acknowledge it briefly and do not call this tool for
that plan anymore. Only resume plan-request progression after a new procedure
plan is explicitly started.

Do not use this to create a plan. Create the plan with `set_procedure_plan`,
then advance one request at a time.
