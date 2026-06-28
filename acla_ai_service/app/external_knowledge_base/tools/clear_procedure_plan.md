---
name: clear_procedure_plan
title: Clearing procedure plan
description: >
  Clear or terminate the visible procedure plan UI when the current plan is
  finished, canceled, stale, or no longer useful.
parameters:
  reason:
    description: Optional short reason the visible plan should be cleared.
---

## Usage notes

Use this when the user asks to cancel, clear, stop, dismiss, hide, end, or
terminate the visible procedure plan.

Use this when you replace a plan with normal conversation and do not intend to
continue advancing the old request list.

Do not use this to advance a step. Use `advance_plan_step` when the current
request is complete and the next request should become active.
