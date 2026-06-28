---
name: advance_plan_step
title: Advancing plan
description: >
  Move the visible procedure plan UI to the next step.
parameters:
  reason:
    description: Optional short reason the current plan step is complete.
---

## Usage notes

Use this only after a procedure plan is visible and the current step has
been completed or is no longer the best focus.

Do not use this to create a plan. Start the plan with
`start_live_performance_analysis`, then advance one step at a time.
