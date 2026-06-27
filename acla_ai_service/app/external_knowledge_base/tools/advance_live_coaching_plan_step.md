---
name: advance_live_coaching_plan_step
title: Advancing coaching plan
description: >
  Move the visible Live Performance Analyst coaching plan UI to the next step.
parameters:
  reason:
    description: Optional short reason the current coaching step is complete.
---

## Usage notes

Use this only after a live coaching plan is visible and the current step has
been completed or is no longer the best focus.

Do not use this to create a plan. Start the plan with
`start_live_performance_analysis`, then advance one step at a time.
