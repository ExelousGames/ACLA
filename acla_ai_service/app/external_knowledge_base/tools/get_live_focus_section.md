---
name: get_live_focus_section
title: Reading focus section
description: >
  Return the current Live Performance Analyst focus section, baseline
  classification, timing estimate, selection reason, score, and show_map
  arguments.
parameters: {}
---

## Usage notes

Use this only after baseline collection is complete and after
`recorded_analysis_plan_ready` or after a focused follow-up classification is
recorded and before coaching. Do not use it during `collecting_baseline`.
If a focus exists and the frontend timing says the coaching window is open,
call show_map with `show_map_arguments`, then speak one concise correction.

If no focus exists, wait for the next recorded-analysis plan observation or
briefly explain that a focus plan is not ready yet.
