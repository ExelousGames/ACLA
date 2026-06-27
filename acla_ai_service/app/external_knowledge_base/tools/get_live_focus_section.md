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

Use this after live section classifications are recorded and before coaching.
If a focus exists and the frontend timing says the coaching window is open,
call show_map with `show_map_arguments`, then speak one concise correction.

If no focus exists, classify more candidate sections or wait for the next
baseline/focus observation.
