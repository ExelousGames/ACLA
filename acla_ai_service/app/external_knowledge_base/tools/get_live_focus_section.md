---
name: get_live_focus_section
title: Analyzing focus section
description: >
  Analyze the current Live Performance Analyst focus section state. Returns
  the selected section, baseline classification, timing estimate, selection
  reason, score, show_map arguments, and any follow-up comparison.
parameters: {}
---

## Usage notes

Use this when you need the current focus-section analysis before coaching or
after a focused follow-up classification has been recorded.

Call it after baseline collection is complete and the live analyst has selected
a focus from recorded-session analysis. If the frontend reports that baseline
collection is incomplete or no focus section is ready, do not retry the same
call immediately; wait for the next live analyst observation or briefly say the
focus analysis is not ready yet.

If the tool returns a focus and the timing says the coaching window is open, use
the returned `show_map_arguments` with `show_map`, then speak one concise
correction tied to the returned labels and section name.

If the result includes a follow-up `comparison`, use it to say whether the
driver improved, stayed similar, or regressed on the active focus section.

If no focus exists, wait for the next recorded-analysis plan observation or
briefly explain that a focus plan is not ready yet.
