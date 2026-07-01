---
name: classify_live_section
title: Classifying live section
description: >
  Classify the active Live Performance Analyst focus section after the driver
  passes through it again. This server-side tool runs the segment classifier,
  records a compact comparison in the frontend section history, and returns
  only compact labels, stats, focus, and comparison data.
parameters:
  section_id:
    description: Known track section id from the live analyst observation or get_live_focus_section result.
  section_name:
    description: Optional section name if an id is not available.
  lap:
    description: Lap to classify for the active focus section. Use "last" for the most recent completed pass, or a specific lap number when supplied by the observation.
---

## Usage notes

Use this only for Live Performance Analyst focus-section follow-up. Do not use
it for baseline analysis, normal one-off live telemetry questions, or broad
recorded-session analysis.

Call it only after the next pass through the active focus section to check
improvement. The baseline focus comes from shared recorded-session AI analysis,
not from classifying every live section. Raw telemetry is not available to the
assistant; use only the compact classification result.

After this tool records a classification, call `get_live_focus_section` when
you need the current focus and map arguments. If the returned `comparison` is
present, speak only the improvement result or one next correction.
