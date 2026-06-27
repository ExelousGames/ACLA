---
name: classify_live_section
title: Classifying live section
description: >
  Classify one known live track section from a completed lap for the Live
  Performance Analyst. This server-side tool runs the segment classifier,
  records a compact classification in the frontend section history, and returns
  only compact labels, stats, focus, and comparison data.
parameters:
  section_id:
    description: Known track section id from the live analyst observation or get_live_focus_section result.
  section_name:
    description: Optional section name if an id is not available.
  lap:
    description: Lap to classify. Use "last" for the most recent completed baseline lap, or a specific lap number when supplied by the observation.
---

## Usage notes

Use this only for Live Performance Analyst workflows. Do not use it for normal
one-off live telemetry questions; use analyze_telemetry for those.

Call it after a `baseline_ready_needs_classification` observation for candidate
sections, or after the next pass through the active focus section to check
improvement. Raw telemetry is not available to the assistant; use only the
compact classification result.

After this tool records a classification, call `get_live_focus_section` when
you need the current focus and map arguments. If the returned `comparison` is
present, speak only the improvement result or one next correction.
