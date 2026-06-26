---
name: analyze_telemetry
title: Analyzing telemetry
description: >
  Classify driving actions over a telemetry scope and return engineer labels
  with definitions and remedies. Use only for live or recorded raw telemetry
  windows, such as "what just happened", "why did I lose time there on this
  lap", or "how was lap N".
parameters:
  scope:
    description: Telemetry time window to classify.
---

## Usage notes

Do not use this for historical user-summary questions about mistake
percentages, weak sections, strong sections, map aggregates, or follow-ups to a
summary finding. Use the user-summary tools for those instead.

When the tool returns labels with definitions and remedies, pick the one or
two that matter most and explain them in natural race-engineer language.
