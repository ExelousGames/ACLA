---
name: analyze_telemetry
title: Analyzing telemetry
description: >
  Classify driving actions over a telemetry scope and return engineer labels
  with definitions and remedies. Use for questions like "what just happened",
  "why did I lose time there", or "how was lap N".
parameters:
  scope:
    description: Telemetry time window to classify.
---

## Usage notes

When the tool returns labels with definitions and remedies, pick the one or
two that matter most and explain them in natural race-engineer language.

