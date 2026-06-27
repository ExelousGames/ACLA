---
name: start_live_performance_analysis
title: Starting live analyst
description: >
  Start the Live Performance Analyst agent in the frontend. The agent observes
  the live ACC session, waits for at least one completed lap, asks the AI
  service to classify completed sections, and emits timed coaching observations.
parameters:
  interval_seconds:
    description: Optional frontend polling interval for checking live analyst state.
---

## Usage notes

Use this when the driver asks to enable, start, watch, monitor, or coach with
live performance analysis. This is not Track Guide; do not use it for requests
that only ask where to brake or how to take the next corner.

After starting, acknowledge that you are collecting a baseline. Do not critique
until the frontend reports baseline readiness and section classifications exist.
