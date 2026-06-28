---
name: start_live_performance_analysis
title: Starting live analyst
description: >
  Start the Live Performance Analyst agent in the frontend. The agent observes
  the live ACC session, waits for at least one completed lap, runs the shared
  recorded-session AI analysis to build one focus goal, and emits timed
  coaching observations.
parameters:
  interval_seconds:
    description: Optional frontend polling interval for checking live analyst state.
---

## Usage notes

Use this when the driver asks to enable, start, watch, monitor, or coach with
live performance analysis. This is not Track Guide; do not use it for requests
that only ask where to brake or how to take the next corner.

After starting, acknowledge that you are collecting a baseline. Do not critique
until the frontend reports `recorded_analysis_plan_ready`. When it does, call
`set_procedure_plan` with the request list you decide on. If recorded-session
analysis is unavailable, say that plainly; do not fall back to
`classify_live_section` for baseline analysis.
