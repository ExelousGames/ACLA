---
name: get_recorded_session_analysis
title: Reading recorded AI analysis
description: >
  Read the shared AI segment analysis for the currently selected recorded
  session. Use for follow-up questions about classified parent segments, child
  labels, mistake/adherence/recovery sections, or what the recorded analysis
  found. This returns compact summaries, not raw telemetry rows.
parameters:
  limit:
    description: Optional maximum number of compact classified segments to return.
---

## Usage notes

Use after run_recorded_ai_analysis has produced a result, or when session
context says recorded_session.ai_analysis.result_ready is true.

In recorded-session mode, use this for generic selected-recording mistake or
improvement questions when analysis is already ready. Use user-summary tools
only when the driver asks for history, trends, percentages, all sessions, or
comparison against prior sessions.

If no analysis is available, tell the driver analysis has not been run yet and
use run_recorded_ai_analysis if they asked you to analyze the recording.
