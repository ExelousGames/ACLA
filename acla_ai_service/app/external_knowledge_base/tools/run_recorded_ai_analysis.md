---
name: run_recorded_ai_analysis
title: Running recorded session AI analysis
description: >
  Run or retrieve the AI segment analysis for the currently selected recorded
  session. Use when the driver asks to analyze this recording, identify
  classified segments, or asks a question that requires the recorded session's
  AI segment analysis and the current analysis status is idle, empty, or stale.
  This returns compact segment summaries and does not expose raw telemetry rows.
parameters:
  force:
    description: Optional. Set true only when the driver explicitly asks to rerun or refresh the recorded analysis.
  limit:
    description: Optional maximum number of compact classified segments to return. Use a small value unless the driver asks for all details.
---

## Usage notes

Use this only in recorded-session mode. If the analysis is already ready and
the driver only asks a follow-up, prefer get_recorded_session_analysis or
get_recorded_session_context.

Do not use live-only telemetry tools for recorded session analysis. If the
result has no segments, say that the AI analysis found no classified segments
and ask what part of the recording the driver wants to inspect next.
