---
name: get_recorded_session_context
title: Reading recorded session context
description: >
  Read compact context for the currently selected recorded session, including
  session metadata, playback position, active AI segment, sample count,
  duration, and AI analysis status. Use when the driver asks about "this
  recording", "where we are", "the current segment", or whether analysis is
  available.
parameters:
  limit:
    description: Optional maximum number of compact classified segments to include inside the nested analysis summary.
---

## Usage notes

Use this before answering questions that depend on current playback position or
selected recording metadata.

Use get_recorded_session_analysis for deeper analysis follow-ups and
run_recorded_ai_analysis when analysis needs to be started or refreshed.
