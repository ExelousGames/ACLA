---
name: analyze_live_recorded_analysis
title: Analyzing baseline lap
description: >
  Submit the completed Live Performance Analyst baseline lap to recorded-session
  classifier analysis. Use this after collect_live_baseline completes to analyze
  the cached live baseline lap.
parameters:
  limit:
    description: Optional maximum number of compact classified baseline segments to return.
---

## Usage notes

Use this only in Live Performance Analyst mode, after `collect_live_baseline`
has returned a complete cached baseline lap. It sends the cached live baseline
lap records to the recorded-session classifier. Do not use
`run_recorded_ai_analysis` here; that tool is for the selected recorded-session
view, not the live baseline lap.

The output is a compact classifier result for the baseline lap. It includes
analysis status, baseline session metadata, telemetry row counts, segment
counts, and a limited list of classified segments with labels. It does not
expose raw telemetry rows.

A classified segment is a short portion of the baseline lap that the classifier
marked with driving labels. Each segment summarizes where the pattern occurred
and which labels apply, such as a mistake, expert-adherence behavior, recovery,
pit-lane event, or racing action. The segment is a compact summary, not the raw
telemetry rows from that portion of the lap.

If the tool says a baseline lap record is required, wait for
`collect_live_baseline` to complete instead of calling live section
classification.

Do not use `classify_live_section` for baseline analysis. Use
`classify_live_section` only for later live section follow-up classification,
not for analyzing the recorded baseline lap.
