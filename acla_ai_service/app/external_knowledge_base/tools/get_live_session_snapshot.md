---
name: get_live_session_snapshot
title: Reading live session
description: >
  Return compact live-session state for the Live Performance Analyst, including
  track, car, current lap, normalized position, completed laps, sample count,
  baseline readiness, and detected live session type.
parameters: {}
---

## Usage notes

Use this when you need to check whether the analyst has a completed baseline or
whether the session is solo practice versus traffic/race. Do not infer session
type yourself if this tool is available.
