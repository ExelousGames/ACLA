---
name: get_live_section_history
title: Reading section history
description: >
  Return compact Live Performance Analyst section classifications already
  recorded by the AI service and frontend.
parameters:
  limit:
    description: Maximum number of recent compact classifications to return.
---

## Usage notes

Use this to understand whether a mistake repeats across laps or to compare the
active focus baseline against a later pass. The result is compact and
driver-safe; it should not contain raw telemetry rows.
