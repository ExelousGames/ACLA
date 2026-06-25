---
name: start_overtake_agent
title: Starting overtake agent
description: >
  Open continuous overtake agent mode. Use only when the driver explicitly asks
  to open, enable, watch, monitor, or plan attack/defense overtake agent mode.
  Do not use for one-off questions like "when can I overtake". The agent uses
  live car coordinates to detect attack windows and defense threats until
  stopped.
parameters:
  interval_seconds:
    description: How often to check while agent mode is active. Default 5; clamped to 2-15.
---

## Usage notes

This is a persistent live monitor. For one-off timing questions, say live timing
needs overtake agent mode opened.

