---
name: start_agent_session
title: Starting agent mode
description: Start a live child agent session for track guide, overtake, or live performance analyst mode.
---

Use `start_agent_session` whenever the driver asks to start, enable, open, watch, monitor, or use a live agent mode.

Arguments:
- `agent_mode`: one of `track_guide`, `overtake`, or `live_performance_analyst`.

Guidance:
- Use `agent_mode: "track_guide"` for corner-by-corner live track guidance.
- Use `agent_mode: "overtake"` for ongoing traffic and passing-opportunity monitoring. Do not start it for a one-off question like "when can I overtake?"; say that live timing needs overtake agent mode opened.
- Use `agent_mode: "live_performance_analyst"` for baseline collection, focus selection, and live performance coaching.
- Do not call any dedicated per-agent start tool.
