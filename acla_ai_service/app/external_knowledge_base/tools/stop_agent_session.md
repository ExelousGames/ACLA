---
name: stop_agent_session
title: Stopping agent mode
description: Stop the active live child agent session and return control to the main assistant.
---

Use `stop_agent_session` when the driver asks to stop, disable, cancel, close, or turn off the active live agent mode.

Guidance:
- This is the only stop tool for Track Guide, Overtake, and Live Performance Analyst agent modes.
- If the driver names a specific agent, stop the currently active child agent session. The frontend tracks which agent session is active.
- Do not call any dedicated per-agent stop tool.
