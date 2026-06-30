---
name: restart_live_baseline
title: Restarting baseline lap
description: >
  Clear the current Live Performance Analyst baseline collection buffer and
  restart the dedicated baseline UI so the next collect_live_baseline call
  records a fresh baseline lap.
parameters: {}
---

## Usage notes

Use this only in Live Performance Analyst mode when the current baseline lap is
bad, stale, incomplete, or from the wrong car or track. Examples include a
crash, off-track lap, pit entry, wrong session, interrupted collection, or a
driver explicitly asking to redo the baseline.

After this tool returns `status: "restarted"`, call `collect_live_baseline` to
wait for the new clean lap. Do not call `analyze_live_recorded_analysis` until
that new `collect_live_baseline` call has completed and returned a cached
baseline lap record.

Do not use this as a routine first step. If no bad or stale baseline exists,
start with `collect_live_baseline`. Restarting discards the current baseline
collection state and resets progress to 0 percent.
