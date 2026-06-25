---
name: get_event_log
title: Searching event log
description: >
  List racing events with their sample-index ranges. Use to find when something
  happened before querying telemetry around it.
parameters:
  eventType:
    description: Event category to search.
  scope:
    description: Which part of the event log to inspect.
  n:
    description: For last_n scope, the number of events.
---

## Usage notes

Pair this with telemetry queries when the driver asks about a specific recent
corner, straight, crash, or overtake.

