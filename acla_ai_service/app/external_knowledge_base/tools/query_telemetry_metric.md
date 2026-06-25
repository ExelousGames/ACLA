---
name: query_telemetry_metric
title: Querying telemetry
description: >
  Read reduced telemetry metrics over a live scope. Use for direct numeric
  questions about recent or scoped telemetry, not for classifier-style driving
  diagnosis.
parameters:
  fields:
    description: Field group names are preferred; raw Physics_* names are allowed when needed.
  scope:
    description: Time window or event window to query.
  reduce:
    description: Aggregation to return. stats means avg, min, max, and stddev.
---

## Usage notes

This tool returns reduced metrics only. It should not expose raw telemetry rows
to the LLM.

