---
name: get_user_summary_map_level
title: Reading user summary by map
description: >
  Retrieve the already-loaded user summary aggregated at the map/track level.
  Use for questions about one specific map only when map_id is known, or for
  explicit all-map comparisons. If the driver asks a map-specific question but
  does not say which map, first call get_available_user_summary_maps and ask
  which map to inspect. This returns aggregate map rows, top sections, and
  section-level mistake/adherence label breakdowns when a specific map is
  requested; it does not return raw telemetry.
parameters:
  map_id:
    description: Optional map or track id, or exact map name, to filter to one map.
---

## Usage notes

Do not use this to infer raw driving telemetry. It summarizes historical user
analysis already loaded by the frontend.

In recorded-session mode, do not use this as the first source for a generic
"find my mistakes" or "what did I do wrong" request. Those refer to the
selected recording unless the driver asks for history, trends, percentages,
all sessions, or comparison against prior sessions.

Use this, not analyze_telemetry, for follow-ups about a mistake percentage,
weak section, strong section, Surtees/Clark Curve-style section finding, or a
previously mentioned user-summary issue. If map_id is known from the prior
turn, pass it so the response includes section breakdowns.
