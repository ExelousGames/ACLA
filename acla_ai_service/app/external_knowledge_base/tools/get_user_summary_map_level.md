---
name: get_user_summary_map_level
title: Reading user summary by map
description: >
  Retrieve the already-loaded user summary aggregated at the map/track level.
  Use for questions about one specific map only when map_id is known, or for
  explicit all-map comparisons. If the driver asks a map-specific question but
  does not say which map, first call get_available_user_summary_maps and ask
  which map to inspect. This returns aggregate map rows and top sections, not
  raw telemetry.
parameters:
  map_id:
    description: Optional map or track id, or exact map name, to filter to one map.
---

## Usage notes

Do not use this to infer raw driving telemetry. It summarizes historical user
analysis already loaded by the frontend.

