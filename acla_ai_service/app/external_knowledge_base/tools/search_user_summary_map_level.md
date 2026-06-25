---
name: search_user_summary_map_level
title: Searching user summary maps
description: >
  Search the already-loaded user summary at the map/track aggregate level.
  Use when the driver asks which maps match a track name, map id, top mistake
  section, or top expert-adherence section. If the user asks a map-specific
  question without naming a map, call get_available_user_summary_maps and ask
  which map instead of searching all maps.
parameters:
  query:
    description: Search text, such as a map name, map id, section, mistake, or adherence phrase.
  limit:
    description: Maximum number of matching maps to return. Default 5; max 10.
---

## Usage notes

This searches summary aggregates only, not raw telemetry.

