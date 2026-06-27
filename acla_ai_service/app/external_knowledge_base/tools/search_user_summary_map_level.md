---
name: search_user_summary_map_level
title: Searching user summary maps
description: >
  Search the already-loaded user summary at the map/track aggregate level.
  Use when the driver asks which maps match a track name, map id, top mistake
  section, top expert-adherence section, or a vague follow-up to a summary
  finding when the map is not known. If the user asks a map-specific question
  without naming a map, call get_available_user_summary_maps and ask which map
  instead of searching all maps.
parameters:
  query:
    description: Search text, such as a map name, map id, section, mistake, or adherence phrase.
  limit:
    description: Maximum number of matching maps to return. Default 5; max 10.
---

## Usage notes

This searches summary aggregates only, not raw telemetry.

In recorded-session mode, do not use this as the first source for a generic
"find my mistakes" or "what did I do wrong" request. Those refer to the
selected recording unless the driver asks for history, trends, percentages,
all sessions, or comparison against prior sessions.

Use this to recover the right map/section from phrases like "Surtees",
"Clark Curve", "that mistake", or "check it now" when the recent conversation
was about user-summary mistakes. Do not use analyze_telemetry unless the
driver explicitly asks about current live/recorded telemetry.
