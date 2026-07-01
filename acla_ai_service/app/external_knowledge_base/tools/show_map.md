---
name: show_map
title: Displaying a circuit map
description: >
  Display a circuit map alongside the assistant message when a spatial
  explanation would be clearer as a graph or map. Use this for track sections,
  corner sequences, racing-line location, incident location, or any answer
  where the driver needs to see where something is happening on the circuit.
parameters:
  map_id:
    description: Optional circuit map id. Prefer this when a map id is known.
  source_track_key:
    description: Optional ACC source track key such as brands_hatch, monza, or spa.
  map_name:
    description: Optional human-readable map or circuit name when no id/key is known.
  section_start:
    description: Optional highlighted section start as normalized lap position from 0 to 1.
  section_end:
    description: Optional highlighted section end as normalized lap position from 0 to 1. It may wrap across start/finish.
  section_label:
    description: Optional short label for the highlighted section.
  title:
    description: Optional short title shown above the map.
  note:
    description: Optional brief note shown below the map.
---

## Usage notes

Call this tool only when the message benefits from a map representation. Send
the explanatory message normally, and call show_map to attach the circuit map.

Use the current session's selected map when the user is already discussing the
current lap/session and no map id is needed. If a specific map is known from a
previous tool result, pass map_id. If only the ACC track key or display name is
known, pass source_track_key or map_name.

Use section_start and section_end when the response points to a section of the
lap. Values are normalized lap position from 0 to 1, so 0.25 means 25 percent
of the way around the lap. For a section that crosses the start/finish line,
section_start may be greater than section_end.

If no map is available, the frontend displays "Map is not available." Do not
apologize at length; continue with the best text explanation.
