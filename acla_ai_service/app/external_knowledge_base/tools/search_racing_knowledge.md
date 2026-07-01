---
name: search_racing_knowledge
title: Searching racing knowledge
description: >
  Semantic search over driver transcripts, race reports, label docs, telemetry
  feature notes, and theory notes. Use for cross-cutting questions where the
  right document is not obvious.
parameters:
  query:
    description: Free-text question or topic.
  top_k:
    description: Number of snippets to return. Defaults to 5.
---

## Usage notes

Use this when keyed tools like label or track lookup are too narrow.

