# RAG knowledge corpus

Drop free-form prose `.md` files in here. Anything in this folder gets
chunked, embedded, and exposed to the LLM via the `search_racing_knowledge`
tool. Good fits:

- Driver transcripts (post-race interviews, radio call notes)
- Race / event reports
- Track-history / circuit-lore prose
- Driving theory passages, technique deep-dives
- Setup discussions where the same topic recurs across many docs

## File format

Plain Markdown. Optional YAML frontmatter:

```markdown
---
title: 2024 Spa GP — Verstappen post-race
source: paddock-radio
track: spa
date: 2024-07-28
tags: [wet, tyre-strategy]
---

## Wet-tyre call

Verstappen reported standing water through Pouhon by lap 6 …

## Final stint

The team gambled on staying out one more lap …
```

## How it gets chunked

Each `## Heading` becomes one chunk. The heading text travels with the
chunk so the embedding sees both context and body. Long sections (over
`racing_kb_max_chunk_chars` in settings) get split on paragraph boundaries.

Files **without** any `##` headings get chunked by paragraph (blank-line
splits).

## When to put something here vs in `tracks/`

- `tracks/<id>.md` = **keyed, structured per-track facts**. The LLM
  calls `get_track_knowledge(track="spa", corner="eau_rouge")` to pull
  exactly that doc. Use for: corner-by-corner notes, recommended gears,
  braking points, track-specific reference data.

- `knowledge/*.md` = **everything else**. The LLM calls
  `search_racing_knowledge(query="what do drivers say about wet setup at Spa")`
  and the index returns the top-k matching chunks.

Same fact can legitimately live in both — `tracks/spa.md` for the
canonical corner data, `knowledge/spa_history.md` for prose history.

## Cache invalidation

The index lives at `acla_ai_service/.cache/racing_knowledge/`. Edits to
any file here trigger a per-file re-embed on next service start
(unchanged files reuse cached vectors). Delete the cache folder to force
a full rebuild.
