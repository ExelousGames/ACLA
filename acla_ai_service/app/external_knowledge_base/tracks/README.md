# Track knowledge (keyed lookup)

One `.md` file per track. Read by the `get_track_knowledge` tool —
the LLM passes a `track` id (and optionally a `corner` name) and gets
the matching doc back as structured fields.

This is the **keyed** side of the racing-engineer knowledge base. For
free-form prose (driver transcripts, race reports, theory), use
`../knowledge/` and the LLM will reach it via `search_racing_knowledge`.

## File format

```markdown
---
id: spa
name: Circuit de Spa-Francorchamps
length_km: 7.004
country: Belgium
---

Overview prose goes here — surface, elevation, notable weather, anything
that applies to the whole circuit. This becomes the `overview` field.

## La Source

Tight hairpin, T1. Brake from 6th to 2nd at the 100m board. Rotate
late — the apex is past the geometric centre.

## Eau Rouge / Raidillon

Flat-out in modern dry pace. The compression at the bottom loads the
front 100% — do not lift mid-arc or the rears will step out at the
top. Stay left for the run up Kemmel.

## Pouhon

Double-left, downhill entry. Single brake, no lift. Late apex on the
second part of the corner unloads the rears for the kink onto Fagnes.
```

Each `## <corner name>` heading becomes a queryable entry. When the LLM
calls `get_track_knowledge(track="spa", corner="Eau Rouge")`, it gets
back just that section. Without a `corner` arg, it gets the overview
plus the list of corner names so it can pick.

## File naming

`tracks/<id>.md` where `<id>` is the lowercase, hyphen-separated track
key (e.g. `spa.md`, `silverstone.md`, `nordschleife.md`). The id in the
frontmatter must match the filename stem.

## When to add prose vs facts

- A short prose paragraph per corner = fine, stays in the keyed doc.
- A 5-page driver narrative about Spa in 1998 = put it in `../knowledge/`.
