# Racing Engineer Corpus - format spec

This folder is the knowledge layer the racing-engineer LLM reaches into via
the server-side `explain_label`, `get_track_knowledge`, and
`search_racing_knowledge` tools. Frontend telemetry tools can also use this
corpus through their own application-side flows. The corpus is plain Markdown
plus optional YAML frontmatter, designed to be hand-edited by a domain expert.

The loader lives in [__init__.py](__init__.py). Label lookups are addressed by
human-readable name, not internal label id.

## Layout

```text
labels/<slugged_name>.md       one per sub-label
main_labels/<slugged_name>.md  one per parent family
features/<NAME>.md             one per telemetry channel
tools/<tool_name>.md           one per LLM-facing tool description
```

The filename stem is the slugged human name: lowercase with non-alphanumeric
characters replaced by `_`. Internal label ids such as `MSP44` live in
`LABEL_MAPPING` in `app/shared/labels.py` and must not appear in filenames or
driver-facing prose. Convert id to name upstream.

## Label Files

The `explain_label` tool returns:

```text
name
definition
solution      only when the label describes a mistake
```

Use only these sections in `labels/`:

| Heading | Purpose |
|---|---|
| `## Definition` | Plain-English statement of what the label means. |
| `## Solution` | Concrete fix for mistake labels only. Omit this for non-mistake labels. |

### Example

```markdown
## Definition

Rear of the car steps out before the apex, requiring countersteer or a
mid-entry lift to keep the line.

## Solution

- Release the brake earlier and let the front bite progressively.
- Soften the initial steering input.
```

## Main Labels

`main_labels/` documents parent families such as `mistake_practice`,
`mistake_racing`, `recovery_merge`, `expert_adherence`, `successful_overtake`,
and `pit_stop`.

Use `## Definition` for every main label. Add `## Solution` only for mistake
families where a generic fix is useful.

## Frontmatter

Frontmatter is optional. Unknown fields pass through unchanged for future
tools. Do not put `id:` in frontmatter; ids are classifier-internal and should
not be addressable from the corpus side.

## What Not To Write

- Raw label codes such as `MSP44` in driver-facing prose.
- Annotation-selection instructions; those belong in the internal annotation
  knowledge base.
- Telemetry signatures, physics analysis, or race-engineer interpretation in
  label files. Label knowledge is for understanding the label, not for
  deciding whether the label applies.
