# Racing Engineer Corpus — format spec

This folder is the knowledge layer the racing-engineer LLM reaches into
via the `explain_label`, `analyze_recent_segment`, and (Phase 2b)
`lookup_label_concept` tools. The corpus is plain Markdown + YAML
frontmatter — designed to be hand-edited by a domain expert without
learning a query DSL.

The loader lives in [__init__.py](__init__.py) and is intentionally
minimal — direct `label(id)` / `feature(name)` lookups, no embedding
index. Drop a `.md` file in the right subfolder, restart, done.

## Layout

```
labels/<slugged_name>.md       one per sub-label (oversteering_at_entry.md, merge_back_to_expert_line.md, …)
main_labels/<slugged_name>.md  one per parent family (mistake_practice.md, recovery_merge.md, …)
features/<NAME>.md             one per telemetry channel (driver_push_to_limit.md, …)
```

The filename stem is the **slugged human name** — lowercase, non-alphanumeric
replaced with `_`. Internal label ids (`MSP44`, `RM7`, …) live in
`LABEL_MAPPING` in `app/domain/labels.py` and never appear in filenames,
prose, or anywhere the LLM can see. Convert id → name via
`LABEL_MAPPING[id]` upstream; the corpus is addressed by name only.

When renaming a label in `LABEL_MAPPING`, also rename its file here so the
slug still matches.

## File format

A file is YAML frontmatter (between `---` lines) followed by Markdown
sections delimited by `## Heading` lines. The loader merges both into
one flat `dict` keyed by lowercase-snake-case names.

### Required for `labels/` entries

The `explain_label` tool returns the slim subset:
`name`, `definition`, `engineer_interpretation`, `remedies`. Author every
sub-label with at least these. The other sections are optional but
make the corpus genuinely useful for analysis.

### Worked example (`labels/oversteering_at_entry.md`)

Frontmatter is optional. The filename stem is the human name (slugged);
add frontmatter only when you want to record extra metadata (graph hints
like co-labels / causes, etc.) — these are internal-only and never
surfaced to the LLM. Internal ids never go in the frontmatter.

```markdown
## Definition
Rear of the car steps out before the apex, requiring countersteer or a
mid-entry lift to keep the line.

## Physics
Rear lateral grip exceeded at turn-in. Typical causes: trail-braking
too aggressively, abrupt steering input, weight transferred too far
forward, cold rear tyres, soft rear anti-roll bar.

## Telemetry signature
- rear slip exceeds front slip on the trajectory_balance chart at entry
- fast steering reversal (countersteer) before the apex
- yaw rate spikes ahead of steering input by ~100-300 ms
- lateral G plateaus or drops momentarily at turn-in

## Engineer interpretation
Driver is rotating the car too aggressively on the brake. Either ease
the trail-off earlier or carry less brake into the turn. If the slide
keeps showing up after corrections, it's a setup question — bring it to
the engineer.

## Remedies
- Release the brake 5–10 m earlier and let the front bite progressively.
- Reduce peak brake pressure ~5%.
- Soften the initial steering input; one smooth motion, no jerk.
- Check rear tyre temps — if they're cold, the warm-up lap was short.
```

## Section conventions

Section names below are the keys the loader produces. Use these exact
headings so tools can look them up by name.

| Heading | Purpose |
|---|---|
| `## Definition` | One-paragraph plain-English statement of what the action *is*. |
| `## Physics` | Why it happens, from a vehicle-dynamics perspective. |
| `## Telemetry signature` | Bullet list of telemetry features that flag it. Reference channel names from `app/domain/telemetry.py` etc. |
| `## Engineer interpretation` | The voice of a real race engineer talking to a driver. 2–4 sentences. This is what the LLM mostly leans on for the "engineer voice." |
| `## Remedies` | Bullet list of concrete next-time actions the driver can take. |

## Frontmatter fields (optional)

| Key | Type | Notes |
|---|---|---|
| `name` | string | Override the filename-derived human name. Rarely needed — usually the slugged filename already matches `LABEL_MAPPING[id]`. |
| `common_co_labels` | list | Internal graph hint. Not LLM-visible. |
| `causes_to_check` | list | Internal graph hint. Not LLM-visible. |

Frontmatter is optional. The loader passes unknown fields through
unchanged — future tools can pick them up. **Never put `id:` in
frontmatter**: ids are classifier-internal and must not be addressable
from the corpus side.

## Main labels (`main_labels/`)

One file per parent family — currently `mistake_practice.md`,
`mistake_racing.md`, `recovery_merge.md`, `expert_adherence.md`,
`successful_overtake.md`, `pit_stop.md`, `missing_data.md`. Per-track
parent files (`silverstone.md`, `brands_hatch.md`) can be added the
same way. These document the *family* — what counts as a "mistake,"
what "expert adherence" means, when "recovery & merge" applies. They
absorb the role the long-unused `MAIN_LABEL_GUIDELINES` dict in
`app/domain/labels.py` was reaching for.

Same section conventions, but `Remedies` typically isn't applicable
(too generic at the family level).

## Telemetry features (`features/`)

One file per channel the engineer might reference: `push_limit.md`,
`speed_difference.md`, `expert_optimal_throttle.md`, `Physics_brake.md`,
etc. Mirrors the column names in `app/domain/telemetry.py`,
`expert_features.py`, `tire_grip_features.py`.

Recommended sections:

| Heading | Purpose |
|---|---|
| `## What it measures` | Plain-English description of the signal. |
| `## Units` | Units + typical range. |
| `## How to read it` | What high / low / spiky values mean. |
| `## Common pairings` | Other channels engineers usually look at alongside this one. |

## Authoring workflow

1. Hand-author each entry — the racing-engineer corpus is data-only and
   meant to be human-written. Match the tone of existing files.
2. After dropping/editing files, the registry picks them up on next
   process start. In a long-running process call
   `racing_engineer.reload()`.

## What NOT to write here

- Code that mutates / validates label data — that's the racing engineer
  LLM's job, not the corpus loader's. The corpus is data-only.
- VLM annotation skills — those live in [../annotation/](../annotation/)
  and are queried via a completely separate registry. The two surfaces
  don't share code paths.
- Raw label code names (`"MSP44"`) in prose. Always use the natural name
  (`"oversteering at entry"`) — the driver hears the prose, never the
  code.
