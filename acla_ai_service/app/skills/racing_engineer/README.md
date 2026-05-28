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
labels/<ID>.md          one per sub-label (MSP44.md, RM7.md, …)
main_labels/<ID>.md     one per parent family (MSP.md, MSR.md, RM.md, EA.md, …)
features/<NAME>.md      one per telemetry channel (push_limit.md, …)
```

The filename stem is the canonical id. Use the same id the classifier
emits (`LABEL_MAPPING` keys in `app/domain/labels.py`).

## File format

A file is YAML frontmatter (between `---` lines) followed by Markdown
sections delimited by `## Heading` lines. The loader merges both into
one flat `dict` keyed by lowercase-snake-case names.

### Required for `labels/` entries

The `explain_label` tool returns the slim subset:
`name`, `definition`, `engineer_interpretation`, `remedies`. Author every
sub-label with at least these. The other sections are optional but
make the corpus genuinely useful for analysis.

### Worked example (`labels/MSP44.md`)

```markdown
---
id: MSP44
name: Oversteering at entry
common_co_labels: [MSP17, MSP22, MSP3]
causes_to_check: [MSP14, MSP22, MSP9]
---

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

## Frontmatter fields

| Key | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | Match the classifier's label id (e.g. `MSP44`). |
| `name` | string | yes | Human-readable label name. Mirrors `LABEL_MAPPING[id]`. |
| `common_co_labels` | list of ids | no | Labels that frequently co-occur with this one. |
| `causes_to_check` | list of ids | no | Labels worth checking as possible root causes. |
| `family` | string | no | Parent family (e.g. `MSP`). The loader doesn't enforce; use it if the LLM needs grouping later. |

You can add any other frontmatter fields — the loader passes them
through unchanged. Future tools can pick them up; existing tools ignore
unknown fields.

## Main labels (`main_labels/`)

One file per parent family: `MSP.md`, `MSR.md`, `RM.md`, `EA.md`, `O.md`,
`OD.md`, `PS.md`, `MD.md` plus per-track ones (`silverstone.md`, `brands_hatch.md`) as
those get written. These document the *family* — what counts as a
"mistake," what "expert adherence" means, when "recovery & merge"
applies. They absorb the role the long-unused `MAIN_LABEL_GUIDELINES`
dict in `app/domain/labels.py` was reaching for.

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
