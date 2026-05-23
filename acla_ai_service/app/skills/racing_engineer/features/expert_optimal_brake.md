---
id: expert_optimal_brake
name: Expert optimal brake
units: 0..1
---

## What it measures
The brake pedal position (0 = off, 1 = max) the expert reference lap
applied at the same track position the driver is currently at.

## Units
Normalised 0..1.

## How to read it
- **Brake-onset point** — find where the expert trace first rises
  from zero. That's the expert's brake reference for the corner. The
  driver's onset (`Physics_brake`) compared to this tells you whether
  the driver braked early, late, or on time.
- **Peak height** — where the expert hit max pressure. Driver peaks
  above this usually indicate MS22 (peak too high).
- **Trail-off slope** — the expert's brake release shape. A long
  gradual ramp is trail-braking; a sharp drop is straight-line
  braking before a chicane.

## Common pairings
- `Physics_brake` — driver brake. The direct comparison.
- `speed_difference` — together they explain why the speed delta
  opened (late onset, high peak, fast release).
- `expert_optimal_speed` — the speed the expert carried *into* the
  brake zone tells you the speed they planned to scrub.

## Engineer interpretation
The single most useful expert channel for diagnosing entry mistakes.
Most MS labels in the entry phase trace back to a difference here.
When suggesting a brake-point change, anchor the suggestion to the
expert's actual reference — "the expert is on the brake at the 150
board" is concrete; "brake earlier" is not.
