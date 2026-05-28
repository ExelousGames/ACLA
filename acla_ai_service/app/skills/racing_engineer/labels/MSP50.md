## Definition
The car runs off the track surface during the corner entry —
either onto the kerb's outer paint, the run-off, or beyond the
white line. Triggers track-limits rules and usually destroys the
corner.

## Physics
Off-track at entry is the visible failure of a chain of upstream
mistakes — almost always too much entry speed, an oversteer slide
that ran out of road (MSP44), or a wide entry that ran out of
track (MSP33). The recovery (RM1) takes time the corner can't
get back.

## Telemetry signature
- Lateral track position outside the track edge during entry.
- Wheels-on-track count drops below threshold.
- Yaw rate and lateral G erratic (the car is on a lower-grip
  surface).
- RM1 typically follows.

## Engineer interpretation
Don't analyse the off itself — analyse what caused it. Off-track
is a symptom; the root cause is the previous several seconds of
inputs. Find that, fix that, and the off won't repeat.

## Remedies
- Identify the trigger MSP label (MSP22 / MSP44 / MSP33 most commonly).
- Reduce entry speed by ~5 km/h on the next attempt at that
  corner.
- If you're going off the same corner twice, you're not adjusting
  — make a real change, not a small one.
