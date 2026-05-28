---
name: Expert optimal throttle
units: 0..1
---

## What it measures
The throttle position (0 = closed, 1 = full) the expert reference lap
applied at the same track position the driver is currently at.

## Units
Normalised 0..1. Multiply by 100 for percentage.

## How to read it
- Compare against the driver's throttle trace to see whether the
  driver is opening throttle earlier, later, or at the same point as
  the expert.
- **A late expert throttle-on point** at corner exit usually means the
  expert was patient — that's a sign the driver shouldn't be on
  throttle earlier either, even if the car feels ready.
- **Expert held throttle through a section the driver lifted** — the
  driver is being cautious where the line allowed flat. Often a
  confidence problem, not a setup problem.

## Common pairings
- The driver's throttle trace — direct comparison.
- Expert optimal brake — together they reveal whether the expert
  overlapped brake/throttle or used a clean transition.
- Expert steering trace — the trio shows the full expert input
  pattern at the point of interest.

## Engineer interpretation
A useful sanity check before suggesting "throttle earlier" — if the
expert wasn't on throttle earlier either, the problem is somewhere
else (line, brake, rotation), not throttle timing.
