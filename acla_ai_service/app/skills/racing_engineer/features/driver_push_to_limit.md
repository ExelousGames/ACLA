---
name: Driver push to limit (slip envelope)
units: 0..1+
---

## What it measures
A scalar that aggregates tyre slip across all four corners and
expresses how close the driver is to the tyre grip limit. Built from
slip angle and slip ratio weighted against the configured front/rear
slip limits.

## Units
- 0 = no slip (tyres well within their grip budget).
- ~1.0 = at the configured slip envelope — the edge of available grip.
- > 1.0 = exceeding the envelope; you're sliding (oversteer /
  understeer depending on which axle).

## How to read it
- **Consistently in the 0.7–0.95 band** through corners — driver is
  using the available grip well without exceeding it. This is what
  fast laps look like.
- **Spikes above 1.0** — momentary loss of grip. Pair with the slip-
  by-axle trace to see whether it's front (understeer) or rear
  (oversteer).
- **Stays under 0.5** through fast corners — driver is leaving grip
  on the table. There's lap time available.

## Common pairings
- Per-axle slip channels — this push value is the summary; the
  per-axle channels tell you *where* the grip went.
- Driver throttle and brake traces — high push values during heavy
  brake or throttle are normal; high push values mid-corner with
  steady inputs mean the driver is asking the tyres for more than
  they have.

## Engineer interpretation
The cleanest single number for "are you driving the car or driving
around it?" Use it as a confidence indicator — if push values are
low through corners the driver could realistically attack, that's a
coaching point about commitment, not a setup problem.
