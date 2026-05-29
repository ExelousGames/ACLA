---
name: Speed difference (driver vs expert)
units: km/h
sign_convention: "positive = driver SLOWER than expert; negative = driver FASTER"
---

## What it measures
The instantaneous gap between the driver's current speed and the
expert reference speed at the same track position. Computed
sample-by-sample along the lap.

## Units
km/h. Positive numbers mean the driver is slower than the expert at
that point; negative means faster.

## How to read it
- **Flat near zero** — driver is matching the reference. This is what
  a clean segment looks like.
- **Sustained positive (slower) that grows** — a mistake is unfolding;
  the driver is losing time and not closing the gap. This is the
  primary trigger for a mistake segment.
- **Sustained positive that closes back to zero** — momentary
  oscillation, not a mistake; the driver recovered.
- **Sustained negative (faster)** — driver carried more speed than
  the expert at this point. Often happens at corner entry just
  before a corresponding positive spike (over-speed → bigger brake →
  slower apex).

## Common pairings
Read the speed difference together with the driver's brake trace and
the expert optimal brake to see *why* the speed gap opened — the
driver was usually braking later or harder. Pair with the lap-time
delta to see whether the speed loss has translated into time loss yet.

## Engineer interpretation
The first chart to glance at when the driver asks "where am I losing
time?" — find the segments where the curve is climbing and stays up,
those are the corners worth analysing.
