---
id: MSP2
name: Initiate the turn too late
family: MSP
common_co_labels: [MSP3, MSP10, MSP16, MSP9]
causes_to_check: [MSP1, MSP22]
---

## Definition
Steering input starts after the expert's turn-in point. The car is
still going straight when it should be rotating — and the corner
gets squeezed into less distance than the expert had.

## Physics
A late turn-in pushes the apex later in the corner and forces a
tighter mid-corner radius to make the geometry work. Tighter radius
means lower available speed, and the catch-up usually shows as a
wider exit (the car can't tuck back to the kerb). It's often a
downstream symptom of carrying too much entry speed — the driver
delays the wheel because the car isn't pointed yet.

## Telemetry signature
- Steering input begins later than expert at the same track position.
- Trajectory stays outside the expert reference deep into the entry.
- Apex hit later in the corner (pairs with MSP3).
- Exit trajectory wider than expert (MSP16) — the line was wrong from
  the start.

## Engineer interpretation
You're rushing the corner because you arrived hot. The wheel input
follows the brake — fix the brake reference and the turn-in lines up
on its own. If the brake was right and you still turned in late,
it's an eye-line issue: look earlier into the corner.

## Remedies
- Earlier brake reference if you're arriving with too much speed.
- Pick a fixed turn-in marker (kerb edge, paint stripe) and commit to
  it.
- Eyes further ahead — the wheel goes where you're looking.
