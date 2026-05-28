## Definition
Steering input begins before the expert's turn-in point. The car
starts rotating too soon — apex comes early, and the exit runs out
of track.

## Physics
Early turn-in pulls the apex toward the entry, leaving the rest of
the corner as a long unwinding tail. The car can't track the kerb
on exit — it's already too far across the corner — and either
understeers wide (MSP47, MSP16) or has to be tucked back with extra
steering correction that scrubs speed.

## Telemetry signature
- Steering input starts earlier than expert.
- Apex position sits earlier in the corner (pairs with MSP7).
- Exit trajectory wider than expert (MSP16).
- Often follows MSP5 — the brake came off early and the driver
  reached for the wheel to fill the gap.

## Engineer interpretation
Eyes too short. You're seeing the apex and turning toward it instead
of seeing the exit and letting the apex come. Pick a turn-in
reference outside the corner — the kerb where the apex starts is
already too late to use as the trigger.

## Remedies
- Move the turn-in reference later — a car length or two.
- Eye-line on the exit, not the apex.
- If brake came off too early (MSP5), fix that first; the wheel
  follows.
