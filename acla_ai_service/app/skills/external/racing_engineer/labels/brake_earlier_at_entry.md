## Definition
A corrective change — the driver moves the brake reference
earlier than the previous lap. Typically a fix after a late
brake (initiate brake too late) caused an off or a destroyed corner.

## Physics
Earlier brake reduces the peak pressure needed (highest brake pressure too high doesn't
fire), gives more distance to bleed off the trail-brake (release brake too quickly
doesn't fire), and lets the car settle at turn-in. Costs a
fraction of straight-line speed but unlocks a clean corner
behind it.

## Telemetry signature
- Brake-onset position earlier than the previous lap.
- Peak brake pressure may be lower (the time bought lets the
  brake be less stabby).
- Apex speed often matches expert better than the previous
  attempt.
- `speed_difference` improves through the corner phase.

## Engineer interpretation
Right call. Bravery on the straight loses time in the corner;
sanity on the straight gains time everywhere. Confirm the new
reference works for two laps, then start working back later in
small increments.

## Remedies
- Lock in the new reference for now.
- Once it feels stable, work the reference back by 2–3 m at a
  time over multiple laps.
- The goal is the expert's reference — not the earliest possible
  brake.
