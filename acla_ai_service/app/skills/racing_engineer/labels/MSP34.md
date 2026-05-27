---
id: MSP34
name: Throttle and brake applied at the same time for too long
family: MSP
common_co_labels: [MSP29, MSP18]
causes_to_check: []
---

## Definition
Both pedals are active simultaneously for longer than expert —
either left-foot braking that overlaps unintentionally, or a
sloppy heel-toe transition that holds the throttle while braking
ramps up.

## Physics
On most cars, overlapping throttle and brake means the engine is
fighting the brakes — wasted energy, heat into the brakes, and
worst of all, a confused weight transfer. The front wants to dive
(brake) and the rear wants to squat (throttle) at the same time,
which produces neither cleanly.

## Telemetry signature
- Both `Physics_gas > 0` and `Physics_brake > 0` simultaneously
  for longer than ~150 ms.
- Engine speed (RPM) stays elevated through the early brake phase.
- Slip on the driven axle elevated mid-brake.
- May coincide with a slow throttle release (MSP29).

## Engineer interpretation
Pick one pedal at a time. Heel-toe is a 100–200 ms throttle blip
during downshift, not a sustained overlap. If you're using left-
foot brake on power exits, that's a different technique and the
overlap is intentional — but the telemetry can't tell, so it
flags it either way.

## Remedies
- Separate pedal commands cleanly: throttle off, then brake.
- If heel-toe, keep the throttle blip short (one downshift, then
  off).
- Left-foot brake is a deliberate choice — if you're using it,
  this label can be ignored when context warrants.
