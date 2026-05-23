---
id: MS7
name: Too early compared to expert Apex
family: MS
common_co_labels: [MS6, MS4, MS11, MS16]
causes_to_check: [MS6]
---

## Definition
The car reaches the apex earlier in the corner than the expert
reference — closer to the entry, further from the exit. The
remainder of the corner becomes an arc the line wasn't planned for.

## Physics
An early apex shortens the entry phase and stretches the exit
phase. The mid-corner geometry forces the car wider as it tries to
unwind — usually showing up as a wide-exit understeer (MS16, MS47)
or a forced lift to stop the run-off.

## Telemetry signature
- Position of minimum speed sits before the expert's apex marker.
- Steering peak happens early in the corner.
- Exit trajectory tracks wide of expert (MS16).
- Throttle-on may be late (MS21) because the line wasn't ready.

## Engineer interpretation
Same diagnosis as MS6 most of the time — early turn-in produced
early apex. Fix the turn-in marker and the apex normalises. Don't
chase the apex with the wheel; let the line take you there.

## Remedies
- Move turn-in reference later (see MS6).
- Eyes on exit reference, not on the apex kerb.
- If you keep clipping the apex early even with a good turn-in, the
  brake might be off too early too (MS5) — check the whole entry
  phase together.
