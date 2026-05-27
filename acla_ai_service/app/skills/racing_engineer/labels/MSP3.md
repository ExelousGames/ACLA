---
id: MSP3
name: Too late compared to expert Apex
family: MSP
common_co_labels: [MSP2, MSP9, MSP10, MSP16]
causes_to_check: [MSP2, MSP9, MSP1]
---

## Definition
The car reaches the apex later in the corner than the expert
reference — further along the geometry, deeper into the turn. A
late apex usually carries a wide exit with it.

## Physics
Apex timing sets the corner's geometric balance. A late apex means
the entry took up too much of the corner's arc, leaving less room
to open the line on exit. The car ends up tracking out wider, exit
speed drops, and any car behind has a tow line.

## Telemetry signature
- Position of minimum speed sits past the expert's apex marker.
- Steering angle peak happens later in the corner.
- Speed delta vs expert grows through the apex phase.
- Exit trajectory wider than expert (often MSP16).

## Engineer interpretation
A late apex is rarely the cause — it's the result. Find what
happened earlier: a late brake (MSP1), a late turn-in (MSP2), or a
tight entry that forced the line out. Fix the upstream and the apex
position normalises.

## Remedies
- Audit the brake and turn-in points first — apex is downstream.
- If the inputs are clean and the apex is still late, you may be
  trying to keep the car too straight at entry — commit to the wheel
  earlier.
