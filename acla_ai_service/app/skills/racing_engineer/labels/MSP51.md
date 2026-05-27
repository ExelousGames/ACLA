---
id: MSP51
name: Off track at exit
family: MSP
common_co_labels: [MSP16, MSP47, MSP46]
causes_to_check: [MSP47, MSP46, MSP16]
---

## Definition
The car runs off the track on the corner exit — beyond the white
line at the exit kerb. Causes a track-limits violation and
usually costs time in the recovery.

## Physics
Exit run-off comes either from understeer (MSP47 — front pushes
wide, car can't tighten back) or oversteer (MSP46 — rear slide,
driver catches it on the outside kerb) or simply from carrying
too wide a line that runs out of track (MSP16). Triggering
condition is usually too much throttle for the line.

## Telemetry signature
- Lateral track position outside the exit kerb.
- Throttle position often at or near 100% just before the off.
- Driven-axle slip elevated or lateral G plateauing then
  dropping.
- RM1 (recover from off-track) follows.

## Engineer interpretation
Look at the throttle commitment. Off-track exits are usually
self-inflicted — too much throttle, too soon, on a line that
couldn't accept it. The fix is patience on the right pedal.

## Remedies
- Delay throttle-on by half a count next attempt.
- Tighter apex (MSP10) gives more exit room.
- If the off is from genuine understeer the front needs help —
  setup question.
