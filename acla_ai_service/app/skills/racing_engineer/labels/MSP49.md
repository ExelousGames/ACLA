---
id: MSP49
name: Gear Too high when accelerating
family: MSP
common_co_labels: [MSP38, MSP19]
causes_to_check: [MSP38]
---

## Definition
On corner exit, the driver is in a higher gear than the expert —
engine RPM is below its torque peak and the car bogs as the
driver tries to accelerate.

## Physics
Below the torque peak the engine produces less torque-per-RPM —
the gear ratio multiplies that lower number, so wheel torque
drops further. The result is sluggish acceleration even at full
throttle. Expert lap was in a shorter (lower) gear by design.

## Telemetry signature
- Engine RPM at throttle-on well below expert's apex RPM.
- Acceleration trace climbs slowly with full throttle applied.
- Time delta vs expert grows through the exit and onto the
  straight.
- Usually downstream of MSP38 (downshift came too late or didn't
  happen).

## Engineer interpretation
You're in the wrong gear. Either you missed a downshift on entry
(MSP38) or your corner-by-corner gear plan is off by one. Add a
downshift in the brake zone next attempt.

## Remedies
- Add the missing downshift in the brake phase.
- Confirm the right gear for the apex — most setups have a clear
  optimal for each corner.
- If the engine is bogging at the limiter going into the corner,
  the gear is way off — review the data with the engineer.
