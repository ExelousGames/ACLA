---
id: MSP46
name: Oversteering at exit
family: MSP
common_co_labels: [MSP15, MSP26, MSP20]
causes_to_check: [MSP15, MSP26, MSP20]
---

## Definition
Rear loses grip on corner exit — power oversteer in RWD cars, or
lift-off oversteer in any car if the throttle came off mid-exit.
The driver catches the slide with countersteer, scrubbing speed.

## Physics
On exit, the rear axle is being asked to deliver longitudinal
grip (power) and lateral grip (still finishing the corner)
simultaneously. Overshoot either side of the budget and the rear
slides. Most often the cause is too much throttle too early
(MSP20) or too aggressive a ramp (MSP15).

## Telemetry signature
- Yaw rate exceeds steering-input expectation on exit.
- Steering-angle reversal (countersteer) after throttle-on.
- Rear-axle slip ratio elevated.
- Throttle trace shows early or steep build (MSP20 / MSP15).

## Engineer interpretation
Smooth the throttle out. The car will take full pedal — but
not all at once and not while the wheel is still loaded. Build
the throttle in proportion to how much the wheel is unwinding.

## Remedies
- Delay throttle-on until steering is unwinding (MSP20).
- Gentler throttle ramp (MSP15).
- If repeatable across corners, the rear may need more grip —
  setup question for the engineer.
- A wider apex (MSP10) often cures power oversteer on RWD cars by
  giving the rear more grip on exit.
