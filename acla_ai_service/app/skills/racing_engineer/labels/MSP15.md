## Definition
Throttle ramp-up rate at corner exit is steeper than the expert's
— the pedal is buried rather than rolled. Often causes wheelspin
on power exits or power oversteer (MSP46) on RWD cars.

## Physics
A sudden throttle ramp asks the rear tyres for full longitudinal
grip in an instant, while they're often still carrying lateral
load from the exit. The combined demand exceeds the tyre's grip
budget; the rear slides (MSP46) or the front pushes wide (MSP47) as
weight transfers back.

## Telemetry signature
- Throttle derivative higher than expert at the throttle-on point.
- Wheel-slip ratios at the driven axle spike upward.
- Steering correction follows the throttle by ~100–300 ms (you're
  catching the slide you caused).
- Pairs with MSP26 (overshoot to full throttle).

## Engineer interpretation
Smooth out the right foot. The car can take 100% — but only when
it's pointing straight. On exit, the throttle is part of the
steering. Build it in proportion to how much the wheel is
unwinding.

## Remedies
- Roll into the throttle over ~0.3 seconds, not one stab.
- Target a smooth 0 → 50% → 100% ramp tied to steering unwind.
- If wheelspin is repeatable, check that throttle-on isn't too
  early (MSP20) — the right *amount* of throttle at the wrong
  *time* is still wrong.
