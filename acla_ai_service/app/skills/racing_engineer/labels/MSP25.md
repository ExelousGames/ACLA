---
id: MSP25
name: Throttle applied too slowly
family: MSP
common_co_labels: [MSP21, MSP19, MSP11]
causes_to_check: [MSP21]
---

## Definition
Throttle ramp-up at corner exit is more gradual than the expert's
— the driver feathers in rather than committing as the car
unwinds.

## Physics
A slow throttle ramp leaves the car accelerating below its
potential through the corner exit. The cost compounds onto the
following straight — every km/h not gained on exit is a km/h not
gained at the next brake zone, and so on.

## Telemetry signature
- Throttle derivative lower than expert at throttle-onset.
- Throttle plateau reached later in the exit phase.
- Peak throttle may also be low (MSP19).
- Exit speed below expert; gap widens onto the next straight.

## Engineer interpretation
Smooth and slow aren't the same thing. Smooth means no spikes;
slow means leaving lap time on the table. Build to 100% over the
same distance the expert does — that's smooth, not aggressive.

## Remedies
- Ramp to full throttle over ~0.3 seconds once steering is
  unwinding.
- Match the expert's ramp shape, not just the peak.
- If you're slow on the throttle because you're nervous about the
  rear, look at apex tightness (MSP9) — wider apex makes
  commitment safer.
