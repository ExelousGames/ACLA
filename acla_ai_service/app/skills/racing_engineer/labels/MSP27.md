---
id: MSP27
name: Initiate brake release too late
family: MSP
common_co_labels: [MSP18, MSP32, MSP3]
causes_to_check: [MSP18]
---

## Definition
Driver holds full brake (or near-peak brake) past the expert's
release-start point. The trail-off phase starts too deep in the
corner.

## Physics
Holding peak brake too long keeps front load high past the point
where the car wants weight to start migrating back to the rear.
The result is a front that's still gripping but a rear that's
under-loaded — pushy at apex, or a forced fast release that costs
stability (MSP17).

## Telemetry signature
- Time spent at peak brake longer than expert before any release
  begins.
- Brake-pressure plateau extends past expert's release-start
  marker.
- Apex speed lower than expert (over-slowed by the long peak).
- May pair with MSP18 (slow release) or MSP17 (fast release once it
  finally starts).

## Engineer interpretation
The release should start almost as soon as the peak hits. A clean
brake is a brief plateau followed by a long taper, not a long
plateau followed by a stab off. Start bleeding the pressure as
soon as the car is rotating into the corner.

## Remedies
- Begin the trail-off at the turn-in point, not later.
- Shorter brake plateau, longer trail-off taper.
- If the peak feels like it has to last to manage the entry speed,
  the peak is too low (MSP13) — go in firmer, off sooner.
