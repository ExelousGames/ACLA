---
id: RM6
name: Recover from small speed gap under 20
family: RM
common_co_labels: [RM7]
causes_to_check: []
---

## Definition
The car is recovering from a moderate speed deficit (less than
20 km/h) — typically following a minor mistake or an
inefficient corner that left a closeable gap.

## Physics
Small gaps close quickly under acceleration since the speed
shortfall isn't so large that drag dominates. Most of a small
recovery is just running a normal racing line; the gap closes by
itself if the driver doesn't add another mistake.

## Telemetry signature
- `speed_difference` between 0 and +20 km/h, closing toward zero.
- Throttle near expert levels.
- Recovery typically completes within one to two corners.
- Pairs with RM7 as the car merges back to the expert line.

## Engineer interpretation
Don't overthink it. Small gaps close on their own with a clean
lap — over-driving to close faster usually adds more time than
it gains. Treat the recovery as a regular lap with a slightly
different reference point.

## Remedies
- Continue normal lap technique; don't reach for extra speed.
- The gap closes inside one or two corners on its own.
- If the gap is opening (not closing), look for an ongoing issue
  — tyre fade, fuel weight, or an unnoticed earlier MSP.
