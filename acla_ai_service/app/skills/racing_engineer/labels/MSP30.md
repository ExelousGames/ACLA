---
id: MSP30
name: Initiate throttle release too early
family: MSP
common_co_labels: [MSP23, MSP5]
causes_to_check: []
---

## Definition
Throttle lift begins before the expert's release point — the
driver eases off ahead of schedule, coasting into the brake zone
or the corner.

## Physics
Coasting between throttle and brake means time without
acceleration *and* without deceleration — neutral pedal positions
are dead time. The expert keeps power on until the brake takes
over to maximise straight-line speed; early release leaves the
car decelerating from drag alone, which is slow.

## Telemetry signature
- Throttle reaches zero before expert's lift-off point.
- Brief plateau between throttle release and brake application
  (the coast).
- Top speed slightly below expert.
- Often pairs with an early brake (MSP5).

## Engineer interpretation
Don't coast. The corner doesn't reward gentleness — it rewards
precise transitions. Stay on the gas until the brake comes on.
There's a fraction of a second where both pedals can swap cleanly
without a coast in between.

## Remedies
- Hold throttle to the expert's lift marker.
- Practice the brake-throttle handoff: lift and brake within the
  same beat, not a count apart.
- If you coast because you're unsure where the brake is, fix the
  reference (MSP1 / MSP5 territory).
