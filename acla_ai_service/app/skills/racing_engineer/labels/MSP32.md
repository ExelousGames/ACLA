---
id: MSP32
name: Highest Brake length too long
family: MSP
common_co_labels: [MSP27, MSP18, MSP13, MSP5]
causes_to_check: [MSP5, MSP13]
---

## Definition
Duration spent at peak brake is longer than the expert's. The
driver lingers on the pedal at full pressure past the point where
the trail-off should have started.

## Physics
A long peak-brake plateau scrubs more speed than needed — the
apex ends up slower than expert, exit speed drops, and the corner
becomes pinched on the inside. Often happens when the driver
braked early (MSP5) but didn't release early to match.

## Telemetry signature
- Time spent at near-peak brake pressure longer than expert.
- Apex speed below expert.
- Total brake duration extended (the plateau, not the trail).
- May pair with MSP5 (early brake-onset) — the long plateau is the
  consequence.

## Engineer interpretation
You're over-braking. The brake's job is to scrub *just enough*
speed for the apex — anything more is dead time. Either you're
braking too early (work the reference later, MSP5) or your peak is
too low (MSP13) and you're compensating with duration.

## Remedies
- Move brake reference later (MSP5) so the natural plateau is
  shorter.
- Increase peak (MSP13) and reduce duration — the same work in
  less time.
- Watch the apex speed — if it's below expert, you're scrubbing
  too much.
