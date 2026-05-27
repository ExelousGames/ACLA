---
id: MSP28
name: Initiate brake release too early
family: MSP
common_co_labels: [MSP17, MSP31, MSP13]
causes_to_check: [MSP13]
---

## Definition
Driver begins releasing brake pressure before the expert does —
the trail-off starts ahead of schedule.

## Physics
Releasing brake too early lets front load come off the front tyres
before the turn-in is established. The front loses bite right as
the driver asks the wheel for direction — understeer at entry
(MSP45) or a forced wider line.

## Telemetry signature
- Brake-release-start position ahead of expert reference.
- Brake duration shorter than expert (MSP31).
- Peak brake may be low (MSP13) since the release came so fast.
- Front slip at turn-in elevated (under-loaded front, slipping
  more easily).

## Engineer interpretation
Hold the brake a beat longer. The release is what carries you
into the turn — let go of it too soon and the front has nothing
to bite with. Think of trail-braking as the link between brake
and steering; cutting it short breaks the chain.

## Remedies
- Hold brake to the turn-in point before starting the release.
- Increase peak slightly (MSP13) so there's more pressure to bleed
  off during the trail phase.
- Practice slow-corner trail-braking first — it's easier to feel
  the cause-effect at lower speeds.
