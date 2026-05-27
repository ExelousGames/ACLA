---
id: MSR2
name: Defense broken (got passed)
family: MSR
common_co_labels: [MSP1, MSP22, MSP9, MSP16]
causes_to_check: []
---

## Definition
Player attempted to hold position against a close primary opponent
and the opponent got through — the signed longitudinal gap flipped
from negative (opponent behind at section entry) to positive (opponent
ahead at section exit). The defensive line or the cover-brake didn't
hold.

## Physics
A defensive line is also a compromised line. Covering the inside
forces a tighter, slower mid-corner radius; lifting early to cover
brake-pressure into the corner sacrifices entry speed; running wide
to "shut the door" opens the inside back up on exit. If the opponent
has a run on you from the previous corner, a static defensive line
won't hold — you need to break the tow before the brake zone, not
react to it inside the corner.

## Telemetry signature
- `find_nearest_opponent` returns a primary candidate with small
  `min_distance_m` or `side_by_side_iloc_count > 0` across the section.
- `got_passed_by_opponent: true` — signed longitudinal gap flips
  negative → positive across the section.
- Player trace typically shows an early-lift / cover-brake (often
  paired with MSP1) and a defensive inside line that produced a tight
  apex and wide exit.
- `time_difference_to_expert` usually grows across the section — the
  cover line cost time and didn't keep the place.

## Engineer interpretation
Two things to separate. First, was the defense the right call here at
all? If the corner doesn't reward an inside line, the cover wasted
time without ever being likely to hold. Second, if it was the right
call, did it start late? A defense that begins inside the corner is
already losing — the place to defend is the straight before, by
breaking the tow or forcing the opponent to commit early.

## Remedies
- Pick the one or two corners on the lap where an inside cover line
  actually rewards the defender; don't cover elsewhere.
- Break the tow on the straight (a small lift / lift-and-go feint)
  before the brake zone, not inside the corner.
- If the opponent is already alongside at the brake board, give the
  place cleanly and set up the undercut on the next corner instead of
  fighting through and losing two places.
