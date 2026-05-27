---
id: MSR1
name: Failed overtake attempt
family: MSR
common_co_labels: [MSP1, MSP44, MSP10, MSP16]
causes_to_check: []
---

## Definition
Player engaged a close opponent with an attacking move — a late brake,
an off-line entry, or a sustained outside / inside sweep — but did not
complete the pass and lost time relative to the expert in the process.
The opponent didn't pass back either; the attempt simply didn't stick.

## Physics
An attacking line is almost always a compromised line. Late-braking
into a corner you weren't going to brake-late on costs entry stability;
sweeping around the outside extends the distance travelled while the
opponent takes the shorter inside line; sitting on the inside without
the position locked in forces a tighter mid-corner radius. Every one
of those moves shaves grip from the rest of the corner, and if the
pass doesn't complete, the time penalty is paid without the position
gain to offset it.

## Telemetry signature
- `find_nearest_opponent` returns a primary candidate with small
  `min_distance_m` or `side_by_side_iloc_count > 0` across the section.
- `passed_by_player: false` AND `got_passed_by_opponent: false` — the
  signed longitudinal gap stays on the same side at section end.
- `time_difference_to_expert` grows across the attempt — the
  compromised line ate the time the move would have saved.
- Player trace usually shows an MSP-style signature too (late brake,
  wide exit, tight entry) — the *cause* of the failed attempt.

## Engineer interpretation
The move was the right intent in the wrong place, or the right place
without the commitment. Don't punish the attempt — flag whether the
corner was actually a passing corner (run-down to a heavy braking
zone, slow apex with traction-limited exit) and whether the player
was alongside before the brake zone. If neither was true, the next
rule is "don't try here." If both were true, the rule is "commit
earlier — be alongside by the brake board, not at it."

## Remedies
- Identify which corners on the lap are actual passing corners and
  rehearse the move only at those.
- Be alongside before the brake reference, not at it — half-attempts
  cost time without buying position.
- If the attempt fails, give the place back cleanly and reset for the
  next opportunity rather than chasing the recovery line.
