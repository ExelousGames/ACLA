## Definition
Fallback label for a failed overtake attempt whose specific
attempt-type signature cannot be read from the player's trace. Player
engaged a close opponent and did not complete the pass; the opponent
didn't pass back either; the attempt simply didn't stick. Prefer a
specific subtype (failed late-brake attack at entry, failed
outside-line sweep, failed switchback, failed slipstream gain on
straight) when the trace identifies one — fall back to this label only
when none of those match.

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
- A close opponent is present and the player ran side-by-side for at
  least part of the section.
- Neither the player nor the opponent completed the pass — the
  positions are unchanged at the section boundary.
- Time difference to the expert grows across the attempt — the
  compromised line ate the time the move would have saved.
- The player's trace usually shows mistake-style symptoms too (late
  brake, wide exit, tight entry) — the *cause* of the failed attempt.

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
