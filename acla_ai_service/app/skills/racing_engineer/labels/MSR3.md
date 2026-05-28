## Definition
Player committed to a late-brake attack at corner entry against a
close opponent — braked deliberately later than the expert reference,
on a tightening line aimed at the inside — but the pass did not stick
and the opponent didn't pass back either. The failure-mirror of O1.

## Physics
A late-brake attack converts straight-line distance into corner-entry
speed at the cost of grip headroom inside the corner. The tyres
arrive at the brake zone already loaded, the brake bias has to do
more work in less distance, and the front axle is asked to slow the
car AND rotate it on a tightening trajectory at the same time. If
the commitment was a hair too optimistic, the car either runs deep
(missed apex, opponent re-takes the inside), pushes wide on exit
(opponent's shorter line beats the player to the next braking zone),
or slides on the limit (over-limit spike on grip utilisation, line
lost). All three outcomes cost time on top of giving the position
back.

## Telemetry signature
- `find_nearest_opponent` returns a primary candidate with
  `side_by_side_iloc_count > 0` or small `min_distance_m` across the
  braking-to-apex window.
- `passed_by_player: false` AND `got_passed_by_opponent: false`.
- Brake **initiation onset is later than** the expert's at entry, with
  **peak brake pressure higher than** expert across the braking phase.
- Trajectory often shows **moving toward negative (tightening)** into
  the apex (player aimed for the inside), sometimes followed by an
  **entry trajectory bulges outside** spike when grip ran out.
- `time_difference_to_expert` grows across the attempt.
- An **over-limit spike** on grip utilisation often co-occurs with
  the brake / turn-in phase.

## Engineer interpretation
The intent was right and the execution was within sight — what
killed it was either the location (the corner doesn't reward
inside-line entries) or the commitment level (alongside *at* the
brake board instead of *before* it). A late brake from a half-car
length back is gambling the corner against a position that wasn't
yours yet. Separate which of the two failed: was the player already
overlapped at the brake reference, or trying to *create* the overlap
inside the brake zone?

## Remedies
- Be alongside before the brake reference. Half a car length earlier
  on the straight, not on the brakes.
- Rehearse the specific corners on the lap where late-brake entries
  actually pay (slow apex with traction-limited exit, downhill brake
  zone with run-up). Don't try this at fast sweepers.
- If the brake is committed and the line goes long, give the place
  back cleanly on exit rather than scrubbing speed across the apex
  and losing the next corner too.
