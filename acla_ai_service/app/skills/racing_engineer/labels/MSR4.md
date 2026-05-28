## Definition
Player committed to an outside-line sweep around a close opponent —
trajectory wider than the expert reference through entry-to-apex by a
sustained margin — but the pass did not complete and the opponent
didn't pass back either. The failure-mirror of O3.

## Physics
The outside line trades distance for grip. By running wider than the
expert into the corner, the player keeps a wider radius and a higher
mid-corner speed, hoping to carry the speed advantage past the
opponent's tighter inside line by exit. The geometry only works if
the corner allows a true cross-over — the outside line has to be
*shorter overall* by the next braking zone — and if the player can
actually get the throttle down earlier on exit. If the corner doesn't
reward the outside (most don't), or the opponent holds the inside
*and* the exit, the extra distance is paid in time without ever
producing the geometric advantage.

## Telemetry signature
- `find_nearest_opponent` returns a primary candidate with
  `side_by_side_iloc_count > 0` across the entry-to-apex window;
  `query_opponent_trajectory` shows the opponent on the inside of the
  player's heading throughout.
- `passed_by_player: false` AND `got_passed_by_opponent: false`.
- Player trajectory offset **wider than expert** sustained across
  entry-to-apex (not a single-frame spike).
- Throttle pickup on exit is often earlier than expert, but the
  speed-delta benefit isn't enough to flip the signed gap.
- `time_difference_to_expert` grows across the section — the wider
  arc cost more than the throttle pickup recovered.

## Engineer interpretation
Outside passes are corner-dependent and they're a commitment from
*before* the corner — by the time the player is alongside on the
outside at entry, the geometry is already decided. If the corner
doesn't have a real outside-line option (no traction advantage on
exit, no next-corner cross-over), the move was never going to work
and the player paid the time to find out. If the corner *does*
reward the outside, the question is whether the player was alongside
early enough to claim the line.

## Remedies
- Identify the two or three corners on the lap that actually reward
  an outside sweep (look for traction-limited exits onto a long
  straight, or a sequence where the outside at corner N is the inside
  at corner N+1). Don't try the outside elsewhere.
- Commit to the outside line *before* turn-in, not during. A
  half-committed outside attempt costs as much as a full one but
  rarely produces the geometric advantage.
- If by mid-corner the cross-over isn't there, settle for second on
  the exit rather than chasing a wider line that loses two corners.
