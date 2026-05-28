## Definition
Player attempted a switchback against a close opponent — conceding
the entry on a wider line to set up a tighter, faster exit that
crosses back inside the opponent — but the pass did not complete by
section end. The failure-mirror of O4.

## Physics
A switchback is a deliberate geometric trade: give up the apex,
brake earlier than the opponent, sacrifice mid-corner speed for a
straighter, earlier-throttle exit. Done right, the player's
trajectory and the opponent's cross — opponent on the inside at
entry, player on the inside at exit — and the player has the better
drive onto the next straight. Done wrong, the player has given up
the corner on entry without ever closing the geometric gap on exit:
the opponent's stronger mid-corner speed plus an okay-enough exit is
enough to stay ahead onto the next braking zone.

## Telemetry signature
- `find_nearest_opponent` returns a primary candidate close through
  entry-to-exit; `query_opponent_trajectory` shows the lateral-offset
  trace started to cross but the signed longitudinal gap never
  flipped.
- `passed_by_player: false` AND `got_passed_by_opponent: false`.
- Brake **initiation onset is earlier than** the expert's; trajectory
  offset goes from **wider than expert** at entry to **tighter than
  expert** at exit (the cross-over shape).
- Throttle **application onset is earlier than** expert around the
  exit.
- `time_difference_to_expert` grows across the section — the
  sacrificed entry cost more than the exit recovered.

## Engineer interpretation
The shape was right but the run-out was wrong. A switchback only
pays if the player can convert the exit advantage into a pass *by
the next braking zone* — if the straight after the corner is short,
or the opponent's defensive line shuts the inside on exit, the
geometry never resolves. Separate the two failure modes: did the
switchback shape happen but the exit advantage was too small (corner
selection issue), or did the player initiate too late so the cross
was still happening at the corner exit (timing issue)?

## Remedies
- Pick corners where the straight after is long enough to convert
  the exit drive — switchbacks at corners followed by another
  immediate brake zone rarely pay.
- Initiate the switchback earlier — be wide at the apex, not at the
  exit. The cross-over needs to be complete before the throttle
  goes back down.
- If the cross-over isn't there mid-corner, abort to a normal exit
  rather than holding the compromised line all the way through.
