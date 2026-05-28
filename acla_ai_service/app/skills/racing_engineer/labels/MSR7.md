## Definition
Player ran an inside-cover defense at corner entry — brake earlier
than the expert and/or trajectory tighter than the expert through
entry-to-apex, closing the inside line — but the close opponent got
through anyway. The failure-mirror of OD1.

## Physics
An inside cover is a compromised line by design: the player brakes
earlier (sacrificing entry speed), holds the tighter inside radius
(sacrificing mid-corner speed), and accepts a wider, slower exit. The
trade buys position by denying the opponent the inside attack
geometry. It breaks in three common ways. First, the cover started
too late — the opponent was already alongside or inside at the brake
board, so the tighter line just produced a slower car on the wrong
side. Second, the cover started in time but the exit ran wide enough
that the opponent's better mid-corner speed plus a clean exit beat
the player to the next braking zone. Third, the cover induced a
mistake — over-tight entry, understeer toward the apex, or a
trail-brake that became an over-limit spike.

## Telemetry signature
- `find_nearest_opponent` returns a close primary candidate
  (`side_by_side_iloc_count > 0` typical) across the entry-to-apex
  window; `query_opponent_trajectory` shows the opponent's signed
  longitudinal gap going negative → positive across the section.
- `got_passed_by_opponent: true`.
- Brake **initiation onset is earlier than** expert AND/OR
  trajectory offset is **tighter than expert** through entry-to-apex.
- Trajectory often shows offset **moving toward positive (widening)**
  post-apex — the compromised entry forced a wide exit.
- `time_difference_to_expert` grows across the section.
- An understeer signature (slip balance trough below −0.02 rad)
  sometimes co-occurs with the tight entry.

## Engineer interpretation
The defense started in the wrong place — either too late
(opponent already had the inside) or in a corner where the
inside-cover geometry never held position to begin with. Inside
covers only work where the corner punishes the outside line *and*
where there's no immediate next-corner re-attack. If the corner
opens to a long straight, the cover that produced a wide exit just
hands the position back on the run-down.

## Remedies
- Pick the one or two corners on the lap where an inside cover
  actually denies the pass *and* doesn't sacrifice the exit. Avoid
  covering corners that open onto long straights.
- Break the tow on the straight before the brake zone, not inside
  the corner. A defense that begins under brakes is already losing.
- If the opponent is already alongside at the brake reference, give
  the place cleanly. Setting up the undercut on the next corner beats
  fighting through and losing two places.
