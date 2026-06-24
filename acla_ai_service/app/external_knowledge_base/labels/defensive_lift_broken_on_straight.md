## Definition
On a straight section, the player attempted a defensive lift / feint
against a close trailing opponent, but the gap to the opponent kept
shrinking and they got through. The failure-mirror of defensive lift
on straight.

## Physics
A defensive lift breaks the trailing car's slipstream rhythm. By
lifting briefly the defender slows the rate at which the attacker
closes in the tow, forces them to either commit to the pass early
(at a sub-optimal moment) or check up themselves, and resets the
straight-line geometry. It fails when the lift is too small, too
late, or too predictable. A small lift in a long tow doesn't slow the
closing rate enough — the attacker absorbs it and keeps coming. A
late lift cedes the geometry before it can disrupt — the attacker is
already pulling out. A predictable lift (same place every lap) gets
pre-empted by an early pull-out and a side-by-side dive at the brake
zone.

## Telemetry signature
- A close opponent sits directly behind the player at the start of
  the straight and the gap closes to zero across the straight (the
  opponent gets through).
- The opponent completed the pass on the player.
- Gap to the opponent shrank on the straight.
- The driver slowed more than the opponent or accelerated worse while
  trying to hold position.
- The gap flipped from driver ahead to opponent ahead.

## Engineer interpretation
The lift either wasn't big enough to break the tow or was timed
after the attacker had already committed to the pass. The decision
to lift is bounded: too big a lift hands the position; too small
doesn't disrupt; too late doesn't matter. The player needs to read
*where* the attacker is in the closing sequence — still building
overlap (lift early), already pulling out (don't lift, force the
side-by-side into the brake zone), already alongside (the straight
is lost, set up the undercut).

## Remedies
- Read the gap before lifting. If the attacker is still building
  overlap, an early lift can stall the close. If they've pulled out
  already, lifting just slows you into the brake zone.
- Vary the defensive position lap-to-lap. A lift in the same place
  every lap gets pre-empted.
- If the lift didn't break the tow, don't compound it with a tight
  defensive line into the corner that becomes inside cover broken (early-brake defense). Set up the
  undercut on the next corner instead.
