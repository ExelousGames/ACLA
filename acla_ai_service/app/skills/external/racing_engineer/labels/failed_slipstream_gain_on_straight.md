## Definition
On a straight section, the player was in a close opponent's slipstream
— speed gain from draft, throttle at or below expert — and the gap
closed but did not flip by the end of the straight. The failure-mirror
of slipstream gain on straight.

## Physics
Slipstreams produce free speed by reducing aero drag for the
trailing car, but the close-following position itself is unstable —
the trailing car is on dirtier air for lateral grip, and the pull-out
to complete the pass costs the draft the moment it begins. The pass
mechanically requires the player to (a) close in the draft until
overlap is reachable, (b) pull out to clean air with enough remaining
straight, (c) cross the opponent before the brake zone. Get any of
the three timings wrong and the geometry doesn't resolve: pull out
too early and the draft dies before overlap; too late and the brake
reference arrives before the cross; never pull out and the pass
becomes a half-overlap dive into the next braking zone (which often
becomes failed late-brake attack at entry — a failed late-brake).

## Telemetry signature
- A close opponent sits directly ahead (in-line, not side-by-side)
  with a small longitudinal gap, typically under ~30 m.
- Neither the player nor the opponent completed the pass.
- The longitudinal gap shrinks monotonically across the straight but
  **never crosses zero**.
- Player speed delta shows **player faster than expert** with
  throttle **at or below expert** — the speed gain is draft, not
  input.
- Sometimes followed by a failed late-brake attack into the next brake
  zone (panic late-brake when the straight runs out).

## Engineer interpretation
Two failure modes worth separating. First: was there enough straight
to complete the pass? If the closing rate plus the remaining
straight didn't algebraically produce a cross, the move was always a
two-lap setup, not a one-lap pass — the player should have stayed
tucked in for the next opportunity instead of pulling out. Second:
if the geometry *was* there, did the pull-out timing kill the draft
before the overlap? Pulling out at half a car length is too early;
pulling out under the brake board is too late.

## Remedies
- Run the geometry on the straight: closing rate × remaining
  straight distance = available overlap. If the number says no,
  don't pull out at all — stay in the draft to the brake zone and
  set up the next-corner attack instead.
- Pull out late but committed — feet ahead of the opponent's rear
  wheel, not a side-by-side dive at the brake board.
- If the straight runs out and the pass isn't there, brake on the
  expert reference. A clean exit from the brake zone usually beats a
  desperate late-brake that becomes failed late-brake attack at entry.
