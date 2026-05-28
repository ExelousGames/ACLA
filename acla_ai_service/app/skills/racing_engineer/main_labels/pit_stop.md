## Definition
The driver is entering the pit lane, stationary at the box, or exiting
the pit lane. Segment characteristics are dominated by pit-lane rules
(speed limit, deliberate trajectory deviation) rather than driving
technique.

## When this applies
Segments inside the pit-lane region of the track, including the
approach (speed reduction onto pit entry) and the exit (acceleration
back onto track speed). Stationary time during a stop is also a pit
stop.

## How to read pit-stop segments
Most analysis isn't useful here — the pit-lane speed limit and the
deliberate line off the racing surface make the telemetry incomparable
to a normal lap. The interesting things are pit-entry timing (did the
driver brake too early / too late for the entry line?), pit-exit
acceleration (did they hit the limiter cleanly?), and merge-out
behaviour (a recovery segment often follows).

## Engineer interpretation
Don't drill into telemetry deltas for pit-stop segments — they will
look "wrong" by design. Useful coaching is at the boundaries: smooth
pit-entry deceleration, clean exit acceleration to track speed, safe
merge into traffic. If the driver speeds in the pit lane, that's a
separate problem the system flags directly.
