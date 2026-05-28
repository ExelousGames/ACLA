## Definition
The segment contains gaps, corrupted samples, or impossible values in
the telemetry stream. Nothing reliable can be said about driving
behaviour because the underlying data is broken.

## When this applies
Sensor dropouts, log truncation, channels stuck at zero or saturated,
timestamp gaps. The segment is flagged as missing data rather than
guessed.

## How to read a missing-data segment
Treat missing data as "no comment" — skip the segment in any debrief
and move on to the next clean segment. If missing data happens
repeatedly in the same spot on the track or with the same channel,
that's a logging / hardware problem worth flagging separately, not a
driving issue.

## Engineer interpretation
Never coach on a missing-data segment. The right response when asked
is "no useful data for that bit — let's look at the next one." Resist
the temptation to interpret partial signals; the cost of being wrong
is high (driver loses trust in the coach) and the upside of being
right on guessed data is small.
