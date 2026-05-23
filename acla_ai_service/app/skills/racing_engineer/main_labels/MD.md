---
id: MD
name: Missing Data
family: MD
---

## Definition
The segment contains gaps, corrupted samples, or impossible values in
the telemetry stream. The classifier can't say anything reliable about
driving behaviour because the underlying data is broken.

## When this applies
Sensor dropouts, log truncation, channels stuck at zero or saturated,
timestamp gaps. The classifier labels the segment MD rather than
guessing.

## How to read MD segments
Treat MD as "no comment" — skip the segment in any debrief and move
on to the next clean segment. If MD happens repeatedly in the same
spot on the track or with the same channel, that's a logging /
hardware problem worth flagging separately, not a driving issue.

## Engineer interpretation
Never coach on an MD segment. The right response when asked is "no
useful data for that bit — let's look at the next one." Resist the
temptation to interpret partial signals; the cost of being wrong is
high (driver loses trust in the coach) and the upside of being right
on guessed data is small.
