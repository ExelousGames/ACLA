## Definition
A driving error that costs time relative to the expert reference for
the same segment of track. Sub-labels (MSP1, MSP9, MSP22, …) describe
the specific *kind* of mistake — late brake, oversteer at entry,
high peak brake pressure, etc.

## When this applies
A segment is labelled MSP when the time-difference-to-expert curve
grows during the segment and doesn't close back by the end. A small
oscillation that recovers isn't a mistake; a sustained widening gap
is.

## How to read a MSP chain
Mistakes rarely arrive alone. A late brake (MSP1) tends to drag a high
peak (MSP22) and an oversteer (MSP44) along with it. When several MSP
sub-labels are present in one segment, find the *root* — usually the
earliest one in the corner phase. Fix the root and the rest often
disappear.

## Engineer interpretation
The driver did something they didn't intend to do, and it cost time.
The job is to translate the telemetry signature into one concrete
change the driver can make next time. Don't list every sub-label;
collapse them into a causal chain and pick the one lever that fixes
the most.
