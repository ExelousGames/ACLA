## Definition
The driver is attempting a pass on another car. Different rules apply:
the optimal line may not be the racing line, brake points typically
move later, and exit priorities change (defending track position vs.
exit speed).

## When this applies
The classifier marks a segment O when the trajectory and speed profile
diverge from the expert in a way consistent with an overtake — late
brake, deliberate line deviation, throttle-on timing tied to opponent
position rather than ideal exit.

## How to read O segments
An overtake segment looks like an MSP to the classifier in some ways
(off the expert line, high peak brake) but it's a deliberate choice.
Don't coach an O segment as if it were a mistake. The right question
is whether the attempt was *successful and safe* — if yes, the line
was correct for the situation. If the overtake didn't stick or
involved contact, then it's worth a debrief.

## Engineer interpretation
Treat O as a context flag, not a critique. The driver knows they were
overtaking. Ask: did it work? did you lose much time on the exit? was
the gap real or did you optimistic-brake? Those are the conversations
that matter. Telemetry deltas to the expert line are mostly noise
during an overtake.
