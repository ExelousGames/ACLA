## Definition
Time or position loss caused by an *interaction* with a close opponent —
not by a technical execution error on an empty piece of track. Sub-labels
(MSR1, MSR2) describe the specific kind of interaction failure: a pass
attempt that didn't stick, or a defensive line that got broken.

## When this applies
A segment is labelled MSR when a close primary opponent is present
(`find_nearest_opponent` returns a small `min_distance_m` or
`side_by_side_iloc_count > 0`) AND the interaction outcome is adverse —
the player tried to pass and didn't, or got passed while trying to hold.
If no close opponent is present, the section is MSP, not MSR; the
technical execution error stands on its own.

## How to read a MSR chain
MSR rarely arrives alone — the player's trace usually shows MSP-style
symptoms too (a late brake, a tight entry) because the *cause* of the
failed interaction is the optimistic move the player tried to make. The
job is to separate the two: the MSR sub-label captures the racing
decision, the co-occurring MSP sub-labels capture how it broke down
technically. Coach the decision first, the execution second.

## Engineer interpretation
The driver got into a position with another car and the position cost
them. Don't moralise the move — every overtake attempt that fails or
defense that breaks once worked for someone in the same spot. Identify
whether the right call was *don't try here* (wrong place) or *try here
but commit harder / earlier* (right place, wrong execution), and give
one concrete next-time rule.
