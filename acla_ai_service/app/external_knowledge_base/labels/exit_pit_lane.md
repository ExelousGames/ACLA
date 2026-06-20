## Definition
The car is exiting the pit lane and rejoining the racing surface
from the circuit's known Pit section. The segment position must overlap
the known Pit section directly or through a Pit-vs-adjacent-straight
ambiguity supported by pit-lane trajectory offset and large player/expert
speed separation or slow pit-exit acceleration.

## Physics
Pit exit is identified by merge / recovery back toward the racing line
while the segment is in the Pit section. The recovery isn't a driving
mistake; it's a procedural phase.

## Telemetry signature
- Segment position overlaps the known Pit section.
- Pit-vs-adjacent-straight ambiguity can count as Pit-section position
  when sustained pit-lane trajectory offset and large player/expert
  speed separation or slow pit-exit acceleration support pit procedure.
- Telemetry shows merge / recovery back toward the racing line.
- Throttle and speed can rise as the car rejoins, but they are not
  enough without Pit-section position or Pit-vs-straight ambiguity.

Do not infer pit exit from low speed plus a wide or merging trajectory
alone when the segment is located in a normal racing section away from
Pit.

## Engineer interpretation
Cold tyres, cold brakes. Don't ask the car for everything on the
out-lap — warm into it through the first sector or two. Most mistakes
after leaving Pit come from racing too hard before the tyres are ready.

## Remedies
- First lap out: build pace gradually, especially on entry to
  fast corners.
- Merge out cleanly from Pit.
- Watch traffic — drivers exiting the pit are slower than the
  field; expect to be passed.
