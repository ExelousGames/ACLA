---
id: MS9
name: Entry trajectory too tight
family: MS
common_co_labels: [MS2, MS33, MS3, MS10, MS44]
causes_to_check: [MS44, MS2]
---

## Definition
Driver turns in tighter than the optimal entry line — the car arrives
inside the expert's trajectory at the entry phase. Usually pairs with
a late or wide apex because the line was wrong from the start.

## Physics
A too-tight entry shortens the radius the car has to follow, which
means a slower speed is required to hold the same lateral G. Most of
the apex problems that follow are forced — the geometry leaves no
room. It also tends to put the car in a bad position for the exit:
either pinched (running out of track) or forced to widen mid-corner
(scrubbing speed).

## Telemetry signature
- Trajectory at entry sits inside the expert reference line.
- Steering input is larger and earlier than the expert's.
- Apex speed lower than expert (the radius forces it).
- Often follows MS44 (oversteer at entry) where the driver over-
  rotated to catch a slide.

## Engineer interpretation
You're turning in too early or too sharp. Two common reasons: the
brake unsettled the car so you rushed the wheel, or you're chasing
the apex with your eyes instead of carrying out to the right entry
point. Trust the wide entry — the car covers less distance overall
when the line is right.

## Remedies
- Hold wider at turn-in by half a car width; let the corner come to
  you.
- Smooth steering input — one progressive motion, not a hand jerk.
- If oversteer is forcing the tight line, fix the brake release first
  (MS17) and the entry will normalise.
- Look further ahead — eyes drive the line; chasing the apex pinches
  the entry.
