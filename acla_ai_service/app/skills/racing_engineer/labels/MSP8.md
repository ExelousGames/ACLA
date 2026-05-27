---
id: MSP8
name: Time to exit too late
family: MSP
common_co_labels: [MSP21, MSP19, MSP3, MSP11]
causes_to_check: [MSP3]
---

## Definition
Driver holds the steering and waits to unwind well past the
expert's exit-start point. The car finishes the corner late —
throttle-on is delayed, exit speed bleeds.

## Physics
Late exit means the car spends more time at low speed in the
slowest part of the corner. Every fraction of a second held on
the wheel after the apex is a fraction not spent accelerating onto
the next straight. The line may look neat from the outside; the
lap time tells the truth.

## Telemetry signature
- Steering unwind begins later than expert.
- Throttle-on point delayed (often pairs with MSP21).
- Peak throttle reached later in the exit (MSP19 if it never gets
  there).
- Speed delta vs expert grows through the exit phase.

## Engineer interpretation
You're babying the exit. The car is finished with the corner
before you are. Commit to the throttle the moment the steering
starts unwinding under load — that's the cue, not a number of
metres past the apex.

## Remedies
- Open the throttle the instant the wheel feels light.
- Smooth not slow — the goal is earlier throttle, not violent
  throttle.
- If the rear breaks loose when you try it, then the line through
  the apex was tighter than it needed to be (MSP9 / MSP10) — widen
  the apex slightly.
