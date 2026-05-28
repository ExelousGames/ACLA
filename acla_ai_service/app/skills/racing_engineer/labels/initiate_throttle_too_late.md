## Definition
Throttle application begins after the expert's throttle-on point.
The car is unwound and ready for power, but the driver is still
coasting through the exit.

## Physics
Every fraction of a second between when the car wants throttle and
when the driver delivers it is a fraction the car spends not
accelerating. On modern grippy tyres the window between "ready"
and "too early" is wide — under-throttling here doesn't gain
stability, it just costs time.

## Telemetry signature
- Throttle-onset position later than expert reference.
- Steering already unwinding at throttle-onset (the car was ready).
- Speed delta vs expert grows through the exit phase.
- Peak throttle reached late (often pairs with highest throttle pressure too low).

## Engineer interpretation
You're hesitating. The car is asking for throttle and you're not
giving it. Commit on the steering unwind — that's the cue. If you
keep arriving late, your eye-line is probably too short; look
further past the corner.

## Remedies
- Throttle-on tied to steering coming back to centre.
- Eyes far past the exit — they pull the right foot.
- If the car oversteers when you commit earlier, the apex needs
  widening (entry trajectory too tight) — there's no point committing to throttle if the
  geometry punishes you.
