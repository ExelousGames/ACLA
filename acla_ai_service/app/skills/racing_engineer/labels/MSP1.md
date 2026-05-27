---
id: MSP1
name: Initiate brake too late
family: MSP
common_co_labels: [MSP22, MSP14, MSP44, MSP3, MSP10]
causes_to_check: []
---

## Definition
Brake application begins after the expert's reference point for the
same corner. Triggers a cascade — almost every other entry mistake
downstream of this one comes from compensating for the late brake.

## Physics
Less distance to scrub the speed you need at the apex means you have to
trade something else: peak brake pressure goes up, brake-release timing
gets compressed, and the car loads the front harder and faster than
intended. The result is a destabilised platform exactly when you want
it most settled — at turn-in.

## Telemetry signature
- Brake-onset distance from corner entry shorter than the expert's by
  more than a few metres.
- Peak brake pressure noticeably higher than expert peak (often paired
  with MSP22).
- Apex speed lower than expert — you bled off speed in less distance,
  so the car arrived slower.
- Speed delta vs expert grows during entry rather than closes.

## Engineer interpretation
You're arriving at the corner with too much speed and the brake is
your panic button. The cure isn't braking harder — it's braking
earlier. Pick a fixed reference (a board, a kerb, a sponsor mark) two
to five car lengths before what you used last lap and try that.

## Remedies
- Move the brake reference earlier by 5–10 m and rebuild from there.
- Aim for a peak brake pressure ~80–85% of max rather than stamping.
- Once the new reference feels normal, work on trail-off rather than
  pushing the reference back later again.
