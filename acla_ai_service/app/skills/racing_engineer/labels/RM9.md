---
id: RM9
name: Accelerate later at exit
family: RM
common_co_labels: [RM7]
causes_to_check: []
---

## Definition
A corrective change — the driver delays throttle-on on exit
relative to the previous lap. The classifier flags it when the
late throttle pattern is recovery from an over-aggressive exit
(MS20) that produced an off or a slide.

## Physics
A later throttle-on settles the car before power applies. Costs
some exit speed compared to a perfectly-timed throttle, but
preserves traction and the line. Sometimes the right answer
isn't "earlier throttle" but "right-time throttle, not earlier
than the car can take."

## Telemetry signature
- Throttle-onset position later than the previous lap.
- Steering more unwound at throttle-on than before.
- Rear stability improved (less correction, lower slip ratio).
- Exit speed might be slightly lower — accepted trade.

## Engineer interpretation
You stopped over-driving the exit. Good — control matters more
than commitment when the previous attempt was wild. Now find
the middle ground: not as early as the mistake, not as cautious
as this fix.

## Remedies
- Don't over-correct: stay closer to the expert's throttle-on
  point than to your previous-lap mistake.
- Aim for matching expert next attempt, not for max caution.
- The right answer is the expert's reference, not a permanent
  defensive throttle.
