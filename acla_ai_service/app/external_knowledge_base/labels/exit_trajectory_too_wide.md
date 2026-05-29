## Definition
Exit line drifts wider than the expert reference — the car uses
more track than it should on the way out, sometimes running onto
the kerb or off the white line.

## Physics
Wide exits happen one of two ways. Either the front can't tighten
the line (understeering at exit understeer at exit), forcing the driver to use more
track to manage the radius, or the driver opened the throttle too
early (initiate throttle too early / throttle applied too quickly) and let weight transfer push the front wide.
Both end at the same telemetry: line outside the expert reference.

## Telemetry signature
- Lateral track position on exit outside the expert line.
- Steering angle stays loaded longer than expert.
- Throttle-on may be earlier or harder than expert.
- Front slip elevated through the exit phase.

## Engineer interpretation
Two questions: is the car not turning (understeering at exit), or are you not letting
it turn (early throttle, initiate throttle too early)? The fix depends on the cause.
Either way the driver feels it as the car "not wanting to come back
in" — which is usually them, not the car.

## Remedies
- Delay throttle-on until the steering is unwinding.
- If the front pushes wide even with patient throttle, talk to the
  engineer about front grip / aero balance.
- Tighter apex (apex too wide) gives the front more room at exit.
