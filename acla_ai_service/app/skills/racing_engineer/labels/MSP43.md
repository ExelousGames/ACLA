## Definition
Tyres on exit are under-used — the driver isn't asking for the
longitudinal grip available. `driver_push_to_limit` stays below 1
through the exit phase when the expert lap is much closer to it.

## Physics
Exit grip utilization is mostly about throttle commitment. The
expert builds throttle to track the available grip on the driven
axle — utilization climbs back toward 1 as the car accelerates.
Under-throttling (MSP19, MSP25) means the tyre has more to give and
the driver isn't using it.

## Telemetry signature
- `driver_push_to_limit` stays well below 1 through exit.
- Peak throttle below expert (MSP19).
- Throttle ramp gentle (MSP25).
- Exit speed below expert; gap widens onto the next straight.

## Engineer interpretation
Exit is where lap time is cheapest. The tyres are usually willing
to take more on exit than the driver gives them. Push the
throttle commitment — full pedal as soon as the line opens.

## Remedies
- Earlier and harder throttle once steering is unwinding.
- Aim utilization at 0.9+ on exit.
- If the rear lets go when you commit, the apex is too tight
  (MSP9 / MSP10) — widen the apex to free the rear.
