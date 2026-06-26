## Definition

Tyres on exit are under-used - the driver isn't asking for the
longitudinal grip available. `driver_push_to_limit` stays below 1
through the exit phase when the expert lap is much closer to it.

## Solution

- Earlier and harder throttle once steering is unwinding.
- Aim utilization at 0.9+ on exit.
- If the rear lets go when you commit, the apex is too tight
  (entry trajectory too tight / apex too wide) - widen the apex to free the rear.
