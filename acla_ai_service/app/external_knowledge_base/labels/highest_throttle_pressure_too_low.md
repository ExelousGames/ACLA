## Definition

Peak throttle on exit doesn't reach what the expert reached. The
driver lifts or modulates when the car can take full throttle.

## Solution

- Commit to 100% throttle once the steering starts unwinding.
- Compare to `expert_optimal_throttle` - if the expert was flat,
  there's no excuse.
- If the car oversteers under full throttle, look at apex tightness
  (apex too wide) and rear stability.
