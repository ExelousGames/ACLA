## Definition
Peak throttle on exit overshoots what the expert applied — usually
hitting 100% where the expert modulated, or going to full sooner
than the line allows. Common in cars where full throttle exceeds
mechanical or aero grip on exit.

## Physics
The expert's throttle ceiling on exit isn't arbitrary — it's the
limit the car's rear axle (RWD) or front axle (FWD) can accept
without slip while still managing lateral load. Exceeding it
trades a brief throttle peak for either wheelspin (oversteering at exit on RWD)
or push-wide understeer (understeering at exit on FWD).

## Telemetry signature
- Peak throttle higher than `expert_optimal_throttle`.
- Driven-axle slip ratio spikes at the throttle peak.
- Steering correction follows (catching the resulting slide or
  push).
- Exit speed often *lower* despite higher peak throttle — slip
  ate the gain.

## Engineer interpretation
The expert modulated for a reason. Burying the throttle past the
expert's ceiling doesn't make you faster — it makes the car slide,
and a sliding tyre is a slow tyre. Match the expert's modulation,
then look for line improvements to lift the ceiling.

## Remedies
- Compare to `expert_optimal_throttle` and respect the ceiling.
- If you need more peak, widen the apex (apex too wide) so the car has
  more grip available.
- Practice short-shifting on power exits — sometimes the car wants
  less torque, not more throttle.
