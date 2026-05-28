## Definition
Driver downshifts before the brake phase has scrubbed enough speed
— the new (lower) gear puts the engine above redline or close to
it. Risks over-rev damage and locks the rear axle on engine brake.

## Physics
Downshifting at too-high a road speed maps to engine RPM beyond
the car's safe rev limit. Modern engines protect with rev cuts but
the engine braking that comes in is sudden and harsh; on a RWD car
it can lock the rear wheels and pitch the car into oversteer.
Even with a working rev-match, the engine braking is the wrong
intensity for the corner phase.

## Telemetry signature
- Downshift timestamp before brake has reduced speed adequately.
- Engine RPM immediately after downshift near or at limiter.
- Rear-axle slip ratio spikes (engine-braking lockup) right after
  the shift.
- Brake pressure may dip from the engine taking over.

## Engineer interpretation
Brake first, downshift second. The downshift should happen when
the road speed has come down enough that the new gear feels right
under your foot. Most modern cars are forgiving on this, but the
rear can still bite under hard engine braking on cold tyres.

## Remedies
- Delay each downshift until the speed has dropped to its target.
- If the car has a downshift-block (refuses too-low gears), use
  that — don't override it.
- On RWD with cold rears, be especially careful — the engine-
  braking lockup hits before the tyres can take it.
