## Definition
Driver upshifts before the engine reaches the optimal RPM for the
power curve — the gear change happens too low in the rev range
and the new gear has the engine below its torque peak.

## Physics
Upshifting too early drops the engine into a gear where it has
less torque to deliver to the wheels. Acceleration stalls
momentarily as the engine climbs back up to its sweet spot — what
drivers feel as "bogging down." Cumulative across a lap, early
upshifts cost meaningful time on straights and exits.

## Telemetry signature
- Upshift timestamp ahead of expert by enough to matter (varies
  per car).
- Engine RPM at shift below expert's shift point.
- Brief acceleration drop immediately after the shift.
- Often pairs with MSP19 (low peak throttle) — driver was timid all
  around.

## Engineer interpretation
Hold the gear. Modern engines pull cleanly to the limiter and
some way past their torque peak. If the dash beeps or a shift
light is on, that's the cue — not the moment you think the engine
sounds done. Trust the data, not the ear.

## Remedies
- Shift on the shift-light or the rev-counter target, not on
  sound.
- Compare your shift RPM to expert's — match the marker.
- If you're shifting early to be smooth, smoothness comes from
  technique (clean throttle lift on upshift), not from
  short-shifting.
