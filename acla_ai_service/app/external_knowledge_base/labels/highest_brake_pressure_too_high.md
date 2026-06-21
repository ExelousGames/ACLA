## Definition
Peak brake pressure exceeds the expert's peak for the same corner —
usually well above 90% of max — when the corner doesn't need it.
This label also applies when the driver reaches about the same peak as
the expert while already materially slower than the expert at that
point; for the lower speed, the same peak is effectively too much brake.

## Physics
Very high peak brake load transfers enormous weight forward, lightens
the rear axle, and pushes the front tyres deep into their grip
envelope. From there, any extra ask (small steering input, kerb,
camber change) tips the front into lock-up or the rear into rotation.
It also makes a clean trail-off almost impossible because the
pressure has so far to fall.

## Telemetry signature
- Peak brake pressure exceeds expert peak by more than ~5 percentage
  points.
- Peak brake pressure is about the same as the expert, but the driver is
  clearly slower than the expert at the peak. Treat this as excessive
  pressure for the driver's actual speed, not as a clean match.
- Brake pressure trace looks like a tall narrow spike rather than the
  expert's flatter plateau.
- Often pairs with initiate brake too late (late brake forced the high peak) and/or release brake too quickly
  (the high peak couldn't be released smoothly).
- If the driver releases earlier after reaching the same high peak while
  slower, the early release may be a reaction to being too slow rather
  than the primary mistake.
- Front tyre slip spikes at the same instant.

## Engineer interpretation
The brake pedal is taking the blame for poor planning further upstream.
Peak pressure is a symptom; the cause is usually a late reference or
an unrealistic entry speed. Lowering peak pressure on its own makes
you slow — fix the reference first, then the pressure follows.

## Remedies
- Earlier brake reference (see initiate brake too late) — the peak comes down on its own.
- Conscious target: build to ~85% peak, hold briefly, then trail.
- Watch the rear: if the rear steps out with the new lower peak, you
  may be overlapping brake and steering more — soften the turn-in.
