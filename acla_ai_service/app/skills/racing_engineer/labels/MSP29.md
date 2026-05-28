## Definition
Throttle lift begins after the expert's throttle-release point —
the driver is still on the gas when they should already be
preparing for the brake zone or the corner.

## Physics
Holding throttle past the expert's release point costs distance —
either the brake comes on earlier and the car ends up scrubbing
the same amount of speed but starting at a higher peak (MSP22), or
the brake gets compressed into less distance with a fast ramp
(MSP14). Either way the brake phase ends up uglier.

## Telemetry signature
- Throttle still positive past expert's lift-off marker.
- May pair with MSP34 (brake+throttle overlap) if the brake came
  on before the throttle finished releasing.
- Top speed before brake zone above expert.
- Brake phase compressed (MSP22, MSP14 downstream).

## Engineer interpretation
Bravery on the straight, debt at the brake zone. Carrying extra
speed feels fast but the corner has to pay it back. Lift at the
same marker the expert lifts — the brake reference depends on
that to land correctly.

## Remedies
- Set the lift point as a fixed marker, same as the brake point.
- If you're trying to gain time on the straight, look upstream —
  the previous corner exit is where straight-line speed actually
  comes from.
