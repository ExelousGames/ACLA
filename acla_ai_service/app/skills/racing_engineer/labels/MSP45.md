## Definition
Front tyres lose grip at corner entry — the car refuses to turn
as much as the driver steers for. The driver winds on extra lock
and the car still ploughs wide of the intended line.

## Physics
Front grip is overwhelmed at entry, usually because the front
isn't loaded enough (brake released too early, MSP28) or the
inputs are coming in too fast for the tyre to find slip (MSP14).
On cold tyres or low-grip surfaces it shows up more easily.

## Telemetry signature
- Steering angle larger than the car's actual yaw response would
  predict.
- Trajectory drifts wide of the expert line at entry.
- Front slip ratio elevated, near or above its envelope limit.
- Often paired with MSP9 (entry too tight — driver compensated for
  the push).

## Engineer interpretation
The front needs more load to bite. Carry brake further into the
turn — let the trail-off plant the front. Don't fight understeer
with more steering; more lock just generates more slip.

## Remedies
- Trail-brake longer (don't release as early — see MSP28).
- Smoother steering input — fast hands create slip.
- Check tyre temps; cold fronts produce phantom understeer.
- If understeer is repeatable across corners, talk to the
  engineer about front grip (camber, pressure, aero).
