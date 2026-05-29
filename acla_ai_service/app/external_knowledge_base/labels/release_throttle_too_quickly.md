## Definition
The driver lifts off the throttle faster than the expert — a sharp
back-off rather than a smooth roll-off. Destabilises the rear and
sends weight forward abruptly.

## Physics
A fast throttle release on a RWD car shifts weight forward in a
moment, lightening the rear and inviting lift-off oversteer (oversteering at entry).
On FWD it can pitch the front into a tuck. Either way the
transition into the brake phase becomes a fight rather than a flow.

## Telemetry signature
- Throttle derivative steeply negative at release.
- Throttle goes from full to zero in fewer milliseconds than expert.
- Yaw-rate disturbance immediately after release.
- Often paired with oversteering at entry (oversteer) if turn-in followed quickly.

## Engineer interpretation
You don't have to slam off the throttle to hit the brake firmly.
Roll off, then build the brake. Two pedals, two motions, no overlap
unless it's intentional (heel-toe). The smoother the lift, the
better the platform for whatever comes next.

## Remedies
- Roll off the throttle over ~0.2 seconds.
- Separate the lift from the brake — finish one before starting the
  other (unless heel-toe is the technique you want).
- If the rear keeps moving on lift, try a tiny brake overlap to
  preload the front before the full brake hit.
