## Definition
Brake pressure falls from peak to zero too fast — the trail-off ramp
is steeper than the expert's. The car loses the front-end load needed
to turn cleanly before it has finished rotating.

## Physics
A clean trail-off is what keeps the front tyres planted while the
driver feeds in steering. Releasing too quickly drops front normal
force suddenly, so the front grip falls just as the driver is asking
the wheel for direction change. The line widens, the apex slips, and
the driver compensates with more steering — adding friction loss.

## Telemetry signature
- Time from peak brake to zero brake is shorter than the expert's by
  more than ~100 ms.
- Brake-pressure trace has a near-vertical falling edge instead of a
  gradual ramp.
- Steering angle increases at the same moment the brake hits zero —
  the driver is fighting the loss of front grip.
- Often pairs with highest brake pressure too high (high peak forced a fast release) or oversteering at entry
  (rear oversteers as load shifts unevenly).

## Engineer interpretation
You're slamming the brake off rather than rolling it off. The brake
isn't a switch — the last 20% of pressure does some of the most
important work. Think of trail-braking as steering with the brake
pedal: the lighter and slower the release, the more the car will
turn for you without extra wheel input.

## Remedies
- Consciously lengthen the release: imagine bleeding pressure over the
  same distance you spent building it.
- Lower the peak so there's less pressure to dump (see highest brake pressure too high).
- Practice on a slow corner first; trail-braking habit transfers up to
  fast corners.
