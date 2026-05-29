## Definition
A corrective change in driving — the driver is consciously
applying throttle earlier on exit than on the previous lap. The
classifier sees the change as a recovery action, often after a
previous segment was flagged with late throttle (initiate throttle too late).

## Physics
Earlier throttle-on extends the acceleration phase out of the
corner, gaining time onto the next straight. The change has to
be paired with a line that can accept the earlier power — too
early on the wrong line gives initiate throttle too early / oversteering at exit.

## Telemetry signature
- Throttle-onset position earlier than the previous lap at the
  same corner.
- May be earlier than expert (over-correction) or matched (clean
  fix).
- `speed_difference` improvement onto the following straight.

## Engineer interpretation
You corrected the right thing. Earlier throttle is one of the
easiest places to find time — if the car accepted it, repeat the
change next lap. If the rear got loose, ease back slightly but
don't undo the whole correction.

## Remedies
- Lock in the new throttle-on point — make it the reference.
- Watch for over-correction (oversteer on exit, oversteering at exit).
- If the line accepts even earlier throttle, push it further.
