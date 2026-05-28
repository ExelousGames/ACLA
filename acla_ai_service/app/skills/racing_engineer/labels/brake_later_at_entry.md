## Definition
A corrective change — the driver moves the brake reference later
than the previous lap. Typically a fix after an early brake
(initiate brake too early) cost time on the straight.

## Physics
Later brake captures more of the available straight-line
acceleration. The brake phase shortens; the peak may climb (highest brake pressure too high
risk if pushed too far). When done well, it removes dead time
without compromising the corner.

## Telemetry signature
- Brake-onset position later than the previous lap.
- Top speed before the brake zone closer to expert.
- Peak brake pressure may rise (manage so it doesn't hit highest brake pressure too high).
- `speed_difference` improves into the brake zone.

## Engineer interpretation
Working the brake reference later in steps is the right way to
find lap time. Two-metre steps per lap; check the apex speed
each time. If the apex speed drops, you've gone too far — back
off one step.

## Remedies
- Move the reference 2–5 m later per lap.
- Watch peak brake (highest brake pressure too high) — don't compensate with stab pressure.
- Stop pushing the reference once apex speed and trajectory
  match expert.
