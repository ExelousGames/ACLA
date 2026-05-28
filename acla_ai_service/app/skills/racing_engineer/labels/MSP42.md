## Definition
At corner entry, the tyres are operating below their available
grip envelope — the driver isn't asking them for the lateral or
longitudinal force they can deliver. The `driver_push_to_limit`
signal stays low when it should be near 1.

## Physics
Tyres have a grip budget that can be spent on braking, cornering,
or a mix of both (the grip circle). At entry, the expert uses
near-full grip — trail-braking spends some grip longitudinally
and some laterally. Inefficient utilization means the driver is
under the curve: the tyre has more to give and the driver isn't
asking.

## Telemetry signature
- `driver_push_to_limit` stays well below 1.0 through entry.
- Brake pressure low (often MSP13) — under-using longitudinal grip.
- Entry speed low — under-using lateral grip too.
- Apex speed acceptable or above expert despite the conservative
  inputs.

## Engineer interpretation
You have more in the tyre. Conservative driving feels safe but
the gap between safe and the limit is where lap time lives. Push
the brake firmer or carry more speed in — pick one to start.

## Remedies
- Build brake pressure (MSP13) — that's the easiest way to push
  utilization up.
- Carry more entry speed in small increments.
- Watch the `driver_push_to_limit` chart: aim for 0.8–0.95
  through entry, not 0.5.
