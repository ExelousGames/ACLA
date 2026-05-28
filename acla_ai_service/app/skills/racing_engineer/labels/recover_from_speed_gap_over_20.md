## Definition
The car is recovering from a speed deficit greater than 20 km/h
relative to the expert reference at the same track position —
typically following a significant mistake, an off-track moment,
or a forced slow-down.

## Physics
A large speed deficit takes longer to close than a small one
because acceleration falls off as speed rises (drag scales with
the square of speed). The expert is pulling away in clean air
while you're climbing back up; the gap compounds before it
closes.

## Telemetry signature
- `speed_difference` exceeds +20 km/h sustained.
- Full or near-full throttle while the gap closes.
- Recovery segment longer than recover from small speed gap under 20 (small-gap counterpart).
- Often follows recover from off-track (off-track) or off track at entry/off track at exit (off-track during
  corner).

## Engineer interpretation
Drive your own race for the recovery. Trying to make up the time
in one corner is how mistakes compound — one off becomes two,
two becomes a DNF. Make up the time in 0.1s increments per lap,
not 2s in one corner.

## Remedies
- Resist over-driving the next corner; recovery rewards patience.
- Watch tyre / brake temps — heat builds when you're chasing.
- If the deficit is from a real incident (off, spin), accept
  the loss and target a clean rest of the lap.
