---
id: MSP14
name: Brake applied too quickly
family: MSP
common_co_labels: [MSP22, MSP44, MSP17]
causes_to_check: [MSP1]
---

## Definition
The rate at which brake pressure ramps up is steeper than the
expert's — the pedal is stamped rather than rolled. Front load
arrives all at once, which spikes front grip demand and unsettles
the rear.

## Physics
Sudden brake ramp dumps weight forward in a fraction of a second.
The front tyres go from cold-grip to peak-load instantly, often
overshooting the optimal slip ratio (lock-up). The rear goes
light at the same instant — any steering input near the same time
turns into oversteer (MSP44).

## Telemetry signature
- Brake-pressure derivative (rate of change) higher than expert at
  brake-onset.
- Brake-pressure curve has a near-vertical leading edge.
- Front slip spikes immediately after brake-onset.
- Often paired with MSP22 (over-shoots the peak too) and MSP44
  (rear lets go).

## Engineer interpretation
Roll the pedal, don't stamp it. The brake works through a curve,
not a switch. A clean ramp gives the tyres time to find their
optimal slip, and gives the rear time to settle before you add
steering.

## Remedies
- Build pressure over ~0.2–0.4 seconds instead of one stab.
- Practise the first 20% of the pedal — the rest follows.
- If you're stamping because you're late (MSP1 upstream), fix the
  reference first.
