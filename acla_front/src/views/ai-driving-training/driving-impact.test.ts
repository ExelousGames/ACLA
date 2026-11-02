import { detectDrivingImpact, ImpactSample } from './driving-impact';

const sample = (measuredAt: number, forward: number, right = 0, steering = 0): ImpactSample =>
    ({ measuredAt, velocity: [right, forward], steering });

it.each([20, 50, 100, 200, 400, 500])('detects a sudden stop over %s ms', (ms) => {
    const impact = detectDrivingImpact(sample(1000, 30), sample(1000 + ms, 0));
    expect(impact.available).toBe(true);
    expect(impact.detected).toBe(true);
    expect(impact.confidence).toBeGreaterThanOrEqual(0.6);
    expect(impact.confidence).toBeLessThanOrEqual(1);
    expect(impact.severity).toBe(1);
});

it.each([50, 100, 200, 500])('allows sustained hard braking and acceleration at %s ms intervals', (ms) => {
    for (const direction of [-1, 1]) {
        for (let i = 0; i < 4; i++) {
            const before = sample(1000 + i * ms, 80 + direction * 2.5 * 9.81 * i * ms / 1000);
            const after = sample(1000 + (i + 1) * ms, 80 + direction * 2.5 * 9.81 * (i + 1) * ms / 1000);
            expect(detectDrivingImpact(before, after)).toMatchObject({ detected: false, confidence: 0, severity: 0 });
        }
    }
});

it('detects sideways impacts, equal-speed reversals and accelerating rear shunts', () => {
    const before = sample(1000, 30);
    for (const after of [sample(1100, 30, 8), sample(1100, -30), sample(1100, 45)]) {
        expect(detectDrivingImpact(before, after).detected).toBe(true);
    }
});

it('reduces direction-change confidence for steering without hiding a severe stop', () => {
    const straight = detectDrivingImpact(sample(1000, 30), sample(1100, 30, 4));
    const steering = detectDrivingImpact(sample(1000, 30), sample(1100, 30, 4, 0.4));
    expect(straight.detected).toBe(true);
    expect(steering.detected).toBe(false);
    expect(steering.confidence).toBeLessThan(straight.confidence);
    expect(detectDrivingImpact(sample(1000, 30, 0, 1), sample(1100, 0, 0, -1)).detected).toBe(true);
});

it.each([-1, 1])('allows steady cornering, gradual sideslip and countersteering with sign %s', (sign) => {
    expect(detectDrivingImpact(sample(1000, 30, 2 * sign, 0.4 * sign), sample(1200, 30, 2 * sign, 0.4 * sign)).confidence).toBe(0);
    expect(detectDrivingImpact(sample(1000, 30, 0, 0.4 * sign), sample(1200, 29, 2 * sign, 0.4 * sign)).detected).toBe(false);
    expect(detectDrivingImpact(sample(1000, 30, -2 * sign, -0.4 * sign), sample(1200, 30, 2 * sign, 0.4 * sign)).detected).toBe(false);
});

it('ignores small noise, stopped wheels and gentle reverse maneuvers', () => {
    for (const [before, after] of [[sample(1000, 30), sample(1020, 29.9, 0.3)],
        [sample(1000, 0), sample(1100, 0)], [sample(1000, -1), sample(1100, -1.2, 0.1)],
        [sample(1000, 0.2), sample(1100, -0.2)]]) {
        expect(detectDrivingImpact(before, after)).toMatchObject({ detected: false, confidence: 0 });
    }
});

it('increases severity with excess impulse and reduces confidence for sparse samples', () => {
    const before = sample(1000, 30);
    const small = detectDrivingImpact(before, sample(1100, 24));
    const large = detectDrivingImpact(before, sample(1100, 10));
    expect(small.detected).toBe(true);
    expect(large.deltaV).toBeGreaterThan(small.deltaV);
    expect(large.severity).toBeGreaterThan(small.severity);
    expect(detectDrivingImpact(before, sample(1500, 0)).confidence).toBeLessThan(large.confidence);
    expect(detectDrivingImpact(sample(1000, -30), sample(1100, -10))).toEqual(large);
});

it.each([-10, 0, 1, 19, 501, 2000, NaN, Infinity])('marks unreliable intervals unavailable: %s ms', (ms) => {
    expect(detectDrivingImpact(sample(1000, 30), sample(1000 + ms, 0)))
        .toEqual({ available: false, detected: false, confidence: 0, severity: 0, deltaV: 0 });
});

it.each([NaN, Infinity, -Infinity])('rejects invalid motion values %s', (value) => {
    for (const bad of [sample(1100, value), sample(1100, 0, value), sample(1100, 0, 0, value)]) {
        expect(detectDrivingImpact(sample(1000, 30), bad).available).toBe(false);
        expect(detectDrivingImpact(bad, sample(1200, 0)).available).toBe(false);
    }
    expect(detectDrivingImpact(sample(1000, 30), sample(1100, 0, 0, 1.01)).available).toBe(false);
});
