import { detectDrivingImpact, ImpactSample } from './driving-impact';

const sample = (measuredAt: number, forward: number, right = 0, steering = 0, brake = 0): ImpactSample =>
    ({ measuredAt, velocity: [right, forward], steering, brake });

it.each([20, 80])('detects an unbraked diagonal bounce and slide from %s km/h', (kph) => {
    const speed = kph / 3.6;
    for (const ms of [100, 200, 300, 500]) {
        for (const sign of [-1, 1]) {
            const after = sample(1000 + ms, speed * (kph === 20 ? 0.6 : 0.9), speed * (kph === 20 ? 0.3 : 0.15) * sign);
            const impact = detectDrivingImpact(sample(1000, speed), after);
            expect(impact.detected).toBe(true);
            expect(impact.severity).toBeGreaterThan(0);
            expect(detectDrivingImpact(after, { ...after, measuredAt: after.measuredAt + 100 }).detected).toBe(false);
        }
    }
});

it.each([18, 20, 30, 80])('detects an unbraked stop from %s km/h without a speed cutoff', (kph) => {
    expect(detectDrivingImpact(sample(1000, kph / 3.6), sample(1500, 0)).detected).toBe(true);
});

it.each([20, 80])('allows coasting, acceleration and velocity noise from %s km/h', (kph) => {
    for (const ms of [20, 100, 300, 500]) {
        const speed = kph / 3.6;
        for (const change of [-3 * ms / 1000, 2 * ms / 1000]) {
            expect(detectDrivingImpact(sample(1000, speed), sample(1000 + ms, speed + change, 0.2)))
                .toMatchObject({ detected: false, confidence: 0 });
        }
    }
});

it.each([20, 50, 100, 200, 400, 500])('detects a sudden stop over %s ms', (ms) => {
    const impact = detectDrivingImpact(sample(1000, 30), sample(1000 + ms, 0));
    expect(impact.available).toBe(true);
    expect(impact.detected).toBe(true);
    expect(impact.confidence).toBeGreaterThanOrEqual(0.6);
    expect(impact.confidence).toBeLessThanOrEqual(1);
    expect(impact.severity).toBe(1);
});

it.each([[20, 2], [50, 3], [100, 4], [200, 7], [400, 14], [500, 18]])(
    'detects smaller jolts over %s ms with a %s m/s speed change', (ms, change) => {
        for (const direction of [-1, 1]) {
            const impact = detectDrivingImpact(sample(1000, 30), sample(1000 + ms, 30 + direction * change));
            expect(impact.detected).toBe(true);
            expect(impact.severity).toBeGreaterThan(0);
            if (direction < 0 && change >= 14) expect(impact.severity).toBe(1);
            else expect(impact.severity).toBeLessThan(1);
        }
    });

it.each([-1, 1])('detects a small lateral jolt with sign %s', (sign) => {
    expect(detectDrivingImpact(sample(1000, 30), sample(1100, 30, 2.6 * sign)).detected).toBe(true);
});

it.each([[50, 2], [100, 3], [200, 6], [400, 12], [500, 15]])(
    'detects diagonal jolts in all four directions while steering over %s ms', (ms, change) => {
        for (const forward of [-1, 1]) {
            for (const right of [-1, 1]) {
                const before = sample(1000, 30, 0, 0.4 * right);
                const after = sample(1000 + ms, 30 + change * forward, change * right, 0.4 * right);
                const impact = detectDrivingImpact(before, after);
                expect(impact.detected).toBe(true);
                expect(impact.severity).toBeGreaterThan(0);
            }
        }
    });

it.each([-1, 1])('detects diagonal jolts during a steering change while travelling with sign %s', (travel) => {
    for (const forward of [-1, 1]) {
        for (const right of [-1, 1]) {
            const before = sample(1000, 30 * travel);
            const after = sample(1100, 30 * travel + 4 * forward, 4 * right, 0.4 * right);
            expect(detectDrivingImpact(before, after).detected).toBe(true);
        }
    }
});

it.each([20, 50, 100, 200, 500])('allows ordinary combined braking and sideslip over %s ms', (ms) => {
    for (const sign of [-1, 1]) {
        const seconds = ms / 1000;
        const speed = 30 - 1.5 * 9.81 * seconds;
        const slip = 0.4 * seconds;
        const before = sample(1000, 30, 0, 0.4 * sign, 0.5);
        const after = sample(1000 + ms, speed * Math.cos(slip), speed * Math.sin(slip) * sign, 0.4 * sign, 0.5);
        expect(detectDrivingImpact(before, after)).toMatchObject({ detected: false, confidence: 0, severity: 0 });
    }
});

it.each([20, 50, 100, 200, 300, 400, 500])('detects unbraked stops around 10 km/h over %s ms', (ms) => {
    for (const speed of [-12, -10, -9, 9, 10, 12]) {
        const impact = detectDrivingImpact(sample(1000, speed / 3.6), sample(1000 + ms, 0));
        expect(impact.detected).toBe(true);
        expect(impact.confidence).toBeGreaterThanOrEqual(0.5);
        expect(impact.severity).toBeGreaterThan(0);
        expect(impact.severity).toBeLessThan(0.3);
    }
});

it('detects a low-speed contact that leaves the car creeping with sparse samples', () => {
    expect(detectDrivingImpact(sample(1000, 10 / 3.6), sample(1500, 1 / 3.6)).detected).toBe(true);
});

it.each([100, 200, 500])('allows a braked stop from 10 km/h over %s ms', (ms) => {
    for (const [beforeBrake, afterBrake] of [[0.1, 0], [0, 0.1], [1, 1]]) {
        const before = sample(1000, 10 / 3.6, 0, 0, beforeBrake);
        const after = sample(1000 + ms, 0, 0, 0, afterBrake);
        expect(detectDrivingImpact(before, after)).toMatchObject({ detected: false, confidence: 0 });
    }
});

it.each([20, 50, 100, 200, 500])('allows low-speed coasting deceleration and noise over %s ms', (ms) => {
    const before = sample(1000, 10 / 3.6);
    const after = sample(1000 + ms, 10 / 3.6 - 3 * ms / 1000 - 0.3);
    expect(detectDrivingImpact(before, after)).toMatchObject({ detected: false, confidence: 0 });
});

it('does not treat pulling away or stopping at a crawl as a low-speed impact', () => {
    expect(detectDrivingImpact(sample(1000, 0), sample(1200, 10 / 3.6)).detected).toBe(false);
    expect(detectDrivingImpact(sample(1000, 2 / 3.6), sample(1200, 0)).detected).toBe(false);
});

it('rejects a gradual braked excess impulse without enough abruptness', () => {
    const impact = detectDrivingImpact(sample(1000, 30, 0, 0, 1), sample(1500, 16, 0, 0, 1));
    expect(impact.deltaV).toBeGreaterThan(1);
    expect(impact.detected).toBe(false);
    expect(impact.confidence).toBe(0);
});

it.each([50, 100, 200, 500])('allows sustained hard braking and acceleration at %s ms intervals', (ms) => {
    for (const direction of [-1, 1]) {
        for (let i = 0; i < 4; i++) {
            const brake = direction < 0 ? 1 : 0;
            const before = sample(1000 + i * ms, 80 + direction * 2.5 * 9.81 * i * ms / 1000, 0, 0, brake);
            const after = sample(1000 + (i + 1) * ms, 80 + direction * 2.5 * 9.81 * (i + 1) * ms / 1000, 0, 0, brake);
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
    for (const bad of [sample(1100, value), sample(1100, 0, value), sample(1100, 0, 0, value), sample(1100, 0, 0, 0, value)]) {
        expect(detectDrivingImpact(sample(1000, 30), bad).available).toBe(false);
        expect(detectDrivingImpact(bad, sample(1200, 0)).available).toBe(false);
    }
    expect(detectDrivingImpact(sample(1000, 30), sample(1100, 0, 0, 1.01)).available).toBe(false);
});

it.each([-0.01, 1.01])('rejects an out-of-range brake pedal value: %s', (brake) => {
    const bad = sample(1100, 0, 0, 0, brake);
    expect(detectDrivingImpact(sample(1000, 10 / 3.6), bad).available).toBe(false);
    expect(detectDrivingImpact(bad, sample(1200, 0)).available).toBe(false);
});
