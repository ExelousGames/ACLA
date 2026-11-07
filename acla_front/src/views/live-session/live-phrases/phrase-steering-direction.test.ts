import type { StandardTelemetrySample } from '../live-session-types';
import { compareSteeringDirection } from './phrase-steering-direction';

const motion = (x: number, y: number): StandardTelemetrySample => ({
    Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0, Physics_velocity_x: x, Physics_velocity_y: y, Physics_velocity_z: y,
});

it.each(['acc', 'iracing'])('classifies both corner directions using unit velocity in %s', (game) => {
    const displacement: [number, number, number] = game === 'acc' ? [0, 10, 0] : [0, 0, 10];
    for (const sign of [-1, 1] as const) {
        for (const scale of [0.01, 1, 100]) {
            expect(compareSteeringDirection(displacement, [0, 0, 0], motion(2 * sign * scale, 10 * scale), game, sign)).toBe('tooWide');
            expect(compareSteeringDirection(displacement, [0, 0, 0], motion(-2 * sign * scale, 10 * scale), game, sign)).toBe('tooTight');
            expect(compareSteeringDirection(displacement, [0, 0, 0], motion(0, 10 * scale), game, sign)).toBe('aligned');
        }
    }
});

it('shifts the starting position using both lateral and forward error, keeping the reference endpoint fixed', () => {
    // ACC at zero heading: model right is world -X. The driver starts at (3, -5), reference ends at (0, 10).
    const error: [number, number, number] = [3, 5, 0];
    expect(compareSteeringDirection([0, 10, 0], error, motion(-3, 15), 'acc', 1)).toBe('aligned');
    expect(compareSteeringDirection([0, 10, 0], error, motion(0, 15), 'acc', 1)).toBe('tooWide');
    expect(compareSteeringDirection([0, 10, 0], error, motion(-6, 15), 'acc', 1)).toBe('tooTight');
    expect(compareSteeringDirection([0, 10, 0], [-3, 5, 0], motion(0, 15), 'acc', 1)).toBe('tooTight');
});

it.each([
    [0, 0, 0, [-2, 3]],
    [Math.PI / 2, 0, 0, [-3, -2]],
    [-Math.PI / 2, 0, 0, [3, 2]],
    [0, Math.PI / 2, 0, [-2, -4]],
    [0, 0, Math.PI / 2, [4, 3]],
    [Math.PI / 2, Math.PI / 2, Math.PI / 2, [2, 4]],
] as const)('rotates ACC model error with heading=%s pitch=%s roll=%s', (heading, pitch, roll, offset) => {
    expect(compareSteeringDirection([0, 10, 0], [2, 3, 4], {
        ...motion(offset[0], 10 + offset[1]), Physics_heading: heading, Physics_pitch: pitch, Physics_roll: roll,
    }, 'acc', 1)).toBe('aligned');
});

it.each([
    [0, 0, 0, [-2, 40]],
    [-Math.PI / 2, 0, 0, [-40, -2]],
    [Math.PI / 2, 0, 0, [40, 2]],
    [0, Math.PI / 2, 0, [-2, 3]],
    [0, 0, -Math.PI / 2, [3, 40]],
    [-Math.PI / 2, Math.PI / 6, -Math.PI / 2, [-20 * Math.sqrt(3) - 1, 3]],
] as const)('uses iRacing world axes with heading=%s pitch=%s roll=%s', (heading, pitch, roll, offset) => {
    expect(compareSteeringDirection([0, 0, 10], [-2, 40, 3], {
        ...motion(offset[0], 10 + offset[1]), Physics_heading: heading, Physics_pitch: pitch, Physics_roll: roll,
    }, 'iracing', -1)).toBe('aligned');
});

it('ignores heading errors within 3 degrees and handles angle wraparound', () => {
    expect(compareSteeringDirection([0, 10, 0], [0, 0, 0], motion(0.5, 10), 'acc', 1)).toBe('aligned');
    expect(compareSteeringDirection([-10, -0.5, 0], [0, 0, 0], motion(-10, 0.5), 'acc', 1)).toBe('tooWide');
});

it.each([
    { Physics_velocity_x: undefined }, { Physics_velocity_y: NaN }, { Physics_heading: undefined },
    { Physics_pitch: Infinity }, { Physics_roll: undefined }, { Physics_velocity_x: 0, Physics_velocity_y: 0 },
    { Physics_velocity_x: 0, Physics_velocity_y: -10 },
])('withholds ambiguous or unavailable motion: %j', (patch) => {
    expect(compareSteeringDirection([0, 10, 0], [0, 0, 0], { ...motion(0, 10), ...patch }, 'acc', 1)).toBeUndefined();
});

it('withholds a coincident target, invalid error and unknown corner direction', () => {
    expect(compareSteeringDirection([0, 10, 0], [0, -10, 0], motion(0, 10), 'acc', 1)).toBeUndefined();
    expect(compareSteeringDirection([0, 10, 0], [NaN, 0, 0], motion(0, 10), 'acc', 1)).toBeUndefined();
    expect(compareSteeringDirection([0, 10, 0], [0, 0, 0], motion(0, 10), 'acc')).toBeUndefined();
});
