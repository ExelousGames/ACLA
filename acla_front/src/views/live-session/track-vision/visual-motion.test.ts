import { estimateRigidMotion, matchVisualFeatures, MotionImage, PointMatch, transformPoint } from './visual-motion';
import type { GroundPoint } from './track-vision-types';

const knownTransform = (p: GroundPoint) => {
    // Independent yaw, pitch and roll, followed by camera translation.
    const yaw = 0.07, pitch = -0.03, roll = 0.02;
    const x = Math.cos(yaw) * p.x - Math.sin(yaw) * p.y;
    const y = Math.sin(yaw) * p.x + Math.cos(yaw) * p.y;
    const z = Math.sin(pitch) * y + Math.cos(pitch) * p.z;
    return { x: Math.cos(roll) * x + Math.sin(roll) * z - 0.7,
        y: Math.cos(pitch) * y - Math.sin(pitch) * p.z - 3,
        z: -Math.sin(roll) * x + Math.cos(roll) * z + 0.2 };
};
const correspondences = (): PointMatch[] => Array.from({ length: 40 }, (_, i) => {
    const previous = { x: (i % 8) - 4, y: 8 + Math.floor(i / 8) * 5, z: (i % 3) * 0.4 };
    return { previous, current: knownTransform(previous) };
});

it('recovers full visual rotation and translation in the previous-to-current direction despite outliers', () => {
    const matches = correspondences();
    matches.forEach((match, i) => { if (i % 5 === 0) match.current = { x: i, y: 60 - i, z: -4 }; });
    const result = estimateRigidMotion(matches)!;
    expect(result).not.toBeNull();
    expect(result.inliers).toBe(32);
    expect(result.rmseM).toBeLessThan(1e-8);
    const probe = { x: 2, y: 14, z: 1 };
    const actual = transformPoint(probe, result.motion), expected = knownTransform(probe);
    for (const axis of ['x', 'y', 'z'] as const) expect(actual[axis]).toBeCloseTo(expected[axis], 7);
});

it('accepts planar static features while rejecting collinear, sparse and nonfinite support', () => {
    const planar = correspondences().map(({ previous }) => {
        const p = { ...previous, z: 0 };
        return { previous: p, current: knownTransform(p) };
    });
    expect(estimateRigidMotion(planar)?.rmseM).toBeLessThan(1e-8);
    expect(estimateRigidMotion(planar.slice(0, 11))).toBeNull();
    expect(estimateRigidMotion(planar.map((m) => ({ ...m, previous: { ...m.previous, x: NaN } })))).toBeNull();
    expect(estimateRigidMotion(planar.map((_, i) => ({ previous: { x: 0, y: i, z: 0 },
        current: { x: 1, y: i - 1, z: 0 } })))).toBeNull();
});

it('rejects depth scale jumps, implausible motion and a majority of inconsistent matches', () => {
    const matches = correspondences();
    expect(estimateRigidMotion(matches.map(({ previous }) => ({ previous,
        current: { x: previous.x * 1.3, y: previous.y * 1.3, z: previous.z * 1.3 } })))).toBeNull();
    expect(estimateRigidMotion(matches.map(({ previous }) => ({ previous,
        current: { ...previous, y: previous.y + 30 } })))).toBeNull();
    expect(estimateRigidMotion(matches.map(({ previous }) => ({ previous,
        current: { x: previous.y, y: -previous.x, z: previous.z } })))).toBeNull();
    matches.forEach((match, i) => { if (i % 3 !== 0) match.current = { x: i % 7, y: 5 + i * 2, z: i % 4 }; });
    expect(estimateRigidMotion(matches)).toBeNull();
});

function image(shift = 0): MotionImage {
    const width = 96, height = 64;
    let seed = 42;
    const texture = Uint8Array.from({ length: width * height }, () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed >>> 24;
    });
    return { width, height, gray: texture.map((_, i) => i % width >= shift ? texture[i - shift] : 0),
        support: new Uint8Array(width * height).fill(1), lift: (x, y) => ({ x: x * 0.2, y: 15, z: y * 0.2 }) };
}

it('matches translated image texture and lifts both observations into metric coordinates', () => {
    const matches = matchVisualFeatures(image(), image(5));
    expect(matches.length).toBeGreaterThanOrEqual(12);
    for (const match of matches) {
        expect(match.current.x - match.previous.x).toBeCloseTo(1);
        expect(match.current.z).toBe(match.previous.z);
    }
    const motion = estimateRigidMotion(matches)!;
    expect(motion.motion.translation.x).toBeCloseTo(1);
});

it('rejects textureless, repeated, unsupported and non-overlapping image evidence', () => {
    const previous = image(), current = image(5);
    current.support.fill(0);
    expect(matchVisualFeatures(previous, current)).toHaveLength(0);
    current.support.fill(1); current.gray.fill(127);
    expect(matchVisualFeatures(previous, current)).toHaveLength(0);
    previous.gray.fill(127);
    expect(matchVisualFeatures(previous, current)).toHaveLength(0);
    for (let i = 0; i < previous.gray.length; i++) previous.gray[i] = ((i % 96) % 4 < 2) === (Math.floor(i / 96) % 4 < 2) ? 0 : 255;
    current.gray.set(previous.gray);
    expect(matchVisualFeatures(previous, current)).toHaveLength(0);
    expect(matchVisualFeatures(image(), { ...image(), width: 95 })).toHaveLength(0);
});
