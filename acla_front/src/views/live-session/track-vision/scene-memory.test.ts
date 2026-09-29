import { RollingSceneMemory, SCENE_MEMORY_MAX_AGE_MS, SCENE_MEMORY_MAX_POINTS, SceneImage } from './scene-memory';
import { DEFAULT_CAMERA } from './camera-projection';
import type { LocalTrackScene, TrackVisionFrame } from './track-vision-types';
import { VISION_MAX_AGE_MS } from './track-vision-types';

const emptyScene = (): LocalTrackScene => ({ leftBoundary: [], rightBoundary: [], cars: [], geometry: null });
const frame = (capturedAt: number): TrackVisionFrame => ({ capturedAt, width: 160, height: 96,
    calibration: { ...DEFAULT_CAMERA, pitchDeg: 0, imageWidth: 160, imageHeight: 96 },
    detections: {
        depth: { task: 'depth', width: 160, height: 160, values: new Float32Array(160 * 160).fill(10), inferenceMs: 1, classNames: [] },
        segment: { task: 'segment', width: 160, height: 160, inferenceMs: 1,
            classNames: ['track', 'car', 'car pack', 'car interior', 'other', 'grass'],
            instances: [{ classId: 0, confidence: 0.9, box: [0, 0, 1, 1], mask: new Uint8Array(160 * 160).fill(1) }] },
    },
});
function pixels(shift = 0): SceneImage {
    let seed = 47;
    const width = 160, height = 96;
    const texture = Uint8Array.from({ length: width * height }, () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed >>> 24;
    });
    return { width, height, gray: texture.map((_, i) => i % width >= shift ? texture[i - shift] : 0) };
}

it('moves remembered geometry into the current frame and fuses repeated static observations', () => {
    const memory = new RollingSceneMemory();
    const scene = emptyScene();
    scene.leftBoundary = [{ x: -12, y: 10, z: 0 }];
    const original = memory.update(frame(0), scene, pixels(), 0)!;
    const snapshot = JSON.stringify(original);
    const result = memory.update(frame(200), emptyScene(), pixels(4), 200)!;
    expect(result.status).toBe('aligned');
    expect(result.inliers).toBeGreaterThanOrEqual(12);
    const edge = result.points.find((p) => p.surface === 'left-edge')!;
    expect(edge.x).toBeCloseTo(-11.5);
    expect(edge.y).toBeCloseTo(10);
    expect(edge.lastSeenAt).toBe(0);
    expect(result.points.some((p) => p.observations === 2)).toBe(true);
    expect(JSON.stringify(original)).toBe(snapshot);
    expect(scene.leftBoundary[0].x).toBe(-12);
    expect(memory.update(frame(200), emptyScene(), pixels(4), 200)).toBe(result);
    const next = memory.update(frame(400), emptyScene(), pixels(8), 400)!;
    expect(next.points.find((p) => p.surface === 'left-edge')!.x).toBeCloseTo(-11);
});

it('bounds memory by age, local range and voxel count', () => {
    const memory = new RollingSceneMemory();
    const scene = emptyScene();
    scene.leftBoundary = [{ x: -12, y: 10, z: 0 }];
    memory.update(frame(0), scene, pixels(), 0);
    for (let t = 1000; t <= SCENE_MEMORY_MAX_AGE_MS; t += 1000) {
        const result = memory.update(frame(t), emptyScene(), pixels(), t)!;
        expect(result.status).toBe('aligned');
        expect(result.points.some((p) => p.surface === 'left-edge')).toBe(t < SCENE_MEMORY_MAX_AGE_MS);
    }
    scene.leftBoundary = Array.from({ length: 5000 }, (_, i) => ({ x: i % 80 - 40, y: Math.floor(i / 80), z: 0 }));
    scene.rightBoundary = [{ x: 0, y: 90, z: 0 }, { x: 0, y: -15, z: 0 }, { x: 50, y: 10, z: 0 }, { x: 0, y: 10, z: NaN }];
    const result = memory.update(frame(4000), scene, pixels(), 4000)!;
    expect(result.points).toHaveLength(SCENE_MEMORY_MAX_POINTS);
    expect(result.points.every((p) => p.y >= -10 && p.y <= 80 && Math.abs(p.x) <= 40)).toBe(true);
    expect(result.points.some((p) => p.surface === 'right-edge')).toBe(false);
});

it.each([1, 2, 3, 4])('excludes overlapping dynamic/interior/unknown class %s even at lower confidence', (classId) => {
    const source = frame(0), segment = source.detections.segment!;
    if (segment.task !== 'segment') throw new Error('Expected segment');
    segment.instances.push({ classId, confidence: 0.4, box: [0, 0, 1, 1], mask: new Uint8Array(160 * 160).fill(1) });
    const scene = emptyScene();
    scene.leftBoundary = [{ x: 0, y: 10, z: 0 }];
    const result = new RollingSceneMemory().update(source, scene, pixels(), 0)!;
    expect(result.points).toHaveLength(0);
});

it('remembers supported roadside surfaces without adding cars to memory', () => {
    const source = frame(0), segment = source.detections.segment!;
    if (segment.task !== 'segment') throw new Error('Expected segment');
    segment.instances[0].classId = 5;
    const scene = emptyScene();
    scene.cars = [{ classId: 1, confidence: 0.9, pack: false, roadSupported: false, points: [{ x: 1, y: 3, z: 1 }],
        center: { x: 1, y: 3, z: 1 }, min: { x: 1, y: 3, z: 1 }, max: { x: 1, y: 3, z: 1 } }];
    const result = new RollingSceneMemory().update(source, scene, pixels(), 0)!;
    expect(result.points.length).toBeGreaterThan(0);
    expect(result.points.every((p) => p.surface === 'roadside' && p.y === 10)).toBe(true);
});

it('does not count inferred boundary points as static observations', () => {
    const scene = emptyScene();
    scene.leftBoundary = [{ x: -2, y: 10, z: 0, estimated: true }, { x: -2, y: 12, z: 0 }];
    scene.rightBoundary = [{ x: 2, y: 10, z: 0, estimated: true }];
    const result = new RollingSceneMemory().update(frame(0), scene, pixels(), 0)!;
    expect(result.points.filter((point) => point.surface.endsWith('edge'))).toEqual([
        { x: -2, y: 12, z: 0, surface: 'left-edge', lastSeenAt: 0, observations: 1 },
    ]);
});

it('drops historical surfaces when current depth proves that space is empty', () => {
    const memory = new RollingSceneMemory(), scene = emptyScene();
    scene.leftBoundary = [{ x: 0, y: 5, z: 1.2 }];
    memory.update(frame(0), scene, pixels(), 0);
    const result = memory.update(frame(200), emptyScene(), pixels(), 200)!;
    expect(result.status).toBe('aligned');
    expect(result.points.some((p) => p.surface === 'left-edge')).toBe(false);
});

it.each(['calibration', 'resolution', 'gap', 'backward', 'texture', 'depth scale'])(
    'restarts rather than smearing history after %s changes', (change) => {
        const memory = new RollingSceneMemory(), scene = emptyScene();
        scene.leftBoundary = [{ x: -12, y: 10, z: 0 }];
        memory.update(frame(100), scene, pixels(), 100);
        const next = frame(change === 'gap' ? 1200 : change === 'backward' ? 50 : 300), image = pixels();
        if (change === 'calibration') next.calibration!.yawDeg = 10;
        if (change === 'resolution') { next.width = 320; next.calibration!.imageWidth = 320; }
        if (change === 'texture') image.gray.fill(127);
        if (change === 'depth scale') {
            const depth = next.detections.depth!;
            if (depth.task === 'depth') depth.values.fill(20);
        }
        const result = memory.update(next, emptyScene(), image, next.capturedAt)!;
        expect(result.status).toBe('reset');
        expect(result.points.some((p) => p.surface === 'left-edge')).toBe(false);
        expect(result.points.every((p) => p.lastSeenAt === next.capturedAt && p.observations === 1)).toBe(true);
    });

it('clears missing, stale and malformed inputs and can seed a new capture', () => {
    const memory = new RollingSceneMemory();
    const seed = () => memory.update(frame(0), emptyScene(), pixels(), 0);
    expect(seed()).not.toBeNull();
    expect(memory.update(null, null, null)).toBeNull();
    expect(seed()!.status).toBe('seeded');
    expect(memory.update(frame(0), emptyScene(), pixels(), VISION_MAX_AGE_MS)).toBeNull();
    seed();
    const missing = frame(200); delete missing.detections.depth;
    expect(memory.update(missing, emptyScene(), pixels(), 200)).toBeNull();
    expect(seed()!.status).toBe('seeded');
    const malformed = frame(200);
    malformed.calibration!.imageWidth++;
    expect(memory.update(malformed, emptyScene(), pixels(), 200)).toBeNull();
    expect(seed()!.status).toBe('seeded');
    memory.reset();
    expect(seed()!.status).toBe('seeded');
});
