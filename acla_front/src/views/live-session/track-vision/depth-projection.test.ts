import { createDepthProjection } from './depth-projection';
import { DEFAULT_CAMERA } from './camera-projection';
import type { TrackVisionFrame } from './track-vision-types';

const frame = (): TrackVisionFrame => ({
    capturedAt: 0, width: 1600, height: 800,
    calibration: { ...DEFAULT_CAMERA, pitchDeg: 0, imageWidth: 1600, imageHeight: 800 },
    detections: { depth: { task: 'depth', width: 8, height: 8, inferenceMs: 1, classNames: [],
        values: Float32Array.from({ length: 64 }, (_, i) => Math.floor(i / 8) >= 2 && Math.floor(i / 8) < 6 ? 10 : 190) } },
});

it('removes depth letterboxing and excludes padded samples at image edges', () => {
    const lift = createDepthProjection(frame())!;
    expect(lift(0.5, 0.5, () => true)).toEqual({ x: 0, y: 10, z: 1.2 });
    expect(lift(0.5, 0, () => true)!.y).toBe(10);
    expect(lift(0.5, 0.999, () => true)!.y).toBe(10);
});

it('never blends other surfaces into an object at a depth discontinuity', () => {
    const source = frame();
    const depth = source.detections.depth!;
    if (depth.task !== 'depth') throw new Error('Expected depth');
    depth.values.forEach((_, i) => { depth.values[i] = i % 8 < 4 ? 10 : 80; });
    const lift = createDepthProjection(source)!;
    expect(lift(0.499, 0.5, (u) => u < 0.5)!.y).toBeCloseTo(10);
    expect(lift(0.501, 0.5, (u) => u > 0.5)!.y).toBeCloseTo(80);
    expect(lift(0.5, 0.5, () => false)).toBeNull();
});

it('rejects mismatched dimensions, malformed maps and out-of-image points', () => {
    const source = frame();
    const depth = source.detections.depth!;
    if (depth.task !== 'depth') throw new Error('Expected depth');
    const lift = createDepthProjection(source)!;
    for (const u of [-0.1, 1.1, NaN]) expect(lift(u, 0.5, () => true)).toBeNull();
    source.calibration!.imageWidth++;
    expect(createDepthProjection(source)).toBeNull();
    source.calibration!.imageWidth--;
    depth.values = new Float32Array(1);
    expect(createDepthProjection(source)).toBeNull();
});
