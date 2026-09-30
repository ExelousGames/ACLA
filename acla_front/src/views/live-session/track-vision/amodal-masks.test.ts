import { predictAmodalMasks } from './amodal-masks';
import type { TrackVisionFrame } from './track-vision-types';

function fixture(label = 'track', overlapping = false, background = false) {
    const width = 20, height = 20;
    const target = (x: number, y: number) => x >= 3 && x < 17 && y >= 3 && y < 17;
    const occluder = (x: number, y: number) => x >= 8 && x < 12 && y >= 6 && y < 14;
    const mask = (contains: (x: number, y: number) => boolean) => Uint8Array.from({ length: width * height }, (_, i) =>
        Number(contains(i % width, Math.floor(i / width))));
    const frame: TrackVisionFrame = { capturedAt: 0, width: 100, height: 100,
        detections: {
            segment: { task: 'segment', width, height, inferenceMs: 0, classNames: [label, 'occluder'], instances: [
                { classId: 0, confidence: 0.9, box: [0.15, 0.15, 0.85, 0.85],
                    mask: mask((x, y) => target(x, y) && (overlapping || !occluder(x, y))) },
                { classId: 1, confidence: 0.9, box: [0.4, 0.3, 0.6, 0.7], mask: mask(occluder) },
            ] },
            depth: { task: 'depth', width: 40, height: 40, inferenceMs: 0, classNames: [],
                values: Float32Array.from({ length: 1600 }, (_, i) => {
                    const x = Math.floor(i % 40 / 2), y = Math.floor(i / 40 / 2);
                    return occluder(x, y) ? 5 : target(x, y) ? 10 : 20;
                }) },
        } };
    const segment = frame.detections.segment!;
    if (background && segment.task === 'segment') {
        segment.classNames.push('background terrain');
        segment.instances.push({ classId: 2, confidence: 0.9, box: [0, 0, 1, 1], mask: mask((x, y) => !target(x, y)) });
    }
    return frame;
}

it.each(['track', 'car', 'car pack', 'curb', 'grass', 'fence', 'sand', 'other', 'Outfield asphalt road', 'new model label'])(
    'predicts a hidden section supported by another mask for %s', (label) => {
        const frame = fixture(label, false, true);
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('segment');
        const original = segment.instances.map((item) => item.mask.slice());
        const prediction = predictAmodalMasks(frame).find((mask) => mask.classId === 0)!;
        expect(prediction.hiddenMask.reduce((sum, active) => sum + active, 0)).toBe(32);
        expect(prediction.depths[10 * 20 + 10]).toBe(10);
        expect(prediction.visibleMask[10 * 20 + 10]).toBe(0);
        expect(segment.instances.map((item) => item.mask)).toEqual(original);
    },
);

it('handles overlapping amodal inputs independently of instance order', () => {
    const frame = fixture('car', true);
    const before = predictAmodalMasks(frame);
    expect(before.find((mask) => mask.classId === 0)!.hiddenMask[210]).toBe(1);
    const segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.instances.reverse();
    expect(predictAmodalMasks(frame)).toEqual(before);
});

it.each(['unlabelled gap', 'missing depth', 'farther instance', 'no depth'])(
    'leaves unsupported regions unresolved with %s', (reason) => {
        const frame = fixture();
        const segment = frame.detections.segment!, depth = frame.detections.depth!;
        if (segment.task !== 'segment' || depth.task !== 'depth') throw new Error('detections');
        if (reason === 'unlabelled gap') segment.instances.pop();
        if (reason === 'missing depth' || reason === 'farther instance') depth.values = depth.values.map((value) =>
            value === 5 ? reason === 'missing depth' ? NaN : 30 : value);
        if (reason === 'no depth') delete frame.detections.depth;
        expect(predictAmodalMasks(frame).find((mask) => mask.classId === 0)!.hiddenMask.some(Boolean)).toBe(false);
    },
);

it('interpolates a sloped hidden surface in inverse depth', () => {
    const frame = fixture();
    const depth = frame.detections.depth!;
    if (depth.task !== 'depth') throw new Error('depth');
    depth.values = depth.values.map((value, i) => value === 10 ? 1 / (0.12 - Math.floor(i % 40 / 2) * 0.003) : value);
    const prediction = predictAmodalMasks(frame).find((mask) => mask.classId === 0)!;
    expect(prediction.depths[210]).toBeCloseTo(1 / (0.12 - 10 * 0.003), 4);
});

it.each(['track', 'car', 'grass', 'new label'])('retains %s and lower-confidence car interior independently', (label) => {
    const frame = fixture(label, true);
    const segment = frame.detections.segment!, depth = frame.detections.depth!;
    if (segment.task !== 'segment' || depth.task !== 'depth') throw new Error('detections');
    segment.classNames[1] = '  CAR  interior ';
    segment.instances[1].confidence = 0.6;
    const original = segment.instances.map(({ mask }) => mask.slice()), originalDepth = depth.values.slice();
    for (const reverse of [false, true]) {
        if (reverse) segment.instances.reverse();
        const masks = predictAmodalMasks(frame, 0.65);
        expect(masks).toHaveLength(2);
        const target = masks.find(({ classId }) => classId === 0)!;
        const interior = masks.find(({ classId }) => classId === 1)!;
        expect(target.mask).toEqual(original[0]);
        expect(interior.mask).toEqual(original[1]);
        expect(target.hiddenMask.some(Boolean)).toBe(false);
        expect(interior.hiddenMask.some(Boolean)).toBe(false);
        expect(target.depths[210]).toBe(5);
        expect(interior.depths[210]).toBe(5);
    }
    segment.instances.reverse();
    expect(segment.instances.map(({ mask }) => mask)).toEqual(original);
    expect(depth.values).toEqual(originalDepth);
});
