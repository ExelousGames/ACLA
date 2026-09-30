import { createWorldMask, resizeMask } from './world-mask';
import type { SegmentResult } from './track-vision-types';

const instance = (classId: number, pixels: number[], confidence = 0.9): SegmentResult['instances'][number] => ({
    classId, confidence, box: [0, 0, 1, 1], mask: new Uint8Array(pixels),
});

it('keeps the full image outside interior masks, including unlabelled pixels and overlapping labels', () => {
    const segment = { task: 'segment' as const, width: 3, height: 2,
        classNames: ['track', 'new label', '  CAR  Interior  '], instances: [
            instance(0, [1, 1, 0, 0, 0, 0], 0.99), instance(1, [0, 1, 1, 1, 0, 0]),
            instance(2, [0, 1, 0, 0, 0, 0], 0.6), instance(2, [0, 0, 0, 1, 0, 0], 0.6),
        ] };
    const original = segment.instances.map(({ mask }) => mask.slice());
    const expected = new Uint8Array([1, 0, 1, 0, 1, 1]);
    expect(createWorldMask(segment, 0.5)?.mask).toEqual(expected);
    expect(segment.instances.map(({ mask }) => mask)).toEqual(original);
    segment.instances.reverse();
    expect(createWorldMask(segment, 0.5)?.mask).toEqual(expected);
});

it('keeps empty segmentation frames and excludes only accepted interior pixels', () => {
    const segment = { task: 'segment' as const, width: 2, height: 2, classNames: ['car interior', 'track'], instances: [
        instance(0, [1, 1, 0, 0]), instance(0, [0, 0, 1, 0], 0.4),
        instance(0, [0, 0, 1, 0], NaN), instance(0, [0, 0, 1, 0], 2), instance(0, [1]),
    ] };
    expect(createWorldMask(segment, 0.5)?.mask).toEqual(new Uint8Array([0, 0, 1, 1]));
    expect(createWorldMask(segment, 0.5)?.carInteriorMask).toEqual(new Uint8Array([1, 1, 0, 0]));
    expect(createWorldMask({ ...segment, instances: [] })?.mask).toEqual(new Uint8Array(4).fill(1));
    expect(createWorldMask({ ...segment, width: 0 })).toBeNull();
    segment.instances.shift();
    expect(createWorldMask(segment, 0.5)?.mask).toEqual(new Uint8Array(4).fill(1));
    expect(createWorldMask({ ...segment, instances: [instance(0, [1, 1, 1, 1])] })?.mask.some(Boolean)).toBe(false);
});

it('resamples coverage at pixel centers without turning excluded pixels back on', () => {
    const region = { width: 3, height: 2, mask: new Uint8Array([1, 0, 1, 0, 1, 0]) };
    expect(resizeMask(region, 6, 4)).toEqual(new Uint8Array([
        1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    ]));
    expect(resizeMask(region, 2, 1)).toEqual(new Uint8Array([0, 0]));
});
