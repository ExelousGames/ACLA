import { drawLabelDepths, filteredFrame, filteredMasks, labelDepthRange, labelDepths } from './pipeline-visuals';
import type { AmodalMask } from './amodal-masks';
import type { TrackVisionFrame } from './track-vision-types';

afterEach(() => { jest.restoreAllMocks(); });

it('retains cockpit masks and their depth for downstream filtering without mutating detections', () => {
    const track = { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array(16).fill(1) };
    const interior = { ...track, classId: 1, confidence: 0.5, mask: Uint8Array.from({ length: 16 }, (_, i) => Number(i % 4 === 0)) };
    const rejected = { ...track, classId: 2, confidence: 0.6 };
    const depth = Float32Array.from([10, NaN, 0, 201, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]);
    const frame: TrackVisionFrame = { capturedAt: 0, width: 640, height: 640, detections: {
        segment: { task: 'segment', width: 4, height: 4, classNames: ['track', 'car interior', 'grass'],
            inferenceMs: 1, instances: [track, interior, rejected] },
        depth: { task: 'depth', width: 4, height: 4, classNames: [], inferenceMs: 1, values: depth },
    } };
    const masks = filteredMasks(frame);
    expect(masks).toHaveLength(2);
    expect(masks.find(({ label }) => label === 'track')!.mask).toEqual(track.mask);
    expect(masks.find(({ label }) => label === 'car interior')!.mask).toEqual(interior.mask);
    expect(labelDepths(masks)).toEqual([{ classId: 0, label: 'track', maskIndex: 1, instance: 1, estimated: 0,
        samples: 13, near: 10, far: 40, median: 20 }, { classId: 1, label: 'car interior', maskIndex: 0, instance: 1, estimated: 0,
        samples: 4, near: 10, far: 10, median: 10 }]);
    const filtered = filteredFrame(frame, masks).detections.segment;
    expect(filtered?.task === 'segment' && filtered.instances.find(({ classId }) => classId === 0)!.box).toEqual([0, 0, 1, 1]);
    expect(track.mask).toEqual(new Uint8Array(16).fill(1));
    expect(frame.detections.segment?.task === 'segment' && frame.detections.segment.instances).toHaveLength(3);
    expect(frame.detections.depth?.task === 'depth' && frame.detections.depth.values).toBe(depth);
});

it('keeps masks with the same label separate and excludes hidden predictions from measured distances', () => {
    const mask = (values: number[], hidden: number[], classId = 0): AmodalMask => ({
        classId, label: classId ? 'fence' : 'track', confidence: 0.9, box: [0, 0, 1, 1], bounds: [0, 0, 2, 2],
        mask: new Uint8Array(4).fill(1), depths: Float32Array.from(values), hiddenMask: Uint8Array.from(hidden),
        visibleMask: Uint8Array.from(hidden, (value) => 1 - value),
    });
    const rows = labelDepths([mask([10, 20, 100, 0], [0, 0, 1, 0]), mask([30, 40, 0, 0], [0, 0, 0, 0]),
        mask([0, NaN, Infinity, 201], [0, 0, 0, 0], 1)]);
    expect(rows).toEqual([
        { classId: 0, label: 'track', maskIndex: 0, instance: 1, estimated: 1, samples: 2, near: 10, far: 20, median: 15 },
        { classId: 0, label: 'track', maskIndex: 1, instance: 2, estimated: 0, samples: 2, near: 30, far: 40, median: 35 },
        { classId: 1, label: 'fence', maskIndex: 2, instance: 1, estimated: 0, samples: 0, near: null, far: null, median: null },
    ]);
    expect(filteredMasks(null)).toEqual([]);
});

it('combines mask depths and draws per-mask captions matching the table, including missing depth', () => {
    const mask = (active: number[], depths: number[]): AmodalMask => ({
        classId: 0, label: 'track', confidence: 0.9, box: [0, 0, 1, 1], bounds: [0, 0, 2, 2],
        mask: Uint8Array.from(active), visibleMask: Uint8Array.from(active), hiddenMask: new Uint8Array(4),
        depths: Float32Array.from(depths),
    });
    const masks = [mask([1, 1, 0, 0], [10, 10, 200, 0]), mask([0, 1, 1, 1], [0, 30, 40, NaN])];
    masks[1].bounds = [1, 0, 2, 2];
    masks[1].hiddenMask[2] = 1;
    masks[1].visibleMask[2] = 0;
    const frame: TrackVisionFrame = { capturedAt: 0, width: 1280, height: 720, detections: {
        segment: { task: 'segment', width: 2, height: 2, classNames: ['track'], inferenceMs: 1, instances: masks },
    } };
    const layers: Uint8ClampedArray[] = [];
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        createImageData: () => {
            const data = new Uint8ClampedArray(16);
            layers.push(data);
            return { data };
        }, putImageData: jest.fn(),
    } as any);
    const context = { save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), fillRect: jest.fn(),
        fillText: jest.fn(), measureText: jest.fn(() => ({ width: 80 })),
    } as unknown as CanvasRenderingContext2D;
    expect(labelDepthRange(masks)).toEqual({ near: 10, far: 40 });
    drawLabelDepths(context, frame, masks);
    expect(layers).toHaveLength(1);
    expect(layers[0].slice(0, 4)).toEqual(layers[0].slice(4, 8));
    expect([layers[0][3], layers[0][7], layers[0][11]]).toEqual([190, 190, 190]);
    expect(layers[0].slice(0, 4)).not.toEqual(layers[0].slice(8, 12));
    expect(Array.from(layers[0].slice(12))).toEqual([0, 0, 0, 0]);
    expect(context.drawImage).toHaveBeenCalledTimes(1);
    expect(context.drawImage).toHaveBeenCalledWith(expect.any(HTMLCanvasElement), 0, 0.4375, 2, 1.125, 0, 0, 1280, 720);
    expect(context.fillText).toHaveBeenNthCalledWith(1, 'track #1 · 10.0 m', 4, expect.any(Number));
    expect(context.fillText).toHaveBeenNthCalledWith(2, 'track #2 · 30.0 m', 644, expect.any(Number));

    const missing = mask([1, 1, 1, 1], [0, NaN, Infinity, 201]);
    expect(labelDepthRange([missing])).toEqual({ near: null, far: null });
    drawLabelDepths(context, frame, [missing]);
    expect(Array.from(layers[1])).toEqual(new Array(16).fill(0));
    expect(context.fillText).toHaveBeenLastCalledWith('track #1 · No depth', 4, expect.any(Number));
    expect(labelDepthRange([])).toEqual({ near: null, far: null });
});
