import { filteredFrame, filteredMasks, labelDepths } from './pipeline-visuals';
import type { AmodalMask } from './amodal-masks';
import type { TrackVisionFrame } from './track-vision-types';

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
    expect(labelDepths(masks)).toEqual([{ classId: 0, label: 'track', instances: 1, estimated: 0,
        samples: 13, near: 10, far: 40, median: 20 }, { classId: 1, label: 'car interior', instances: 1, estimated: 0,
        samples: 4, near: 10, far: 10, median: 10 }]);
    const filtered = filteredFrame(frame, masks).detections.segment;
    expect(filtered?.task === 'segment' && filtered.instances.find(({ classId }) => classId === 0)!.box).toEqual([0, 0, 1, 1]);
    expect(track.mask).toEqual(new Uint8Array(16).fill(1));
    expect(frame.detections.segment?.task === 'segment' && frame.detections.segment.instances).toHaveLength(3);
    expect(frame.detections.depth?.task === 'depth' && frame.detections.depth.values).toBe(depth);
});

it('groups each label across instances and excludes hidden predictions from measured distances', () => {
    const mask = (values: number[], hidden: number[], classId = 0): AmodalMask => ({
        classId, label: classId ? 'fence' : 'track', confidence: 0.9, box: [0, 0, 1, 1], bounds: [0, 0, 2, 2],
        mask: new Uint8Array(4).fill(1), depths: Float32Array.from(values), hiddenMask: Uint8Array.from(hidden),
        visibleMask: Uint8Array.from(hidden, (value) => 1 - value),
    });
    const rows = labelDepths([mask([10, 20, 100, 0], [0, 0, 1, 0]), mask([30, 40, 0, 0], [0, 0, 0, 0]),
        mask([0, NaN, Infinity, 201], [0, 0, 0, 0], 1)]);
    expect(rows[0]).toEqual({ classId: 0, label: 'track', instances: 2, estimated: 1, samples: 4, near: 10, far: 40, median: 25 });
    expect(rows[1]).toEqual({ classId: 1, label: 'fence', instances: 1, estimated: 0, samples: 0, near: null, far: null, median: null });
    expect(filteredMasks(null)).toEqual([]);
});
