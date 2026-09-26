import { createSegmentationLayers } from './segmentation-layers';
import type { SegmentResult } from './track-vision-types';

it.each(['car', 'car pack', ' RACE\tCAR ', ' Car  PACK '])
('preserves track beneath %s in the shared working-map and preview layers', (label) => {
    const track = { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([1, 0, 1, 0]) };
    const car = { ...track, classId: 1, mask: new Uint8Array([1, 1, 0, 0]) };
    const segment: SegmentResult & { classNames: string[] } = {
        task: 'segment', width: 2, height: 2, classNames: [' ROAD ', label], instances: [car, track],
    };
    for (const minimumConfidence of [0, 0.65]) {
        for (const instances of [[car, track], [track, car]]) {
            segment.instances = instances;
            const layers = createSegmentationLayers(segment, minimumConfidence)!;
            expect(layers.trackMask).toEqual(new Uint8Array([1, 0, 1, 0]));
            expect(layers.trafficMask).toEqual(new Uint8Array([1, 1, 0, 0]));
            expect(layers.excludedMask).toEqual(new Uint8Array(4));
            expect(layers.instances.map(({ classId }) => classId)).toEqual([0, 1]);
            expect(layers.hasCarLabels).toBe(true);
            expect(segment.instances).toBe(instances);
            expect(track.mask).toEqual(new Uint8Array([1, 0, 1, 0]));
            expect(car.mask).toEqual(new Uint8Array([1, 1, 0, 0]));
        }
    }
});

it('keeps excluded surfaces separate from both overlapping track and traffic', () => {
    const instance = { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([1]) };
    const layers = createSegmentationLayers({
        task: 'segment', width: 1, height: 1, classNames: ['track', 'car pack', 'grass'],
        instances: [instance, { ...instance, classId: 1 }, { ...instance, classId: 2 }],
    })!;
    expect(layers.trackMask).toEqual(new Uint8Array([1]));
    expect(layers.trafficMask).toEqual(new Uint8Array([1]));
    expect(layers.excludedMask).toEqual(new Uint8Array([1]));
});
