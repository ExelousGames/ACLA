import type { SegmentResult } from './track-vision-types';
import { isTrackLabel } from './vision-labels';

const CAR_LABELS = ['car', 'cars', 'vehicle', 'vehicles', 'race car', 'racecar', 'opponent', 'opponent car'];
type MaskKind = 'track' | 'car' | 'car pack' | 'excluded';
const maskKind = (label: string | undefined): MaskKind => isTrackLabel(label) ? 'track'
    : label === 'car pack' ? 'car pack' : CAR_LABELS.includes(label ?? '') ? 'car' : 'excluded';

/** Shared by the capture overlay and working map: traffic never erases track coverage. */
export function createSegmentationLayers(segment: SegmentResult & { classNames: string[] }, minimumConfidence = 0) {
    if (!Number.isInteger(segment.width) || !Number.isInteger(segment.height) || segment.width <= 0 || segment.height <= 0) return null;
    const size = segment.width * segment.height;
    const kinds = segment.classNames.map((label) => maskKind(label.trim().toLowerCase().replace(/\s+/g, ' ')));
    const instances = segment.instances.filter((item) => item.confidence >= minimumConfidence && item.confidence <= 1)
        .map((item) => ({ ...item, kind: kinds[item.classId] ?? 'excluded' }))
        .sort((a, b) => Number(b.kind === 'track') - Number(a.kind === 'track'));
    const trackMask = new Uint8Array(size), trafficMask = new Uint8Array(size), excludedMask = new Uint8Array(size);
    for (const instance of instances) {
        if (instance.mask.length !== size) continue;
        const mask = instance.kind === 'track' ? trackMask : instance.kind === 'excluded' ? excludedMask : trafficMask;
        instance.mask.forEach((active, i) => { if (active) mask[i] = 1; });
    }
    return { instances, trackMask, trafficMask, excludedMask, hasCarLabels: kinds.some((kind) => kind === 'car' || kind === 'car pack') };
}
