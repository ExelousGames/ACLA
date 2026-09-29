import type { SegmentResult } from './track-vision-types';
import { isTrackLabel } from './vision-labels';

const CAR_LABELS = ['car', 'cars', 'vehicle', 'vehicles', 'race car', 'racecar', 'opponent', 'opponent car'];
const ROADSIDE_LABELS = ['curb', 'grass', 'sand', 'outfield asphalt road'];
type MaskKind = 'track' | 'car' | 'car pack' | 'excluded';
const maskKind = (label: string | undefined): MaskKind => isTrackLabel(label) ? 'track'
    : label === 'car pack' ? 'car pack' : CAR_LABELS.includes(label ?? '') ? 'car' : 'excluded';

/** Shared by the capture overlay and working map: traffic never erases track coverage. */
export function createSegmentationLayers(segment: SegmentResult & { classNames: string[] }, minimumConfidence = 0) {
    if (!Number.isInteger(segment.width) || !Number.isInteger(segment.height) || segment.width <= 0 || segment.height <= 0) return null;
    const size = segment.width * segment.height;
    const labels = segment.classNames.map((label) => label.trim().toLowerCase().replace(/\s+/g, ' '));
    const kinds = labels.map(maskKind);
    const instances = segment.instances.filter((item) => item.confidence >= minimumConfidence && item.confidence <= 1)
        .map((item) => ({ ...item, kind: kinds[item.classId] ?? 'excluded' }))
        .sort((a, b) => Number(b.kind === 'track') - Number(a.kind === 'track'));
    const trackMask = new Uint8Array(size), trafficMask = new Uint8Array(size), excludedMask = new Uint8Array(size);
    const roadsideMask = new Uint8Array(size), obstacleMask = new Uint8Array(size);
    const carInteriorMask = new Uint8Array(size);
    for (const instance of instances) {
        if (instance.mask.length !== size) continue;
        const mask = instance.kind === 'track' ? trackMask : instance.kind === 'excluded' ? excludedMask : trafficMask;
        instance.mask.forEach((active, i) => { if (active) mask[i] = 1; });
        if (instance.kind === 'excluded') {
            const surface = ROADSIDE_LABELS.includes(labels[instance.classId]) ? roadsideMask : obstacleMask;
            instance.mask.forEach((active, i) => { if (active) surface[i] = 1; });
            if (labels[instance.classId] === 'car interior') {
                instance.mask.forEach((active, i) => { if (active) carInteriorMask[i] = 1; });
            }
        }
    }
    return { instances, trackMask, trafficMask, excludedMask, roadsideMask, obstacleMask, carInteriorMask,
        hasCarLabels: kinds.some((kind) => kind === 'car' || kind === 'car pack') };
}
