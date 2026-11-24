import { getCircuitMapCenterlineSegments } from 'views/circuit-maps/centerline-segments';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { DrivingModelConfig } from './driving-model-config';
import type { DrivingObservation } from './driving-observation';
import { isConsecutiveRecordedObservation } from './recorded-driving-replay';
import { recordedTrackKey } from './recorded-driving-reference';

type SavedRange = { start: number; end: number; length: number; key: string };
const savedRanges = new WeakMap<CircuitMapDto, SavedRange[]>();

function rangeAt(map: CircuitMapDto | null, position: number): string {
    if (!map) return '';
    let ranges = savedRanges.get(map);
    if (!ranges) {
        ranges = getCircuitMapCenterlineSegments(map).flatMap(({ start_position: rawStart, end_position: rawEnd }) => {
            if (![rawStart, rawEnd].every((point) => Number.isFinite(point) && point >= 0 && point <= 1)
                || rawStart === rawEnd) return [];
            const start = rawStart === 1 ? 0 : rawStart;
            const end = rawEnd === 1 && start > 0 ? 0 : rawEnd;
            const length = end < start ? 1 - start + end : end - start;
            return length > 0 ? [{ start, end, length, key: `${start}:${end}` }] : [];
        }).sort((a, b) => a.length - b.length || a.start - b.start || a.end - b.end);
        savedRanges.set(map, ranges);
    }
    // The shortest matching range owns an overlap. A shared endpoint belongs
    // to the range starting there, including at the start/finish line.
    const point = position === 1 ? 0 : position;
    for (const { start, end, key } of ranges) {
        if (start < end ? start <= point && point < end : start <= point || point < end) return key;
    }
    return '';
}

export function recordedSampleGroup(observation: DrivingObservation, clipLabel: string | undefined,
    map: CircuitMapDto | null): string {
    const { game, map: track } = observation.context;
    return JSON.stringify([clipLabel?.trim().toLowerCase() ?? '', game, recordedTrackKey(game, track),
        rangeAt(map, observation.normalizedPosition)]);
}

interface SampleBucket {
    count: number;
    quota: number;
    seen: number;
}

/** Count learnable origins before applying updates, then keep an evenly spaced
 * subset in each observed combination. Cameras and replay order stay streaming. */
export class RecordedDrivingBalance {
    private buckets = new Map<string, SampleBucket>();
    private previous?: { observation: DrivingObservation; group: string };

    constructor(private config: DrivingModelConfig) {}

    count(observation: DrivingObservation | undefined, group: string | undefined, startsRange?: boolean) {
        const previous = this.previous;
        if (observation && group !== undefined) {
            if (!startsRange && previous && isConsecutiveRecordedObservation(previous.observation, observation, this.config)) {
                const bucket = this.buckets.get(previous.group);
                if (bucket) bucket.count++;
                else this.buckets.set(previous.group, { count: 1, quota: 0, seen: 0 });
            }
            this.previous = { observation, group };
        } else this.previous = undefined;
    }

    prepare() {
        this.previous = undefined;
        let quota = Infinity;
        this.buckets.forEach((bucket) => { quota = Math.min(quota, bucket.count); });
        this.buckets.forEach((bucket) => { bucket.quota = quota; });
    }

    select(group: string) {
        const bucket = this.buckets.get(group);
        if (!bucket || bucket.seen >= bucket.count) return false;
        const { quota, count } = bucket;
        const before = Math.round(bucket.seen * quota / count);
        bucket.seen++;
        return Math.round(bucket.seen * quota / count) > before;
    }
}
