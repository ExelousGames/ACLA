import type { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapMiddleLine, getLiveMapPosition } from 'views/live-session/live-map-data';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import type { Vec3 } from 'views/session-shared/visualization/charts/mapTelemetry';
import { DEFAULT_DRIVING_CONFIG } from './driving-model-config';

export const DRIVING_CENTERLINE_LOOKAHEAD_SECONDS = 6;

interface Centerline {
    line: CircuitMapBinSample[];
    distances: number[];
    length: number;
    lineByDistance: (Vec3 & { normalized_position: number })[];
}

const middleLines = new WeakMap<CircuitMapDto, Centerline | null>();
const distanceBetween = (a: Vec3, b: Vec3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

function prepareCenterline(map: CircuitMapDto): Centerline | null {
    const line = getLiveMapMiddleLine(map);
    if (line.length < 2) return null;
    let distance = 0;
    const distances = line.map((point, i) => {
        if (i > 0) distance += distanceBetween(line[i - 1], point);
        return distance;
    });
    const length = distance + distanceBetween(line[line.length - 1], line[0]);
    if (!Number.isFinite(length) || length <= 0) return null;
    const lineByDistance = line.map(({ x, y, z }, i) => ({ x, y, z, normalized_position: distances[i] / length }))
        .filter((point, i, points) => i === 0 || point.normalized_position > points[i - 1].normalized_position);
    return { line, distances, length, lineByDistance };
}

function sampleAtOrBefore(line: CircuitMapBinSample[], position: number): number {
    let low = 0, high = line.length;
    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (line[middle].normalized_position <= position) low = middle + 1;
        else high = middle;
    }
    return low === 0 ? line.length - 1 : low - 1;
}

/** Encode map-resolution samples behind and six seconds of travel ahead as XYZ offsets
 * from the centerline at capture-time lap progress. */
export function drivingCenterlineInput(map: CircuitMapDto | null, game: string, track: string,
    position: number, speedKmh: number, config = DEFAULT_DRIVING_CONFIG): number[] | null {
    const trackKey = (value: string) => game === 'acc' ? getAccTelemetryTrackKey(value) || value.trim() : value.trim();
    const lookaheadDistance = speedKmh / 3.6 * DRIVING_CENTERLINE_LOOKAHEAD_SECONDS;
    if (!map || map.game !== game || trackKey(map.source_track_key || map.circuit_name) !== trackKey(track)
        || !Number.isSafeInteger(map.resolution) || map.resolution < 2
        || !Number.isFinite(lookaheadDistance) || lookaheadDistance < 0) return null;
    let centerline = middleLines.get(map);
    if (centerline === undefined) {
        centerline = prepareCenterline(map);
        middleLines.set(map, centerline);
    }
    if (!centerline) return null;
    const { line, distances, length, lineByDistance } = centerline;
    const origin = getLiveMapPosition(line, position);
    if (!origin) return null;
    const previous = sampleAtOrBefore(line, position);
    const originProgressByDistance = (distances[previous] + distanceBetween(line[previous], origin)) / length;
    const { centerlineStepsPerSide: steps, centerlineScale: scale } = config.observation;
    const features: number[] = [];
    const append = (point: Vec3) => {
        features.push((point.x - origin.x) / scale, (point.y - origin.y) / scale, (point.z - origin.z) / scale);
    };
    for (let step = -steps; step < 0; step++) {
        const progress = position + step / map.resolution;
        append(getLiveMapPosition(line, ((progress % 1) + 1) % 1)!);
    }
    for (let step = 1; step <= steps; step++) {
        const distance = lookaheadDistance * (step / steps);
        const progress = (originProgressByDistance + (distance % length) / length) % 1;
        append(lookaheadDistance === 0 ? origin : getLiveMapPosition(lineByDistance, progress)!);
    }
    return features.every(Number.isFinite) ? features : null;
}
