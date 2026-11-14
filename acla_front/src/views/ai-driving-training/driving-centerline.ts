import type { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getLiveMapMiddleLine, getLiveMapPosition } from 'views/live-session/live-map-data';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import { DEFAULT_DRIVING_CONFIG } from './driving-model-config';

const middleLines = new WeakMap<CircuitMapDto, CircuitMapBinSample[]>();

/** Map-resolution steps -50..-1, then +1..+50, each stored as relative XYZ.
 * The origin is the Live Map centerline at capture-time lap progress, not the car
 * or reference racing line. Preserve map axes and scale without clipping bends. */
export function drivingCenterlineInput(map: CircuitMapDto | null, game: string, track: string,
    position: number, config = DEFAULT_DRIVING_CONFIG): number[] | null {
    const trackKey = (value: string) => game === 'acc' ? getAccTelemetryTrackKey(value) || value.trim() : value.trim();
    if (!map || map.game !== game || trackKey(map.source_track_key || map.circuit_name) !== trackKey(track)
        || !Number.isSafeInteger(map.resolution) || map.resolution < 2) return null;
    let line = middleLines.get(map);
    if (!line) {
        line = getLiveMapMiddleLine(map);
        middleLines.set(map, line);
    }
    const origin = getLiveMapPosition(line, position);
    if (!origin) return null;
    const { centerlineStepsPerSide: steps, centerlineScale: scale } = config.observation;
    const features: number[] = [];
    for (let step = -steps; step <= steps; step++) {
        if (step === 0) continue;
        const progress = position + step / map.resolution;
        const point = getLiveMapPosition(line, ((progress % 1) + 1) % 1)!;
        features.push((point.x - origin.x) / scale, (point.y - origin.y) / scale, (point.z - origin.z) / scale);
    }
    return features.every(Number.isFinite) ? features : null;
}
