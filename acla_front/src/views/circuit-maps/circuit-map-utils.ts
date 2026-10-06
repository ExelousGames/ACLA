import type { Vec3 } from 'views/session-shared/visualization/charts/mapTelemetry';
import { ACCMemoeryTracks, ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { StandardTelemetrySample } from 'views/live-session/live-session-types';
import {
    CIRCUIT_MAP_CAPTURE_MODES,
    CircuitMapAlignedRow,
    CircuitMapBinSample,
    CircuitMapCaptureMode,
    CircuitMapGame,
    CircuitMapSamplesByMode
} from './circuit-map-types';

export const CIRCUIT_MAP_BIN_RESOLUTION = 1000;

export const getCircuitMapTrackKey = (row: Pick<StandardTelemetrySample, 'Static_track'>): string | null => (
    typeof row.Static_track === 'string' && row.Static_track.trim() ? row.Static_track : null
);

export const getCircuitMapName = (trackKey: string | null, game: CircuitMapGame): string => (
    trackKey ? (game === 'acc' ? ACCMemoeryTracks.get(trackKey) || trackKey : trackKey) : ''
);

export const getCircuitMapBin = (
    normalizedPosition: number,
    resolution = CIRCUIT_MAP_BIN_RESOLUTION
): number | null => {
    if (!Number.isFinite(normalizedPosition) || normalizedPosition < 0 || normalizedPosition > 1) {
        return null;
    }

    const maxBin = Math.max(0, resolution - 1);
    return Math.min(maxBin, Math.floor(normalizedPosition * resolution));
};

// Live readers and file converters supply the same standard player fields.
// Never substitute another car when the player's coordinates are unavailable.
export const extractCircuitMapCaptureSample = (
    row: StandardTelemetrySample,
    resolution = CIRCUIT_MAP_BIN_RESOLUTION
): { bin: number; normalizedPosition: number; position: Vec3 } | null => {
    const normalizedPosition = row.Graphics_normalized_car_position;
    const playerId = row.Graphics_player_car_id;
    if (row.Graphics_status !== ACC_STATUS.ACC_LIVE
        || typeof normalizedPosition !== 'number' || !Number.isFinite(normalizedPosition)
        || typeof playerId !== 'number' || !Number.isSafeInteger(playerId) || playerId < 0) {
        return null;
    }

    const bin = getCircuitMapBin(normalizedPosition, resolution);
    if (bin === null) {
        return null;
    }

    const slot = row.Graphics_car_id?.indexOf(playerId) ?? -1;
    const position = slot >= 0 ? row.Graphics_car_coordinates?.[slot] : undefined;
    if (!position || ![position.x, position.y, position.z].every((value) => (
        typeof value === 'number' && Number.isFinite(value)
    ))) {
        return null;
    }

    return {
        bin,
        normalizedPosition,
        position
    };
};

export const upsertCircuitMapSample = (
    samples: CircuitMapBinSample[],
    capture: { bin: number; normalizedPosition: number; position: Vec3 },
    updatedAt = new Date().toISOString()
): CircuitMapBinSample[] => {
    const index = samples.findIndex((sample) => sample.bin === capture.bin);

    if (index >= 0) {
        const existing = samples[index];
        if (existing.locked) {
            return samples;
        }

        const nextCount = existing.sample_count + 1;
        const nextSample: CircuitMapBinSample = {
            ...existing,
            normalized_position: capture.normalizedPosition,
            x: (existing.x * existing.sample_count + capture.position.x) / nextCount,
            y: (existing.y * existing.sample_count + capture.position.y) / nextCount,
            z: (existing.z * existing.sample_count + capture.position.z) / nextCount,
            sample_count: nextCount,
            updated_at: updatedAt
        };

        return [
            ...samples.slice(0, index),
            nextSample,
            ...samples.slice(index + 1)
        ];
    }

    const nextSample: CircuitMapBinSample = {
        bin: capture.bin,
        normalized_position: capture.normalizedPosition,
        x: capture.position.x,
        y: capture.position.y,
        z: capture.position.z,
        sample_count: 1,
        updated_at: updatedAt
    };

    return [...samples, nextSample].sort((a, b) => a.bin - b.bin);
};

export const upsertCaptureModeSample = (
    samplesByMode: CircuitMapSamplesByMode,
    mode: CircuitMapCaptureMode,
    capture: { bin: number; normalizedPosition: number; position: Vec3 },
    updatedAt?: string
): CircuitMapSamplesByMode => ({
    ...samplesByMode,
    [mode]: upsertCircuitMapSample(samplesByMode[mode] || [], capture, updatedAt)
});

export const mergeCircuitMapSamples = (
    samples: CircuitMapBinSample[],
    rows: StandardTelemetrySample[],
    updatedAt = new Date().toISOString()
): { samples: CircuitMapBinSample[]; capturedRows: number } => {
    const bins = new Map(samples.map((sample) => [sample.bin, sample]));
    let capturedRows = 0;
    rows.forEach((row) => {
        const capture = extractCircuitMapCaptureSample(row);
        if (!capture) return;
        const existing = bins.get(capture.bin);
        bins.set(capture.bin, upsertCircuitMapSample(existing ? [existing] : [], capture, updatedAt)[0]);
        capturedRows += 1;
    });
    return { samples: Array.from(bins.values()).sort((a, b) => a.bin - b.bin), capturedRows };
};

export const alignCircuitMapSamples = (
    samplesByMode: CircuitMapSamplesByMode,
    resolution = CIRCUIT_MAP_BIN_RESOLUTION
): CircuitMapAlignedRow[] => {
    const rows = new Map<number, CircuitMapAlignedRow>();

    CIRCUIT_MAP_CAPTURE_MODES.forEach(({ value: mode }) => {
        (samplesByMode[mode] || []).forEach((sample) => {
            const existing = rows.get(sample.bin) || {
                bin: sample.bin,
                normalized_position: sample.bin / resolution
            };

            rows.set(sample.bin, {
                ...existing,
                normalized_position: sample.normalized_position,
                [mode]: sample
            });
        });
    });

    return Array.from(rows.values()).sort((a, b) => a.bin - b.bin);
};

export const getCircuitMapDrawSegments = (
    samples: CircuitMapBinSample[],
    mode: CircuitMapCaptureMode,
    resolution = CIRCUIT_MAP_BIN_RESOLUTION
): CircuitMapBinSample[][] => {
    const sortedSamples = [...samples].sort((a, b) => a.bin - b.bin);

    if (mode !== 'pit_lane' || sortedSamples.length < 2) {
        return sortedSamples.length > 0 ? [sortedSamples] : [];
    }

    const wrapEdgeWindow = resolution * 0.2;
    const crossesLapStart = sortedSamples[0].bin <= wrapEdgeWindow
        && sortedSamples[sortedSamples.length - 1].bin >= resolution - wrapEdgeWindow;

    if (!crossesLapStart) {
        return [sortedSamples];
    }

    const maxPitLaneGap = resolution * 0.1;
    let largestGap = 0;
    let splitIndex = -1;

    for (let index = 1; index < sortedSamples.length; index += 1) {
        const previous = sortedSamples[index - 1];
        const sample = sortedSamples[index];
        const gap = sample.bin - previous.bin;

        if (gap > largestGap) {
            largestGap = gap;
            splitIndex = index;
        }
    }

    if (splitIndex < 0 || largestGap <= maxPitLaneGap) {
        return [sortedSamples];
    }

    return [
        sortedSamples.slice(0, splitIndex),
        sortedSamples.slice(splitIndex)
    ].filter((segment) => segment.length > 0);
};

export const countCircuitMapSamples = (samplesByMode: CircuitMapSamplesByMode): number => (
    Object.values(samplesByMode).reduce((sum, samples) => sum + (samples?.length || 0), 0)
);

export const cloneSamplesByMode = (samplesByMode: CircuitMapSamplesByMode): CircuitMapSamplesByMode => ({
    left_boundary: [...(samplesByMode.left_boundary || [])],
    middle_line: [...(samplesByMode.middle_line || [])],
    right_boundary: [...(samplesByMode.right_boundary || [])],
    pit_lane: [...(samplesByMode.pit_lane || [])]
});
