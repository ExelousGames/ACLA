import { parseTelemetryFrame, Vec3 } from 'views/lap-analysis/visualization/charts/mapTelemetry';
import {
    CIRCUIT_MAP_CAPTURE_MODES,
    CircuitMapAlignedRow,
    CircuitMapBinSample,
    CircuitMapCaptureMode,
    CircuitMapSamplesByMode
} from './circuit-map-types';

export const CIRCUIT_MAP_BIN_RESOLUTION = 1000;

const toFiniteNumber = (value: unknown): number | null => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
};

export const getCircuitMapBin = (
    normalizedPosition: number,
    resolution = CIRCUIT_MAP_BIN_RESOLUTION
): number | null => {
    if (!Number.isFinite(normalizedPosition) || normalizedPosition < 0) {
        return null;
    }

    const maxBin = Math.max(0, resolution - 1);
    return Math.min(maxBin, Math.floor(normalizedPosition * resolution));
};

export const extractAccCaptureSample = (
    row: Record<string, any>,
    sourceIndex = 0,
    resolution = CIRCUIT_MAP_BIN_RESOLUTION
): { bin: number; normalizedPosition: number; position: Vec3 } | null => {
    const normalizedPosition = toFiniteNumber(row.Graphics_normalized_car_position);
    if (normalizedPosition === null) {
        return null;
    }

    const bin = getCircuitMapBin(normalizedPosition, resolution);
    if (bin === null) {
        return null;
    }

    const frame = parseTelemetryFrame(row, sourceIndex);
    if (!frame) {
        return null;
    }

    const playerKey = frame.playerKey || 'slot:0';
    const playerCar = frame.cars.find((car) => car.key === playerKey) || frame.cars[0];
    if (!playerCar) {
        return null;
    }

    return {
        bin,
        normalizedPosition: Math.min(1, Math.max(0, normalizedPosition)),
        position: playerCar.position
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

export const countCircuitMapSamples = (samplesByMode: CircuitMapSamplesByMode): number => (
    Object.values(samplesByMode).reduce((sum, samples) => sum + (samples?.length || 0), 0)
);

export const cloneSamplesByMode = (samplesByMode: CircuitMapSamplesByMode): CircuitMapSamplesByMode => ({
    left_boundary: [...(samplesByMode.left_boundary || [])],
    right_boundary: [...(samplesByMode.right_boundary || [])],
    pit_lane: [...(samplesByMode.pit_lane || [])]
});
