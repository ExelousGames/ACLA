import { ACCMemoeryTracks } from 'data/live-analysis/live-map-data';
import { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getCircuitMapDrawSegments } from 'views/circuit-maps/circuit-map-utils';
import { Vec3 } from './mapTelemetry';

export type CircuitTrackLayout = {
    surface: Vec3[][];
    leftBoundary: Vec3[][];
    rightBoundary: Vec3[][];
    pitLane: Vec3[][];
    centerLine: Vec3[][];
    allPoints: Vec3[];
};

export const EMPTY_CIRCUIT_TRACK_LAYOUT: CircuitTrackLayout = {
    surface: [],
    leftBoundary: [],
    rightBoundary: [],
    pitLane: [],
    centerLine: [],
    allPoints: []
};

const sampleToVec3 = (sample: CircuitMapBinSample): Vec3 | null => {
    const x = Number(sample.x);
    const y = Number(sample.y);
    const z = Number(sample.z);

    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        return null;
    }

    return { x, y, z };
};

const samplesToVec3 = (samples: CircuitMapBinSample[]): Vec3[] => (
    samples
        .map(sampleToVec3)
        .filter((point): point is Vec3 => point !== null)
);

const getCircuitMapSegments = (
    samples: CircuitMapBinSample[] | undefined,
    mode: 'left_boundary' | 'right_boundary' | 'pit_lane',
    resolution: number
): Vec3[][] => (
    getCircuitMapDrawSegments(samples || [], mode, resolution)
        .map(samplesToVec3)
        .filter((segment) => segment.length > 1)
);

export const getAccTelemetryTrackKey = (...values: unknown[]): string | null => {
    for (const value of values) {
        if (typeof value !== 'string') continue;

        const trimmed = value.trim();
        if (!trimmed) continue;

        if (ACCMemoeryTracks.has(trimmed)) {
            return trimmed;
        }

        const normalizedValue = trimmed.toLocaleLowerCase();
        let matchedTrackKey: string | null = null;
        ACCMemoeryTracks.forEach((trackName, trackKey) => {
            if (trackName.toLocaleLowerCase() === normalizedValue) {
                matchedTrackKey = trackKey;
            }
        });
        if (matchedTrackKey) return matchedTrackKey;
    }

    return null;
};

export const buildCircuitTrackLayout = (map: CircuitMapDto | null): CircuitTrackLayout => {
    if (!map) return EMPTY_CIRCUIT_TRACK_LAYOUT;

    const resolution = Number.isFinite(map.resolution) && map.resolution > 0 ? map.resolution : 1000;
    const leftBoundary = getCircuitMapSegments(map.samples.left_boundary, 'left_boundary', resolution);
    const rightBoundary = getCircuitMapSegments(map.samples.right_boundary, 'right_boundary', resolution);
    const pitLane = getCircuitMapSegments(map.samples.pit_lane, 'pit_lane', resolution);
    const leftSamples = [...(map.samples.left_boundary || [])].sort((a, b) => a.bin - b.bin);
    const rightSamples = [...(map.samples.right_boundary || [])].sort((a, b) => a.bin - b.bin);
    const rightByBin = new Map(rightSamples.map((sample) => [sample.bin, sample]));
    const centerLine = samplesToVec3(
        leftSamples
            .map((leftSample): CircuitMapBinSample | null => {
                const rightSample = rightByBin.get(leftSample.bin);
                if (!rightSample) return null;

                return {
                    ...leftSample,
                    x: (leftSample.x + rightSample.x) / 2,
                    y: (leftSample.y + rightSample.y) / 2,
                    z: (leftSample.z + rightSample.z) / 2
                };
            })
            .filter((sample): sample is CircuitMapBinSample => sample !== null)
    );
    const surface = samplesToVec3(leftSamples);
    const reversedRight = samplesToVec3(rightSamples).reverse();
    const surfaceSegments = surface.length > 1 && reversedRight.length > 1
        ? [[...surface, ...reversedRight]]
        : [];
    const allPoints = [
        ...surfaceSegments.flat(),
        ...leftBoundary.flat(),
        ...rightBoundary.flat(),
        ...pitLane.flat(),
        ...centerLine
    ];

    return {
        surface: surfaceSegments,
        leftBoundary,
        rightBoundary,
        pitLane,
        centerLine: centerLine.length > 1 ? [centerLine] : [],
        allPoints
    };
};

export const buildSessionPointsTrackLayout = (points?: { position_x: number; position_y: number }[]): CircuitTrackLayout => {
    if (!points || points.length === 0) return EMPTY_CIRCUIT_TRACK_LAYOUT;

    const centerLine = points.map((point) => ({
        x: point.position_x,
        y: 0,
        z: point.position_y
    }));

    return {
        ...EMPTY_CIRCUIT_TRACK_LAYOUT,
        centerLine: centerLine.length > 1 ? [centerLine] : [],
        allPoints: centerLine
    };
};
