import {
    DriverExpertComparisonData,
    DriverExpertComparisonSample,
    DriverExpertTrajectoryPoint,
    normalizeDriverExpertComparisonData,
} from 'components/driver-expert-comparison';
import { parseTelemetryFrame } from './mapTelemetry';

export interface AnalysisResultsComparisonAdapterInput {
    baselineRecords: readonly Record<string, any>[];
    expertReferenceData: readonly unknown[];
    startIndex: number;
    endIndex: number;
}

interface JoinedComparisonRow {
    rawIndex: number;
    driver: Record<string, any>;
    expert: Record<string, unknown>;
    driverArrayIndex: number;
    driverTrackPosition?: number;
    expertTrackPosition?: number;
}

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const finiteNumber = (value: unknown): number | undefined => {
    if (value === null || value === undefined || value === '') return undefined;
    const parsed = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
};

const clamp = (value: number, minimum: number, maximum: number): number => (
    Math.min(maximum, Math.max(minimum, value))
);

const finiteIndex = (value: unknown): number | undefined => {
    const parsed = finiteNumber(value);
    return parsed === undefined ? undefined : Math.trunc(parsed);
};

const arrayValue = (value: unknown): unknown[] => {
    if (Array.isArray(value)) return value;
    if (typeof value !== 'string') return [];
    try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
};

const normalizedInput = (value: unknown): number | undefined => {
    const parsed = finiteNumber(value);
    return parsed === undefined ? undefined : clamp(parsed, 0, 1);
};

const normalizedPosition = (value: unknown): number | undefined => {
    const parsed = finiteNumber(value);
    return parsed !== undefined && parsed >= 0 && parsed <= 1 ? parsed : undefined;
};

const trajectoryPoint = (
    xValue: unknown,
    yValue: unknown,
    zValue: unknown,
): DriverExpertTrajectoryPoint | undefined => {
    const x = finiteNumber(xValue);
    const y = finiteNumber(yValue);
    const z = finiteNumber(zValue);
    if (x === undefined || (y === undefined && z === undefined)) return undefined;
    return {
        x,
        ...(y !== undefined ? { y } : {}),
        ...(z !== undefined ? { z } : {}),
    };
};

const getDriverTrajectory = (
    row: Record<string, any>,
    sourceIndex: number,
): DriverExpertTrajectoryPoint | undefined => {
    const frame = parseTelemetryFrame(row, sourceIndex);
    if (!frame?.playerKey) return undefined;
    const player = frame.cars.find((car) => car.key === frame.playerKey);
    if (!player) return undefined;
    const sourcePosition = arrayValue(row.Graphics_car_coordinates)[player.slot];
    if (!isRecord(sourcePosition)) return undefined;
    return trajectoryPoint(sourcePosition.x, sourcePosition.y, sourcePosition.z);
};

const getDriverTrackPosition = (
    driver: Record<string, any>,
): number | undefined => normalizedPosition(
    driver.Graphics_normalized_car_position
    ?? driver.normalized_position
    ?? driver.normalizedPosition,
);

const getExpertTrackPosition = (
    expert: Record<string, unknown>,
): number | undefined => normalizedPosition(
    expert.Graphics_normalized_car_position
    ?? expert.normalized_position
    ?? expert.normalizedPosition,
);

export const adaptAnalysisResultsComparison = ({
    baselineRecords,
    expertReferenceData,
    startIndex,
    endIndex,
}: AnalysisResultsComparisonAdapterInput): DriverExpertComparisonData => {
    const firstIndex = finiteIndex(startIndex);
    const lastIndex = finiteIndex(endIndex);
    if (firstIndex === undefined || lastIndex === undefined || lastIndex < firstIndex) {
        return { samples: [] };
    }

    const driverByRawIndex = new Map<number, {
        row: Record<string, any>;
        arrayIndex: number;
    }>();
    baselineRecords.forEach((row, arrayIndex) => {
        if (!row || typeof row !== 'object') return;
        const rawIndex = finiteIndex(row.raw_index ?? row.rawIndex) ?? arrayIndex;
        if (!driverByRawIndex.has(rawIndex)) {
            driverByRawIndex.set(rawIndex, { row, arrayIndex });
        }
    });

    const joinedRows: JoinedComparisonRow[] = expertReferenceData.flatMap((value) => {
        if (!isRecord(value)) return [];
        const rawIndex = finiteIndex(value.raw_index);
        if (rawIndex === undefined || rawIndex < firstIndex || rawIndex > lastIndex) return [];
        const driverEntry = driverByRawIndex.get(rawIndex);
        if (!driverEntry) return [];
        return [{
            rawIndex,
            driver: driverEntry.row,
            expert: value,
            driverArrayIndex: driverEntry.arrayIndex,
            driverTrackPosition: getDriverTrackPosition(driverEntry.row),
            expertTrackPosition: getExpertTrackPosition(value),
        }];
    }).sort((left, right) => left.rawIndex - right.rawIndex);

    const timing = joinedRows.map((row) => ({
        driverTimeMs: finiteNumber(row.driver.Graphics_current_time),
        expertTimeMs: finiteNumber(row.expert.expert_optimal_time),
    }));
    const invalidTiming = timing.some((sample, index) => (
        sample.driverTimeMs === undefined
        || sample.expertTimeMs === undefined
        || (index > 0 && sample.driverTimeMs <= timing[index - 1].driverTimeMs!)
        || (index > 0 && sample.expertTimeMs <= timing[index - 1].expertTimeMs!)
    ));
    const invalidPositions = joinedRows.some((row) => (
        row.driverTrackPosition === undefined || row.expertTrackPosition === undefined
    ));
    if (invalidTiming || invalidPositions) return { samples: [] };

    const samples: DriverExpertComparisonSample[] = joinedRows.map((row, index) => {
        const driverGas = normalizedInput(row.driver.Physics_gas);
        const expertGas = normalizedInput(row.expert.expert_optimal_throttle);
        const driverBrake = normalizedInput(row.driver.Physics_brake);
        const expertBrake = normalizedInput(row.expert.expert_optimal_brake);
        const driverGear = finiteNumber(row.driver.Physics_gear);
        const expertGear = finiteNumber(row.expert.expert_optimal_gear);
        const driverTrajectory = getDriverTrajectory(row.driver, row.driverArrayIndex);
        const expertTrajectory = trajectoryPoint(
            row.expert.expert_optimal_player_pos_x,
            row.expert.expert_optimal_player_pos_y,
            row.expert.expert_optimal_player_pos_z,
        );

        return {
            driverTimeMs: timing[index].driverTimeMs!,
            expertTimeMs: timing[index].expertTimeMs!,
            driverTrackPosition: row.driverTrackPosition!,
            expertTrackPosition: row.expertTrackPosition!,
            ...(driverTrajectory ? { driverTrajectory } : {}),
            ...(expertTrajectory ? { expertTrajectory } : {}),
            ...(driverGas !== undefined ? { driverGas } : {}),
            ...(expertGas !== undefined ? { expertGas } : {}),
            ...(driverBrake !== undefined ? { driverBrake } : {}),
            ...(expertBrake !== undefined ? { expertBrake } : {}),
            ...(driverGear !== undefined ? { driverGear } : {}),
            ...(expertGear !== undefined ? { expertGear } : {}),
        };
    });

    return normalizeDriverExpertComparisonData({ samples }) ?? { samples: [] };
};
