import {
    DriverExpertComparisonData,
    DriverExpertComparisonSample,
    DriverExpertTrajectoryPoint,
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
    trackPosition?: number;
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

const normalizedInput = (value: unknown): number | undefined => {
    const parsed = finiteNumber(value);
    return parsed === undefined ? undefined : clamp(parsed, 0, 1);
};

const trajectoryPoint = (xValue: unknown, zValue: unknown): DriverExpertTrajectoryPoint | undefined => {
    const x = finiteNumber(xValue);
    const z = finiteNumber(zValue);
    return x === undefined || z === undefined ? undefined : { x, z };
};

const getDriverTrajectory = (
    row: Record<string, any>,
    sourceIndex: number,
): DriverExpertTrajectoryPoint | undefined => {
    const frame = parseTelemetryFrame(row, sourceIndex);
    if (!frame?.playerKey) return undefined;
    const player = frame.cars.find((car) => car.key === frame.playerKey);
    return player ? { x: player.position.x, z: player.position.z } : undefined;
};

const getTrackPosition = (
    driver: Record<string, any>,
    expert: Record<string, unknown>,
): number | undefined => normalizedInput(
    driver.Graphics_normalized_car_position
    ?? driver.normalized_position
    ?? driver.normalizedPosition
    ?? expert.Graphics_normalized_car_position,
);

const computeProgress = (rows: readonly JoinedComparisonRow[]): number[] => {
    if (rows.length <= 1) return rows.map(() => 0);
    const positions = rows.map(({ trackPosition }) => trackPosition);
    if (positions.every((position): position is number => position !== undefined)) {
        let wrapOffset = 0;
        let previous = positions[0];
        const unwrapped = positions.map((position, index) => {
            if (index > 0 && previous - position > 0.5) wrapOffset += 1;
            previous = position;
            return position + wrapOffset;
        });
        const distance = unwrapped[unwrapped.length - 1] - unwrapped[0];
        if (distance > 0) {
            return unwrapped.map((position) => clamp(
                ((position - unwrapped[0]) / distance) * 100,
                0,
                100,
            ));
        }
    }

    return rows.map((_, index) => (index / (rows.length - 1)) * 100);
};

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
            trackPosition: getTrackPosition(driverEntry.row, value),
        }];
    }).sort((left, right) => left.rawIndex - right.rawIndex);

    const progress = computeProgress(joinedRows);
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
            row.expert.expert_optimal_player_pos_z,
        );

        return {
            progress: progress[index],
            ...(row.trackPosition !== undefined ? { trackPosition: row.trackPosition } : {}),
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

    return { samples };
};
