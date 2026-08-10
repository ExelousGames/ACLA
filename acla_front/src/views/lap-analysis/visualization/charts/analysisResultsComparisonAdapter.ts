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
}

interface DriverSourcePoint {
    timeMs: number;
    normalizedPosition: number;
    unwrappedPosition: number;
    trajectory?: DriverExpertTrajectoryPoint;
    gas?: number;
    brake?: number;
    gear?: number;
}

interface ExpertSourcePoint {
    row: Record<string, unknown>;
    timeMs: number;
    normalizedPosition: number;
    unwrappedPosition: number;
}

interface InterpolatedDriverPoint {
    point: Omit<DriverSourcePoint, 'unwrappedPosition'>;
    rightIndex: number;
}

const POSITION_EPSILON = 1e-9;
const FINISH_LINE_BACKWARD_JUMP = 0.5;

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
    driver: Record<string, unknown>,
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

const unwrapPositions = (positions: readonly number[]): number[] | undefined => {
    const unwrapped: number[] = [];
    let lapOffset = 0;
    let previousNormalizedPosition: number | undefined;

    for (const position of positions) {
        if (previousNormalizedPosition !== undefined && position < previousNormalizedPosition) {
            if (previousNormalizedPosition - position <= FINISH_LINE_BACKWARD_JUMP) {
                return undefined;
            }
            lapOffset += 1;
        }
        unwrapped.push(position + lapOffset);
        previousNormalizedPosition = position;
    }

    return unwrapped;
};

const buildDriverPoints = (
    baselineRecords: readonly Record<string, any>[],
): DriverSourcePoint[] | undefined => {
    if (!baselineRecords.length) return undefined;

    const rows: Array<{
        row: Record<string, any>;
        timeMs: number;
        normalizedPosition: number;
        sourceIndex: number;
    }> = [];
    let previousTimeMs: number | undefined;

    for (let sourceIndex = 0; sourceIndex < baselineRecords.length; sourceIndex += 1) {
        const row = baselineRecords[sourceIndex];
        if (!isRecord(row)) return undefined;
        const timeMs = finiteNumber(row.Graphics_current_time);
        const position = getDriverTrackPosition(row);
        if (
            timeMs === undefined
            || position === undefined
            || (previousTimeMs !== undefined && timeMs <= previousTimeMs)
        ) {
            return undefined;
        }
        rows.push({ row, timeMs, normalizedPosition: position, sourceIndex });
        previousTimeMs = timeMs;
    }

    const unwrappedPositions = unwrapPositions(rows.map((entry) => entry.normalizedPosition));
    if (!unwrappedPositions) return undefined;

    return rows.map(({ row, timeMs, normalizedPosition: position, sourceIndex }, index) => {
        const trajectory = getDriverTrajectory(row, sourceIndex);
        const gas = normalizedInput(row.Physics_gas);
        const brake = normalizedInput(row.Physics_brake);
        const gear = finiteNumber(row.Physics_gear);
        return {
            timeMs,
            normalizedPosition: position,
            unwrappedPosition: unwrappedPositions[index],
            ...(trajectory ? { trajectory } : {}),
            ...(gas !== undefined ? { gas } : {}),
            ...(brake !== undefined ? { brake } : {}),
            ...(gear !== undefined ? { gear } : {}),
        };
    });
};

const buildExpertPoints = (
    expertReferenceData: readonly unknown[],
): ExpertSourcePoint[] | undefined => {
    if (!expertReferenceData.length) return undefined;

    const rows: Array<{
        row: Record<string, unknown>;
        timeMs: number;
        normalizedPosition: number;
    }> = [];
    let previousTimeMs: number | undefined;

    for (const value of expertReferenceData) {
        if (!isRecord(value)) return undefined;
        const timeMs = finiteNumber(value.expert_optimal_time);
        const position = getExpertTrackPosition(value);
        if (
            timeMs === undefined
            || position === undefined
            || (previousTimeMs !== undefined && timeMs <= previousTimeMs)
        ) {
            return undefined;
        }
        rows.push({ row: value, timeMs, normalizedPosition: position });
        previousTimeMs = timeMs;
    }

    const unwrappedPositions = unwrapPositions(rows.map((entry) => entry.normalizedPosition));
    if (!unwrappedPositions) return undefined;

    return rows.map((entry, index) => ({
        ...entry,
        unwrappedPosition: unwrappedPositions[index],
    }));
};

const interpolateValue = (
    previous: number | undefined,
    next: number | undefined,
    ratio: number,
): number | undefined => {
    if (ratio <= POSITION_EPSILON) return previous;
    if (ratio >= 1 - POSITION_EPSILON) return next;
    return previous !== undefined && next !== undefined
        ? previous + ((next - previous) * ratio)
        : undefined;
};

const interpolateTrajectory = (
    previous: DriverExpertTrajectoryPoint | undefined,
    next: DriverExpertTrajectoryPoint | undefined,
    ratio: number,
): DriverExpertTrajectoryPoint | undefined => {
    if (ratio <= POSITION_EPSILON) return previous;
    if (ratio >= 1 - POSITION_EPSILON) return next;
    if (!previous || !next) return undefined;
    return trajectoryPoint(
        interpolateValue(previous.x, next.x, ratio),
        interpolateValue(previous.y, next.y, ratio),
        interpolateValue(previous.z, next.z, ratio),
    );
};

const getPrecedingGear = (
    driver: readonly DriverSourcePoint[],
    rightIndex: number,
): number | undefined => {
    let gearIndex = rightIndex === 0 ? 0 : rightIndex - 1;
    while (gearIndex >= 0) {
        const gear = finiteNumber(driver[gearIndex].gear);
        if (gear !== undefined) return gear;
        gearIndex -= 1;
    }
    return undefined;
};

const interpolateDriverAtPosition = (
    driver: readonly DriverSourcePoint[],
    targetPosition: number,
    minimumRightIndex: number,
): InterpolatedDriverPoint | undefined => {
    let rightIndex = minimumRightIndex;
    while (
        rightIndex < driver.length
        && driver[rightIndex].unwrappedPosition < targetPosition - POSITION_EPSILON
    ) {
        rightIndex += 1;
    }
    if (rightIndex >= driver.length) return undefined;

    if (rightIndex === 0) {
        const exact = driver[0];
        if (Math.abs(exact.unwrappedPosition - targetPosition) > POSITION_EPSILON) {
            return undefined;
        }
        const { unwrappedPosition: _unwrappedPosition, ...point } = exact;
        return { point, rightIndex };
    }

    const previous = driver[rightIndex - 1];
    const next = driver[rightIndex];
    if (
        targetPosition < previous.unwrappedPosition - POSITION_EPSILON
        || targetPosition > next.unwrappedPosition + POSITION_EPSILON
    ) {
        return undefined;
    }

    const positionSpan = next.unwrappedPosition - previous.unwrappedPosition;
    let ratio: number;
    if (Math.abs(next.unwrappedPosition - targetPosition) <= POSITION_EPSILON) {
        ratio = 1;
    } else if (Math.abs(previous.unwrappedPosition - targetPosition) <= POSITION_EPSILON) {
        ratio = 0;
    } else {
        if (positionSpan <= POSITION_EPSILON) return undefined;
        ratio = (targetPosition - previous.unwrappedPosition) / positionSpan;
    }

    const trajectory = interpolateTrajectory(previous.trajectory, next.trajectory, ratio);
    const gas = interpolateValue(previous.gas, next.gas, ratio);
    const brake = interpolateValue(previous.brake, next.brake, ratio);
    const gear = getPrecedingGear(driver, rightIndex);
    return {
        rightIndex,
        point: {
            timeMs: previous.timeMs + ((next.timeMs - previous.timeMs) * ratio),
            normalizedPosition: ((targetPosition % 1) + 1) % 1,
            ...(trajectory ? { trajectory } : {}),
            ...(gas !== undefined ? { gas } : {}),
            ...(brake !== undefined ? { brake } : {}),
            ...(gear !== undefined ? { gear } : {}),
        },
    };
};

const interpolateDriverSequence = (
    driver: readonly DriverSourcePoint[],
    expert: readonly ExpertSourcePoint[],
    lapOffset: number,
): Array<Omit<DriverSourcePoint, 'unwrappedPosition'>> | undefined => {
    const interpolated: Array<Omit<DriverSourcePoint, 'unwrappedPosition'>> = [];
    let rightIndex = 0;
    let previousTargetPosition: number | undefined;

    for (const expertPoint of expert) {
        const targetPosition = expertPoint.unwrappedPosition + lapOffset;
        const repeatedPosition = previousTargetPosition !== undefined
            && Math.abs(targetPosition - previousTargetPosition) <= POSITION_EPSILON;
        const match = interpolateDriverAtPosition(
            driver,
            targetPosition,
            repeatedPosition ? rightIndex + 1 : rightIndex,
        );
        if (!match) return undefined;
        interpolated.push(match.point);
        rightIndex = match.rightIndex;
        previousTargetPosition = targetPosition;
    }

    return interpolated;
};

const buildComparisonSamples = (
    driver: readonly DriverSourcePoint[],
    expert: readonly ExpertSourcePoint[],
): DriverExpertComparisonSample[] | undefined => {
    const driverStart = driver[0].unwrappedPosition;
    const driverEnd = driver[driver.length - 1].unwrappedPosition;
    const expertStart = expert[0].unwrappedPosition;
    const expertEnd = expert[expert.length - 1].unwrappedPosition;
    const firstLapOffset = Math.ceil(driverStart - expertStart - POSITION_EPSILON);
    const lastLapOffset = Math.floor(driverEnd - expertEnd + POSITION_EPSILON);

    for (let lapOffset = firstLapOffset; lapOffset <= lastLapOffset; lapOffset += 1) {
        const interpolatedDriver = interpolateDriverSequence(driver, expert, lapOffset);
        if (!interpolatedDriver) continue;

        const samples = expert.map((expertPoint, index): DriverExpertComparisonSample => {
            const driverPoint = interpolatedDriver[index];
            const expertTrajectory = trajectoryPoint(
                expertPoint.row.expert_optimal_player_pos_x,
                expertPoint.row.expert_optimal_player_pos_y,
                expertPoint.row.expert_optimal_player_pos_z,
            );
            const expertGas = normalizedInput(expertPoint.row.expert_optimal_throttle);
            const expertBrake = normalizedInput(expertPoint.row.expert_optimal_brake);
            const expertGear = finiteNumber(expertPoint.row.expert_optimal_gear);

            return {
                driverTimeMs: driverPoint.timeMs,
                expertTimeMs: expertPoint.timeMs,
                driverTrackPosition: expertPoint.normalizedPosition,
                expertTrackPosition: expertPoint.normalizedPosition,
                ...(driverPoint.trajectory ? { driverTrajectory: driverPoint.trajectory } : {}),
                ...(expertTrajectory ? { expertTrajectory } : {}),
                ...(driverPoint.gas !== undefined ? { driverGas: driverPoint.gas } : {}),
                ...(expertGas !== undefined ? { expertGas } : {}),
                ...(driverPoint.brake !== undefined ? { driverBrake: driverPoint.brake } : {}),
                ...(expertBrake !== undefined ? { expertBrake } : {}),
                ...(driverPoint.gear !== undefined ? { driverGear: driverPoint.gear } : {}),
                ...(expertGear !== undefined ? { expertGear } : {}),
            };
        });
        if (normalizeDriverExpertComparisonData({ samples })) return samples;
    }

    return undefined;
};

export const adaptAnalysisResultsComparison = ({
    baselineRecords,
    expertReferenceData,
}: AnalysisResultsComparisonAdapterInput): DriverExpertComparisonData => {
    const driver = buildDriverPoints(baselineRecords);
    const expert = buildExpertPoints(expertReferenceData);
    if (!driver || !expert) return { samples: [] };

    const samples = buildComparisonSamples(driver, expert);
    return samples ? { samples } : { samples: [] };
};
