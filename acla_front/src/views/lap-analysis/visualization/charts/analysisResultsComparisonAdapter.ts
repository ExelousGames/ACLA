import {
    DriverExpertComparisonData,
    DriverExpertComparisonDiagnostic,
    DriverExpertComparisonSample,
    DriverExpertTrajectoryPoint,
    normalizeDriverExpertComparisonData,
} from 'components/driver-expert-comparison';
import { unwrapLapTelemetrySequence } from 'components/driver-expert-comparison/lapTelemetrySequence';
import { parseTelemetryFrame } from './mapTelemetry';

export interface AnalysisResultsComparisonAdapterInput {
    baselineRecords: readonly Record<string, any>[];
    expertReferenceData: readonly unknown[];
}

export interface AnalysisResultsComparisonResolution {
    comparison?: DriverExpertComparisonData;
    diagnostics: DriverExpertComparisonDiagnostic[];
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

const countReason = (counts: Record<string, number>, code: string): void => {
    counts[code] = (counts[code] ?? 0) + 1;
};

const appendCountedDiagnostics = (
    diagnostics: DriverExpertComparisonDiagnostic[],
    counts: Record<string, number>,
    definitions: Record<string, string>,
    totalRows: number,
): void => {
    Object.entries(counts).forEach(([code, count]) => {
        if (!count) return;
        diagnostics.push({
            code,
            message: definitions[code] ?? code,
            details: { affected_rows: count, total_rows: totalRows },
        });
    });
};

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

const buildDriverPoints = (
    baselineRecords: readonly Record<string, any>[],
    diagnostics: DriverExpertComparisonDiagnostic[],
): DriverSourcePoint[] | undefined => {
    if (!baselineRecords.length) {
        diagnostics.push({
            code: 'driver_records_missing',
            message: 'The recorded lap contains no Driver telemetry rows.',
        });
        return undefined;
    }

    const rows: Array<{
        row: Record<string, any>;
        timeMs: number;
        normalizedPosition: number;
        sourceIndex: number;
    }> = [];
    const rejected: Record<string, number> = {};
    let previousTimeMs: number | undefined;
    let previousPosition: number | undefined;
    for (let sourceIndex = 0; sourceIndex < baselineRecords.length; sourceIndex += 1) {
        const row = baselineRecords[sourceIndex];
        if (!isRecord(row)) {
            countReason(rejected, 'driver_row_invalid');
            continue;
        }
        const timeMs = finiteNumber(row.Graphics_current_time);
        const position = getDriverTrackPosition(row);
        if (
            timeMs === undefined
            || position === undefined
        ) {
            if (timeMs === undefined) countReason(rejected, 'driver_time_missing_or_invalid');
            if (position === undefined) countReason(rejected, 'driver_position_missing_or_invalid');
            continue;
        }
        if (previousPosition !== undefined && previousTimeMs !== undefined) {
            const crossedFinishLine = previousPosition - position > FINISH_LINE_BACKWARD_JUMP;
            const reversedAwayFromFinish = position < previousPosition && !crossedFinishLine;
            const nonIncreasingAwayFromFinish = timeMs <= previousTimeMs && !crossedFinishLine;
            if (reversedAwayFromFinish || nonIncreasingAwayFromFinish) {
                if (reversedAwayFromFinish) countReason(rejected, 'driver_position_reversed');
                if (nonIncreasingAwayFromFinish) countReason(rejected, 'driver_time_non_increasing');
                continue;
            }
        }
        rows.push({ row, timeMs, normalizedPosition: position, sourceIndex });
        previousTimeMs = timeMs;
        previousPosition = position;
    }
    appendCountedDiagnostics(diagnostics, rejected, {
        driver_row_invalid: 'Driver telemetry contains rows that are not objects.',
        driver_time_missing_or_invalid: 'Driver telemetry rows are missing a finite lap clock.',
        driver_position_missing_or_invalid: 'Driver telemetry rows are missing a normalized track position between 0 and 1.',
        driver_position_reversed: 'Driver track position moves backward without a finish-line crossing.',
        driver_time_non_increasing: 'Driver lap time repeats or decreases without a finish-line crossing.',
    }, baselineRecords.length);
    if (!rows.length) {
        diagnostics.push({
            code: 'driver_samples_unusable',
            message: 'No Driver telemetry rows remain after validation.',
        });
        return undefined;
    }

    const sequence = unwrapLapTelemetrySequence(
        rows.map((entry) => entry.timeMs),
        rows.map((entry) => entry.normalizedPosition),
    );
    if (!sequence) {
        diagnostics.push({
            code: 'driver_sequence_invalid',
            message: 'Driver telemetry cannot be converted to an increasing lap timeline.',
            details: { retained_rows: rows.length },
        });
        return undefined;
    }

    return rows.map(({ row, normalizedPosition: position, sourceIndex }, index) => {
        const trajectory = getDriverTrajectory(row, sourceIndex);
        const gas = normalizedInput(row.Physics_gas);
        const brake = normalizedInput(row.Physics_brake);
        const gear = finiteNumber(row.Physics_gear);
        return {
            timeMs: sequence.timesMs[index],
            normalizedPosition: position,
            unwrappedPosition: sequence.positions[index],
            ...(trajectory ? { trajectory } : {}),
            ...(gas !== undefined ? { gas } : {}),
            ...(brake !== undefined ? { brake } : {}),
            ...(gear !== undefined ? { gear } : {}),
        };
    });
};

const buildExpertPoints = (
    expertReferenceData: readonly unknown[],
    diagnostics: DriverExpertComparisonDiagnostic[],
): ExpertSourcePoint[] | undefined => {
    if (!expertReferenceData.length) {
        diagnostics.push({
            code: 'expert_reference_missing',
            message: 'The analysis segment contains no Expert reference telemetry.',
        });
        return undefined;
    }

    const rows: Array<{
        row: Record<string, unknown>;
        timeMs: number;
        normalizedPosition: number;
    }> = [];
    const rejected: Record<string, number> = {};
    for (const value of expertReferenceData) {
        if (!isRecord(value)) {
            countReason(rejected, 'expert_row_invalid');
            continue;
        }
        const timeMs = finiteNumber(value.expert_optimal_time);
        const position = getExpertTrackPosition(value);
        if (
            timeMs === undefined
            || position === undefined
        ) {
            if (timeMs === undefined) countReason(rejected, 'expert_time_missing_or_invalid');
            if (position === undefined) countReason(rejected, 'expert_position_missing_or_invalid');
            continue;
        }
        if (timeMs < 0) countReason(rejected, 'expert_time_negative');
        rows.push({ row: value, timeMs, normalizedPosition: position });
    }
    appendCountedDiagnostics(diagnostics, rejected, {
        expert_row_invalid: 'Expert reference telemetry contains rows that are not objects.',
        expert_time_missing_or_invalid: 'Expert reference rows are missing a finite optimal lap clock.',
        expert_position_missing_or_invalid: 'Expert reference rows are missing a normalized track position between 0 and 1.',
        expert_time_negative: 'Expert reference rows contain a negative optimal lap clock.',
    }, expertReferenceData.length);
    if (Object.keys(rejected).length > 0) {
        diagnostics.push({
            code: 'expert_reference_invalid',
            message: 'Expert reference telemetry must be complete; invalid rows cannot be skipped.',
        });
        return undefined;
    }

    const sequence = unwrapLapTelemetrySequence(
        rows.map((entry) => entry.timeMs),
        rows.map((entry) => entry.normalizedPosition),
    );
    if (!sequence) {
        diagnostics.push({
            code: 'expert_sequence_invalid',
            message: 'Expert reference telemetry does not have increasing time and track-position order.',
            details: { rows: rows.length },
        });
        return undefined;
    }

    return rows.map((entry, index) => ({
        ...entry,
        timeMs: sequence.timesMs[index],
        unwrappedPosition: sequence.positions[index],
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
    diagnostics: DriverExpertComparisonDiagnostic[],
): DriverExpertComparisonSample[] | undefined => {
    const driverStart = driver[0].unwrappedPosition;
    const driverEnd = driver[driver.length - 1].unwrappedPosition;
    const expertStart = expert[0].unwrappedPosition;
    const expertEnd = expert[expert.length - 1].unwrappedPosition;
    const firstLapOffset = Math.ceil(driverStart - expertStart - POSITION_EPSILON);
    const lastLapOffset = Math.floor(driverEnd - expertEnd + POSITION_EPSILON);
    if (firstLapOffset > lastLapOffset) {
        diagnostics.push({
            code: 'driver_coverage_incomplete',
            message: 'No complete Driver lap covers the Expert segment from start to end.',
            details: {
                driver_range: [driverStart, driverEnd],
                expert_range: [expertStart, expertEnd],
            },
        });
        return undefined;
    }

    let interpolationFailures = 0;
    let validationFailures = 0;
    for (let lapOffset = firstLapOffset; lapOffset <= lastLapOffset; lapOffset += 1) {
        const interpolatedDriver = interpolateDriverSequence(driver, expert, lapOffset);
        if (!interpolatedDriver) {
            interpolationFailures += 1;
            continue;
        }

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
        validationFailures += 1;
    }

    if (interpolationFailures > 0) diagnostics.push({
        code: 'driver_interpolation_failed',
        message: 'Driver telemetry cannot be interpolated at every Expert track position.',
        details: { attempted_laps: interpolationFailures },
    });
    if (validationFailures > 0) diagnostics.push({
        code: 'comparison_samples_invalid',
        message: 'Aligned Driver and Expert samples do not form valid increasing replay timelines.',
        details: { attempted_laps: validationFailures },
    });

    return undefined;
};

export const resolveAnalysisResultsComparison = ({
    baselineRecords,
    expertReferenceData,
}: AnalysisResultsComparisonAdapterInput): AnalysisResultsComparisonResolution => {
    const diagnostics: DriverExpertComparisonDiagnostic[] = [];
    const driver = buildDriverPoints(baselineRecords, diagnostics);
    const expert = buildExpertPoints(expertReferenceData, diagnostics);
    if (!driver || !expert) return { diagnostics };

    const samples = buildComparisonSamples(driver, expert, diagnostics);
    return samples
        ? { comparison: { samples }, diagnostics: [] }
        : { diagnostics };
};

export const adaptAnalysisResultsComparison = ({
    baselineRecords,
    expertReferenceData,
}: AnalysisResultsComparisonAdapterInput): DriverExpertComparisonData => {
    const resolution = resolveAnalysisResultsComparison({ baselineRecords, expertReferenceData });
    return resolution.comparison ?? { samples: [] };
};
