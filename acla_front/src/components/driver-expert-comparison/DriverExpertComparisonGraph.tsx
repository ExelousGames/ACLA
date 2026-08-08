import React from 'react';
import { useDesktopGame } from 'contexts/DesktopGameContext';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import styles from './DriverExpertComparisonGraph.module.css';

export const DRIVER_COMPARISON_COLOR = '#00e676';
export const EXPERT_COMPARISON_COLOR = '#448aff';

const THROTTLE_COLOR = '#21e58b';
const BRAKE_COLOR = '#ff4d62';
const TRACK_VIEWBOX_WIDTH = 760;
const TRACK_VIEWBOX_HEIGHT = 220;
const TRACK_PADDING = 28;
const PEDAL_START_ANGLE = -60;
const PEDAL_SWEEP_ANGLE = 60;
const PEDAL_CENTER = { x: 66, y: 58 };
const PEDAL_RADIUS = 42;

export interface DriverExpertTrajectoryPoint {
    x: number;
    y?: number;
    z?: number;
}

export interface DriverExpertComparisonSample {
    driverTimeMs: number;
    expertTimeMs: number;
    driverTrackPosition: number;
    expertTrackPosition: number;
    driverTrajectory?: DriverExpertTrajectoryPoint;
    expertTrajectory?: DriverExpertTrajectoryPoint;
    driverGas?: number;
    expertGas?: number;
    driverBrake?: number;
    expertBrake?: number;
    driverGear?: number;
    expertGear?: number;
}

export interface DriverExpertComparisonData {
    samples: readonly DriverExpertComparisonSample[];
}

export interface DriverExpertComparisonAvailability {
    trajectory: boolean;
    gas: boolean;
    brake: boolean;
    gear: boolean;
}

export interface DriverExpertComparisonLayout {
    chartHeight?: number | string;
    trajectoryHeight?: number | string;
    minColumnWidth?: number | string;
}

export interface DriverExpertComparisonGraphProps {
    data: DriverExpertComparisonData;
    title?: string;
    className?: string;
    width?: number | string;
    layout?: DriverExpertComparisonLayout;
}

type ReplayContinuousKey = 'trackPosition' | 'gas' | 'brake';
type OptionalSampleScalarKey = (
    'driverGas'
    | 'expertGas'
    | 'driverBrake'
    | 'expertBrake'
    | 'driverGear'
    | 'expertGear'
);

interface PlottingTrajectoryPoint {
    x: number;
    y: number;
}

interface ReplayStreamPoint<TTrajectory = DriverExpertTrajectoryPoint> {
    timeMs: number;
    trackPosition: number;
    trajectory?: TTrajectory;
    gas?: number;
    brake?: number;
    gear?: number;
}

interface DriverExpertReplay<TTrajectory = DriverExpertTrajectoryPoint> {
    driver: ReplayStreamPoint<TTrajectory>[];
    expert: ReplayStreamPoint<TTrajectory>[];
    durationMs: number;
}

interface ReplayFrame {
    driverTrackPosition?: number;
    expertTrackPosition?: number;
    driverGas?: number;
    expertGas?: number;
    driverBrake?: number;
    expertBrake?: number;
    driverGear?: number;
    expertGear?: number;
    driverTrajectory?: PlottingTrajectoryPoint;
    expertTrajectory?: PlottingTrajectoryPoint;
}

interface PositionedTrajectoryPoint extends PlottingTrajectoryPoint {
    svgX: number;
    svgY: number;
}

interface TrackGeometry {
    driver: PositionedTrajectoryPoint[];
    expert: PositionedTrajectoryPoint[];
    project: (point: PlottingTrajectoryPoint | undefined) => PositionedTrajectoryPoint | undefined;
}

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const finiteNumber = (value: unknown): number | undefined => (
    typeof value === 'number' && Number.isFinite(value) ? value : undefined
);

const clamp = (value: number, minimum: number, maximum: number): number => (
    Math.min(maximum, Math.max(minimum, value))
);

const POSITION_EPSILON = 1e-9;
const FINISH_LINE_BACKWARD_JUMP = 0.5;

const normalizeSourceTrajectory = (value: unknown): DriverExpertTrajectoryPoint | undefined => {
    if (!isRecord(value)) return undefined;
    const x = finiteNumber(value.x);
    const y = finiteNumber(value.y);
    const z = finiteNumber(value.z);
    if (x === undefined || (y === undefined && z === undefined)) return undefined;
    return {
        x,
        ...(y !== undefined ? { y } : {}),
        ...(z !== undefined ? { z } : {}),
    };
};

const toPlottingTrajectory = (
    value: unknown,
    verticalAxis: 'y' | 'z',
): PlottingTrajectoryPoint | undefined => {
    const source = normalizeSourceTrajectory(value);
    const vertical = source ? finiteNumber(source[verticalAxis]) : undefined;
    return source && vertical !== undefined ? { x: source.x, y: vertical } : undefined;
};

const selectPlottingReplay = (
    replay: DriverExpertReplay | undefined,
    detectedGame: DesktopGame | null,
): DriverExpertReplay<PlottingTrajectoryPoint> | undefined => {
    if (!replay) return undefined;
    const driverVerticalAxis = detectedGame === 'acc' ? 'z' : 'y';
    const mapStream = (
        stream: readonly ReplayStreamPoint[],
        verticalAxis: 'y' | 'z',
    ): ReplayStreamPoint<PlottingTrajectoryPoint>[] => stream.map((point) => {
        const { trajectory, ...values } = point;
        const plottingPoint = toPlottingTrajectory(trajectory, verticalAxis);
        return {
            ...values,
            ...(plottingPoint ? { trajectory: plottingPoint } : {}),
        };
    });
    return {
        driver: mapStream(replay.driver, driverVerticalAxis),
        expert: mapStream(replay.expert, 'y'),
        durationMs: replay.durationMs,
    };
};

export const normalizeDriverExpertComparisonData = (
    value: unknown,
): DriverExpertComparisonData | undefined => {
    if (!isRecord(value) || !Array.isArray(value.samples)) return undefined;

    const samples: DriverExpertComparisonSample[] = [];
    let previousDriverTimeMs: number | undefined;
    let previousExpertTimeMs: number | undefined;
    for (const sample of value.samples) {
        if (!isRecord(sample)) return undefined;
        const driverTimeMs = finiteNumber(sample.driverTimeMs);
        const expertTimeMs = finiteNumber(sample.expertTimeMs);
        if (
            driverTimeMs === undefined
            || expertTimeMs === undefined
            || (previousDriverTimeMs !== undefined && driverTimeMs <= previousDriverTimeMs)
            || (previousExpertTimeMs !== undefined && expertTimeMs <= previousExpertTimeMs)
        ) {
            return undefined;
        }

        const normalized: DriverExpertComparisonSample = {
            driverTimeMs,
            expertTimeMs,
            driverTrackPosition: 0,
            expertTrackPosition: 0,
        };
        const driverTrackPosition = finiteNumber(sample.driverTrackPosition);
        const expertTrackPosition = finiteNumber(sample.expertTrackPosition);
        if (
            driverTrackPosition === undefined
            || expertTrackPosition === undefined
            || driverTrackPosition < 0
            || driverTrackPosition > 1
            || expertTrackPosition < 0
            || expertTrackPosition > 1
        ) {
            return undefined;
        }
        normalized.driverTrackPosition = driverTrackPosition;
        normalized.expertTrackPosition = expertTrackPosition;
        const driverTrajectory = normalizeSourceTrajectory(sample.driverTrajectory);
        const expertTrajectory = normalizeSourceTrajectory(sample.expertTrajectory);
        if (driverTrajectory) normalized.driverTrajectory = driverTrajectory;
        if (expertTrajectory) normalized.expertTrajectory = expertTrajectory;

        const scalarKeys: OptionalSampleScalarKey[] = [
            'driverGas',
            'expertGas',
            'driverBrake',
            'expertBrake',
            'driverGear',
            'expertGear',
        ];
        scalarKeys.forEach((key) => {
            const scalar = finiteNumber(sample[key]);
            if (scalar !== undefined) normalized[key] = scalar;
        });
        samples.push(normalized);
        previousDriverTimeMs = driverTimeMs;
        previousExpertTimeMs = expertTimeMs;
    }

    const data = { samples };
    return buildDriverExpertReplay(data) ? data : undefined;
};

const buildUnwrappedReplayStream = (
    samples: readonly DriverExpertComparisonSample[],
    identity: 'driver' | 'expert',
): ReplayStreamPoint[] | undefined => {
    const points: ReplayStreamPoint[] = [];
    let lapOffset = 0;
    let previousNormalizedPosition: number | undefined;
    let previousUnwrappedPosition: number | undefined;
    let previousTimeMs: number | undefined;

    for (const sample of samples) {
        const timeMs = finiteNumber(
            identity === 'driver' ? sample.driverTimeMs : sample.expertTimeMs,
        );
        const normalizedPosition = finiteNumber(
            identity === 'driver'
                ? sample.driverTrackPosition
                : sample.expertTrackPosition,
        );
        if (
            timeMs === undefined
            || normalizedPosition === undefined
            || normalizedPosition < 0
            || normalizedPosition > 1
            || (previousTimeMs !== undefined && timeMs <= previousTimeMs)
        ) {
            return undefined;
        }

        if (
            previousNormalizedPosition !== undefined
            && normalizedPosition < previousNormalizedPosition
        ) {
            const backwardJump = previousNormalizedPosition - normalizedPosition;
            if (backwardJump <= FINISH_LINE_BACKWARD_JUMP) return undefined;
            lapOffset += 1;
        }

        const trackPosition = normalizedPosition + lapOffset;
        if (
            previousUnwrappedPosition !== undefined
            && trackPosition + POSITION_EPSILON < previousUnwrappedPosition
        ) {
            return undefined;
        }

        const trajectory = normalizeSourceTrajectory(
            identity === 'driver' ? sample.driverTrajectory : sample.expertTrajectory,
        );
        const gas = finiteNumber(identity === 'driver' ? sample.driverGas : sample.expertGas);
        const brake = finiteNumber(identity === 'driver' ? sample.driverBrake : sample.expertBrake);
        const gear = finiteNumber(identity === 'driver' ? sample.driverGear : sample.expertGear);
        points.push({
            timeMs,
            trackPosition,
            ...(trajectory ? { trajectory } : {}),
            ...(gas !== undefined ? { gas } : {}),
            ...(brake !== undefined ? { brake } : {}),
            ...(gear !== undefined ? { gear } : {}),
        });
        previousTimeMs = timeMs;
        previousNormalizedPosition = normalizedPosition;
        previousUnwrappedPosition = trackPosition;
    }

    return points.length ? points : undefined;
};

const shiftReplayStream = (
    stream: readonly ReplayStreamPoint[],
    lapOffset: number,
): ReplayStreamPoint[] => stream.map((point) => ({
    ...point,
    trackPosition: point.trackPosition + lapOffset,
}));

const interpolateTrajectoryAtPosition = (
    previous: DriverExpertTrajectoryPoint | undefined,
    next: DriverExpertTrajectoryPoint | undefined,
    ratio: number,
): DriverExpertTrajectoryPoint | undefined => {
    if (!previous || !next) return undefined;
    const interpolateAxis = (axis: 'x' | 'y' | 'z'): number | undefined => {
        const previousValue = finiteNumber(previous[axis]);
        const nextValue = finiteNumber(next[axis]);
        return previousValue !== undefined && nextValue !== undefined
            ? previousValue + ((nextValue - previousValue) * ratio)
            : undefined;
    };
    const x = interpolateAxis('x');
    const y = interpolateAxis('y');
    const z = interpolateAxis('z');
    if (x === undefined || (y === undefined && z === undefined)) return undefined;
    return {
        x,
        ...(y !== undefined ? { y } : {}),
        ...(z !== undefined ? { z } : {}),
    };
};

const interpolateValueAtPosition = (
    previous: number | undefined,
    next: number | undefined,
    ratio: number,
): number | undefined => (
    previous !== undefined && next !== undefined
        ? previous + ((next - previous) * ratio)
        : undefined
);

const trimReplayStreamToPosition = (
    stream: readonly ReplayStreamPoint[],
    startPosition: number,
): ReplayStreamPoint[] | undefined => {
    const firstAtOrAfter = stream.findIndex((point) => (
        point.trackPosition + POSITION_EPSILON >= startPosition
    ));
    if (firstAtOrAfter < 0) return undefined;

    const exactPoint = stream[firstAtOrAfter];
    if (Math.abs(exactPoint.trackPosition - startPosition) <= POSITION_EPSILON) {
        return [
            { ...exactPoint, trackPosition: startPosition },
            ...stream.slice(firstAtOrAfter + 1),
        ];
    }
    if (firstAtOrAfter === 0) return undefined;

    const previous = stream[firstAtOrAfter - 1];
    const next = stream[firstAtOrAfter];
    const positionSpan = next.trackPosition - previous.trackPosition;
    if (positionSpan <= 0) return undefined;
    const ratio = (startPosition - previous.trackPosition) / positionSpan;
    const gas = interpolateValueAtPosition(previous.gas, next.gas, ratio);
    const brake = interpolateValueAtPosition(previous.brake, next.brake, ratio);
    const trajectory = interpolateTrajectoryAtPosition(
        previous.trajectory,
        next.trajectory,
        ratio,
    );
    let gearIndex = firstAtOrAfter - 1;
    while (gearIndex >= 0 && finiteNumber(stream[gearIndex].gear) === undefined) {
        gearIndex -= 1;
    }
    const gear = gearIndex >= 0 ? finiteNumber(stream[gearIndex].gear) : undefined;
    return [{
        timeMs: previous.timeMs + ((next.timeMs - previous.timeMs) * ratio),
        trackPosition: startPosition,
        ...(trajectory ? { trajectory } : {}),
        ...(gas !== undefined ? { gas } : {}),
        ...(brake !== undefined ? { brake } : {}),
        ...(gear !== undefined ? { gear } : {}),
    }, ...stream.slice(firstAtOrAfter)];
};

const chooseExpertLapOffset = (
    driver: readonly ReplayStreamPoint[],
    expert: readonly ReplayStreamPoint[],
): number | undefined => {
    const driverStart = driver[0].trackPosition;
    const driverEnd = driver[driver.length - 1].trackPosition;
    const expertStart = expert[0].trackPosition;
    const expertEnd = expert[expert.length - 1].trackPosition;
    const firstOffset = Math.ceil(driverStart - expertEnd - POSITION_EPSILON);
    const lastOffset = Math.floor(driverEnd - expertStart + POSITION_EPSILON);
    let best: { offset: number; overlap: number; initialGap: number } | undefined;

    for (let offset = firstOffset; offset <= lastOffset; offset += 1) {
        const shiftedStart = expertStart + offset;
        const shiftedEnd = expertEnd + offset;
        const overlapStart = Math.max(driverStart, shiftedStart);
        const overlapEnd = Math.min(driverEnd, shiftedEnd);
        if (overlapEnd + POSITION_EPSILON < overlapStart) continue;
        const candidate = {
            offset,
            overlap: Math.max(0, overlapEnd - overlapStart),
            initialGap: Math.abs(driverStart - shiftedStart),
        };
        if (
            !best
            || candidate.overlap > best.overlap + POSITION_EPSILON
            || (
                Math.abs(candidate.overlap - best.overlap) <= POSITION_EPSILON
                && candidate.initialGap < best.initialGap - POSITION_EPSILON
            )
            || (
                Math.abs(candidate.overlap - best.overlap) <= POSITION_EPSILON
                && Math.abs(candidate.initialGap - best.initialGap) <= POSITION_EPSILON
                && Math.abs(candidate.offset) < Math.abs(best.offset)
            )
        ) {
            best = candidate;
        }
    }
    return best?.offset;
};

const getReplayStreamDurationMs = (stream: readonly ReplayStreamPoint[]): number => (
    stream.length <= 1 ? 0 : Math.max(0, stream[stream.length - 1].timeMs - stream[0].timeMs)
);

const buildDriverExpertReplay = (
    data: DriverExpertComparisonData | null | undefined,
): DriverExpertReplay | undefined => {
    const samples = data && Array.isArray(data.samples) ? data.samples : [];
    const driver = buildUnwrappedReplayStream(samples, 'driver');
    const unshiftedExpert = buildUnwrappedReplayStream(samples, 'expert');
    if (!driver || !unshiftedExpert) return undefined;

    const expertLapOffset = chooseExpertLapOffset(driver, unshiftedExpert);
    if (expertLapOffset === undefined) return undefined;
    const expert = shiftReplayStream(unshiftedExpert, expertLapOffset);
    const sharedStartPosition = Math.max(
        driver[0].trackPosition,
        expert[0].trackPosition,
    );
    const alignedDriver = trimReplayStreamToPosition(driver, sharedStartPosition);
    const alignedExpert = trimReplayStreamToPosition(expert, sharedStartPosition);
    if (!alignedDriver || !alignedExpert) return undefined;

    return {
        driver: alignedDriver,
        expert: alignedExpert,
        durationMs: Math.max(
            getReplayStreamDurationMs(alignedDriver),
            getReplayStreamDurationMs(alignedExpert),
        ),
    };
};

const streamHasValue = (
    stream: readonly ReplayStreamPoint[],
    key: 'gas' | 'brake' | 'gear',
): boolean => stream.some((point) => finiteNumber(point[key]) !== undefined);

export const getDriverExpertComparisonAvailability = (
    data: DriverExpertComparisonData | null | undefined,
    detectedGame: DesktopGame | null = null,
): DriverExpertComparisonAvailability => {
    const replay = buildDriverExpertReplay(data);
    if (!replay) return { trajectory: false, gas: false, brake: false, gear: false };
    const driverVerticalAxis = detectedGame === 'acc' ? 'z' : 'y';
    const hasDriverTrajectory = replay.driver.some((point) => (
        toPlottingTrajectory(point.trajectory, driverVerticalAxis) !== undefined
    ));
    const hasExpertTrajectory = replay.expert.some((point) => (
        toPlottingTrajectory(point.trajectory, 'y') !== undefined
    ));
    return {
        trajectory: hasDriverTrajectory && hasExpertTrajectory,
        gas: streamHasValue(replay.driver, 'gas') && streamHasValue(replay.expert, 'gas'),
        brake: streamHasValue(replay.driver, 'brake') && streamHasValue(replay.expert, 'brake'),
        gear: streamHasValue(replay.driver, 'gear') && streamHasValue(replay.expert, 'gear'),
    };
};

export const hasComparableDriverExpertData = (
    data: DriverExpertComparisonData | null | undefined,
    detectedGame: DesktopGame | null = null,
): boolean => Object.values(getDriverExpertComparisonAvailability(data, detectedGame)).some(Boolean);

const toCssSize = (value: number | string | undefined): number | string | undefined => (
    typeof value === 'number' ? `${value}px` : value
);

const getPedalGaugeAngle = (value: number): number => (
    PEDAL_START_ANGLE + (clamp(value, 0, 1) * PEDAL_SWEEP_ANGLE)
);

const polarPoint = (
    angle: number,
    radius = PEDAL_RADIUS,
): { x: number; y: number } => {
    const radians = (angle * Math.PI) / 180;
    return {
        x: PEDAL_CENTER.x + (Math.cos(radians) * radius),
        y: PEDAL_CENTER.y + (Math.sin(radians) * radius),
    };
};

const formatNumber = (value: number): string => Number(value.toFixed(3)).toString();
const formatReplayTimeMs = (value: number): string => `${(value / 1000).toFixed(2)}s`;

const buildPedalArc = (): string => {
    const start = polarPoint(PEDAL_START_ANGLE);
    const end = polarPoint(PEDAL_START_ANGLE + PEDAL_SWEEP_ANGLE);
    return [
        `M ${formatNumber(start.x)} ${formatNumber(start.y)}`,
        `A ${PEDAL_RADIUS} ${PEDAL_RADIUS} 0 0 1 ${formatNumber(end.x)} ${formatNumber(end.y)}`,
    ].join(' ');
};

const PEDAL_ARC_PATH = buildPedalArc();

const prefersReducedMotion = (): boolean => (
    typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches
);

export const getDriverExpertReplayDurationMs = (
    data: DriverExpertComparisonData | null | undefined,
): number => buildDriverExpertReplay(data)?.durationMs ?? 0;

const normalizedSampleTime = (
    stream: readonly ReplayStreamPoint<PlottingTrajectoryPoint>[],
    index: number,
): number => stream[index].timeMs - stream[0].timeMs;

const getFrameIndexes = (
    stream: readonly ReplayStreamPoint<PlottingTrajectoryPoint>[],
    elapsedTimeMs: number,
): { lower: number; upper: number } => {
    if (stream.length <= 1) return { lower: 0, upper: 0 };
    let lower = 0;
    while (
        lower + 1 < stream.length
        && normalizedSampleTime(stream, lower + 1) <= elapsedTimeMs
    ) {
        lower += 1;
    }
    return { lower, upper: Math.min(stream.length - 1, lower + 1) };
};

const interpolateScalar = (
    stream: readonly ReplayStreamPoint<PlottingTrajectoryPoint>[],
    key: ReplayContinuousKey,
    elapsedTimeMs: number,
    lower: number,
    upper: number,
): number | undefined => {
    let previousIndex = lower;
    while (previousIndex >= 0 && finiteNumber(stream[previousIndex][key]) === undefined) {
        previousIndex -= 1;
    }
    let nextIndex = upper;
    while (nextIndex < stream.length && finiteNumber(stream[nextIndex][key]) === undefined) {
        nextIndex += 1;
    }
    const previous = previousIndex >= 0 ? finiteNumber(stream[previousIndex][key]) : undefined;
    const next = nextIndex < stream.length ? finiteNumber(stream[nextIndex][key]) : undefined;
    if (previous === undefined) return next;
    if (next === undefined) return previous;
    const previousTime = normalizedSampleTime(stream, previousIndex);
    const nextTime = normalizedSampleTime(stream, nextIndex);
    const span = nextTime - previousTime;
    if (span <= 0) return next;
    const ratio = clamp((elapsedTimeMs - previousTime) / span, 0, 1);
    return previous + ((next - previous) * ratio);
};

const interpolateTrajectory = (
    stream: readonly ReplayStreamPoint<PlottingTrajectoryPoint>[],
    elapsedTimeMs: number,
    lower: number,
    upper: number,
): PlottingTrajectoryPoint | undefined => {
    let previousIndex = lower;
    while (previousIndex >= 0 && !stream[previousIndex].trajectory) {
        previousIndex -= 1;
    }
    let nextIndex = upper;
    while (nextIndex < stream.length && !stream[nextIndex].trajectory) {
        nextIndex += 1;
    }
    const previous = previousIndex >= 0
        ? stream[previousIndex].trajectory
        : undefined;
    const next = nextIndex < stream.length
        ? stream[nextIndex].trajectory
        : undefined;
    if (!previous) return next;
    if (!next) return previous;
    const previousTime = normalizedSampleTime(stream, previousIndex);
    const nextTime = normalizedSampleTime(stream, nextIndex);
    const span = nextTime - previousTime;
    if (span <= 0) return next;
    const ratio = clamp((elapsedTimeMs - previousTime) / span, 0, 1);
    return {
        x: previous.x + ((next.x - previous.x) * ratio),
        y: previous.y + ((next.y - previous.y) * ratio),
    };
};

const steppedGear = (
    stream: readonly ReplayStreamPoint<PlottingTrajectoryPoint>[],
    lower: number,
): number | undefined => {
    for (let index = lower; index >= 0; index -= 1) {
        const value = finiteNumber(stream[index].gear);
        if (value !== undefined) return value;
    }
    for (let index = lower + 1; index < stream.length; index += 1) {
        const value = finiteNumber(stream[index].gear);
        if (value !== undefined) return value;
    }
    return undefined;
};

const buildReplayFrame = (
    replay: DriverExpertReplay<PlottingTrajectoryPoint> | undefined,
    elapsedTimeMs: number,
): ReplayFrame => {
    if (!replay) return {};
    const driverIndexes = getFrameIndexes(replay.driver, elapsedTimeMs);
    const expertIndexes = getFrameIndexes(replay.expert, elapsedTimeMs);
    return {
        driverTrackPosition: interpolateScalar(
            replay.driver,
            'trackPosition',
            elapsedTimeMs,
            driverIndexes.lower,
            driverIndexes.upper,
        ),
        expertTrackPosition: interpolateScalar(
            replay.expert,
            'trackPosition',
            elapsedTimeMs,
            expertIndexes.lower,
            expertIndexes.upper,
        ),
        driverGas: interpolateScalar(
            replay.driver,
            'gas',
            elapsedTimeMs,
            driverIndexes.lower,
            driverIndexes.upper,
        ),
        expertGas: interpolateScalar(
            replay.expert,
            'gas',
            elapsedTimeMs,
            expertIndexes.lower,
            expertIndexes.upper,
        ),
        driverBrake: interpolateScalar(
            replay.driver,
            'brake',
            elapsedTimeMs,
            driverIndexes.lower,
            driverIndexes.upper,
        ),
        expertBrake: interpolateScalar(
            replay.expert,
            'brake',
            elapsedTimeMs,
            expertIndexes.lower,
            expertIndexes.upper,
        ),
        driverGear: steppedGear(replay.driver, driverIndexes.lower),
        expertGear: steppedGear(replay.expert, expertIndexes.lower),
        driverTrajectory: interpolateTrajectory(
            replay.driver,
            elapsedTimeMs,
            driverIndexes.lower,
            driverIndexes.upper,
        ),
        expertTrajectory: interpolateTrajectory(
            replay.expert,
            elapsedTimeMs,
            expertIndexes.lower,
            expertIndexes.upper,
        ),
    };
};

const useReplayElapsedTime = (
    replay: DriverExpertReplay<PlottingTrajectoryPoint> | undefined,
    durationMs: number,
): number => {
    const shouldFinishImmediately = !replay || durationMs <= 0 || prefersReducedMotion();
    const [elapsedTimeMs, setElapsedTimeMs] = React.useState(
        shouldFinishImmediately ? durationMs : 0,
    );

    React.useEffect(() => {
        if (!replay || durationMs <= 0 || prefersReducedMotion()) {
            setElapsedTimeMs(durationMs);
            return undefined;
        }

        let animationFrame: number | null = null;
        let startedAt: number | null = null;
        setElapsedTimeMs(0);

        const animate = (timestamp: number) => {
            if (startedAt === null) startedAt = timestamp;
            const nextElapsedTimeMs = clamp(timestamp - startedAt, 0, durationMs);
            setElapsedTimeMs(nextElapsedTimeMs);
            if (nextElapsedTimeMs < durationMs) {
                animationFrame = window.requestAnimationFrame(animate);
            } else {
                animationFrame = null;
            }
        };

        animationFrame = window.requestAnimationFrame(animate);
        return () => {
            if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
        };
    }, [durationMs, replay]);

    return elapsedTimeMs;
};

const createTrackGeometry = (
    replay: DriverExpertReplay<PlottingTrajectoryPoint> | undefined,
): TrackGeometry => {
    const driverPoints = (replay?.driver ?? []).flatMap((sample) => {
        const point = sample.trajectory;
        return point ? [point] : [];
    });
    const expertPoints = (replay?.expert ?? []).flatMap((sample) => {
        const point = sample.trajectory;
        return point ? [point] : [];
    });
    const allPoints = [...driverPoints, ...expertPoints];
    if (!allPoints.length) {
        return { driver: [], expert: [], project: () => undefined };
    }

    const minX = Math.min(...allPoints.map(({ x }) => x));
    const maxX = Math.max(...allPoints.map(({ x }) => x));
    const minY = Math.min(...allPoints.map(({ y }) => y));
    const maxY = Math.max(...allPoints.map(({ y }) => y));
    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const usableWidth = TRACK_VIEWBOX_WIDTH - (TRACK_PADDING * 2);
    const usableHeight = TRACK_VIEWBOX_HEIGHT - (TRACK_PADDING * 2);
    const scaleCandidates = [
        spanX > 0 ? usableWidth / spanX : Number.POSITIVE_INFINITY,
        spanY > 0 ? usableHeight / spanY : Number.POSITIVE_INFINITY,
    ];
    const finiteScales = scaleCandidates.filter(Number.isFinite);
    const scale = finiteScales.length ? Math.min(...finiteScales) : 1;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    const project = (
        point: PlottingTrajectoryPoint | undefined,
    ): PositionedTrajectoryPoint | undefined => {
        if (!point) return undefined;
        return {
            ...point,
            svgX: (TRACK_VIEWBOX_WIDTH / 2) + ((point.x - centerX) * scale),
            svgY: (TRACK_VIEWBOX_HEIGHT / 2) - ((point.y - centerY) * scale),
        };
    };

    return {
        driver: driverPoints.map(project).filter((point): point is PositionedTrajectoryPoint => !!point),
        expert: expertPoints.map(project).filter((point): point is PositionedTrajectoryPoint => !!point),
        project,
    };
};

const trajectoryPath = (points: readonly PositionedTrajectoryPoint[]): string => points
    .map(({ svgX, svgY }, index) => (
        `${index === 0 ? 'M' : 'L'} ${formatNumber(svgX)} ${formatNumber(svgY)}`
    ))
    .join(' ');

const PedalGauge: React.FC<{
    available: boolean;
    identity: 'driver' | 'expert';
    label: 'Throttle' | 'Brake';
    value: number | undefined;
}> = ({ available, identity, label, value }) => {
    const normalizedValue = clamp(value ?? 0, 0, 1);
    const angle = getPedalGaugeAngle(normalizedValue);
    const armEnd = polarPoint(angle, PEDAL_RADIUS - 4);
    const marker = polarPoint(angle);
    const color = label === 'Throttle' ? THROTTLE_COLOR : BRAKE_COLOR;
    const testId = `${identity}-${label.toLowerCase()}-gauge`;

    if (!available) {
        return (
            <div
                className={[styles.gauge, styles.gaugeUnavailable].join(' ')}
                data-testid={testId}
                data-state="unavailable"
                aria-label={`${identity} ${label} unavailable`}
            >
                <svg className={styles.gaugeSvg} viewBox="0 0 132 112" aria-hidden="true">
                    <path className={styles.gaugeTrack} d={PEDAL_ARC_PATH} pathLength={100} />
                </svg>
                <span className={styles.gaugeLabel}>{label}</span>
                <span className={styles.gaugeUnavailableValue}>N/A</span>
            </div>
        );
    }

    const percentage = Math.round(normalizedValue * 100);
    return (
        <div
            className={styles.gauge}
            role="meter"
            aria-label={`${identity} ${label}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percentage}
            data-testid={testId}
            data-value={formatNumber(normalizedValue)}
            data-gauge-angle={formatNumber(angle)}
        >
            <svg className={styles.gaugeSvg} viewBox="0 0 132 112" aria-hidden="true">
                <path className={styles.gaugeTrack} d={PEDAL_ARC_PATH} pathLength={100} />
                <path
                    className={styles.gaugeFill}
                    d={PEDAL_ARC_PATH}
                    pathLength={100}
                    stroke={color}
                    strokeDasharray={`${normalizedValue * 100} 100`}
                />
                <line
                    className={styles.gaugeArm}
                    x1={PEDAL_CENTER.x}
                    y1={PEDAL_CENTER.y}
                    x2={armEnd.x}
                    y2={armEnd.y}
                    stroke={color}
                />
                <circle cx={PEDAL_CENTER.x} cy={PEDAL_CENTER.y} r="4" fill={color} />
                <circle
                    className={styles.gaugeMarker}
                    cx={marker.x}
                    cy={marker.y}
                    r="4.5"
                    fill={color}
                />
            </svg>
            <span className={styles.gaugeLabel}>{label}</span>
            <span className={styles.gaugeValue}>{percentage}<small>%</small></span>
        </div>
    );
};

const CompetitorPanel: React.FC<{
    identity: 'driver' | 'expert';
    gasAvailable: boolean;
    brakeAvailable: boolean;
    gearAvailable: boolean;
    gas: number | undefined;
    brake: number | undefined;
    gear: number | undefined;
}> = ({ identity, gasAvailable, brakeAvailable, gearAvailable, gas, brake, gear }) => {
    const isDriver = identity === 'driver';
    const label = isDriver ? 'Driver' : 'Expert';
    const color = isDriver ? DRIVER_COMPARISON_COLOR : EXPERT_COMPARISON_COLOR;
    const gearValue = gearAvailable && gear !== undefined ? Math.round(gear) : '—';

    return (
        <section
            className={styles.competitorPanel}
            aria-label={`${label} live pedal telemetry`}
            style={{ '--identity-color': color } as React.CSSProperties}
            data-testid={`${identity}-panel`}
        >
            <header className={styles.competitorHeader}>
                <div className={styles.identity}>
                    <span className={styles.identityMarker} />
                    <span>{label}</span>
                </div>
                <div
                    className={[styles.gear, gearAvailable ? '' : styles.gearUnavailable]
                        .filter(Boolean).join(' ')}
                    data-testid={`${identity}-gear`}
                    data-state={gearAvailable ? 'available' : 'unavailable'}
                >
                    <span className={styles.gearLabel}>Gear</span>
                    <strong>{gearValue}</strong>
                </div>
            </header>
            <div className={styles.gaugeRow}>
                <PedalGauge
                    available={gasAvailable}
                    identity={identity}
                    label="Throttle"
                    value={gas}
                />
                <PedalGauge
                    available={brakeAvailable}
                    identity={identity}
                    label="Brake"
                    value={brake}
                />
            </div>
        </section>
    );
};

const TrackReplay: React.FC<{
    available: boolean;
    frame: ReplayFrame;
    geometry: TrackGeometry;
    height: number | string;
    filterId: string;
}> = ({ available, frame, geometry, height, filterId }) => {
    const driverMarker = geometry.project(frame.driverTrajectory);
    const expertMarker = geometry.project(frame.expertTrajectory);
    const driverPath = trajectoryPath(geometry.driver);
    const expertPath = trajectoryPath(geometry.expert);

    return (
        <section
            className={styles.trackPanel}
            style={{ height: toCssSize(height) }}
            aria-label="Track replay"
        >
            <header className={styles.panelHeader}>
                <span>Track replay</span>
                <div className={styles.traceLabels} aria-label="Track trace identities">
                    <span style={{ color: DRIVER_COMPARISON_COLOR }}>Driver trace</span>
                    <span style={{ color: EXPERT_COMPARISON_COLOR }}>Expert trace</span>
                </div>
            </header>
            {available ? (
                <svg
                    className={styles.trackSvg}
                    viewBox={`0 0 ${TRACK_VIEWBOX_WIDTH} ${TRACK_VIEWBOX_HEIGHT}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label="Track replay showing Driver and Expert trajectories"
                    data-testid="comparison-track-map"
                >
                    <defs>
                        <filter id={`${filterId}-driver-glow`} x="-40%" y="-40%" width="180%" height="180%">
                            <feGaussianBlur stdDeviation="4" result="blur" />
                            <feMerge>
                                <feMergeNode in="blur" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                        <filter id={`${filterId}-expert-glow`} x="-40%" y="-40%" width="180%" height="180%">
                            <feGaussianBlur stdDeviation="3" result="blur" />
                            <feMerge>
                                <feMergeNode in="blur" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                    </defs>
                    <rect className={styles.trackViewport} width="760" height="220" rx="9" />
                    {driverPath && (
                        <>
                            <path className={styles.trackShadow} d={driverPath} />
                            <path
                                className={styles.driverPath}
                                d={driverPath}
                                filter={`url(#${filterId}-driver-glow)`}
                                data-testid="driver-track-path"
                            />
                        </>
                    )}
                    {expertPath && (
                        <path
                            className={styles.expertPath}
                            d={expertPath}
                            filter={`url(#${filterId}-expert-glow)`}
                            data-testid="expert-track-path"
                        />
                    )}
                    {driverMarker && (
                        <g
                            className={styles.positionMarker}
                            data-testid="driver-position-marker"
                            data-x={formatNumber(driverMarker.x)}
                            data-y={formatNumber(driverMarker.y)}
                            data-track-position={frame.driverTrackPosition === undefined
                                ? undefined
                                : formatNumber(frame.driverTrackPosition)}
                        >
                            <circle
                                className={styles.markerHalo}
                                cx={driverMarker.svgX}
                                cy={driverMarker.svgY}
                                r="11"
                                fill={DRIVER_COMPARISON_COLOR}
                            />
                            <circle
                                cx={driverMarker.svgX}
                                cy={driverMarker.svgY}
                                r="5.5"
                                fill={DRIVER_COMPARISON_COLOR}
                            />
                        </g>
                    )}
                    {expertMarker && (
                        <g
                            className={styles.positionMarker}
                            data-testid="expert-position-marker"
                            data-x={formatNumber(expertMarker.x)}
                            data-y={formatNumber(expertMarker.y)}
                            data-track-position={frame.expertTrackPosition === undefined
                                ? undefined
                                : formatNumber(frame.expertTrackPosition)}
                        >
                            <circle
                                className={styles.markerHalo}
                                cx={expertMarker.svgX}
                                cy={expertMarker.svgY}
                                r="10"
                                fill={EXPERT_COMPARISON_COLOR}
                            />
                            <circle
                                cx={expertMarker.svgX}
                                cy={expertMarker.svgY}
                                r="5"
                                fill={EXPERT_COMPARISON_COLOR}
                            />
                        </g>
                    )}
                </svg>
            ) : (
                <div
                    className={styles.trackUnavailable}
                    role="status"
                    data-testid="trajectory-unavailable"
                >
                    <span className={styles.placeholderTrack} aria-hidden="true" />
                    <span>Track data unavailable</span>
                </div>
            )}
        </section>
    );
};

export const DriverExpertComparisonGraph: React.FC<DriverExpertComparisonGraphProps> = ({
    data,
    title,
    className,
    width = '100%',
    layout,
}) => {
    const { detectedGame } = useDesktopGame();
    const rawSamples = data?.samples;
    const sourceSamples = React.useMemo(
        () => (Array.isArray(rawSamples) ? [...rawSamples] : []),
        [rawSamples],
    );
    const replay = React.useMemo(
        () => buildDriverExpertReplay({ samples: sourceSamples }),
        [sourceSamples],
    );
    const plottingReplay = React.useMemo(
        () => selectPlottingReplay(replay, detectedGame),
        [detectedGame, replay],
    );
    const availability = React.useMemo(
        () => getDriverExpertComparisonAvailability(data, detectedGame),
        [data, detectedGame],
    );
    const hasAnyComparison = Object.values(availability).some(Boolean);
    const chartHeight = layout?.chartHeight ?? 190;
    const trajectoryHeight = layout?.trajectoryHeight ?? 220;
    const rootClassName = [styles.root, className].filter(Boolean).join(' ');
    const rootStyle = {
        width: toCssSize(width),
        '--driver-expert-min-column-width': toCssSize(layout?.minColumnWidth ?? 260),
    } as React.CSSProperties;
    const replayDurationMs = replay?.durationMs ?? 0;
    const elapsedTimeMs = useReplayElapsedTime(plottingReplay, replayDurationMs);
    const frame = React.useMemo(
        () => buildReplayFrame(plottingReplay, elapsedTimeMs),
        [elapsedTimeMs, plottingReplay],
    );
    const geometry = React.useMemo(() => createTrackGeometry(plottingReplay), [plottingReplay]);
    const reactId = React.useId();
    const filterId = React.useMemo(() => `driver-expert-${reactId.replace(/:/g, '')}`, [reactId]);
    const isComplete = !replay || replayDurationMs <= 0 || elapsedTimeMs >= replayDurationMs;
    const replayStatus = !replay ? 'No data' : isComplete ? 'Replay complete' : 'Replaying';

    return (
        <section
            className={rootClassName}
            style={rootStyle}
            aria-label={title ?? 'Driver and Expert comparison'}
            data-testid="driver-expert-comparison"
        >
            <header className={styles.header}>
                <div className={styles.titleBlock}>
                    <span className={styles.eyebrow}>Driver / Expert</span>
                    <h2 className={styles.title}>{title ?? 'Segment pedal replay'}</h2>
                </div>
                <div className={styles.replayReadout}>
                    <span
                        className={[styles.statusDot, isComplete ? styles.statusDotComplete : '']
                            .filter(Boolean).join(' ')}
                        aria-hidden="true"
                    />
                    <span className={styles.replayStatus} data-testid="replay-status">
                        {replayStatus}
                    </span>
                    <span className={styles.progressDivider} aria-hidden="true" />
                    <span
                        className={styles.progressValue}
                        role="progressbar"
                        aria-label="Replay elapsed time"
                        aria-valuemin={0}
                        aria-valuemax={Math.round(replayDurationMs)}
                        aria-valuenow={Math.round(elapsedTimeMs)}
                        data-testid="replay-progress"
                    >
                        {formatReplayTimeMs(elapsedTimeMs)} / {formatReplayTimeMs(replayDurationMs)}
                    </span>
                </div>
            </header>

            {!hasAnyComparison && (
                <div className={styles.overallUnavailable} role="status">
                    Expert comparison unavailable
                </div>
            )}

            <div className={styles.hud} data-testid="driver-expert-comparison-grid">
                <TrackReplay
                    available={availability.trajectory}
                    frame={frame}
                    geometry={geometry}
                    height={trajectoryHeight}
                    filterId={filterId}
                />
                <div
                    className={styles.competitorGrid}
                    style={{ minHeight: toCssSize(chartHeight) }}
                    data-testid="pedal-panel-region"
                >
                    <CompetitorPanel
                        identity="driver"
                        gasAvailable={availability.gas}
                        brakeAvailable={availability.brake}
                        gearAvailable={availability.gear}
                        gas={frame.driverGas}
                        brake={frame.driverBrake}
                        gear={frame.driverGear}
                    />
                    <CompetitorPanel
                        identity="expert"
                        gasAvailable={availability.gas}
                        brakeAvailable={availability.brake}
                        gearAvailable={availability.gear}
                        gas={frame.expertGas}
                        brake={frame.expertBrake}
                        gear={frame.expertGear}
                    />
                </div>
            </div>
        </section>
    );
};

export default DriverExpertComparisonGraph;
