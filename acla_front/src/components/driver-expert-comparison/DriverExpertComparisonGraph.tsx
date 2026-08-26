import React from 'react';
import { useDesktopGame } from 'contexts/DesktopGameContext';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import {
    isOverlayNonEmptyString,
    isOverlayRecord,
} from 'views/floating-chat/overlay-renderer-validation';
import type { DriverExpertComparisonSnapshot } from './DriverExpertComparisonOverlay';
import { unwrapLapTelemetrySequence } from './lapTelemetrySequence';
import styles from './DriverExpertComparisonGraph.module.css';

export const DRIVER_COMPARISON_COLOR = '#00e676';
export const EXPERT_COMPARISON_COLOR = '#448aff';

const THROTTLE_COLOR = '#21e58b';
const BRAKE_COLOR = '#ff4d62';
const TRACK_VIEWBOX_WIDTH = 760;
const TRACK_VIEWBOX_HEIGHT = 220;
const TRACK_PADDING = 28;
const TELEMETRY_POD_BASE_HORIZONTAL_TRIM = 12;
const TELEMETRY_POD_BASE_WIDTH = 184 - (TELEMETRY_POD_BASE_HORIZONTAL_TRIM * 2);
const TELEMETRY_POD_BASE_HEIGHT = 102;
const TELEMETRY_POD_SCALE = 2.25;
const TELEMETRY_POD_WIDTH = TELEMETRY_POD_BASE_WIDTH * TELEMETRY_POD_SCALE;
const TELEMETRY_POD_HEIGHT = TELEMETRY_POD_BASE_HEIGHT * TELEMETRY_POD_SCALE;
const TELEMETRY_POD_HORIZONTAL_TRIM = TELEMETRY_POD_BASE_HORIZONTAL_TRIM * TELEMETRY_POD_SCALE;
const TELEMETRY_POD_GAP = 14;
const DRIVER_MARKER_HALO_RADIUS = 11;
const EXPERT_MARKER_HALO_RADIUS = 10;
const FOLLOW_CAMERA_SCALE = 4;
const CAMERA_HEADING_SMOOTHING_RADIUS = 2;
const OVERVIEW_HOLD_DURATION_MS = 1_000;
const CAMERA_FOCUS_DURATION_MS = 750;
const DRIVER_CAMERA_ANCHOR_Y_RATIO = 2 / 3;
const EXPERT_CAMERA_ANCHOR_Y_RATIO = 1 / 3;
const PEDAL_START_ANGLE = -60;
const PEDAL_SWEEP_ANGLE = 60;
const PEDAL_CENTER = { x: 66, y: 58 };
const PEDAL_RADIUS = 42;
const PEDAL_GAUGE_SCALE = 0.62;
const THROTTLE_GAUGE_X = 12;
const BRAKE_GAUGE_X = 66;

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
    /** @deprecated Retained as a no-op for backwards compatibility. */
    chartHeight?: number | string;
    trajectoryHeight?: number | string;
    /** @deprecated Retained as a no-op for backwards compatibility. */
    minColumnWidth?: number | string;
}

export interface DriverExpertComparisonGraphProps {
    data: DriverExpertComparisonData;
    title?: string;
    className?: string;
    width?: number | string;
    layout?: DriverExpertComparisonLayout;
    game?: DesktopGame | null;
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
    driverDirection?: PlottingTrajectoryPoint;
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

type ComparisonIdentity = 'driver' | 'expert';

interface TelemetryPodPosition {
    x: number;
    y: number;
}

interface FollowCamera {
    transform: string;
    target: ComparisonIdentity;
    anchorX: number;
    anchorY: number;
    rotationRadians: number;
    project: (point: PositionedTrajectoryPoint | undefined) => PositionedTrajectoryPoint | undefined;
}

type TrackPresentationPhase = 'overview' | 'focusing' | 'following';

interface ReplayTimeline {
    elapsedTimeMs: number;
    cameraProgress: number;
    statusOpacity: number;
    presentationPhase: TrackPresentationPhase;
    isComplete: boolean;
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
    flipVerticalAxis = false,
): PlottingTrajectoryPoint | undefined => {
    const source = normalizeSourceTrajectory(value);
    const vertical = source ? finiteNumber(source[verticalAxis]) : undefined;
    return source && vertical !== undefined
        ? { x: source.x, y: flipVerticalAxis ? -vertical : vertical }
        : undefined;
};

const selectPlottingReplay = (
    replay: DriverExpertReplay | undefined,
    detectedGame: DesktopGame | null,
): DriverExpertReplay<PlottingTrajectoryPoint> | undefined => {
    if (!replay) return undefined;
    const isAcc = detectedGame === 'acc';
    const driverVerticalAxis = isAcc ? 'z' : 'y';
    const mapStream = (
        stream: readonly ReplayStreamPoint[],
        verticalAxis: 'y' | 'z',
    ): ReplayStreamPoint<PlottingTrajectoryPoint>[] => stream.map((point) => {
        const { trajectory, ...values } = point;
        const plottingPoint = toPlottingTrajectory(
            trajectory,
            verticalAxis,
            isAcc,
        );
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
    for (const sample of value.samples) {
        if (!isRecord(sample)) return undefined;
        const driverTimeMs = finiteNumber(sample.driverTimeMs);
        const expertTimeMs = finiteNumber(sample.expertTimeMs);
        if (
            driverTimeMs === undefined
            || expertTimeMs === undefined
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
    }

    const data = { samples };
    return buildDriverExpertReplay(data) ? data : undefined;
};

const buildUnwrappedReplayStream = (
    samples: readonly DriverExpertComparisonSample[],
    identity: 'driver' | 'expert',
): ReplayStreamPoint[] | undefined => {
    const rawTimesMs: number[] = [];
    const normalizedPositions: number[] = [];
    for (const sample of samples) {
        const rawTimeMs = finiteNumber(
            identity === 'driver' ? sample.driverTimeMs : sample.expertTimeMs,
        );
        const normalizedPosition = finiteNumber(
            identity === 'driver'
                ? sample.driverTrackPosition
                : sample.expertTrackPosition,
        );
        if (
            rawTimeMs === undefined
            || normalizedPosition === undefined
        ) {
            return undefined;
        }
        rawTimesMs.push(rawTimeMs);
        normalizedPositions.push(normalizedPosition);
    }

    const sequence = unwrapLapTelemetrySequence(rawTimesMs, normalizedPositions);
    if (!sequence) return undefined;

    return samples.map((sample, index) => {
        const trajectory = normalizeSourceTrajectory(
            identity === 'driver' ? sample.driverTrajectory : sample.expertTrajectory,
        );
        const gas = finiteNumber(identity === 'driver' ? sample.driverGas : sample.expertGas);
        const brake = finiteNumber(identity === 'driver' ? sample.driverBrake : sample.expertBrake);
        const gear = finiteNumber(identity === 'driver' ? sample.driverGear : sample.expertGear);
        return {
            timeMs: sequence.timesMs[index],
            trackPosition: sequence.positions[index],
            ...(trajectory ? { trajectory } : {}),
            ...(gas !== undefined ? { gas } : {}),
            ...(brake !== undefined ? { brake } : {}),
            ...(gear !== undefined ? { gear } : {}),
        };
    });
};

const getReplayStreamDurationMs = (stream: readonly ReplayStreamPoint[]): number => (
    stream.length <= 1 ? 0 : Math.max(0, stream[stream.length - 1].timeMs - stream[0].timeMs)
);

const buildDriverExpertReplay = (
    data: DriverExpertComparisonData | null | undefined,
): DriverExpertReplay | undefined => {
    const samples = data && Array.isArray(data.samples) ? data.samples : [];
    const driver = buildUnwrappedReplayStream(samples, 'driver');
    const expert = buildUnwrappedReplayStream(samples, 'expert');
    if (!driver || !expert) return undefined;

    return {
        driver,
        expert,
        durationMs: Math.max(
            getReplayStreamDurationMs(driver),
            getReplayStreamDurationMs(expert),
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

const trajectoryDirection = (
    stream: readonly ReplayStreamPoint<PlottingTrajectoryPoint>[],
    current: PlottingTrajectoryPoint | undefined,
    lower: number,
    upper: number,
    elapsedTimeMs: number,
): PlottingTrajectoryPoint | undefined => {
    if (!current) return undefined;

    const directionBetween = (
        from: PlottingTrajectoryPoint | undefined,
        to: PlottingTrajectoryPoint | undefined,
    ): PlottingTrajectoryPoint | undefined => {
        if (!from || !to) return undefined;
        const x = to.x - from.x;
        const y = to.y - from.y;
        const magnitude = Math.hypot(x, y);
        return magnitude > POSITION_EPSILON
            ? { x: x / magnitude, y: y / magnitude }
            : undefined;
    };

    const findDistinctTrajectoryIndex = (
        startIndex: number,
        step: -1 | 1,
        origin: PlottingTrajectoryPoint,
    ): number | undefined => {
        for (
            let index = startIndex;
            index >= 0 && index < stream.length;
            index += step
        ) {
            if (directionBetween(origin, stream[index].trajectory)) return index;
        }
        return undefined;
    };

    const adjacentDistinctTrajectoryIndexes = (
        anchorIndex: number,
        step: -1 | 1,
    ): number[] => {
        const indexes: number[] = [];
        let index = anchorIndex;
        let point = stream[index].trajectory;
        if (!point) return indexes;

        while (indexes.length < CAMERA_HEADING_SMOOTHING_RADIUS) {
            const adjacentIndex = findDistinctTrajectoryIndex(index + step, step, point);
            if (adjacentIndex === undefined) break;
            indexes.push(adjacentIndex);
            index = adjacentIndex;
            point = stream[index].trajectory;
            if (!point) break;
        }
        return indexes;
    };

    const smoothedDirectionAt = (
        anchorIndex: number,
    ): PlottingTrajectoryPoint | undefined => {
        const anchor = stream[anchorIndex].trajectory;
        if (!anchor) return undefined;

        const previousIndexes = adjacentDistinctTrajectoryIndexes(anchorIndex, -1);
        const nextIndexes = adjacentDistinctTrajectoryIndexes(anchorIndex, 1);
        const centeredRadius = Math.min(previousIndexes.length, nextIndexes.length);
        if (centeredRadius > 0) {
            return directionBetween(
                stream[previousIndexes[centeredRadius - 1]].trajectory,
                stream[nextIndexes[centeredRadius - 1]].trajectory,
            );
        }

        if (nextIndexes.length > 0) {
            return directionBetween(anchor, stream[nextIndexes[0]].trajectory);
        }
        if (previousIndexes.length > 0) {
            return directionBetween(stream[previousIndexes[0]].trajectory, anchor);
        }
        return undefined;
    };

    let previousIndex = lower;
    while (previousIndex >= 0 && !stream[previousIndex].trajectory) previousIndex -= 1;
    let nextIndex = upper;
    while (nextIndex < stream.length && !stream[nextIndex].trajectory) nextIndex += 1;

    const previous = previousIndex >= 0 ? stream[previousIndex].trajectory : undefined;
    const next = nextIndex < stream.length ? stream[nextIndex].trajectory : undefined;
    if (previous && next) {
        const forwardIndex = findDistinctTrajectoryIndex(nextIndex, 1, previous);
        if (forwardIndex !== undefined) {
            const forwardPoint = stream[forwardIndex].trajectory;
            const forward = smoothedDirectionAt(previousIndex)
                ?? directionBetween(previous, forwardPoint);
            const following = smoothedDirectionAt(forwardIndex)
                ?? directionBetween(previous, forwardPoint);
            if (!forward || !following) return forward;

            const startTime = normalizedSampleTime(stream, previousIndex);
            const endTime = normalizedSampleTime(stream, forwardIndex);
            const segmentProgress = endTime <= startTime
                ? 1
                : clamp((elapsedTimeMs - startTime) / (endTime - startTime), 0, 1);
            const turnProgress = easeInOut(segmentProgress);
            const forwardAngle = Math.atan2(forward.y, forward.x);
            const followingAngle = Math.atan2(following.y, following.x);
            const shortestTurn = Math.atan2(
                Math.sin(followingAngle - forwardAngle),
                Math.cos(followingAngle - forwardAngle),
            );
            const smoothedAngle = forwardAngle + (shortestTurn * turnProgress);
            return { x: Math.cos(smoothedAngle), y: Math.sin(smoothedAngle) };
        }
    }

    const originIndex = previous ? previousIndex : nextIndex;
    const origin = previous ?? next;
    if (!origin) return undefined;
    const smoothedDirection = smoothedDirectionAt(originIndex);
    if (smoothedDirection) return smoothedDirection;
    const laterIndex = findDistinctTrajectoryIndex(originIndex + 1, 1, origin);
    if (laterIndex !== undefined) {
        return directionBetween(origin, stream[laterIndex].trajectory);
    }
    const earlierIndex = findDistinctTrajectoryIndex(originIndex - 1, -1, origin);
    return earlierIndex === undefined
        ? undefined
        : directionBetween(stream[earlierIndex].trajectory, origin);
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
    const driverTrajectory = interpolateTrajectory(
        replay.driver,
        elapsedTimeMs,
        driverIndexes.lower,
        driverIndexes.upper,
    );
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
        driverTrajectory,
        expertTrajectory: interpolateTrajectory(
            replay.expert,
            elapsedTimeMs,
            expertIndexes.lower,
            expertIndexes.upper,
        ),
        driverDirection: trajectoryDirection(
            replay.driver,
            driverTrajectory,
            driverIndexes.lower,
            driverIndexes.upper,
            elapsedTimeMs,
        ),
    };
};

const easeInOut = (value: number): number => value * value * (3 - (2 * value));

const useReplayTimeline = (
    replay: DriverExpertReplay<PlottingTrajectoryPoint> | undefined,
    durationMs: number,
    introduceCamera: boolean,
): ReplayTimeline => {
    const reduceMotion = prefersReducedMotion();
    const introductionDurationMs = introduceCamera
        ? OVERVIEW_HOLD_DURATION_MS + CAMERA_FOCUS_DURATION_MS
        : 0;
    const animationDurationMs = replay ? introductionDurationMs + durationMs : 0;
    const shouldFinishImmediately = !replay || animationDurationMs <= 0 || reduceMotion;
    const [animationElapsedTimeMs, setAnimationElapsedTimeMs] = React.useState(
        shouldFinishImmediately ? animationDurationMs : 0,
    );

    React.useEffect(() => {
        if (!replay || animationDurationMs <= 0 || prefersReducedMotion()) {
            setAnimationElapsedTimeMs(animationDurationMs);
            return undefined;
        }

        let animationFrame: number | null = null;
        let startedAt: number | null = null;
        setAnimationElapsedTimeMs(0);

        const animate = (timestamp: number) => {
            if (startedAt === null) startedAt = timestamp;
            const nextElapsedTimeMs = clamp(timestamp - startedAt, 0, animationDurationMs);
            setAnimationElapsedTimeMs(nextElapsedTimeMs);
            if (nextElapsedTimeMs < animationDurationMs) {
                animationFrame = window.requestAnimationFrame(animate);
            } else {
                animationFrame = null;
            }
        };

        animationFrame = window.requestAnimationFrame(animate);
        return () => {
            if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
        };
    }, [animationDurationMs, replay]);

    const focusProgress = !introduceCamera || reduceMotion
        ? 1
        : clamp(
            (animationElapsedTimeMs - OVERVIEW_HOLD_DURATION_MS) / CAMERA_FOCUS_DURATION_MS,
            0,
            1,
        );
    const cameraProgress = easeInOut(focusProgress);
    const presentationPhase: TrackPresentationPhase = focusProgress <= 0
        ? 'overview'
        : focusProgress < 1 ? 'focusing' : 'following';

    return {
        elapsedTimeMs: clamp(animationElapsedTimeMs - introductionDurationMs, 0, durationMs),
        cameraProgress,
        statusOpacity: cameraProgress,
        presentationPhase,
        isComplete: !replay || animationElapsedTimeMs >= animationDurationMs,
    };
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
    const referencePoints = driverPoints.length ? driverPoints : expertPoints;
    if (!referencePoints.length) {
        return { driver: [], expert: [], project: () => undefined };
    }

    const minX = Math.min(...referencePoints.map(({ x }) => x));
    const maxX = Math.max(...referencePoints.map(({ x }) => x));
    const minY = Math.min(...referencePoints.map(({ y }) => y));
    const maxY = Math.max(...referencePoints.map(({ y }) => y));
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

const positionTelemetryPod = (
    marker: PositionedTrajectoryPoint,
    identity: ComparisonIdentity,
): TelemetryPodPosition => (identity === 'driver' ? {
    x: marker.svgX + TELEMETRY_POD_GAP + TELEMETRY_POD_HORIZONTAL_TRIM,
    y: marker.svgY - TELEMETRY_POD_HEIGHT - TELEMETRY_POD_GAP,
} : {
    x: marker.svgX
        - TELEMETRY_POD_WIDTH
        - TELEMETRY_POD_GAP
        - TELEMETRY_POD_HORIZONTAL_TRIM,
    y: marker.svgY + TELEMETRY_POD_GAP,
});

const getFollowCamera = (
    driverMarker: PositionedTrajectoryPoint | undefined,
    expertMarker: PositionedTrajectoryPoint | undefined,
    driverDirection: PlottingTrajectoryPoint | undefined,
    viewportHeight: number,
    progress = 1,
): FollowCamera | undefined => {
    const target = driverMarker ?? expertMarker;
    if (!target) return undefined;

    const targetIdentity: ComparisonIdentity = driverMarker ? 'driver' : 'expert';
    const haloRadius = driverMarker ? DRIVER_MARKER_HALO_RADIUS : EXPERT_MARKER_HALO_RADIUS;
    const anchorX = TRACK_VIEWBOX_WIDTH / 2;
    const requestedAnchorY = viewportHeight * (
        driverMarker ? DRIVER_CAMERA_ANCHOR_Y_RATIO : EXPERT_CAMERA_ANCHOR_Y_RATIO
    );
    const minimumAnchorY = driverMarker
        ? TRACK_PADDING + TELEMETRY_POD_HEIGHT + TELEMETRY_POD_GAP
        : TRACK_PADDING + haloRadius;
    const maximumAnchorY = driverMarker
        ? viewportHeight - TRACK_PADDING - haloRadius
        : viewportHeight - TRACK_PADDING - TELEMETRY_POD_HEIGHT - TELEMETRY_POD_GAP;
    const anchorY = minimumAnchorY <= maximumAnchorY
        ? clamp(requestedAnchorY, minimumAnchorY, maximumAnchorY)
        : viewportHeight / 2;

    // Plotting Y points up while projected SVG Y points down. Rotate the projected
    // driver tangent onto the top of the viewport so the camera always looks ahead.
    const projectedHeadingAngle = driverMarker && driverDirection
        ? Math.atan2(-driverDirection.y, driverDirection.x)
        : undefined;
    const requestedRotationRadians = projectedHeadingAngle === undefined
        ? 0
        : (-Math.PI / 2) - projectedHeadingAngle;
    const rotationRadians = Math.atan2(
        Math.sin(requestedRotationRadians),
        Math.cos(requestedRotationRadians),
    );
    const cameraProgress = clamp(progress, 0, 1);
    const currentScale = 1 + ((FOLLOW_CAMERA_SCALE - 1) * cameraProgress);
    const currentRotationRadians = rotationRadians * cameraProgress;
    const currentAnchorX = target.svgX + ((anchorX - target.svgX) * cameraProgress);
    const currentAnchorY = target.svgY + ((anchorY - target.svgY) * cameraProgress);
    const cosine = Math.cos(currentRotationRadians) * currentScale;
    const sine = Math.sin(currentRotationRadians) * currentScale;
    const a = cosine;
    const b = sine;
    const c = -sine;
    const d = cosine;
    const e = currentAnchorX - ((a * target.svgX) + (c * target.svgY));
    const f = currentAnchorY - ((b * target.svgX) + (d * target.svgY));
    const cameraNumber = (value: number): string => Number(value.toFixed(6)).toString();

    return {
        transform: `matrix(${cameraNumber(a)} ${cameraNumber(b)} ${cameraNumber(c)} ${cameraNumber(d)} ${cameraNumber(e)} ${cameraNumber(f)})`,
        target: targetIdentity,
        anchorX: currentAnchorX,
        anchorY: currentAnchorY,
        rotationRadians: currentRotationRadians,
        project: (point) => point ? {
            ...point,
            svgX: (a * point.svgX) + (c * point.svgY) + e,
            svgY: (b * point.svgX) + (d * point.svgY) + f,
        } : undefined,
    };
};

const getTelemetryLeaderEnd = (
    marker: PositionedTrajectoryPoint,
    position: TelemetryPodPosition,
): { x: number; y: number } => {
    const markerX = marker.svgX - position.x;
    const markerY = marker.svgY - position.y;
    let x = clamp(markerX, 0, TELEMETRY_POD_WIDTH);
    let y = clamp(markerY, 0, TELEMETRY_POD_HEIGHT);
    if (
        markerX >= 0
        && markerX <= TELEMETRY_POD_WIDTH
        && markerY >= 0
        && markerY <= TELEMETRY_POD_HEIGHT
    ) {
        const edgeDistances = [markerX, TELEMETRY_POD_WIDTH - markerX, markerY,
            TELEMETRY_POD_HEIGHT - markerY];
        const nearestEdge = edgeDistances.indexOf(Math.min(...edgeDistances));
        if (nearestEdge === 0) x = 0;
        if (nearestEdge === 1) x = TELEMETRY_POD_WIDTH;
        if (nearestEdge === 2) y = 0;
        if (nearestEdge === 3) y = TELEMETRY_POD_HEIGHT;
    }
    return { x, y };
};

const PedalGauge: React.FC<{
    available: boolean;
    identity: ComparisonIdentity;
    label: 'Throttle' | 'Brake';
    value: number | undefined;
    x: number;
}> = ({ available, identity, label, value, x }) => {
    const normalizedValue = clamp(value ?? 0, 0, 1);
    const percentage = Math.round(normalizedValue * 100);
    const angle = getPedalGaugeAngle(normalizedValue);
    const armEnd = polarPoint(angle, PEDAL_RADIUS - 4);
    const marker = polarPoint(angle);
    const color = label === 'Throttle' ? THROTTLE_COLOR : BRAKE_COLOR;
    const channel = label.toLowerCase();

    return (
        <g
            className={!available ? styles.gaugeUnavailable : undefined}
            transform={`translate(${x} 27) scale(${PEDAL_GAUGE_SCALE})`}
            role={available ? 'meter' : undefined}
            aria-label={available ? `${identity} ${label}` : `${identity} ${label} unavailable`}
            aria-valuemin={available ? 0 : undefined}
            aria-valuemax={available ? 100 : undefined}
            aria-valuenow={available ? percentage : undefined}
            data-testid={`${identity}-${channel}-gauge`}
            data-state={available ? 'available' : 'unavailable'}
            data-value={available ? formatNumber(normalizedValue) : undefined}
            data-gauge-angle={available ? formatNumber(angle) : undefined}
        >
            <path className={styles.gaugeTrack} d={PEDAL_ARC_PATH} pathLength={100} />
            {available && (
                <>
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
                </>
            )}
            <text
                className={styles.gaugeLabel}
                x={PEDAL_CENTER.x}
                y="102"
                textAnchor="middle"
            >
                {label}
            </text>
            <text
                className={available ? styles.gaugeValue : styles.gaugeUnavailableValue}
                x={PEDAL_CENTER.x}
                y={PEDAL_CENTER.y}
                textAnchor="middle"
                dominantBaseline="middle"
            >
                {available ? `${percentage}%` : 'N/A'}
            </text>
        </g>
    );
};

const TelemetryPod: React.FC<{
    identity: ComparisonIdentity;
    position: TelemetryPodPosition;
    marker: PositionedTrajectoryPoint;
    gasAvailable: boolean;
    brakeAvailable: boolean;
    gearAvailable: boolean;
    gas: number | undefined;
    brake: number | undefined;
    gear: number | undefined;
}> = ({
    identity,
    position,
    marker,
    gasAvailable,
    brakeAvailable,
    gearAvailable,
    gas,
    brake,
    gear,
}) => {
    const isDriver = identity === 'driver';
    const label = isDriver ? 'Driver' : 'Expert';
    const color = isDriver ? DRIVER_COMPARISON_COLOR : EXPERT_COMPARISON_COLOR;
    const leaderEnd = getTelemetryLeaderEnd(marker, position);
    const gearValue = gearAvailable && gear !== undefined ? Math.round(gear) : 'N/A';
    const transform = `translate(${formatNumber(position.x)} ${formatNumber(position.y)})`;

    return (
        <g
            className={styles.telemetryPod}
            transform={transform}
            role="group"
            aria-label={`${label} live telemetry`}
            style={{ '--identity-color': color } as React.CSSProperties}
            data-testid={`${identity}-telemetry-pod`}
            data-card-width={formatNumber(TELEMETRY_POD_WIDTH)}
            data-card-height={formatNumber(TELEMETRY_POD_HEIGHT)}
        >
            <line
                className={styles.telemetryLeader}
                x1={marker.svgX - position.x}
                y1={marker.svgY - position.y}
                x2={leaderEnd.x}
                y2={leaderEnd.y}
                stroke={color}
                data-testid={`${identity}-telemetry-leader`}
            />
            <rect
                className={styles.telemetryPodBody}
                width={TELEMETRY_POD_WIDTH}
                height={TELEMETRY_POD_HEIGHT}
                rx={7 * TELEMETRY_POD_SCALE}
            />
            <g transform={`scale(${TELEMETRY_POD_SCALE})`}>
                <rect width="3" height={TELEMETRY_POD_BASE_HEIGHT} rx="1.5" fill={color} />
                <circle
                    className={styles.telemetryIdentityDot}
                    cx="13"
                    cy="14"
                    r="3"
                    fill={color}
                />
                <text className={styles.telemetryIdentity} x="22" y="17" fill={color}>
                    {label}
                </text>
                <g
                    className={!gearAvailable ? styles.telemetryGearUnavailable : undefined}
                    data-testid={`${identity}-gear`}
                    data-state={gearAvailable ? 'available' : 'unavailable'}
                >
                    <text
                        className={styles.telemetryGearLabel}
                        x={TELEMETRY_POD_BASE_WIDTH - 55}
                        y="17"
                    >
                        Gear
                    </text>
                    <text
                        className={styles.telemetryGearValue}
                        x={TELEMETRY_POD_BASE_WIDTH - 11}
                        y="18"
                        textAnchor="end"
                    >
                        {gearValue}
                    </text>
                </g>
                <line
                    className={styles.telemetryDivider}
                    x1="10"
                    y1="25"
                    x2={TELEMETRY_POD_BASE_WIDTH - 10}
                    y2="25"
                />
                <PedalGauge
                    available={gasAvailable}
                    identity={identity}
                    label="Throttle"
                    value={gas}
                    x={THROTTLE_GAUGE_X}
                />
                <PedalGauge
                    available={brakeAvailable}
                    identity={identity}
                    label="Brake"
                    value={brake}
                    x={BRAKE_GAUGE_X}
                />
            </g>
        </g>
    );
};

const TrackReplay: React.FC<{
    availability: DriverExpertComparisonAvailability;
    frame: ReplayFrame;
    geometry: TrackGeometry;
    height: number | string;
    filterId: string;
    cameraProgress: number;
    statusOpacity: number;
    presentationPhase: TrackPresentationPhase;
}> = ({
    availability,
    frame,
    geometry,
    height,
    filterId,
    cameraProgress,
    statusOpacity,
    presentationPhase,
}) => {
    const svgRef = React.useRef<SVGSVGElement>(null);
    const [viewportHeight, setViewportHeight] = React.useState(TRACK_VIEWBOX_HEIGHT);

    React.useLayoutEffect(() => {
        const svg = svgRef.current;
        if (!svg) return undefined;

        const updateViewportHeight = ({ width, height: renderedHeight }: {
            width: number;
            height: number;
        }) => {
            if (width <= 0 || renderedHeight <= 0) return;
            const minimumHeight = (TRACK_PADDING * 2) + 1;
            const nextHeight = Math.max(
                minimumHeight,
                (TRACK_VIEWBOX_WIDTH * renderedHeight) / width,
            );
            setViewportHeight((currentHeight) => (
                Math.abs(currentHeight - nextHeight) < 0.01 ? currentHeight : nextHeight
            ));
        };

        updateViewportHeight(svg.getBoundingClientRect());
        if (typeof ResizeObserver === 'undefined') return undefined;

        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (entry) updateViewportHeight(entry.contentRect);
        });
        observer.observe(svg);
        return () => observer.disconnect();
    }, []);

    const driverWorldMarker = geometry.project(frame.driverTrajectory);
    const expertWorldMarker = geometry.project(frame.expertTrajectory);
    const driverPath = trajectoryPath(geometry.driver);
    const expertPath = trajectoryPath(geometry.expert);
    const camera = getFollowCamera(
        driverWorldMarker,
        expertWorldMarker,
        frame.driverDirection,
        viewportHeight,
        cameraProgress,
    );
    const driverMarker = camera?.project(driverWorldMarker);
    const expertMarker = camera?.project(expertWorldMarker);
    const driverPod = driverMarker ? positionTelemetryPod(driverMarker, 'driver') : undefined;
    const expertPod = expertMarker ? positionTelemetryPod(expertMarker, 'expert') : undefined;
    const hasTrajectory = Boolean(driverPath || expertPath);
    const unavailableOffsetY = (viewportHeight - TRACK_VIEWBOX_HEIGHT) / 2;

    const renderPod = (
        identity: ComparisonIdentity,
        position: TelemetryPodPosition,
        marker: PositionedTrajectoryPoint,
    ) => (
        <TelemetryPod
            identity={identity}
            position={position}
            marker={marker}
            gasAvailable={availability.gas}
            brakeAvailable={availability.brake}
            gearAvailable={availability.gear}
            gas={identity === 'driver' ? frame.driverGas : frame.expertGas}
            brake={identity === 'driver' ? frame.driverBrake : frame.expertBrake}
            gear={identity === 'driver' ? frame.driverGear : frame.expertGear}
        />
    );

    return (
        <section
            className={styles.trackPanel}
            style={{ height: toCssSize(height) }}
            aria-label="Track replay"
        >
            <svg
                ref={svgRef}
                className={styles.trackSvg}
                viewBox={`0 0 ${TRACK_VIEWBOX_WIDTH} ${formatNumber(viewportHeight)}`}
                preserveAspectRatio="xMidYMid meet"
                role="img"
                aria-label={hasTrajectory
                    ? 'Track replay showing Driver and Expert trajectories'
                    : 'Trajectory data unavailable'}
                data-testid={hasTrajectory
                    ? 'comparison-track-map'
                    : 'comparison-trajectory-unavailable'}
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
                <rect
                    className={styles.trackViewport}
                    width={TRACK_VIEWBOX_WIDTH}
                    height={viewportHeight}
                    rx="9"
                />
                {hasTrajectory && camera ? (
                    <>
                        <g
                            className={styles.cameraLayer}
                            transform={camera.transform}
                            data-testid="comparison-camera-layer"
                            data-camera-target={camera.target}
                            data-camera-phase={presentationPhase}
                            data-camera-progress={formatNumber(cameraProgress)}
                            data-camera-anchor-x={formatNumber(camera.anchorX)}
                            data-camera-anchor-y={formatNumber(camera.anchorY)}
                            data-camera-rotation={formatNumber(
                                (camera.rotationRadians * 180) / Math.PI,
                            )}
                            data-heading-x={frame.driverDirection
                                ? formatNumber(frame.driverDirection.x)
                                : undefined}
                            data-heading-y={frame.driverDirection
                                ? formatNumber(frame.driverDirection.y)
                                : undefined}
                        >
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
                        </g>
                        <g
                            className={styles.cameraOverlay}
                            data-testid="comparison-camera-overlay"
                            data-status-visibility={statusOpacity <= 0
                                ? 'hidden'
                                : statusOpacity < 1 ? 'fading' : 'visible'}
                            aria-hidden={statusOpacity <= 0 ? true : undefined}
                            style={{ opacity: statusOpacity }}
                        >
                            {driverMarker && driverPod
                                && renderPod('driver', driverPod, driverMarker)}
                            {expertMarker && expertPod
                                && renderPod('expert', expertPod, expertMarker)}
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
                                        r={DRIVER_MARKER_HALO_RADIUS}
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
                                        r={EXPERT_MARKER_HALO_RADIUS}
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
                        </g>
                    </>
                ) : (
                    <g
                        className={styles.trackUnavailable}
                        transform={`translate(0 ${formatNumber(unavailableOffsetY)})`}
                        role="status"
                        data-testid="trajectory-unavailable"
                    >
                        <path
                            className={styles.placeholderTrack}
                            d="M 310 87 C 337 68 411 68 446 87 C 467 99 452 113 410 111 C 367 109 333 119 306 104 C 296 98 297 92 310 87 Z"
                            aria-hidden="true"
                        />
                        <text x="380" y="58" textAnchor="middle">
                            Trajectory data unavailable
                        </text>
                    </g>
                )}
            </svg>
        </section>
    );
};

export const DriverExpertComparisonGraph: React.FC<DriverExpertComparisonGraphProps> = ({
    data,
    title,
    className,
    width = '100%',
    layout,
    game,
}) => {
    const { detectedGame } = useDesktopGame();
    const comparisonGame = game === undefined ? detectedGame : game;
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
        () => selectPlottingReplay(replay, comparisonGame),
        [comparisonGame, replay],
    );
    const availability = React.useMemo(
        () => getDriverExpertComparisonAvailability(data, comparisonGame),
        [comparisonGame, data],
    );
    const hasAnyComparison = Object.values(availability).some(Boolean);
    const trajectoryHeight = layout?.trajectoryHeight ?? 300;
    const rootClassName = [styles.root, className].filter(Boolean).join(' ');
    const rootStyle = { width: toCssSize(width) } as React.CSSProperties;
    const replayDurationMs = replay?.durationMs ?? 0;
    const geometry = React.useMemo(() => createTrackGeometry(plottingReplay), [plottingReplay]);
    const hasTrajectory = geometry.driver.length > 0 || geometry.expert.length > 0;
    const timeline = useReplayTimeline(plottingReplay, replayDurationMs, hasTrajectory);
    const { elapsedTimeMs } = timeline;
    const frame = React.useMemo(
        () => buildReplayFrame(plottingReplay, elapsedTimeMs),
        [elapsedTimeMs, plottingReplay],
    );
    const reactId = React.useId();
    const filterId = React.useMemo(() => `driver-expert-${reactId.replace(/:/g, '')}`, [reactId]);
    const isComplete = timeline.isComplete;
    const replayStatus = !replay ? 'No data' : isComplete ? 'Replay complete' : 'Replaying';

    return (
        <section
            className={rootClassName}
            style={rootStyle}
            aria-label={title ?? 'Segment comparison replay'}
            data-testid="driver-expert-comparison"
        >
            <header className={styles.header}>
                <div className={styles.titleBlock}>
                    <h2 className={styles.title}>{title ?? 'Segment comparison replay'}</h2>
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
                    availability={availability}
                    frame={frame}
                    geometry={geometry}
                    height={trajectoryHeight}
                    filterId={filterId}
                    cameraProgress={timeline.cameraProgress}
                    statusOpacity={timeline.statusOpacity}
                    presentationPhase={timeline.presentationPhase}
                />
            </div>
        </section>
    );
};

export const driverExpertComparisonOverlayRenderer: AiOverlayRenderer<DriverExpertComparisonSnapshot> = {
    componentType: 'driver_expert_comparison',
    validateSnapshot: (snapshot): snapshot is DriverExpertComparisonSnapshot => (
        isOverlayRecord(snapshot)
        && isOverlayNonEmptyString(snapshot.title)
        && Boolean(normalizeDriverExpertComparisonData(snapshot.comparison))
        && (
            snapshot.game === undefined
            || snapshot.game === null
            || snapshot.game === 'ac'
            || snapshot.game === 'acc'
            || snapshot.game === 'iracing'
        )
    ),
    renderOverlay: (snapshot, status) => {
        if (status === 'folded') return snapshot.title;
        return (
            <DriverExpertComparisonGraph
                className={status === 'full_size'
                    ? 'floating-pill-comparison floating-pill-comparison--full-size'
                    : 'floating-pill-comparison'}
                data={snapshot.comparison}
                game={snapshot.game}
                title={snapshot.title}
                layout={{ trajectoryHeight: status === 'full_size' ? '100%' : 280 }}
            />
        );
    },
    dimensions: {
        expanded: { width: 760, height: 500 },
        folded: { width: 360, height: 58 },
        full_size: { width: 760, height: 500 },
    },
};

export default DriverExpertComparisonGraph;
