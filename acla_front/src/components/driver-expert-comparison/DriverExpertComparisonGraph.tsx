import React from 'react';
import styles from './DriverExpertComparisonGraph.module.css';

export const DRIVER_COMPARISON_COLOR = '#00e676';
export const EXPERT_COMPARISON_COLOR = '#448aff';

const THROTTLE_COLOR = '#21e58b';
const BRAKE_COLOR = '#ff4d62';
const REPLAY_DURATION_MS = 3000;
const TRACK_VIEWBOX_WIDTH = 760;
const TRACK_VIEWBOX_HEIGHT = 220;
const TRACK_PADDING = 28;
const PEDAL_START_ANGLE = -60;
const PEDAL_SWEEP_ANGLE = 60;
const PEDAL_CENTER = { x: 66, y: 58 };
const PEDAL_RADIUS = 42;

export interface DriverExpertTrajectoryPoint {
    x: number;
    z: number;
}

export interface DriverExpertComparisonSample {
    progress: number;
    trackPosition?: number;
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

type OptionalScalarKey = Exclude<
    keyof DriverExpertComparisonSample,
    'progress' | 'trackPosition' | 'driverTrajectory' | 'expertTrajectory'
>;
type ContinuousScalarKey = 'driverGas' | 'expertGas' | 'driverBrake' | 'expertBrake';
type GearKey = 'driverGear' | 'expertGear';
type TrajectoryKey = 'driverTrajectory' | 'expertTrajectory';

interface ReplayFrame {
    progress: number;
    driverGas?: number;
    expertGas?: number;
    driverBrake?: number;
    expertBrake?: number;
    driverGear?: number;
    expertGear?: number;
    driverTrajectory?: DriverExpertTrajectoryPoint;
    expertTrajectory?: DriverExpertTrajectoryPoint;
}

interface PositionedTrajectoryPoint extends DriverExpertTrajectoryPoint {
    svgX: number;
    svgY: number;
}

interface TrackGeometry {
    driver: PositionedTrajectoryPoint[];
    expert: PositionedTrajectoryPoint[];
    project: (point: DriverExpertTrajectoryPoint | undefined) => PositionedTrajectoryPoint | undefined;
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

const normalizeTrajectory = (value: unknown): DriverExpertTrajectoryPoint | undefined => {
    if (!isRecord(value)) return undefined;
    const x = finiteNumber(value.x);
    const z = finiteNumber(value.z);
    return x === undefined || z === undefined ? undefined : { x, z };
};

export const normalizeDriverExpertComparisonData = (
    value: unknown,
): DriverExpertComparisonData | undefined => {
    if (!isRecord(value) || !Array.isArray(value.samples)) return undefined;

    const samples = value.samples.flatMap((sample): DriverExpertComparisonSample[] => {
        if (!isRecord(sample)) return [];
        const progress = finiteNumber(sample.progress);
        if (progress === undefined) return [];

        const normalized: DriverExpertComparisonSample = {
            progress: clamp(progress, 0, 100),
        };
        const trackPosition = finiteNumber(sample.trackPosition);
        if (trackPosition !== undefined) {
            normalized.trackPosition = clamp(trackPosition, 0, 1);
        }
        const driverTrajectory = normalizeTrajectory(sample.driverTrajectory);
        const expertTrajectory = normalizeTrajectory(sample.expertTrajectory);
        if (driverTrajectory) normalized.driverTrajectory = driverTrajectory;
        if (expertTrajectory) normalized.expertTrajectory = expertTrajectory;

        const scalarKeys: OptionalScalarKey[] = [
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
        return [normalized];
    });

    return { samples };
};

const hasScalarPair = (
    samples: readonly DriverExpertComparisonSample[],
    driverKey: OptionalScalarKey,
    expertKey: OptionalScalarKey,
): boolean => samples.some((sample) => (
    finiteNumber(sample[driverKey]) !== undefined
    && finiteNumber(sample[expertKey]) !== undefined
));

export const getDriverExpertComparisonAvailability = (
    data: DriverExpertComparisonData | null | undefined,
): DriverExpertComparisonAvailability => {
    const samples = data && Array.isArray(data.samples) ? data.samples : [];
    return {
        trajectory: samples.some((sample) => (
            normalizeTrajectory(sample.driverTrajectory) !== undefined
            && normalizeTrajectory(sample.expertTrajectory) !== undefined
        )),
        gas: hasScalarPair(samples, 'driverGas', 'expertGas'),
        brake: hasScalarPair(samples, 'driverBrake', 'expertBrake'),
        gear: hasScalarPair(samples, 'driverGear', 'expertGear'),
    };
};

export const hasComparableDriverExpertData = (
    data: DriverExpertComparisonData | null | undefined,
): boolean => Object.values(getDriverExpertComparisonAvailability(data)).some(Boolean);

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

const orderSamples = (
    samples: readonly DriverExpertComparisonSample[],
): DriverExpertComparisonSample[] => samples
    .map((sample, index) => ({ sample, index }))
    .sort((left, right) => (
        left.sample.progress - right.sample.progress || left.index - right.index
    ))
    .map(({ sample }) => sample);

const getFrameIndexes = (
    samples: readonly DriverExpertComparisonSample[],
    progress: number,
): { lower: number; upper: number } => {
    if (samples.length <= 1) return { lower: 0, upper: 0 };
    let lower = 0;
    while (lower + 1 < samples.length && samples[lower + 1].progress <= progress) {
        lower += 1;
    }
    return { lower, upper: Math.min(samples.length - 1, lower + 1) };
};

const interpolateScalar = (
    samples: readonly DriverExpertComparisonSample[],
    key: ContinuousScalarKey,
    progress: number,
    lower: number,
    upper: number,
): number | undefined => {
    let previousIndex = lower;
    while (previousIndex >= 0 && finiteNumber(samples[previousIndex][key]) === undefined) {
        previousIndex -= 1;
    }
    let nextIndex = upper;
    while (nextIndex < samples.length && finiteNumber(samples[nextIndex][key]) === undefined) {
        nextIndex += 1;
    }
    const previous = previousIndex >= 0 ? finiteNumber(samples[previousIndex][key]) : undefined;
    const next = nextIndex < samples.length ? finiteNumber(samples[nextIndex][key]) : undefined;
    if (previous === undefined) return next;
    if (next === undefined) return previous;
    const span = samples[nextIndex].progress - samples[previousIndex].progress;
    if (span <= 0) return next;
    const ratio = clamp((progress - samples[previousIndex].progress) / span, 0, 1);
    return previous + ((next - previous) * ratio);
};

const interpolateTrajectory = (
    samples: readonly DriverExpertComparisonSample[],
    key: TrajectoryKey,
    progress: number,
    lower: number,
    upper: number,
): DriverExpertTrajectoryPoint | undefined => {
    let previousIndex = lower;
    while (previousIndex >= 0 && !normalizeTrajectory(samples[previousIndex][key])) {
        previousIndex -= 1;
    }
    let nextIndex = upper;
    while (nextIndex < samples.length && !normalizeTrajectory(samples[nextIndex][key])) {
        nextIndex += 1;
    }
    const previous = previousIndex >= 0
        ? normalizeTrajectory(samples[previousIndex][key])
        : undefined;
    const next = nextIndex < samples.length
        ? normalizeTrajectory(samples[nextIndex][key])
        : undefined;
    if (!previous) return next;
    if (!next) return previous;
    const span = samples[nextIndex].progress - samples[previousIndex].progress;
    if (span <= 0) return next;
    const ratio = clamp((progress - samples[previousIndex].progress) / span, 0, 1);
    return {
        x: previous.x + ((next.x - previous.x) * ratio),
        z: previous.z + ((next.z - previous.z) * ratio),
    };
};

const steppedGear = (
    samples: readonly DriverExpertComparisonSample[],
    key: GearKey,
    lower: number,
): number | undefined => {
    for (let index = lower; index >= 0; index -= 1) {
        const value = finiteNumber(samples[index][key]);
        if (value !== undefined) return value;
    }
    for (let index = lower + 1; index < samples.length; index += 1) {
        const value = finiteNumber(samples[index][key]);
        if (value !== undefined) return value;
    }
    return undefined;
};

const buildReplayFrame = (
    samples: readonly DriverExpertComparisonSample[],
    playhead: number,
): ReplayFrame => {
    if (!samples.length) return { progress: 0 };
    const firstProgress = samples[0].progress;
    const lastProgress = samples[samples.length - 1].progress;
    const progress = firstProgress + ((lastProgress - firstProgress) * clamp(playhead, 0, 1));
    const { lower, upper } = getFrameIndexes(samples, progress);
    return {
        progress,
        driverGas: interpolateScalar(samples, 'driverGas', progress, lower, upper),
        expertGas: interpolateScalar(samples, 'expertGas', progress, lower, upper),
        driverBrake: interpolateScalar(samples, 'driverBrake', progress, lower, upper),
        expertBrake: interpolateScalar(samples, 'expertBrake', progress, lower, upper),
        driverGear: steppedGear(samples, 'driverGear', lower),
        expertGear: steppedGear(samples, 'expertGear', lower),
        driverTrajectory: interpolateTrajectory(
            samples,
            'driverTrajectory',
            progress,
            lower,
            upper,
        ),
        expertTrajectory: interpolateTrajectory(
            samples,
            'expertTrajectory',
            progress,
            lower,
            upper,
        ),
    };
};

const useReplayPlayhead = (samples: readonly DriverExpertComparisonSample[]): number => {
    const shouldFinishImmediately = samples.length <= 1 || prefersReducedMotion();
    const [playhead, setPlayhead] = React.useState(shouldFinishImmediately ? 1 : 0);

    React.useEffect(() => {
        if (samples.length <= 1 || prefersReducedMotion()) {
            setPlayhead(1);
            return undefined;
        }

        let animationFrame: number | null = null;
        let startedAt: number | null = null;
        setPlayhead(0);

        const animate = (timestamp: number) => {
            if (startedAt === null) startedAt = timestamp;
            const nextPlayhead = clamp((timestamp - startedAt) / REPLAY_DURATION_MS, 0, 1);
            setPlayhead(nextPlayhead);
            if (nextPlayhead < 1) {
                animationFrame = window.requestAnimationFrame(animate);
            } else {
                animationFrame = null;
            }
        };

        animationFrame = window.requestAnimationFrame(animate);
        return () => {
            if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
        };
    }, [samples]);

    return playhead;
};

const createTrackGeometry = (
    samples: readonly DriverExpertComparisonSample[],
): TrackGeometry => {
    const driverPoints = samples.flatMap((sample) => {
        const point = normalizeTrajectory(sample.driverTrajectory);
        return point ? [point] : [];
    });
    const expertPoints = samples.flatMap((sample) => {
        const point = normalizeTrajectory(sample.expertTrajectory);
        return point ? [point] : [];
    });
    const allPoints = [...driverPoints, ...expertPoints];
    if (!allPoints.length) {
        return { driver: [], expert: [], project: () => undefined };
    }

    const minX = Math.min(...allPoints.map(({ x }) => x));
    const maxX = Math.max(...allPoints.map(({ x }) => x));
    const minZ = Math.min(...allPoints.map(({ z }) => z));
    const maxZ = Math.max(...allPoints.map(({ z }) => z));
    const spanX = maxX - minX;
    const spanZ = maxZ - minZ;
    const usableWidth = TRACK_VIEWBOX_WIDTH - (TRACK_PADDING * 2);
    const usableHeight = TRACK_VIEWBOX_HEIGHT - (TRACK_PADDING * 2);
    const scaleCandidates = [
        spanX > 0 ? usableWidth / spanX : Number.POSITIVE_INFINITY,
        spanZ > 0 ? usableHeight / spanZ : Number.POSITIVE_INFINITY,
    ];
    const finiteScales = scaleCandidates.filter(Number.isFinite);
    const scale = finiteScales.length ? Math.min(...finiteScales) : 1;
    const centerX = (minX + maxX) / 2;
    const centerZ = (minZ + maxZ) / 2;

    const project = (
        point: DriverExpertTrajectoryPoint | undefined,
    ): PositionedTrajectoryPoint | undefined => {
        const normalized = normalizeTrajectory(point);
        if (!normalized) return undefined;
        return {
            ...normalized,
            svgX: (TRACK_VIEWBOX_WIDTH / 2) + ((normalized.x - centerX) * scale),
            svgY: (TRACK_VIEWBOX_HEIGHT / 2) - ((normalized.z - centerZ) * scale),
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
                            data-z={formatNumber(driverMarker.z)}
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
                            data-z={formatNumber(expertMarker.z)}
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
    const rawSamples = data?.samples;
    const samples = React.useMemo(
        () => orderSamples(Array.isArray(rawSamples) ? rawSamples : []),
        [rawSamples],
    );
    const availability = React.useMemo(() => getDriverExpertComparisonAvailability(data), [data]);
    const hasAnyComparison = Object.values(availability).some(Boolean);
    const chartHeight = layout?.chartHeight ?? 190;
    const trajectoryHeight = layout?.trajectoryHeight ?? 220;
    const rootClassName = [styles.root, className].filter(Boolean).join(' ');
    const rootStyle = {
        width: toCssSize(width),
        '--driver-expert-min-column-width': toCssSize(layout?.minColumnWidth ?? 260),
    } as React.CSSProperties;
    const playhead = useReplayPlayhead(samples);
    const frame = React.useMemo(() => buildReplayFrame(samples, playhead), [playhead, samples]);
    const geometry = React.useMemo(() => createTrackGeometry(samples), [samples]);
    const reactId = React.useId();
    const filterId = React.useMemo(() => `driver-expert-${reactId.replace(/:/g, '')}`, [reactId]);
    const isComplete = samples.length <= 1 || playhead >= 1;
    const progressPercentage = Math.round(clamp(frame.progress, 0, 100));
    const replayStatus = !samples.length ? 'No data' : isComplete ? 'Replay complete' : 'Replaying';

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
                        aria-label="Segment progress"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={progressPercentage}
                        data-testid="replay-progress"
                    >
                        {progressPercentage}%
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
