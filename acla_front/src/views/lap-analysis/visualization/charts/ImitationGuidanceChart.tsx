import React, { useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, Text, Box, Grid, Badge, Separator, Flex, Button } from '@radix-ui/themes';
import { AnalysisContext } from '../../analysis-context';
import { VisualizationProps } from '../VisualizationRegistry';
import { visualizationController } from '../VisualizationController';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

// New backend response structure
interface SequencePrediction {
    step: number;
    time_delta_ms?: number | string;
    time_delta_milliseconds?: number | string;
    all_targets: Record<string, unknown>;
    [key: string]: unknown;
}

interface CurrentSituation {
    speed?: string;
    track_position?: string;
    racing_line?: string;
    tire_grip?: string;
    [key: string]: unknown;
}

interface ContextualInfo {
    track_sector?: string;
    weather_impact?: string;
    optimal_speed_estimate?: string;
    [key: string]: unknown;
}

interface GuidanceResponse {
    status: "success" | "error";
    timestamp: string;
    current_situation?: CurrentSituation;
    sequence_predictions: SequencePrediction[];
    contextual_info?: ContextualInfo;
    [key: string]: unknown;
}

interface GuidanceData {
    message?: string;
    guidance_result: GuidanceResponse;
    timestamp?: string;
    success?: boolean;
    [key: string]: unknown;
}

interface TelemetryData {
    speed?: number;
    speed_kmh?: number;
    speed_mph?: number;
    braking?: number;
    steering?: number;
    throttle?: number;
    gas?: number;
    brake?: number;
    steer_angle?: number;
    steering_angle?: number;
    gear?: number;
    rpm?: number;
    Physics_gas?: number;
    Physics_brake?: number;
    Physics_steer_angle?: number;
    Physics_gear?: number;
    Physics_speed_kmh?: number;
    [key: string]: any;
}

type Keyframe = {
    t: number;
    throttle: number;
    brake: number;
    steering: number;
    gear?: number;
    target_speed?: number;
    normalized_position?: number;
    action?: string;
};

const NORMALIZED_POSITION_KEYS = [
    'Graphics_normalized_car_position',
    'graphics_normalized_car_position',
    'normalized_car_position',
    'car_position'
] as const;

const parseGuidanceTimestamp = (value?: string): number | null => {
    if (!value) {
        return null;
    }

    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
};

const PREFETCH_BUFFER_MS = 400;
const DEFAULT_FETCH_DURATION_MS = 600;
const PROGRESS_EPSILON = 0.01;
const MIN_TIMELINE_INCREMENT = 0.01;
const AUTO_START_HEADROOM_SECONDS = 1;
const MAX_LATENCY_SAMPLES = 10;

// Lightweight logger so we can collect runtime details from the user.
const debugLog = () => undefined;

const getNormalizedPositionFromRecord = (record?: Record<string, unknown> | null): number | undefined => {
    if (!record) return undefined;
    return getNumericFromTargets(record, Array.from(NORMALIZED_POSITION_KEYS));
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
const clampMinus1To1 = (n: number) => Math.max(-1, Math.min(1, n));

const toFiniteNumber = (value: unknown): number | undefined => {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string') {
        const parsed = parseFloat(value);
        if (!Number.isNaN(parsed) && Number.isFinite(parsed)) return parsed;
    }
    return undefined;
};

const clampSteering = (raw?: number): number => {
    if (raw == null || Number.isNaN(raw)) return 0;
    return clampMinus1To1(raw);
};

const normalizePedalInput = (value?: number | null): number => {
    if (value == null || Number.isNaN(value) || !Number.isFinite(value)) {
        return 0;
    }

    return clamp01(value);
};

const toGear = (value?: number): number | undefined => {
    if (value == null || Number.isNaN(value)) return undefined;
    const rounded = Math.round(value);
    if (Math.abs(value - rounded) < 0.3) return rounded;
    return rounded;
};

const getNumericFromTargets = (targets: Record<string, unknown>, keys: string[]): number | undefined => {
    for (const key of keys) {
        if (key in targets) {
            const candidate = toFiniteNumber(targets[key]);
            if (candidate != null) return candidate;
        }
    }
    return undefined;
};

const TIME_DELTA_KEYS = ['time_delta_milliseconds', 'time_delta_ms', 'time_delta_ms', 'time_delta'] as const;

const extractDurationSeconds = (record: Record<string, unknown>, keys: readonly string[]): number | null => {
    for (const key of keys) {
        if (!(key in record)) {
            continue;
        }

        const numeric = toFiniteNumber((record as Record<string, unknown>)[key]);
        if (numeric == null) {
            continue;
        }

        const lowerKey = key.toLowerCase();
        const treatAsMillisecond =
            lowerKey.includes('millisecond') ||
            lowerKey.endsWith('_ms');

        const seconds = treatAsMillisecond ? numeric / 1000 : numeric;
        if (seconds > 0) {
            return seconds;
        }
    }

    return null;
};

const keyframeFromPrediction = (prediction: SequencePrediction, timelineTime = 0): Keyframe => {
    const targets = (prediction.all_targets ?? {}) as Record<string, unknown>;
    const rawThrottle = toFiniteNumber(targets['Physics_gas']);
    const rawBrake = toFiniteNumber(targets['Physics_brake']);
    const rawSteering = toFiniteNumber(targets['Physics_steer_angle']);
    const throttle = normalizePedalInput(rawThrottle);
    const brake = normalizePedalInput(rawBrake);
    const steering = clampSteering(rawSteering);
    const gear = toGear(toFiniteNumber(targets['Physics_gear']));
    const targetSpeed = getNumericFromTargets(targets, ['Physics_speed_kmh', 'speed_kmh', 'speed']);
    const normalizedPosition = getNormalizedPositionFromRecord(targets);
    const predictedActionCandidate = (prediction as { action?: unknown }).action;
    const predictedAction = typeof predictedActionCandidate === 'string' ? predictedActionCandidate : undefined;
    const action = predictedAction && predictedAction.trim().length > 0
        ? predictedAction
        : undefined;

    return {
        t: Number.isFinite(timelineTime) && timelineTime >= 0 ? timelineTime : 0,
        throttle,
        brake,
        steering,
        gear,
        target_speed: targetSpeed,
        normalized_position: normalizedPosition,
        action,
    };
};

const keyframeFromTelemetry = (telemetry?: TelemetryData | null): Keyframe | null => {
    if (!telemetry) return null;
    const rawThrottle = toFiniteNumber(telemetry.Physics_gas ?? telemetry.throttle ?? telemetry.gas);
    const rawBrake = toFiniteNumber(telemetry.Physics_brake ?? telemetry.braking ?? telemetry.brake);
    const rawSteering = toFiniteNumber(
        telemetry.Physics_steer_angle ?? telemetry.steering ?? telemetry.steer_angle ?? telemetry.steering_angle
    );
    const throttle = normalizePedalInput(rawThrottle);
    const brake = normalizePedalInput(rawBrake);
    const steering = clampSteering(rawSteering);
    const gear = toGear(toFiniteNumber(telemetry.Physics_gear ?? telemetry.gear));
    const targetSpeed = toFiniteNumber(
        telemetry.Physics_speed_kmh ?? telemetry.speed ?? telemetry.speed_kmh ?? telemetry.speed_mph
    );
    const normalizedPosition = getNormalizedPositionFromRecord(telemetry as unknown as Record<string, unknown>);

    return { t: 0, throttle, brake, steering, gear, target_speed: targetSpeed, normalized_position: normalizedPosition };
};

const buildKeyframesFromPredictions = (predictions: SequencePrediction[]): Keyframe[] => {
    if (!predictions || predictions.length === 0) {
        return [];
    }

    const sorted = [...predictions].sort((a, b) => a.step - b.step);
    let cumulativeTime = 0;
    let loggedMissingDelta = false;

    const frames = sorted
        .map((prediction, index) => {
            const record = prediction as Record<string, unknown>;
            const targetsRecord = (prediction.all_targets ?? {}) as Record<string, unknown>;
            const deltaSecondsFromPrediction = extractDurationSeconds(record, TIME_DELTA_KEYS);
            const deltaSecondsFromTargets = extractDurationSeconds(targetsRecord, TIME_DELTA_KEYS);

            let deltaSeconds = deltaSecondsFromPrediction ?? deltaSecondsFromTargets;

            if (deltaSeconds == null || deltaSeconds <= 0) {
                if (!loggedMissingDelta) {
                    loggedMissingDelta = true;
                }
                return null;
            }

            cumulativeTime = index === 0 ? deltaSeconds : cumulativeTime + deltaSeconds;

            return keyframeFromPrediction(prediction, cumulativeTime);
        })
        .filter((frame): frame is Keyframe => frame != null && Number.isFinite(frame.t));

    return frames;
};

const ImitationGuidanceChart: React.FC<VisualizationProps> = (props) => {
    const analysisContext = useContext(AnalysisContext);
    const [guidanceData, setGuidanceData] = useState<GuidanceData | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [autoUpdate, setAutoUpdate] = useState<boolean>(false);
    const [progress, setProgress] = useState<number>(0); // seconds along timeline
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const animationFrameRef = useRef<number | null>(null);
    const playbackStartRef = useRef<number | null>(null);
    const fetchDurationMsRef = useRef<number>(DEFAULT_FETCH_DURATION_MS);
    const prefetchTriggeredRef = useRef<boolean>(false);
    const progressRef = useRef<number>(0);
    const playbackDurationRef = useRef<number>(0);
    const lastFetchStartedAtRef = useRef<number | null>(null);
    const autoUpdateRef = useRef<boolean>(autoUpdate);
    const latencySamplesRef = useRef<number[]>([]);

    // Extract track and car information from session data
    const trackName = analysisContext.recordedSessioStaticsData?.track || 'Unknown Track';
    const carName = analysisContext.recordedSessioStaticsData?.car_model || 'Unknown Car';
    const liveData = analysisContext.liveData as TelemetryData;

    const liveDataRef = useRef<TelemetryData | null>(liveData ?? null);
    const trackNameRef = useRef(trackName);
    const carNameRef = useRef(carName);
    const requestInFlightRef = useRef(false);

    const keyframes: Keyframe[] = useMemo(() => {
        const predictions = guidanceData?.guidance_result?.sequence_predictions ?? [];
        const frames = buildKeyframesFromPredictions(predictions);

        if (frames.length === 0) {
            const currentFrame = keyframeFromTelemetry(liveData);
            return currentFrame ? [{ ...currentFrame, t: 0 }] : [];
        }

        return frames;
    }, [guidanceData, liveData]);

    const totalDuration = useMemo(() => (keyframes.length ? keyframes[keyframes.length - 1].t : 0), [keyframes]);

    useEffect(() => {
        autoUpdateRef.current = autoUpdate;
    }, [autoUpdate]);

    function getInterpolatedValues(timeS: number) {
        if (keyframes.length === 0) {
            return {
                throttle: 0,
                brake: 0,
                steering: 0,
                gear: undefined as number | undefined,
                target_speed: undefined as number | undefined,
                action: undefined as string | undefined,
            };
        }

        let closest = keyframes[0];
        let smallestDiff = Math.abs(timeS - closest.t);

        for (let i = 1; i < keyframes.length; i++) {
            const frame = keyframes[i];
            const diff = Math.abs(timeS - frame.t);
            if (diff < smallestDiff) {
                closest = frame;
                smallestDiff = diff;
                if (smallestDiff <= MIN_TIMELINE_INCREMENT) {
                    break;
                }
            }
        }

        return {
            throttle: clamp01(closest.throttle),
            brake: clamp01(closest.brake),
            steering: clampMinus1To1(closest.steering),
            gear: closest.gear,
            target_speed: closest.target_speed,
            action: closest.action,
        };
    }

    // Function to call the imitation learning guidance API
    useEffect(() => {
        liveDataRef.current = liveData ?? null;
    }, [liveData]);

    useEffect(() => {
        trackNameRef.current = trackName;
    }, [trackName]);

    useEffect(() => {
        carNameRef.current = carName;
    }, [carName]);

    const fetchGuidance = useCallback(async () => {
        const currentLiveData = liveDataRef.current;
        const currentTrackName = trackNameRef.current;
        const currentCarName = carNameRef.current;
        let latestServerLatencyMs = 0;
        let latestAverageLatencySeconds = fetchDurationMsRef.current / 1000;

        if (!currentLiveData || Object.keys(currentLiveData).length === 0) {
            setError('No live telemetry data available');
            return;
        }

        if (requestInFlightRef.current) {
            return;
        }

        requestInFlightRef.current = true;
        setLoading(true);
        setError(null);
        const fetchStart = typeof performance !== 'undefined' ? performance.now() : Date.now();
        lastFetchStartedAtRef.current = fetchStart;

        try {
            const response = await apiService.post('/racing-session/imitation-learning-guidance', {
                current_telemetry: currentLiveData,
                track_name: currentTrackName,
                car_name: currentCarName
            });
            if (!response.data) {
                setError('No response data received');
                return;
            }

            const raw = response.data as GuidanceData;
            const result = raw?.guidance_result;
            console.log('Imitation learning guidance response:', raw);
            if (!result) {
                setError('Malformed response: missing guidance_result');
                return;
            }

            if (result.status !== 'success') {
                setError('Failed to get guidance: API returned error status');
                return;
            }

            const frames = buildKeyframesFromPredictions(result.sequence_predictions ?? []);
            const finalFrameTime = frames.length > 0 ? frames[frames.length - 1].t : 0;
            const nowPerf = typeof performance !== 'undefined' ? performance.now() : Date.now();
            const fetchStartedAt = lastFetchStartedAtRef.current;
            const elapsedMs = fetchStartedAt != null ? nowPerf - fetchStartedAt : fetchDurationMsRef.current;
            const measuredLatencySeconds = Math.max(0, elapsedMs / 1000);

            const guidanceTimestampMs = parseGuidanceTimestamp(result.timestamp ?? raw.timestamp);
            const nowWallMs = Date.now();
            latestServerLatencyMs = guidanceTimestampMs != null ? Math.max(0, nowWallMs - guidanceTimestampMs) : 0;
            const latencySampleSeconds = Math.max(measuredLatencySeconds, latestServerLatencyMs / 1000);
            latencySamplesRef.current.push(latencySampleSeconds);
            if (latencySamplesRef.current.length > MAX_LATENCY_SAMPLES) {
                latencySamplesRef.current.shift();
            }
            const totalLatency = latencySamplesRef.current.reduce((acc, value) => acc + value, 0);
            latestAverageLatencySeconds = latencySamplesRef.current.length > 0
                ? totalLatency / latencySamplesRef.current.length
                : latencySampleSeconds;
            const effectiveLatencySeconds = latestAverageLatencySeconds;
            const autoMode = autoUpdateRef.current;
            // When auto-updating, advance the playback to account for time spent waiting on the response.
            // Add a small headroom so the driver can see the action slightly ahead of real time.
            const latencyAdjustedStart = effectiveLatencySeconds + AUTO_START_HEADROOM_SECONDS;
            const initialProgress = autoMode ? Math.min(finalFrameTime, latencyAdjustedStart) : 0;
            const safeProgress = Number.isFinite(initialProgress) && initialProgress >= 0 ? initialProgress : 0;

            setGuidanceData({
                message: raw.message,
                success: raw.success,
                timestamp: raw.timestamp ?? result.timestamp,
                guidance_result: result,
            });
            setProgress(safeProgress);
            progressRef.current = safeProgress;
            playbackDurationRef.current = finalFrameTime;
            playbackStartRef.current = nowPerf - safeProgress * 1000;
            prefetchTriggeredRef.current = false;
            const shouldPlay = autoMode || (!autoMode && frames.length > 0);
            setIsPlaying(shouldPlay);

            const guidanceText = extractGuidanceText(raw, result);
            if (guidanceText) {
                analysisContext.sendGuidanceToChat(guidanceText);
            }
        } catch (err: any) {
            console.error('Imitation learning guidance error:', err);
            setError('API call failed: ' + (err.response?.data?.message || err.message));
            prefetchTriggeredRef.current = false;
        } finally {
            setLoading(false);
            requestInFlightRef.current = false;
            const fetchEndPerf = typeof performance !== 'undefined' ? performance.now() : Date.now();
            const measured = fetchEndPerf - fetchStart;
            if (Number.isFinite(measured) && measured > 0) {
                const averageLatencyMs = latestAverageLatencySeconds * 1000;
                fetchDurationMsRef.current = Math.max(measured, latestServerLatencyMs, averageLatencyMs);
            }
        }
    }, [analysisContext]);

    // Keep human-readable copy fully backend-driven.
    const extractGuidanceText = (raw: GuidanceData, guidanceResult: GuidanceResponse): string | null => {
        const topLevelMessage = typeof raw.message === 'string' ? raw.message.trim() : '';
        if (topLevelMessage) {
            return topLevelMessage;
        }

        const resultRecord = guidanceResult as Record<string, unknown>;
        const resultMessage = typeof resultRecord.message === 'string' ? resultRecord.message.trim() : '';
        if (resultMessage) {
            return resultMessage;
        }

        return null;
    };

    // Toggle auto-update mode
    const toggleAutoUpdate = useCallback(() => {
        setAutoUpdate(prev => {
            const next = !prev;
            if (next) {
                setIsPlaying(true);
                const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
                playbackStartRef.current = now - (progressRef.current ?? 0) * 1000;
            }
            return next;
        });
    }, []);

    const setAutoUpdateMode = useCallback((enabled: boolean) => {
        setAutoUpdate(enabled);
        if (enabled) {
            setIsPlaying(true);
            const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
            playbackStartRef.current = now - (progressRef.current ?? 0) * 1000;
        } else {
            setIsPlaying(false);
        }
    }, []);

    useEffect(() => {
        visualizationController.registerInstanceControls(
            props.id,
            'imitation-guidance-chart',
            [
                {
                    name: 'refresh_once',
                    description: 'Refresh the imitation guidance data once'
                },
                {
                    name: 'set_auto_update',
                    description: 'Toggle automatic updating of imitation guidance data',
                    params: { enabled: 'boolean' }
                }
            ],
            {
                refresh_once: async () => {
                    await fetchGuidance();
                    return {
                        autoUpdate,
                        status: 'guidance refreshed'
                    };
                },
                set_auto_update: ({ enabled } = {}) => {
                    const nextEnabled = typeof enabled === 'boolean' ? enabled : true;
                    setAutoUpdateMode(nextEnabled);
                    return {
                        autoUpdate: nextEnabled
                    };
                }
            }
        );

        return () => {
            visualizationController.unregisterInstanceControls(props.id);
        };
    }, [props.id, fetchGuidance, autoUpdate, setAutoUpdateMode]);

    useEffect(() => {
        if (autoUpdate && !guidanceData && !requestInFlightRef.current) {
            fetchGuidance();
        }
    }, [autoUpdate, guidanceData, fetchGuidance]);

    useEffect(() => {
        progressRef.current = progress;
    }, [progress]);

    useEffect(() => {
        playbackDurationRef.current = totalDuration;
    }, [totalDuration]);

    useEffect(() => {
        if (isPlaying) {
            const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
            playbackStartRef.current = now - (progressRef.current ?? 0) * 1000;
            prefetchTriggeredRef.current = false;
        }
    }, [isPlaying]);

    useEffect(() => {
        if (!(autoUpdate || isPlaying)) {
            if (animationFrameRef.current != null) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
            return;
        }

        const currentDuration = playbackDurationRef.current;
        if (!autoUpdate && isPlaying && (!Number.isFinite(currentDuration) || currentDuration <= PROGRESS_EPSILON)) {
            setIsPlaying(false);
            return;
        }

        const step = () => {
            const duration = playbackDurationRef.current;
            const start = playbackStartRef.current;

            if (duration > 0 && start != null) {
                const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
                const elapsedSeconds = Math.max(0, (now - start) / 1000);
                const clamped = Math.min(duration, elapsedSeconds);

                if (Math.abs(clamped - progressRef.current) > PROGRESS_EPSILON) {
                    setProgress(clamped);
                    progressRef.current = clamped;
                }

                if (autoUpdate && !prefetchTriggeredRef.current) {
                    const remainingMs = Math.max(0, (duration - clamped) * 1000);
                    const leadTimeMs = fetchDurationMsRef.current + PREFETCH_BUFFER_MS;

                    if (remainingMs <= leadTimeMs && !requestInFlightRef.current) {
                        prefetchTriggeredRef.current = true;
                        fetchGuidance();
                    }
                }

                if (!autoUpdate && duration > 0 && clamped >= duration - PROGRESS_EPSILON) {
                    setIsPlaying(false);
                    animationFrameRef.current = null;
                    return;
                }
            }

            animationFrameRef.current = requestAnimationFrame(step);
        };

        animationFrameRef.current = requestAnimationFrame(step);

        return () => {
            if (animationFrameRef.current != null) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
        };
    }, [autoUpdate, fetchGuidance, isPlaying]);

    useEffect(() => {
        return () => {
            if (animationFrameRef.current != null) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
        };
    }, []);

    const onScrub = useCallback((v: number) => {
        const clamped = Math.max(0, Math.min(totalDuration || 0, v));
        setProgress(clamped);
        progressRef.current = clamped;
        const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
        playbackStartRef.current = now - clamped * 1000;
        prefetchTriggeredRef.current = false;
        setIsPlaying(false);
    }, [totalDuration]);

    // Render telemetry data section
    const renderTelemetryData = () => {
        if (!liveData || Object.keys(liveData).length === 0) {
            return (
                <Box p="3">
                    <Text size="2" color="gray">No live telemetry data available</Text>
                </Box>
            );
        }

        const rawThrottle = toFiniteNumber(liveData.Physics_gas ?? liveData.throttle ?? liveData.gas);
        const rawBrake = toFiniteNumber(liveData.Physics_brake ?? liveData.braking ?? liveData.brake);
        const rawSteering = toFiniteNumber(
            liveData.Physics_steer_angle ?? liveData.steering ?? liveData.steer_angle ?? liveData.steering_angle
        );
        const throttlePercent = rawThrottle != null ? `${Math.round(normalizePedalInput(rawThrottle) * 100)}%` : undefined;
        const brakePercent = rawBrake != null ? `${Math.round(normalizePedalInput(rawBrake) * 100)}%` : undefined;
        const steeringPercent = rawSteering != null ? `${Math.round(clampSteering(rawSteering) * 100)}%` : undefined;
        const gearValue = toGear(toFiniteNumber(liveData.Physics_gear ?? liveData.gear));
        const speedValue = toFiniteNumber(
            liveData.Physics_speed_kmh ?? liveData.speed ?? liveData.speed_kmh ?? liveData.speed_mph
        );
        const rpmValue = toFiniteNumber(liveData.rpm);

        const telemetryItems = {
            Throttle: throttlePercent,
            Brake: brakePercent,
            Steering: steeringPercent,
            Speed: speedValue != null ? `${speedValue.toFixed(1)} km/h` : undefined,
            Gear: gearValue != null ? gearValue : undefined,
            RPM: rpmValue != null ? Math.round(rpmValue) : undefined,
        } as Record<string, string | number | undefined>;

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">Current Telemetry</Text>
                <Grid columns="2" gap="2">
                    {Object.entries(telemetryItems)
                        .filter(([, value]) => value !== undefined)
                        .map(([label, value]) => (
                            <Box key={label}>
                                <Text size="1" color="gray">{label}</Text>
                                <Text size="2" weight="medium">{value}</Text>
                            </Box>
                        ))}
                </Grid>
            </Box>
        );
    };

    // Render current situation section
    const renderCurrentSituation = () => {
        if (!guidanceData?.guidance_result?.current_situation) {
            return null;
        }

        const { current_situation } = guidanceData.guidance_result;
        const speed = current_situation.speed ?? 'Unknown';
        const trackPosition = current_situation.track_position ?? 'Unknown';
        const racingLine = current_situation.racing_line ?? 'Unknown';
        const tireGrip = current_situation.tire_grip ?? 'Unknown';
        const racingLineColor = current_situation.racing_line
            ? (/optimal/i.test(current_situation.racing_line) ? 'green' : 'orange')
            : 'gray';
        const tireGripColor = current_situation.tire_grip
            ? (/good/i.test(current_situation.tire_grip) ? 'green' : 'red')
            : 'gray';

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">Current Situation</Text>
                <Grid columns="2" gap="3">
                    <Box>
                        <Text size="1" color="gray">Speed</Text>
                        <Text size="2" weight="medium">{speed}</Text>
                    </Box>
                    <Box>
                        <Text size="1" color="gray">Track Position</Text>
                        <Text size="2" weight="medium">{trackPosition}</Text>
                    </Box>
                    <Box>
                        <Text size="1" color="gray">Racing Line</Text>
                        <Badge
                            color={racingLineColor}
                            size="1"
                        >
                            {racingLine}
                        </Badge>
                    </Box>
                    <Box>
                        <Text size="1" color="gray">Tire Grip</Text>
                        <Badge
                            color={tireGripColor}
                            size="1"
                        >
                            {tireGrip}
                        </Badge>
                    </Box>
                </Grid>
            </Box>
        );
    };

    // Render animated timeline section
    const renderAnimatedTimeline = () => {
        const hasData = keyframes.length > 0 && totalDuration > 0;
        const cur = getInterpolatedValues(progress);
        const targetSpeedDisplay = typeof cur.target_speed === 'number' ? `${cur.target_speed.toFixed(1)} km/h` : '-';
        const gearDisplay = typeof cur.gear === 'number' ? cur.gear : '-';
        const actionDisplay = cur.action ?? '-';

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">Action Timeline</Text>

                {/* Current values */}
                <div className={styles.animatorSection}>
                    <div className={styles.gaugeRow}>
                        <div className={styles.gaugeLabel}>Throttle</div>
                        <div className={styles.barOuter}>
                            <div className={styles.barFillThrottle} style={{ width: `${(cur.throttle * 100).toFixed(2)}%` }} />
                        </div>
                        <div className={styles.gaugeValue}>{Math.round(cur.throttle * 100)}%</div>
                    </div>

                    <div className={styles.gaugeRow}>
                        <div className={styles.gaugeLabel}>Brake</div>
                        <div className={styles.barOuter}>
                            <div className={styles.barFillBrake} style={{ width: `${(cur.brake * 100).toFixed(2)}%` }} />
                        </div>
                        <div className={styles.gaugeValue}>{Math.round(cur.brake * 100)}%</div>
                    </div>

                    <div className={styles.gaugeRow}>
                        <div className={styles.gaugeLabel}>Steering</div>
                        <div className={styles.steeringOuter}>
                            <div className={styles.steeringLeft} style={{ width: `${Math.max(0, -cur.steering) * 100}%` }} />
                            <div className={styles.steeringCenterLine} />
                            <div className={styles.steeringRight} style={{ width: `${Math.max(0, cur.steering) * 100}%` }} />
                        </div>
                        <div className={styles.gaugeValue}>{Math.round(cur.steering * 100)}%</div>
                    </div>
                </div>

                {/* Extra info */}
                <Grid columns="3" gap="3" mt="2">
                    <Box style={{ textAlign: 'center' }}>
                        <Text size="1" color="gray">Gear</Text>
                        <Text size="2" weight="bold">{gearDisplay}</Text>
                    </Box>
                    <Box style={{ textAlign: 'center' }}>
                        <Text size="1" color="gray">Target Speed</Text>
                        <Text size="2" weight="bold">{targetSpeedDisplay}</Text>
                    </Box>
                    <Box style={{ textAlign: 'center' }}>
                        <Text size="1" color="gray">Action</Text>
                        <Text size="2" weight="bold">{actionDisplay}</Text>
                    </Box>
                </Grid>

                {/* Timeline controls */}
                <div className={styles.timelineControls}>
                    <div className={styles.controlsLeft}>
                        <Text size="1" color="gray">Drag the slider to inspect keyframes</Text>
                    </div>
                    <div className={styles.controlsRight}>
                        <Text size="1" color="gray">{progress.toFixed(2)}s / {totalDuration.toFixed(2)}s</Text>
                    </div>
                </div>
                <div className={styles.timelineSlider}>
                    <input
                        type="range"
                        min={0}
                        max={Math.max(0.01, totalDuration)}
                        step={0.01}
                        value={Number.isFinite(progress) ? progress : 0}
                        onChange={(e) => onScrub(parseFloat(e.target.value))}
                        disabled={!hasData}
                    />
                    {/* Keyframe markers */}
                    <div className={styles.keyframeTrack}>
                        {keyframes.map((k, idx) => (
                            <span key={idx} className={styles.keyframeDot} style={{ left: `${(k.t / (totalDuration || 1)) * 100}%` }} />
                        ))}
                    </div>
                </div>
            </Box>
        );
    };

    // Render contextual information section  
    const renderContextualInfo = () => {
        if (!guidanceData?.guidance_result?.contextual_info) {
            return null;
        }

        const { contextual_info } = guidanceData.guidance_result;
        const trackSector = contextual_info.track_sector ?? 'Unknown sector';
        const weatherImpact = contextual_info.weather_impact ?? 'Unknown conditions';
        const optimalSpeed = contextual_info.optimal_speed_estimate ?? 'Not provided';

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">Track Context</Text>
                <Grid gap="2">
                    <Box p="2" style={{
                        border: '1px solid var(--blue-6)',
                        borderRadius: '4px',
                        backgroundColor: 'var(--blue-1)'
                    }}>
                        <Text size="1" color="gray">Location</Text>
                        <Text size="2" weight="medium" style={{ display: 'block' }}>
                            {trackSector}
                        </Text>
                    </Box>

                    <Box p="2" style={{
                        border: '1px solid var(--green-6)',
                        borderRadius: '4px',
                        backgroundColor: 'var(--green-1)'
                    }}>
                        <Text size="1" color="gray">Weather Conditions</Text>
                        <Text size="2" weight="medium" style={{ display: 'block' }}>
                            {weatherImpact}
                        </Text>
                    </Box>

                    <Box p="2" style={{
                        border: '1px solid var(--orange-6)',
                        borderRadius: '4px',
                        backgroundColor: 'var(--orange-1)'
                    }}>
                        <Text size="1" color="gray">Optimal Speed Estimate</Text>
                        <Text size="2" weight="medium" style={{ display: 'block' }}>
                            {optimalSpeed}
                        </Text>
                    </Box>
                </Grid>
            </Box>
        );
    };

    return (
        <Card className={styles.imitationGuidanceChart} style={{ height: '100%', minHeight: 0 }}>
            <Flex direction="column" height="100%" style={{ minHeight: 0, flex: 1 }}>
                {/* Header */}
                <Box p="3" style={{ borderBottom: '1px solid var(--gray-6)' }}>
                    <Flex justify="between" align="center">
                        <Box>
                            <Text size="3" weight="bold">AI Track Guidance</Text>
                            <Text size="1" color="gray">{trackName} • {carName}</Text>
                        </Box>
                        <Flex gap="2" align="center">
                            <Button
                                size="1"
                                variant={autoUpdate ? "solid" : "soft"}
                                color={autoUpdate ? "green" : "gray"}
                                onClick={toggleAutoUpdate}
                                disabled={loading}
                            >
                                {autoUpdate ? 'Auto' : 'Manual'}
                            </Button>
                            <Button
                                size="1"
                                onClick={fetchGuidance}
                                disabled={loading || autoUpdate}
                                variant="soft"
                            >
                                {loading ? 'Loading...' : 'Update'}
                            </Button>
                        </Flex>
                    </Flex>
                </Box>

                {/* Content */}
                {/* Scroll container: ensure minHeight:0 so flexbox allows overflow, add padding bottom so last item is reachable */}
                <Flex
                    direction="column"
                    flexGrow="1"
                    style={{
                        overflowY: 'auto',
                        flexBasis: 0,
                        minHeight: 0,
                        paddingBottom: '12px'
                    }}
                >
                    {error && (
                        <Box p="3" style={{ borderBottom: '1px solid var(--red-6)', backgroundColor: 'var(--red-2)' }}>
                            <Text size="2" color="red">{error}</Text>
                        </Box>
                    )}

                    {renderTelemetryData()}

                    {guidanceData?.guidance_result?.status === 'success' && (
                        <>
                            <Separator size="4" />
                            {renderCurrentSituation()}

                            <Separator size="4" />
                            {renderAnimatedTimeline()}

                            <Separator size="4" />
                            {renderContextualInfo()}
                        </>
                    )}

                    {guidanceData?.guidance_result?.timestamp && (
                        <Box p="3" pt="2">
                            <Text size="1" color="gray">
                                Last updated: {new Date(guidanceData.guidance_result.timestamp).toLocaleTimeString()}
                            </Text>
                        </Box>
                    )}

                    {!guidanceData && !loading && !error && (
                        <Box p="4" style={{ textAlign: 'center' }}>
                            <Text size="2" color="gray">Click "Update" to get AI guidance</Text>
                        </Box>
                    )}
                </Flex>
            </Flex>
        </Card>
    );
};

export default ImitationGuidanceChart;
