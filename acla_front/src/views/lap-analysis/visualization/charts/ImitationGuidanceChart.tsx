import React, { useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, Text, Box, Grid, Badge, Separator, Flex, Button } from '@radix-ui/themes';
import { AnalysisContext } from '../../analysis-context';
import { VisualizationProps } from '../VisualizationRegistry';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

// New backend response structure
interface SequencePrediction {
    step: number;
    time_ahead: string;
    time_delta_seconds?: number | string;
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
    acceleration?: number;
    braking?: number;
    steering?: number;
    throttle?: number;
    gear?: number;
    rpm?: number;
    [key: string]: any;
}

type Keyframe = {
    t: number;
    throttle: number;
    brake: number;
    steering: number;
    gear?: number;
    target_speed?: number;
    action?: string;
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

const normalizeSteering = (raw?: number): number => {
    if (raw == null || Number.isNaN(raw)) return 0;
    let normalized = raw;
    if (Math.abs(normalized) > 1.5) {
        normalized = normalized / 90; // assume degrees, map roughly into [-1,1]
    }
    return clampMinus1To1(normalized);
};

const deriveAction = (throttle: number, brake: number, steering: number): string => {
    if (brake > 0.75) return 'Brake hard';
    if (brake > 0.35) return 'Start braking';
    if (throttle > 0.8 && brake < 0.1) return 'Full throttle';
    if (throttle > 0.45 && brake < 0.2) return 'Accelerate';
    if (Math.abs(steering) > 0.6) return steering > 0 ? 'Steer right' : 'Steer left';
    if (Math.abs(steering) > 0.3) return steering > 0 ? 'Turn right' : 'Turn left';
    return 'Hold steady';
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

const parseTimeAhead = (timeStr?: string): number => {
    if (!timeStr) return 0;
    const trimmed = String(timeStr).trim();
    const match = trimmed.match(/([0-9]*\.?[0-9]+)/);
    if (!match) return 0;
    const val = parseFloat(match[1]);
    if (/ms/i.test(trimmed)) return val / 1000;
    return val;
};

const keyframeFromPrediction = (prediction: SequencePrediction, timeOverride?: number): Keyframe => {
    const targets = (prediction.all_targets ?? {}) as Record<string, unknown>;
    const throttle = clamp01(getNumericFromTargets(targets, ['Physics_gas', 'gas', 'throttle']) ?? 0);
    const brake = clamp01(getNumericFromTargets(targets, ['Physics_brake', 'brake']) ?? 0);
    const rawSteering = getNumericFromTargets(targets, ['Physics_steer_angle', 'steering']) ?? 0;
    const steering = normalizeSteering(rawSteering);
    const gear = toGear(getNumericFromTargets(targets, ['Physics_gear', 'gear']));
    const targetSpeed = getNumericFromTargets(targets, ['Physics_speed_kmh', 'speed_kmh', 'speed']);
    const predictedActionCandidate = (prediction as { action?: unknown }).action;
    const predictedAction = typeof predictedActionCandidate === 'string' ? predictedActionCandidate : undefined;
    const action = predictedAction && predictedAction.trim().length > 0
        ? predictedAction
        : deriveAction(throttle, brake, steering);

    const timelineTime = typeof timeOverride === 'number' && Number.isFinite(timeOverride)
        ? timeOverride
        : parseTimeAhead(prediction.time_ahead);

    return {
        t: timelineTime,
        throttle,
        brake,
        steering,
        gear,
        target_speed: targetSpeed,
        action,
    };
};

const keyframeFromTelemetry = (telemetry?: TelemetryData | null): Keyframe | null => {
    if (!telemetry) return null;
    const throttle = clamp01(toFiniteNumber(telemetry.throttle ?? telemetry.gas) ?? 0);
    const brake = clamp01(toFiniteNumber(telemetry.braking ?? telemetry.brake) ?? 0);
    const steering = normalizeSteering(toFiniteNumber(telemetry.steering ?? telemetry.steer_angle ?? telemetry.steering_angle) ?? 0);
    const gear = toGear(toFiniteNumber(telemetry.gear));
    const targetSpeed = toFiniteNumber(telemetry.speed ?? telemetry.speed_kmh ?? telemetry.speed_mph);
    const action = deriveAction(throttle, brake, steering);

    return { t: 0, throttle, brake, steering, gear, target_speed: targetSpeed, action };
};

const ImitationGuidanceChart: React.FC<VisualizationProps> = (props) => {
    const analysisContext = useContext(AnalysisContext);
    const [guidanceData, setGuidanceData] = useState<GuidanceData | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [autoUpdate, setAutoUpdate] = useState<boolean>(false);
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const rafRef = useRef<number | null>(null);

    // Animation state
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [progress, setProgress] = useState<number>(0); // seconds along timeline
    const playStartRef = useRef<number | null>(null); // performance.now baseline
    const progressAtPlayStartRef = useRef<number>(0); // progress when play started

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
        const frames: Keyframe[] = [];
        let cumulativeTime = 0;

        for (const prediction of predictions) {
            const deltaSecondsRaw = (prediction as { time_delta_seconds?: unknown }).time_delta_seconds;
            const deltaSeconds = toFiniteNumber(deltaSecondsRaw);
            let timeOverride: number | undefined;

            if (deltaSeconds != null && Number.isFinite(deltaSeconds)) {
                cumulativeTime += Math.max(0, deltaSeconds);
                timeOverride = cumulativeTime;
            }

            const frame = keyframeFromPrediction(prediction, timeOverride);
            if (Number.isFinite(frame.t)) {
                frames.push(frame);
            }
        }

        frames.sort((a, b) => a.t - b.t);

        if (frames.length === 0) {
            const currentFrame = keyframeFromTelemetry(liveData);
            return currentFrame ? [{ ...currentFrame, t: 0, action: 'Current' }] : [];
        }

        return frames;
    }, [guidanceData, liveData]);

    const totalDuration = useMemo(() => (keyframes.length ? keyframes[keyframes.length - 1].t : 0), [keyframes]);

    function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

    function getInterpolatedValues(timeS: number) {
        if (keyframes.length === 0) {
            return {
                throttle: 0, brake: 0, steering: 0,
                gear: undefined as number | undefined,
                target_speed: undefined as number | undefined,
                action: undefined as string | undefined,
            };
        }
        if (timeS <= keyframes[0].t) {
            const k = keyframes[0];
            return { throttle: k.throttle, brake: k.brake, steering: k.steering, gear: k.gear, target_speed: k.target_speed, action: k.action };
        }
        if (timeS >= keyframes[keyframes.length - 1].t) {
            const k = keyframes[keyframes.length - 1];
            return { throttle: k.throttle, brake: k.brake, steering: k.steering, gear: k.gear, target_speed: k.target_speed, action: k.action };
        }
        // find segment
        let i = 0;
        for (; i < keyframes.length - 1; i++) {
            if (timeS >= keyframes[i].t && timeS <= keyframes[i + 1].t) break;
        }
        const a = keyframes[i];
        const b = keyframes[i + 1];
        const segDur = b.t - a.t || 1e-6;
        const lt = (timeS - a.t) / segDur;
        return {
            throttle: clamp01(lerp(a.throttle, b.throttle, lt)),
            brake: clamp01(lerp(a.brake, b.brake, lt)),
            steering: clampMinus1To1(lerp(a.steering, b.steering, lt)),
            gear: lt < 0.5 ? (a.gear ?? b.gear) : (b.gear ?? a.gear),
            target_speed: lt < 0.5 ? (a.target_speed ?? b.target_speed) : (b.target_speed ?? a.target_speed),
            action: lt < 0.5 ? (a.action ?? b.action) : (b.action ?? a.action),
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

            if (!result) {
                setError('Malformed response: missing guidance_result');
                return;
            }

            if (result.status !== 'success') {
                setError('Failed to get guidance: API returned error status');
                return;
            }

            setGuidanceData({
                message: raw.message,
                success: raw.success,
                timestamp: raw.timestamp ?? result.timestamp,
                guidance_result: result,
            });

            setProgress(0);
            setIsPlaying(true);
            playStartRef.current = null;
            progressAtPlayStartRef.current = 0;

            if (result.sequence_predictions && result.sequence_predictions.length > 0) {
                const guidanceText = formatGuidanceForChat(result);
                analysisContext.sendGuidanceToChat(guidanceText);
            }
        } catch (err: any) {
            console.error('Imitation learning guidance error:', err);
            setError('API call failed: ' + (err.response?.data?.message || err.message));
        } finally {
            setLoading(false);
            requestInFlightRef.current = false;
        }
    }, [analysisContext]);

    // Format guidance data for AI chat
    const formatGuidanceForChat = (guidanceResult: GuidanceResponse): string => {
        if (!guidanceResult || !guidanceResult.sequence_predictions) return 'AI guidance received';

        const { current_situation, sequence_predictions, contextual_info } = guidanceResult;
        const predictionsCount = sequence_predictions.length;
        const firstPrediction = sequence_predictions[0];
        const firstAction = firstPrediction ? keyframeFromPrediction(firstPrediction, 0).action : undefined;

        let base = `AI Guidance: ${predictionsCount} predictions`;
        if (current_situation?.racing_line) {
            base += ` (racing line: ${current_situation.racing_line})`;
        }
        if (firstAction) {
            base += `. First: ${firstAction}`;
        }
        if (contextual_info?.track_sector) {
            base += ` at ${contextual_info.track_sector}`;
        }
        return base;
    };

    // Toggle auto-update mode
    const toggleAutoUpdate = useCallback(() => {
        setAutoUpdate(prev => !prev);
    }, []);

    // Cleanup interval on unmount
    useEffect(() => {
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
            if (rafRef.current) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (!autoUpdate) {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            return;
        }

        if (intervalRef.current) {
            clearInterval(intervalRef.current);
        }

        intervalRef.current = setInterval(fetchGuidance, 1000);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        };
    }, [autoUpdate, fetchGuidance]);

    // Auto-fetch when auto-update is enabled or session context changes
    useEffect(() => {
        if (!autoUpdate) return;
        fetchGuidance();
    }, [autoUpdate, trackName, carName, fetchGuidance]);

    // Animation loop
    useEffect(() => {
        if (!isPlaying || totalDuration <= 0) return;

        const step = (now: number) => {
            if (playStartRef.current == null) {
                playStartRef.current = now;
            }
            const elapsed = (now - playStartRef.current) / 1000; // seconds
            const newProgress = Math.min(progressAtPlayStartRef.current + elapsed, totalDuration);
            setProgress(newProgress);

            if (newProgress >= totalDuration) {
                setIsPlaying(false); // stop at end
                playStartRef.current = null;
                progressAtPlayStartRef.current = totalDuration;
                return;
            }
            rafRef.current = requestAnimationFrame(step);
        };

        rafRef.current = requestAnimationFrame(step);
        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
        };
    }, [isPlaying, totalDuration]);

    const onPlayPause = useCallback(() => {
        if (totalDuration <= 0) return;
        if (isPlaying) {
            // pause
            setIsPlaying(false);
            playStartRef.current = null;
            progressAtPlayStartRef.current = progress;
        } else {
            // resume
            setIsPlaying(true);
            playStartRef.current = null; // will be set on next RAF
        }
    }, [isPlaying, progress, totalDuration]);

    const onRestart = useCallback(() => {
        setProgress(0);
        progressAtPlayStartRef.current = 0;
        playStartRef.current = null;
        if (totalDuration > 0) setIsPlaying(true);
    }, [totalDuration]);

    const onScrub = useCallback((v: number) => {
        const clamped = Math.max(0, Math.min(totalDuration || 0, v));
        setProgress(clamped);
        progressAtPlayStartRef.current = clamped;
        playStartRef.current = null;
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

        const displayKeys = ['speed', 'throttle', 'braking', 'steering', 'gear', 'rpm'];
        const availableData = displayKeys.filter(key => liveData[key] !== undefined);

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">Current Telemetry</Text>
                <Grid columns="2" gap="2">
                    {availableData.map(key => (
                        <Box key={key}>
                            <Text size="1" color="gray">{key.toUpperCase()}</Text>
                            <Text size="2" weight="medium">
                                {typeof liveData[key] === 'number' ? liveData[key].toFixed(2) : liveData[key]}
                            </Text>
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
                <Text size="2" weight="bold" mb="2">Action Animator</Text>

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
                        <Button size="1" onClick={onPlayPause} disabled={!hasData} variant="soft">
                            {isPlaying ? 'Pause' : 'Play'}
                        </Button>
                        <Button size="1" onClick={onRestart} disabled={!hasData} variant="soft">Restart</Button>
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
