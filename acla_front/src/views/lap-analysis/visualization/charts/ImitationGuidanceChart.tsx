import React, { useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, Text, Box, Grid, Badge, Separator, Progress, Flex, Button } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

// New backend response structure
interface SequencePrediction {
    step: number;
    time_ahead: string; // e.g., "0.1s"
    action: string; // e.g., "Begin braking"
    throttle: number; // 0..1
    brake: number; // 0..1
    steering: number; // -1..1
    // New optional fields from backend
    gear?: number;
    target_speed?: number;
}

interface CurrentSituation {
    speed: string; // e.g., "120 km/h"
    track_position: string; // e.g., "mid-corner"
    // Relax types to accept arbitrary strings from backend (e.g., "good grip")
    racing_line: string;
    tire_grip: string;
}

interface ContextualInfo {
    track_sector: string; // e.g., "Sector 2, Turn 5"
    weather_impact: string; // e.g., "Dry conditions, full grip"
    optimal_speed_estimate: string; // e.g., "95 km/h for current section"
}

interface GuidanceResponse {
    status: "success" | "error";
    timestamp: string; // ISO timestamp
    current_situation: CurrentSituation;
    sequence_predictions: SequencePrediction[];
    contextual_info: ContextualInfo;
}

interface GuidanceData {
    message?: string;
    guidance_result?: GuidanceResponse; // keeping old property name for compatibility
    timestamp?: string;
    success?: boolean; // envelope compatibility
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

    // Build keyframes from predictions
    type Keyframe = {
        t: number; // seconds from now
        throttle: number;
        brake: number;
        steering: number;
        gear?: number;
        target_speed?: number;
        action?: string;
    };

    const parseTimeAhead = (timeStr?: string): number => {
        if (!timeStr) return 0;
        const trimmed = String(timeStr).trim();
        const m = trimmed.match(/([0-9]*\.?[0-9]+)/);
        if (!m) return 0;
        const val = parseFloat(m[1]);
        // assume seconds by default
        if (/ms/i.test(trimmed)) return val / 1000;
        return val;
    };

    const keyframes: Keyframe[] = useMemo(() => {
        const preds = guidanceData?.guidance_result?.sequence_predictions ?? [];
        const frames = preds.map(p => ({
            t: parseTimeAhead(p.time_ahead),
            throttle: clamp01(p.throttle ?? 0),
            brake: clamp01(p.brake ?? 0),
            steering: clampMinus1To1(p.steering ?? 0),
            gear: p.gear,
            target_speed: p.target_speed,
            action: p.action,
        }))
            .sort((a, b) => a.t - b.t);

        // Ensure we have an initial frame at t=0 by inferring from liveData if needed
        if (frames.length > 0 && frames[0].t > 0) {
            frames.unshift({
                t: 0,
                throttle: clamp01(Number(liveData?.throttle ?? 0)),
                brake: clamp01(Number(liveData?.braking ?? 0)),
                steering: clampMinus1To1(Number(liveData?.steering ?? 0)),
                gear: typeof liveData?.gear === 'number' ? liveData.gear : undefined,
                target_speed: undefined,
                action: 'Current',
            });
        }

        return frames;
    }, [guidanceData, liveData]);

    const totalDuration = useMemo(() => (keyframes.length ? keyframes[keyframes.length - 1].t : 0), [keyframes]);

    // Helpers
    function clamp01(n: number) { return Math.max(0, Math.min(1, n)); }
    function clampMinus1To1(n: number) { return Math.max(-1, Math.min(1, n)); }

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
            gear: lt < 0.5 ? a.gear : b.gear,
            target_speed: lt < 0.5 ? a.target_speed : b.target_speed,
            action: lt < 0.5 ? a.action : b.action,
        };
    }

    // Function to call the imitation learning guidance API
    const fetchGuidance = useCallback(async () => {
        if (!liveData || Object.keys(liveData).length === 0) {
            setError('No live telemetry data available');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await apiService.post('/racing-session/imitation-learning-guidance', {
                current_telemetry: liveData,
                track_name: trackName,
                car_name: carName
            });
            if (response.data) {
                // Support both legacy (GuidanceResponse at root) and new envelope
                const raw = response.data as any;
                const maybeEnvelope = (raw && (raw.guidance_result || raw.success !== undefined)) ? raw : null;
                const result: GuidanceResponse | null = maybeEnvelope ? raw.guidance_result : raw;

                if (result && result.status === 'success') {
                    setGuidanceData({
                        guidance_result: result,
                        timestamp: result.timestamp,
                        success: maybeEnvelope ? raw.success : undefined,
                        message: maybeEnvelope ? raw.message : undefined,
                    });

                    // Reset animation with new data
                    setProgress(0);
                    setIsPlaying(true);
                    playStartRef.current = null;
                    progressAtPlayStartRef.current = 0;

                    // Send guidance message to AI chat if available
                    if (result.sequence_predictions && result.sequence_predictions.length > 0) {
                        const guidanceText = formatGuidanceForChat(result);
                        analysisContext.sendGuidanceToChat(guidanceText);
                    }
                } else if (result) {
                    setError('Failed to get guidance: API returned error status');
                } else {
                    setError('Malformed response: missing guidance_result');
                }
            } else {
                setError('No response data received');
            }
        } catch (err: any) {
            console.error('Imitation learning guidance error:', err);
            setError('API call failed: ' + (err.response?.data?.message || err.message));
        } finally {
            setLoading(false);
        }
    }, [liveData, trackName, carName, analysisContext]);

    // Format guidance data for AI chat
    const formatGuidanceForChat = (guidanceResult: GuidanceResponse): string => {
        if (!guidanceResult || !guidanceResult.sequence_predictions) return 'AI guidance received';

        const { current_situation, sequence_predictions, contextual_info } = guidanceResult;
        const predictionsCount = sequence_predictions.length;
        const firstAction = sequence_predictions.length > 0 ? sequence_predictions[0].action : undefined;

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
        setAutoUpdate(prev => {
            const newValue = !prev;
            if (newValue) {
                // Start polling every 2 seconds
                intervalRef.current = setInterval(fetchGuidance, 2000);
                fetchGuidance(); // Fetch immediately
            } else {
                // Stop polling
                if (intervalRef.current) {
                    clearInterval(intervalRef.current);
                    intervalRef.current = null;
                }
            }
            return newValue;
        });
    }, [fetchGuidance]);

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

    // Auto-update when live data changes (if auto-update is enabled)
    useEffect(() => {
        if (autoUpdate && liveData && Object.keys(liveData).length > 0) {
            fetchGuidance();
        }
    }, [liveData, autoUpdate, fetchGuidance]);

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

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">Current Situation</Text>
                <Grid columns="2" gap="3">
                    <Box>
                        <Text size="1" color="gray">Speed</Text>
                        <Text size="2" weight="medium">{current_situation.speed}</Text>
                    </Box>
                    <Box>
                        <Text size="1" color="gray">Track Position</Text>
                        <Text size="2" weight="medium">{current_situation.track_position}</Text>
                    </Box>
                    <Box>
                        <Text size="1" color="gray">Racing Line</Text>
                        <Badge
                            color={/optimal/i.test(current_situation.racing_line) ? 'green' : 'orange'}
                            size="1"
                        >
                            {current_situation.racing_line}
                        </Badge>
                    </Box>
                    <Box>
                        <Text size="1" color="gray">Tire Grip</Text>
                        <Badge
                            color={/good/i.test(current_situation.tire_grip) ? 'green' : 'red'}
                            size="1"
                        >
                            {current_situation.tire_grip}
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
                        <Text size="2" weight="bold">{cur.gear ?? '-'}</Text>
                    </Box>
                    <Box style={{ textAlign: 'center' }}>
                        <Text size="1" color="gray">Target Speed</Text>
                        <Text size="2" weight="bold">{cur.target_speed ?? '-'}</Text>
                    </Box>
                    <Box style={{ textAlign: 'center' }}>
                        <Text size="1" color="gray">Action</Text>
                        <Text size="2" weight="bold">{cur.action ?? '-'}</Text>
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
                            {contextual_info.track_sector}
                        </Text>
                    </Box>

                    <Box p="2" style={{
                        border: '1px solid var(--green-6)',
                        borderRadius: '4px',
                        backgroundColor: 'var(--green-1)'
                    }}>
                        <Text size="1" color="gray">Weather Conditions</Text>
                        <Text size="2" weight="medium" style={{ display: 'block' }}>
                            {contextual_info.weather_impact}
                        </Text>
                    </Box>

                    <Box p="2" style={{
                        border: '1px solid var(--orange-6)',
                        borderRadius: '4px',
                        backgroundColor: 'var(--orange-1)'
                    }}>
                        <Text size="1" color="gray">Optimal Speed Estimate</Text>
                        <Text size="2" weight="medium" style={{ display: 'block' }}>
                            {contextual_info.optimal_speed_estimate}
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
