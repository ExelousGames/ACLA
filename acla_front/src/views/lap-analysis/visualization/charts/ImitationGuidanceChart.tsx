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

    // Extract track and car information from session data
    const trackName = analysisContext.recordedSessioStaticsData?.track || 'Unknown Track';
    const carName = analysisContext.recordedSessioStaticsData?.car_model || 'Unknown Car';
    const liveData = analysisContext.liveData as TelemetryData;

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
        };
    }, []);

    // Auto-update when live data changes (if auto-update is enabled)
    useEffect(() => {
        if (autoUpdate && liveData && Object.keys(liveData).length > 0) {
            fetchGuidance();
        }
    }, [liveData, autoUpdate, fetchGuidance]);

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

    // Render sequence predictions section
    const renderSequencePredictions = () => {
        if (!guidanceData?.guidance_result?.sequence_predictions) {
            return null;
        }

        const { sequence_predictions } = guidanceData.guidance_result;

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">Action Sequence ({sequence_predictions.length} steps)</Text>
                <Grid gap="2">
                    {sequence_predictions.slice(0, 5).map((prediction) => (
                        <Box key={prediction.step} p="3" style={{
                            border: '1px solid var(--gray-6)',
                            borderRadius: '6px',
                            backgroundColor: 'var(--gray-1)'
                        }}>
                            <Flex justify="between" align="center" mb="2">
                                <Badge color="blue" size="1">Step {prediction.step}</Badge>
                                <Text size="1" color="gray">{prediction.time_ahead}</Text>
                            </Flex>

                            <Text size="2" weight="medium" mb="2" style={{ display: 'block' }}>
                                {prediction.action}
                            </Text>

                            <Grid columns="3" gap="2">
                                <Box style={{ textAlign: 'center' }}>
                                    <Text size="1" color="gray">Throttle</Text>
                                    <Text size="2" weight="bold" color={prediction.throttle > 0.5 ? 'green' : 'gray'}>
                                        {(prediction.throttle * 100).toFixed(0)}%
                                    </Text>
                                </Box>
                                <Box style={{ textAlign: 'center' }}>
                                    <Text size="1" color="gray">Brake</Text>
                                    <Text size="2" weight="bold" color={prediction.brake > 0.1 ? 'red' : 'gray'}>
                                        {(prediction.brake * 100).toFixed(0)}%
                                    </Text>
                                </Box>
                                <Box style={{ textAlign: 'center' }}>
                                    <Text size="1" color="gray">Steering</Text>
                                    <Text size="2" weight="bold" color={Math.abs(prediction.steering) > 0.1 ? 'blue' : 'gray'}>
                                        {(prediction.steering * 100).toFixed(0)}%
                                    </Text>
                                </Box>
                            </Grid>
                            {(prediction.gear !== undefined || prediction.target_speed !== undefined) && (
                                <Grid columns="2" gap="2" mt="2">
                                    {prediction.gear !== undefined && (
                                        <Box style={{ textAlign: 'center' }}>
                                            <Text size="1" color="gray">Gear</Text>
                                            <Text size="2" weight="bold">{prediction.gear}</Text>
                                        </Box>
                                    )}
                                    {prediction.target_speed !== undefined && (
                                        <Box style={{ textAlign: 'center' }}>
                                            <Text size="1" color="gray">Target Speed</Text>
                                            <Text size="2" weight="bold">{prediction.target_speed}</Text>
                                        </Box>
                                    )}
                                </Grid>
                            )}
                        </Box>
                    ))}
                </Grid>
                {sequence_predictions.length > 5 && (
                    <Text size="1" color="gray" mt="2" style={{ textAlign: 'center' }}>
                        ... and {sequence_predictions.length - 5} more steps
                    </Text>
                )}
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
                            {renderSequencePredictions()}

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
