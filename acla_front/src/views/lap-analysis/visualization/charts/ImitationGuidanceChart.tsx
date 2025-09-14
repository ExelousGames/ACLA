import React, { useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, Text, Box, Grid, Badge, Separator, Progress, Flex, Button } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

interface GuidanceData {
    message: string;
    guidance_result: {
        success: boolean;
        predicted_actions: number[][];
        performance_scores: number[];
        metadata: {
            track_name: string;
            car_name: string;
            sequence_length: number;
            temperature: number;
            input_features_count: number;
            avg_predicted_performance: number;
            prediction_confidence: number;
            prediction_timestamp: string;
        };
        error?: string;
    };
    timestamp?: string;
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
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    // Extract track and car information from session data
    const trackName = analysisContext.recordedSessioStaticsData?.track || 'Unknown Track';
    const carName = analysisContext.recordedSessioStaticsData?.car_model || 'Unknown Car';
    const liveData = analysisContext.liveData as TelemetryData;
    console.log('Live telemetry data:', analysisContext.recordedSessioStaticsData);
    // Debug logging
    console.log('Session data debug:', {
        sessionSelected: analysisContext.sessionSelected,
        mapSelected: analysisContext.mapSelected,
        trackName,
        carName,
        mapValue: analysisContext.sessionSelected?.map,
        carValue: analysisContext.sessionSelected?.car
    });

    // Function to call the imitation learning guidance API
    const fetchGuidance = useCallback(async () => {
        if (!liveData || Object.keys(liveData).length === 0) {
            setError('No live telemetry data available');
            return;
        }

        setLoading(true);
        setError(null);

        console.log('Fetching imitation guidance with:', {
            trackName,
            carName,
            telemetryKeys: Object.keys(liveData)
        });

        try {
            const response = await apiService.post('/racing-session/imitation-learning-guidance', {
                current_telemetry: liveData,
                track_name: trackName,
                car_name: carName
            });

            if (response.data) {
                const data = response.data as GuidanceData;
                if (data.guidance_result?.success) {
                    setGuidanceData(data);

                    // Send guidance message to AI chat if available
                    if (data.guidance_result?.predicted_actions && data.guidance_result.predicted_actions.length > 0) {
                        const guidanceText = formatGuidanceForChat(data.guidance_result);
                        analysisContext.sendGuidanceToChat(guidanceText);
                    }
                } else {
                    setError('Failed to get guidance: ' + (data.guidance_result?.error || data.message || 'Unknown error'));
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
    const formatGuidanceForChat = (guidanceResult: any): string => {
        if (!guidanceResult || !guidanceResult.predicted_actions) return 'AI guidance received';

        const metadata = guidanceResult.metadata || {};
        const confidence = metadata.prediction_confidence || 0;
        const avgPerformance = metadata.avg_predicted_performance || 0;
        const actionsCount = guidanceResult.predicted_actions.length;

        return `AI Guidance: ${actionsCount} predicted actions with ${(confidence * 100).toFixed(1)}% confidence (avg performance: ${avgPerformance.toFixed(2)})`;
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

    // Render guidance predictions and performance
    const renderGuidancePredictions = () => {
        if (!guidanceData?.guidance_result?.predicted_actions || !guidanceData.guidance_result.success) {
            return null;
        }

        const { predicted_actions, performance_scores, metadata } = guidanceData.guidance_result;
        const confidence = metadata?.prediction_confidence || 0;
        const avgPerformance = metadata?.avg_predicted_performance || 0;

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">AI Predictions</Text>

                {/* Summary stats */}
                <Grid columns="2" gap="2" mb="3">
                    <Box>
                        <Text size="1" color="gray">Predicted Actions</Text>
                        <Text size="2" weight="medium">{predicted_actions.length}</Text>
                    </Box>
                    <Box>
                        <Text size="1" color="gray">Avg Performance</Text>
                        <Text size="2" weight="medium">{avgPerformance.toFixed(3)}</Text>
                    </Box>
                </Grid>

                {/* Confidence indicator */}
                {metadata?.prediction_confidence !== undefined && (
                    <Box mb="3">
                        <Text size="1" color="gray">Prediction Confidence</Text>
                        <Flex align="center" gap="2" mt="1">
                            <Progress
                                value={confidence * 100}
                                max={100}
                                size="2"
                                style={{ flex: 1 }}
                                color={confidence > 0.7 ? 'green' : confidence > 0.5 ? 'yellow' : 'red'}
                            />
                            <Text size="2" weight="medium">
                                {(confidence * 100).toFixed(1)}%
                            </Text>
                        </Flex>
                    </Box>
                )}

                {/* First few predicted actions */}
                {predicted_actions.length > 0 && (
                    <Box>
                        <Text size="1" color="gray" mb="2">Next Recommended Actions</Text>
                        <Grid gap="2">
                            {predicted_actions.slice(0, 3).map((action, index) => (
                                <Box key={index} p="2" style={{ border: '1px solid var(--gray-6)', borderRadius: '4px' }}>
                                    <Flex justify="between" align="center">
                                        <Badge color="blue" size="1">Step {index + 1}</Badge>
                                        {performance_scores && performance_scores[index] !== undefined && (
                                            <Text size="1" color="gray">
                                                Score: {performance_scores[index].toFixed(3)}
                                            </Text>
                                        )}
                                    </Flex>
                                    <Text size="1" mt="1" style={{ fontFamily: 'monospace' }}>
                                        [{action.map(val => val.toFixed(3)).join(', ')}]
                                    </Text>
                                </Box>
                            ))}
                        </Grid>

                        {predicted_actions.length > 3 && (
                            <Text size="1" color="gray" mt="2" style={{ textAlign: 'center' }}>
                                ... and {predicted_actions.length - 3} more actions
                            </Text>
                        )}
                    </Box>
                )}
            </Box>
        );
    };

    return (
        <Card className={styles.imitationGuidanceChart} style={{ height: '100%' }}>
            <Flex direction="column" height="100%">
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
                <Flex direction="column" flexGrow="1" style={{ overflowY: 'auto' }}>
                    {error && (
                        <Box p="3" style={{ borderBottom: '1px solid var(--red-6)', backgroundColor: 'var(--red-2)' }}>
                            <Text size="2" color="red">{error}</Text>
                        </Box>
                    )}

                    {renderTelemetryData()}

                    {guidanceData && (
                        <>
                            <Separator size="4" />
                            {renderGuidancePredictions()}

                            {guidanceData.timestamp && (
                                <Box p="3" pt="2">
                                    <Text size="1" color="gray">
                                        Last updated: {new Date(guidanceData.timestamp).toLocaleTimeString()}
                                    </Text>
                                </Box>
                            )}
                        </>
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
