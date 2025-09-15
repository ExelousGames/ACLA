import React, { useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, Text, Box, Grid, Badge, Separator, Progress, Flex, Button } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

// New backend response structure after switching to generate_expert_action_instructions()
interface PredictedAction {
    t: number;
    normalized_position: number; // 0..1 along predicted horizon
    steering_angle_deg: number;
    steering_direction: string; // left/right/straight
    steering_percent: number;   // 0..1 fraction of max steering
    throttle: number;           // 0..1
    brake: number;              // 0..1
}

interface InstructionStep extends PredictedAction {
    deltas: {
        steer_delta_deg: number;
        throttle_delta: number;
        brake_delta: number;
    };
    instruction_text: string;
    substep_index?: number;
    total_substeps?: number;
}

interface GuidanceResult {
    success: boolean;
    predicted_actions: PredictedAction[];
    steps: InstructionStep[];
    instructions: string[]; // human readable lines
    metadata: {
        track_name: string;
        car_name: string;
        sequence_length: number;
        temperature: number;
        input_features_count: number;
        expected_input_features?: number;
        feature_filtering?: string;
        missing_features_count?: number;
        instruction_generation?: any;
        prediction_timestamp: string;
        // Legacy fields may or may not exist
        avg_predicted_performance?: number;
        prediction_confidence?: number;
    };
    error?: string;
}

interface GuidanceData {
    message?: string;
    guidance_result: GuidanceResult;
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
        const confidence = metadata.prediction_confidence;
        const actionsCount = guidanceResult.predicted_actions.length;
        const stepsCount = guidanceResult.steps ? guidanceResult.steps.length : 0;
        const firstInstruction = guidanceResult.instructions && guidanceResult.instructions.length > 0
            ? guidanceResult.instructions[0]
            : undefined;
        let base = `AI Guidance: ${actionsCount} action points, ${stepsCount} significant adjustments`;
        if (typeof confidence === 'number') {
            base += ` (confidence ${(confidence * 100).toFixed(1)}%)`;
        }
        if (firstInstruction) {
            base += `. First: ${firstInstruction}`;
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

    // Render guidance predictions and performance
    const renderGuidancePredictions = () => {
        if (!guidanceData?.guidance_result?.predicted_actions || !guidanceData.guidance_result.success) {
            return null;
        }
        const { predicted_actions, steps, instructions, metadata } = guidanceData.guidance_result;
        const confidence = metadata?.prediction_confidence; // may be undefined now

        return (
            <Box p="3">
                <Text size="2" weight="bold" mb="2">AI Predicted Expert Actions</Text>

                {/* Summary stats */}
                <Grid columns="2" gap="2" mb="3">
                    <Box>
                        <Text size="1" color="gray">Predicted Actions</Text>
                        <Text size="2" weight="medium">{predicted_actions.length}</Text>
                    </Box>
                    {steps && (
                        <Box>
                            <Text size="1" color="gray">Significant Steps</Text>
                            <Text size="2" weight="medium">{steps.length}</Text>
                        </Box>
                    )}
                </Grid>
                {/* Confidence indicator (legacy / optional) */}
                {typeof confidence === 'number' && (
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

                {/* First few predicted action points */}
                {predicted_actions.length > 0 && (
                    <Box mb="3">
                        <Text size="1" color="gray" mb="2">Sample Predicted Action Points</Text>
                        <Grid gap="2">
                            {predicted_actions.slice(0, 3).map((pa, index) => (
                                <Box key={index} p="2" style={{ border: '1px solid var(--gray-6)', borderRadius: '4px' }}>
                                    <Flex justify="between" align="center" mb="1">
                                        <Badge color="blue" size="1">Pt {index + 1}</Badge>
                                        <Text size="1" color="gray">pos {(pa.normalized_position * 100).toFixed(0)}%</Text>
                                    </Flex>
                                    <Text size="1" style={{ fontFamily: 'monospace' }}>
                                        steer {pa.steering_direction} {(pa.steering_percent * 100).toFixed(0)}% • thr {(pa.throttle * 100).toFixed(0)}% • brk {(pa.brake * 100).toFixed(0)}%
                                    </Text>
                                </Box>
                            ))}
                        </Grid>
                        {predicted_actions.length > 3 && (
                            <Text size="1" color="gray" mt="2" style={{ textAlign: 'center' }}>
                                ... and {predicted_actions.length - 3} more points
                            </Text>
                        )}
                    </Box>
                )}

                {/* Significant instruction steps */}
                {steps && steps.length > 0 && (
                    <Box>
                        <Text size="1" color="gray" mb="2">Significant Action Adjustments</Text>
                        <Grid gap="2">
                            {steps.slice(0, 5).map((st, i) => (
                                <Box key={i} p="2" style={{ border: '1px solid var(--gray-6)', borderRadius: '4px' }}>
                                    <Flex justify="between" align="center" mb="1">
                                        <Badge color="green" size="1">Step {i + 1}</Badge>
                                        <Text size="1" color="gray">Δ steer {st.deltas.steer_delta_deg.toFixed(1)}°</Text>
                                    </Flex>
                                    <Text size="1">{st.instruction_text}</Text>
                                </Box>
                            ))}
                        </Grid>
                        {steps.length > 5 && (
                            <Text size="1" color="gray" mt="2" style={{ textAlign: 'center' }}>
                                ... and {steps.length - 5} more steps
                            </Text>
                        )}
                    </Box>
                )}

                {/* Raw instruction text list (collapsed version) */}
                {instructions && instructions.length > 0 && (
                    <Box mt="3">
                        <Text size="1" color="gray" mb="1">Instruction Summary</Text>
                        <Text size="1" style={{ display: 'block', whiteSpace: 'pre-wrap' }}>
                            {instructions.slice(0, 3).join('\n')}
                        </Text>
                        {instructions.length > 3 && (
                            <Text size="1" color="gray" mt="1" style={{ textAlign: 'center' }}>
                                ... {instructions.length - 3} more instructions
                            </Text>
                        )}
                    </Box>
                )}
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
