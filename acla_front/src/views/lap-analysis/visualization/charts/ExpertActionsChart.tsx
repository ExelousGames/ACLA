import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Card, Text, Box, Flex, Button, Badge, Separator, Table, ScrollArea } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';
import { useEnvironment } from 'contexts/EnvironmentContext';
import {
    acquirePersistentImitationModel,
    releasePersistentImitationModel,
    runExpertActionPrediction,
    ExpertPredictionResult
} from 'services/expertActionsService';

const MAX_TELEMETRY_SAMPLES = 2000;
const MODEL_CAR_NAME = 'AllCars';

const METRIC_LABELS: Record<string, string> = {
    expert_optimal_speed: 'Speed (km/h)',
    expert_optimal_throttle: 'Throttle',
    expert_optimal_brake: 'Brake',
    expert_optimal_steering: 'Steering Angle',
    expert_optimal_gear: 'Gear',
    expert_optimal_velocity_x: 'Velocity X',
    expert_optimal_velocity_y: 'Velocity Y',
    expert_optimal_velocity_z: 'Velocity Z'
};

const formatNumber = (value: unknown, digits = 2): string => {
    const num = typeof value === 'number' ? value : Number(value);
    if (Number.isNaN(num)) {
        return '-';
    }
    if (Math.abs(num) >= 1000) {
        return num.toFixed(0);
    }
    return num.toFixed(digits);
};

const extractKeySubset = (prediction?: Record<string, number>): Array<{ key: string; label: string; value: string }> => {
    if (!prediction) return [];
    return Object.entries(prediction)
        .filter(([key]) => METRIC_LABELS[key])
        .map(([key, value]) => ({
            key,
            label: METRIC_LABELS[key] ?? key,
            value: formatNumber(value)
        }));
};

const ExpertActionsChart: React.FC<VisualizationProps> = ({ width = '100%', height = 420 }) => {
    const analysisContext = useContext(AnalysisContext);
    const environment = useEnvironment();

    const [telemetrySamples, setTelemetrySamples] = useState<any[] | null>(null);
    const [prediction, setPrediction] = useState<ExpertPredictionResult | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    type ModelHandle = {
        cacheKey: string;
        data: any;
        trackName?: string;
        carName?: string;
    };

    const modelHandleRef = useRef<ModelHandle | null>(null);

    const trackName = analysisContext.recordedSessioStaticsData?.track || 'Unknown Track';
    const carName = analysisContext.recordedSessioStaticsData?.car_model || 'Unknown Car';
    const sanitizedTrackName = trackName === 'Unknown Track' ? undefined : trackName;
    const sanitizedCarName = MODEL_CAR_NAME; // Always use the pooled AllCars imitation model variant.

    const ensureTelemetrySamples = useCallback(async (): Promise<any[]> => {
        if (telemetrySamples && telemetrySamples.length > 0) {
            return telemetrySamples;
        }

        if (analysisContext.liveData && Object.keys(analysisContext.liveData).length > 0) {
            const samples = [analysisContext.liveData];
            setTelemetrySamples(samples);
            return samples;
        }

        if (analysisContext.recordedSessionDataFilePath) {
            const samples = await analysisContext.readRecordedSessionData();
            setTelemetrySamples(samples);
            return samples;
        }

        throw new Error('No telemetry data available. Start a recording or live session to capture data.');
    }, [
        telemetrySamples,
        analysisContext.recordedSessionDataFilePath,
        analysisContext.liveData,
        analysisContext.readRecordedSessionData
    ]);

    const handleRefreshTelemetry = useCallback(async () => {
        try {
            setError(null);
            const samples = await analysisContext.readRecordedSessionData();
            setTelemetrySamples(samples);
        } catch (refreshError) {
            console.error(refreshError);
            setError((refreshError as Error).message);
        }
    }, [analysisContext.readRecordedSessionData]);

    const handleComputePrediction = useCallback(async () => {
        if (environment !== 'electron') {
            setError('Expert action predictions are available on the desktop application only.');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const samples = await ensureTelemetrySamples();
            if (!samples || samples.length === 0) {
                throw new Error('No telemetry samples available for prediction.');
            }

            const sanitizedSamples = samples
                .filter((sample) => sample && typeof sample === 'object')
                .slice(-MAX_TELEMETRY_SAMPLES);

            if (sanitizedSamples.length === 0) {
                throw new Error('Telemetry samples could not be processed.');
            }

            let handle = modelHandleRef.current;

            if (!handle || handle.trackName !== sanitizedTrackName || handle.carName !== sanitizedCarName) {
                if (handle?.cacheKey) {
                    releasePersistentImitationModel(handle.cacheKey);
                    modelHandleRef.current = null;
                }

                const acquiredHandle = await acquirePersistentImitationModel({
                    trackName: sanitizedTrackName,
                    carName: sanitizedCarName,
                    modelType: 'imitation_learning'
                });

                handle = {
                    cacheKey: acquiredHandle.cacheKey,
                    data: acquiredHandle.data,
                    trackName: sanitizedTrackName,
                    carName: sanitizedCarName
                };
                modelHandleRef.current = handle;
            }

            if (!handle) {
                throw new Error('Model handle is not available after acquisition');
            }

            const result = await runExpertActionPrediction(handle.data, sanitizedSamples);
            setPrediction(result);
        } catch (predictionError) {
            console.error(predictionError);
            setError((predictionError as Error).message);
        } finally {
            setLoading(false);
        }
    }, [environment, ensureTelemetrySamples, sanitizedTrackName, sanitizedCarName]);

    const summaryMetrics = useMemo(() => extractKeySubset(prediction?.prediction), [prediction]);

    const sampledSeries = useMemo(() => {
        const series = prediction?.position_series ?? [];
        if (!series || series.length === 0) return [];
        const step = Math.max(1, Math.floor(series.length / 15));
        return series.filter((_, index) => index % step === 0 || index === series.length - 1);
    }, [prediction]);

    useEffect(() => {
        return () => {
            if (modelHandleRef.current?.cacheKey) {
                releasePersistentImitationModel(modelHandleRef.current.cacheKey);
                modelHandleRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (!modelHandleRef.current) {
            return;
        }

        const { trackName: cachedTrack, carName: cachedCar, cacheKey } = modelHandleRef.current;
        if (cachedTrack !== sanitizedTrackName || cachedCar !== sanitizedCarName) {
            releasePersistentImitationModel(cacheKey);
            modelHandleRef.current = null;
            setPrediction(null);
        }
    }, [sanitizedTrackName, sanitizedCarName]);

    return (
        <Card style={{ width, height, padding: '16px', display: 'flex', flexDirection: 'column' }}>
            <Flex justify="between" align="center" mb="3">
                <Box>
                    <Text size="3" weight="bold">Expert Action Insights</Text>
                    <Text size="1" color="gray">Track: {trackName} • Car: {carName}</Text>
                </Box>
                <Flex gap="2">
                    <Button variant="soft" size="2" onClick={handleRefreshTelemetry} disabled={loading}>
                        Refresh Telemetry
                    </Button>
                    <Button variant="solid" size="2" onClick={handleComputePrediction} disabled={loading}>
                        {loading ? 'Computing…' : 'Compute Predictions'}
                    </Button>
                </Flex>
            </Flex>

            <Separator size="4" mb="3" />

            {environment !== 'electron' && (
                <Box mb="3">
                    <Badge color="amber">Desktop Only</Badge>
                    <Text size="2" color="gray" style={{ marginLeft: '8px' }}>
                        Predictions require the Electron desktop environment.
                    </Text>
                </Box>
            )}

            {error && (
                <Box mb="3" style={{ background: 'var(--red-3)', borderRadius: 8, padding: '12px' }}>
                    <Text size="2" color="red" weight="medium">{error}</Text>
                </Box>
            )}

            {prediction?.status === 'success' ? (
                <Box style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <Box>
                        <Text size="2" weight="medium">Aggregated Expert Actions</Text>
                        <GridMetrics metrics={summaryMetrics} />
                    </Box>

                    <Box>
                        <Text size="2" weight="medium" mb="2">Trajectory Samples</Text>
                        <ScrollArea type="hover" style={{ maxHeight: 180 }}>
                            <Table.Root>
                                <Table.Header>
                                    <Table.Row>
                                        <Table.ColumnHeaderCell>Normalized Position</Table.ColumnHeaderCell>
                                        <Table.ColumnHeaderCell>Speed</Table.ColumnHeaderCell>
                                        <Table.ColumnHeaderCell>Throttle</Table.ColumnHeaderCell>
                                        <Table.ColumnHeaderCell>Brake</Table.ColumnHeaderCell>
                                        <Table.ColumnHeaderCell>Steering</Table.ColumnHeaderCell>
                                    </Table.Row>
                                </Table.Header>
                                <Table.Body>
                                    {sampledSeries.map((entry, index) => (
                                        <Table.Row key={`series-${index}`}>
                                            <Table.Cell>{formatNumber(entry.normalized_position, 3)}</Table.Cell>
                                            <Table.Cell>{formatNumber(entry.expert_optimal_speed)}</Table.Cell>
                                            <Table.Cell>{formatNumber(entry.expert_optimal_throttle)}</Table.Cell>
                                            <Table.Cell>{formatNumber(entry.expert_optimal_brake)}</Table.Cell>
                                            <Table.Cell>{formatNumber(entry.expert_optimal_steering)}</Table.Cell>
                                        </Table.Row>
                                    ))}
                                </Table.Body>
                            </Table.Root>
                        </ScrollArea>
                    </Box>

                    <Box>
                        <Text size="1" color="gray">
                            Samples used: {prediction.metadata?.telemetry_samples_used ?? 'N/A'} • Normalized position available: {prediction.metadata?.has_normalized_positions ? 'Yes' : 'No'}
                        </Text>
                    </Box>
                </Box>
            ) : (
                <Box style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Text color="gray" align="center">
                        {loading ? 'Loading telemetry and computing expert actions…' : 'Run predictions to view expert guidance based on your telemetry.'}
                    </Text>
                </Box>
            )}
        </Card>
    );
};

interface MetricProps {
    metrics: Array<{ key: string; label: string; value: string }>;
}

const GridMetrics: React.FC<MetricProps> = ({ metrics }) => {
    if (!metrics || metrics.length === 0) {
        return (
            <Box mt="2">
                <Text size="1" color="gray">No aggregated predictions available yet.</Text>
            </Box>
        );
    }

    return (
        <Flex wrap="wrap" gap="3" mt="3">
            {metrics.map((metric) => (
                <Box
                    key={metric.key}
                    style={{
                        minWidth: 140,
                        padding: '12px',
                        borderRadius: '8px',
                        background: 'var(--gray-2)'
                    }}
                >
                    <Text size="1" color="gray">{metric.label}</Text>
                    <Text size="3" weight="medium">{metric.value}</Text>
                </Box>
            ))}
        </Flex>
    );
};

export default ExpertActionsChart;
