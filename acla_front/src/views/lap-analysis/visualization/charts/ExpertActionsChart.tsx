import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Card, Text, Box, Flex, Badge, Separator, Table, ScrollArea } from '@radix-ui/themes';
import { AnalysisContext } from '../../analysis-context';
import { VisualizationProps } from '../VisualizationRegistry';
import { useEnvironment } from 'contexts/EnvironmentContext';
import {
    acquirePersistentImitationModel,
    releasePersistentImitationModel,
    createExpertActionsRunner,
    ExpertActionsRunner,
    ExpertPredictionResult
} from 'services/expertActionsService';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';

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

    const [prediction, setPrediction] = useState<ExpertPredictionResult | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [sessionReady, setSessionReady] = useState<boolean>(false);
    const [initializing, setInitializing] = useState<boolean>(false);
    const [sessionNonce, setSessionNonce] = useState<number>(0);

    type ModelHandle = {
        cacheKey: string;
        data: any;
        trackName?: string;
        carName?: string;
    };

    const modelHandleRef = useRef<ModelHandle | null>(null);
    const sessionRef = useRef<ExpertActionsRunner | null>(null);
    const lastAutoPredictionSignatureRef = useRef<string | null>(null);

    const teardownSession = useCallback(async () => {
        if (sessionRef.current) {
            try {
                await sessionRef.current.dispose();
            } catch (disposeError) {
                console.warn('Failed to dispose expert prediction session', disposeError);
            }
            sessionRef.current = null;
        }

        if (modelHandleRef.current?.cacheKey) {
            releasePersistentImitationModel(modelHandleRef.current.cacheKey);
        }

        modelHandleRef.current = null;
        lastAutoPredictionSignatureRef.current = null;
    }, []);

    const trackName = analysisContext.recordedSessioStaticsData?.track || 'Unknown Track';
    const carName = analysisContext.recordedSessioStaticsData?.car_model || 'Unknown Car';
    const sanitizedTrackName = trackName === 'Unknown Track' ? undefined : trackName;
    const sanitizedCarName = MODEL_CAR_NAME; // Always use the pooled AllCars imitation model variant.
    const liveStatus = analysisContext.liveStatus;
    const hasLiveTelemetry = useMemo(() => {
        if (!analysisContext.liveData || Object.keys(analysisContext.liveData).length === 0) {
            return false;
        }
        return liveStatus === ACC_STATUS.ACC_LIVE;
    }, [analysisContext.liveData, liveStatus]);
    const statusSummary = useMemo(() => {
        switch (liveStatus) {
            case ACC_STATUS.ACC_LIVE:
                return { label: 'Live', color: 'green' as const };
            case ACC_STATUS.ACC_PAUSE:
                return { label: 'Paused', color: 'amber' as const };
            case ACC_STATUS.ACC_REPLAY:
                return { label: 'Replay', color: 'blue' as const };
            case ACC_STATUS.ACC_OFF:
                return { label: 'Idle', color: 'gray' as const };
            default:
                return { label: 'Unknown', color: 'gray' as const };
        }
    }, [liveStatus]);

    const statusMessage = useMemo(() => {
        switch (liveStatus) {
            case ACC_STATUS.ACC_PAUSE:
                return 'Game paused. Waiting for live telemetry to resume.';
            case ACC_STATUS.ACC_REPLAY:
                return 'Replay detected. Expert predictions pause until live driving continues.';
            case ACC_STATUS.ACC_OFF:
                return 'Waiting for Assetto Corsa Competizione telemetry.';
            default:
                return null;
        }
    }, [liveStatus]);

    const telemetryNotice = useMemo(() => {
        if (statusMessage) {
            return statusMessage;
        }
        return 'Waiting for live telemetry. Start or resume a live session to enable expert predictions.';
    }, [statusMessage]);

    const fallbackText = useMemo(() => {
        if (environment !== 'electron') {
            return 'Predictions require the Electron desktop environment.';
        }
        if (loading) {
            return 'Loading telemetry and computing expert actions…';
        }
        if (initializing) {
            return 'Preparing expert model. This may take a moment…';
        }
        if (statusMessage) {
            return statusMessage;
        }
        if (!hasLiveTelemetry) {
            return 'Waiting for live telemetry. Start or resume a live session to enable expert predictions.';
        }
        return 'Start a live session and run predictions to view expert guidance based on your telemetry.';
    }, [environment, hasLiveTelemetry, initializing, loading, statusMessage]);

    useEffect(() => {
        if (liveStatus !== ACC_STATUS.ACC_LIVE) {
            setLoading(false);
            setPrediction(null);
            lastAutoPredictionSignatureRef.current = null;
        }

        if (liveStatus !== ACC_STATUS.ACC_LIVE && error) {
            setError(null);
        }
    }, [liveStatus, error]);

    const handleComputePrediction = useCallback(async () => {
        if (environment !== 'electron') {
            setError('Expert action predictions are available on the desktop application only.');
            return;
        }

        const runner = sessionRef.current;

        if (initializing || !sessionReady || !runner) {
            setError('Expert prediction session is still preparing. Please try again shortly.');
            return;
        }

        if (liveStatus !== ACC_STATUS.ACC_LIVE) {
            setLoading(false);
            return;
        }

        setLoading(true);
        setError(null);

        try {
            if (!hasLiveTelemetry) {
                setLoading(false);
                return;
            }

            const liveTelemetry = analysisContext.liveData;

            const sanitizedSamples = [liveTelemetry].filter((sample) => sample && typeof sample === 'object');
            if (sanitizedSamples.length === 0) {
                throw new Error('Telemetry samples could not be processed.');
            }
            console.log('Computing expert actions for telemetry sample:', sanitizedSamples);
            const result = await runner.predict(sanitizedSamples);
            console.log('Received expert prediction result:', result);
            setPrediction(result);
        } catch (predictionError) {
            console.error(predictionError);
            const message = (predictionError as Error).message || 'Failed to compute expert actions.';
            setError(message);

            const normalized = message.toLowerCase();
            if (normalized.includes('session')) {
                await teardownSession();
                setSessionReady(false);
                setSessionNonce((value) => value + 1);
            }
        } finally {
            setLoading(false);
        }
    }, [analysisContext.liveData, environment, hasLiveTelemetry, initializing, liveStatus, sessionReady, teardownSession]);

    const summaryMetrics = useMemo(() => extractKeySubset(prediction?.prediction), [prediction]);

    const sampledSeries = useMemo(() => {
        const series = prediction?.position_series ?? [];
        if (!series || series.length === 0) return [];
        const step = Math.max(1, Math.floor(series.length / 15));
        return series.filter((_, index) => index % step === 0 || index === series.length - 1);
    }, [prediction]);

    useEffect(() => {
        let cancelled = false;

        lastAutoPredictionSignatureRef.current = null;

        const initializeSession = async () => {
            if (environment !== 'electron') {
                await teardownSession();
                if (!cancelled) {
                    setSessionReady(false);
                    setInitializing(false);
                    setPrediction(null);
                }
                return;
            }

            setInitializing(true);
            setSessionReady(false);
            setError(null);
            setPrediction(null);

            await teardownSession();

            try {
                const acquiredHandle = await acquirePersistentImitationModel({
                    trackName: sanitizedTrackName,
                    carName: sanitizedCarName,
                    modelType: 'imitation_learning'
                });

                if (cancelled) {
                    releasePersistentImitationModel(acquiredHandle.cacheKey);
                    return;
                }

                const newHandle: ModelHandle = {
                    cacheKey: acquiredHandle.cacheKey,
                    data: acquiredHandle.data,
                    trackName: sanitizedTrackName,
                    carName: sanitizedCarName
                };

                modelHandleRef.current = newHandle;

                const runner = await createExpertActionsRunner(newHandle.data);

                if (cancelled) {
                    await runner.dispose().catch(() => undefined);
                    releasePersistentImitationModel(newHandle.cacheKey);
                    modelHandleRef.current = null;
                    return;
                }

                sessionRef.current = runner;
                setSessionReady(true);
            } catch (initError) {
                if (!cancelled) {
                    console.error('Failed to initialize expert prediction session', initError);
                    setError((initError as Error).message);
                }
                await teardownSession();
            } finally {
                if (!cancelled) {
                    setInitializing(false);
                }
            }
        };

        void initializeSession();

        return () => {
            cancelled = true;
            void teardownSession();
        };
    }, [environment, sanitizedTrackName, sanitizedCarName, sessionNonce, teardownSession]);

    useEffect(() => {
        if (!sessionReady) {
            lastAutoPredictionSignatureRef.current = null;
        }
    }, [sessionReady, sanitizedTrackName, sanitizedCarName, sessionNonce]);

    useEffect(() => {
        if (environment !== 'electron') {
            lastAutoPredictionSignatureRef.current = null;
        }
    }, [environment]);

    useEffect(() => {
        if (environment !== 'electron') {
            return;
        }

        if (!sessionReady || initializing || loading) {
            return;
        }

        if (liveStatus !== ACC_STATUS.ACC_LIVE) {
            return;
        }

        const liveTelemetry = analysisContext.liveData;
        if (!liveTelemetry || typeof liveTelemetry !== 'object') {
            return;
        }

        const liveKeys = Object.keys(liveTelemetry);
        if (liveKeys.length === 0) {
            return;
        }

        let signature: string;
        try {
            signature = JSON.stringify(liveTelemetry);
        } catch (error) {
            signature = `fallback-${Date.now()}`;
        }

        if (lastAutoPredictionSignatureRef.current === signature) {
            return;
        }

        lastAutoPredictionSignatureRef.current = signature;
        void handleComputePrediction();
    }, [analysisContext.liveData, environment, sessionReady, initializing, loading, liveStatus, handleComputePrediction]);

    return (
        <Card style={{ width, height, padding: '16px', display: 'flex', flexDirection: 'column' }}>
            <Flex justify="between" align="center" mb="3">
                <Box>
                    <Text size="3" weight="bold">Expert Action Insights</Text>
                    <Text size="1" color="gray">Track: {trackName} • Car: {carName}</Text>
                </Box>
                <Flex align="center" gap="2">
                    <Badge color={statusSummary.color}>{statusSummary.label}</Badge>
                    {loading && <Text size="1" color="gray">Computing…</Text>}
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

            {!error && environment === 'electron' && initializing && (
                <Box mb="3" style={{ background: 'var(--gray-2)', borderRadius: 8, padding: '12px' }}>
                    <Text size="2" color="gray">Loading imitation model and preparing expert prediction session…</Text>
                </Box>
            )}

            {!error && !hasLiveTelemetry && (
                <Box mb="3" style={{ background: 'var(--gray-2)', borderRadius: 8, padding: '12px' }}>
                    <Text size="2" color="gray">{telemetryNotice}</Text>
                </Box>
            )}

            {prediction?.status === 'success' ? (
                <Box style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <Box>
                        <Text size="2" weight="medium">Aggregated Expert Actions</Text>
                        <GridMetrics metrics={summaryMetrics} />
                    </Box>

                    <Box>
                        <Text size="2" weight="medium" mb="2">Latest Expert Sample</Text>
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
                        {fallbackText}
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
