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

interface ComparisonField {
    label: string;
    expertKey: string;
    liveKey?: string;
    liveAccessor?: (liveSample: Record<string, unknown>) => unknown;
    digits?: number;
}

const COMPARISON_FIELDS: ComparisonField[] = [
    {
        label: 'Speed (km/h)',
        expertKey: 'expert_optimal_speed',
        liveKey: 'Physics_speed_kmh',
        digits: 1
    },
    {
        label: 'Throttle',
        expertKey: 'expert_optimal_throttle',
        liveKey: 'Physics_gas',
        digits: 3
    },
    {
        label: 'Brake',
        expertKey: 'expert_optimal_brake',
        liveKey: 'Physics_brake',
        digits: 3
    },
    {
        label: 'Steering Angle',
        expertKey: 'expert_optimal_steering',
        liveKey: 'Physics_steer_angle',
        digits: 3
    },
    {
        label: 'Gear',
        expertKey: 'expert_optimal_gear',
        liveKey: 'Physics_gear',
        digits: 0
    },
    {
        label: 'Track Position',
        expertKey: 'expert_optimal_track_position',
        liveKey: 'Graphics_normalized_car_position',
        digits: 3
    },
    {
        label: 'Velocity X',
        expertKey: 'expert_optimal_velocity_x',
        liveKey: 'Physics_velocity_x',
        digits: 3
    },
    {
        label: 'Velocity Y',
        expertKey: 'expert_optimal_velocity_y',
        liveKey: 'Physics_velocity_y',
        digits: 3
    },
    {
        label: 'Velocity Z',
        expertKey: 'expert_optimal_velocity_z',
        liveKey: 'Physics_velocity_z',
        digits: 3
    },
    {
        label: 'Player Position X',
        expertKey: 'expert_optimal_player_pos_x',
        digits: 3,
        liveAccessor: (liveSample) => {
            const coordinates = liveSample['Graphics_car_coordinates'];
            if (!Array.isArray(coordinates) || coordinates.length === 0) {
                return undefined;
            }
            const first = coordinates[0] as Record<string, unknown>;
            const value = first ? first['x'] : undefined;
            return typeof value === 'number' ? value : undefined;
        }
    },
    {
        label: 'Player Position Y',
        expertKey: 'expert_optimal_player_pos_y',
        digits: 3,
        liveAccessor: (liveSample) => {
            const coordinates = liveSample['Graphics_car_coordinates'];
            if (!Array.isArray(coordinates) || coordinates.length === 0) {
                return undefined;
            }
            const first = coordinates[0] as Record<string, unknown>;
            const value = first ? first['y'] : undefined;
            return typeof value === 'number' ? value : undefined;
        }
    },
    {
        label: 'Player Position Z',
        expertKey: 'expert_optimal_player_pos_z',
        digits: 3,
        liveAccessor: (liveSample) => {
            const coordinates = liveSample['Graphics_car_coordinates'];
            if (!Array.isArray(coordinates) || coordinates.length === 0) {
                return undefined;
            }
            const first = coordinates[0] as Record<string, unknown>;
            const value = first ? first['z'] : undefined;
            return typeof value === 'number' ? value : undefined;
        }
    }
];

interface ComparisonRow {
    label: string;
    liveValue?: number;
    expertValue?: number;
    delta: number | null;
    digits?: number;
}

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
    const TelemetryDataLiveStatus = analysisContext.TelemetryDataLiveStatus;
    const hasLiveTelemetry = useMemo(() => {
        if (!analysisContext.liveData || Object.keys(analysisContext.liveData).length === 0) {
            return false;
        }
        return TelemetryDataLiveStatus === ACC_STATUS.ACC_LIVE;
    }, [analysisContext.liveData, TelemetryDataLiveStatus]);
    const statusSummary = useMemo(() => {
        switch (TelemetryDataLiveStatus) {
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
    }, [TelemetryDataLiveStatus]);

    const statusMessage = useMemo(() => {
        switch (TelemetryDataLiveStatus) {
            case ACC_STATUS.ACC_PAUSE:
                return 'Game paused. Waiting for live telemetry to resume.';
            case ACC_STATUS.ACC_REPLAY:
                return 'Replay detected. Expert predictions pause until live driving continues.';
            case ACC_STATUS.ACC_OFF:
                return 'Waiting for Assetto Corsa Competizione telemetry.';
            default:
                return null;
        }
    }, [TelemetryDataLiveStatus]);

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
        if (TelemetryDataLiveStatus !== ACC_STATUS.ACC_LIVE) {
            setLoading(false);
            setPrediction(null);
            lastAutoPredictionSignatureRef.current = null;
        }

        if (TelemetryDataLiveStatus !== ACC_STATUS.ACC_LIVE && error) {
            setError(null);
        }
    }, [TelemetryDataLiveStatus, error]);

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

        if (TelemetryDataLiveStatus !== ACC_STATUS.ACC_LIVE) {
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
    }, [analysisContext.liveData, environment, hasLiveTelemetry, initializing, TelemetryDataLiveStatus, sessionReady, teardownSession]);

    const summaryMetrics = useMemo(() => extractKeySubset(prediction?.prediction), [prediction]);

    const sampledSeries = useMemo(() => {
        const series = prediction?.position_series ?? [];
        if (!series || series.length === 0) return [];
        const step = Math.max(1, Math.floor(series.length / 15));
        return series.filter((_, index) => index % step === 0 || index === series.length - 1);
    }, [prediction]);

    const comparisonRows = useMemo<ComparisonRow[]>(() => {
        if (!prediction?.prediction) {
            return [];
        }

        const liveTelemetry = analysisContext.liveData;
        if (!liveTelemetry || typeof liveTelemetry !== 'object') {
            return [];
        }

        const liveSample = liveTelemetry as Record<string, unknown>;
        const expertValues = prediction.prediction as Record<string, number>;

        const rows: ComparisonRow[] = [];

        COMPARISON_FIELDS.forEach((field) => {
            const liveRaw = field.liveAccessor
                ? field.liveAccessor(liveSample)
                : field.liveKey
                    ? liveSample[field.liveKey]
                    : undefined;
            const liveValue = typeof liveRaw === 'number' ? liveRaw : undefined;

            const expertRaw = expertValues[field.expertKey];
            const expertValue = typeof expertRaw === 'number' ? expertRaw : undefined;

            if (liveValue === undefined && expertValue === undefined) {
                return;
            }

            const delta = liveValue !== undefined && expertValue !== undefined ? liveValue - expertValue : null;

            rows.push({
                label: field.label,
                liveValue,
                expertValue,
                delta,
                digits: field.digits
            });
        });

        return rows;
    }, [analysisContext.liveData, prediction]);

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

        if (TelemetryDataLiveStatus !== ACC_STATUS.ACC_LIVE) {
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
    }, [analysisContext.liveData, environment, sessionReady, initializing, loading, TelemetryDataLiveStatus, handleComputePrediction]);

    return (
        <Card
            style={{
                width,
                height,
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                overflowY: 'auto',
                overflowX: 'hidden'
            }}
        >
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
                    <Flex
                        wrap="wrap"
                        gap="3"
                        style={{ width: '100%', alignItems: 'stretch' }}
                    >
                        <SectionContainer title="Aggregated Expert Actions" style={{ flex: '1 1 280px' }}>
                            <GridMetrics metrics={summaryMetrics} />
                        </SectionContainer>

                        {comparisonRows.length > 0 && (
                            <SectionContainer title="Live vs Expert Comparison" style={{ flex: '1 1 320px' }}>
                                <LiveVsExpertTable rows={comparisonRows} />
                            </SectionContainer>
                        )}
                    </Flex>

                    <SectionContainer title="Latest Expert Sample">
                        <ScrollArea type="hover" style={{ maxHeight: 240 }}>
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
                        <Text size="1" color="gray">
                            Samples used: {prediction.metadata?.telemetry_samples_used ?? 'N/A'} • Normalized position available: {prediction.metadata?.has_normalized_positions ? 'Yes' : 'No'}
                        </Text>
                    </SectionContainer>
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
        <Box
            mt="3"
            style={{
                display: 'grid',
                gap: '12px',
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))'
            }}
        >
            {metrics.map((metric) => (
                <Box
                    key={metric.key}
                    style={{
                        padding: '12px',
                        borderRadius: '8px',
                        background: 'var(--gray-3)'
                    }}
                >
                    <Text size="1" color="gray">{metric.label}</Text>
                    <Text size="3" weight="medium">{metric.value}</Text>
                </Box>
            ))}
        </Box>
    );
};

interface ComparisonTableProps {
    rows: ComparisonRow[];
}

const LiveVsExpertTable: React.FC<ComparisonTableProps> = ({ rows }) => {
    if (!rows || rows.length === 0) {
        return (
            <Box mt="2">
                <Text size="1" color="gray">No live telemetry available for comparison.</Text>
            </Box>
        );
    }

    const renderValue = (value: number | undefined, digits?: number) => {
        return typeof value === 'number' ? formatNumber(value, digits) : '-';
    };

    const renderDelta = (delta: number | null, digits?: number) => {
        return typeof delta === 'number' ? formatNumber(delta, digits ?? 3) : '-';
    };

    return (
        <ScrollArea type="hover" style={{ width: '100%' }}>
            <Box mt="2" style={{ minWidth: 520 }}>
                <Table.Root>
                    <Table.Header>
                        <Table.Row>
                            <Table.ColumnHeaderCell>Metric</Table.ColumnHeaderCell>
                            <Table.ColumnHeaderCell>Live Telemetry</Table.ColumnHeaderCell>
                            <Table.ColumnHeaderCell>Expert Prediction</Table.ColumnHeaderCell>
                            <Table.ColumnHeaderCell>Δ (Live - Expert)</Table.ColumnHeaderCell>
                        </Table.Row>
                    </Table.Header>
                    <Table.Body>
                        {rows.map((row) => (
                            <Table.Row key={row.label}>
                                <Table.Cell>{row.label}</Table.Cell>
                                <Table.Cell>{renderValue(row.liveValue, row.digits)}</Table.Cell>
                                <Table.Cell>{renderValue(row.expertValue, row.digits)}</Table.Cell>
                                <Table.Cell>{renderDelta(row.delta, row.digits)}</Table.Cell>
                            </Table.Row>
                        ))}
                    </Table.Body>
                </Table.Root>
            </Box>
        </ScrollArea>
    );
};

interface SectionContainerProps {
    title: string;
    description?: string;
    children: React.ReactNode;
    style?: React.CSSProperties;
}

const SectionContainer: React.FC<SectionContainerProps> = ({ title, description, children, style }) => (
    <Box
        style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            padding: '16px',
            borderRadius: '12px',
            background: 'var(--gray-2)',
            boxShadow: 'inset 0 0 0 1px var(--gray-4)',
            ...style
        }}
    >
        <Box>
            <Text size="2" weight="medium">{title}</Text>
            {description && (
                <Text size="1" color="gray">{description}</Text>
            )}
        </Box>
        {children}
    </Box>
);

export default ExpertActionsChart;
