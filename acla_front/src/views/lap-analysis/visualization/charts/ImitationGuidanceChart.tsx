import React, { useContext, useState, useEffect, useCallback, useRef } from 'react';
import { Card, Text, Box, Flex, Button, TextField, Table, IconButton } from '@radix-ui/themes';
import { Cross2Icon } from '@radix-ui/react-icons';
import { AnalysisContext } from '../../analysis-context';
import { VisualizationProps } from '../VisualizationRegistry';
import apiService from 'services/api.service';
import styles from './ImitationGuidanceChart.module.css';

const getNormalizedCarPos = (telemetry: Record<string, any> | null): number | undefined => {
    if (!telemetry) return undefined;
    const keys = [
        'Graphics_normalized_car_position',
        'graphics_normalized_car_position',
        'normalized_car_position',
        'car_position'
    ];
    for (const key of keys) {
        if (key in telemetry) {
            const val = Number(telemetry[key]);
            if (!isNaN(val) && isFinite(val)) return val;
        }
    }
    return undefined;
};

const extractGuidanceText = (raw: any, guidanceResult: any): string | null => {
    if (typeof guidanceResult?.llm?.raw_output === 'string') {
        return guidanceResult.llm.raw_output;
    }
    return null;
};

const ImitationGuidanceChart: React.FC<VisualizationProps> = (props) => {
    const analysisContext = useContext(AnalysisContext);

    const [pacebook, setPacebook] = useState<number[]>([0.1, 0.5, 0.8]);
    const [newValue, setNewValue] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const liveData = analysisContext.liveData as Record<string, any> | null;
    const trackName = analysisContext.recordedSessioStaticsData?.track || 'Unknown Track';
    const carName = analysisContext.recordedSessioStaticsData?.car_model || 'Unknown Car';

    const liveDataRef = useRef(liveData);
    const trackNameRef = useRef(trackName);
    const carNameRef = useRef(carName);
    const lastPosRef = useRef<number | undefined>(undefined);
    const requestInFlightRef = useRef(false);

    useEffect(() => {
        liveDataRef.current = liveData;
    }, [liveData]);

    useEffect(() => {
        trackNameRef.current = trackName;
    }, [trackName]);

    useEffect(() => {
        carNameRef.current = carName;
    }, [carName]);

    const fetchGuidance = useCallback(async () => {
        const currentLiveData = liveDataRef.current;
        if (!currentLiveData || Object.keys(currentLiveData).length === 0) {
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
                track_name: trackNameRef.current,
                car_name: carNameRef.current
            });

            const raw = response.data as any;
            const result = raw?.guidance_result;
            if (result?.status === 'success') {
                const guidanceText = extractGuidanceText(raw, result);
                if (guidanceText && analysisContext.sendGuidanceToChat) {
                    analysisContext.sendGuidanceToChat(guidanceText);
                }
            } else {
                setError('Failed to get guidance: API returned error status');
            }
        } catch (err: any) {
            console.error('Imitation learning guidance error:', err);
            setError('API call failed: ' + (err.response?.data?.message || err.message));
        } finally {
            setLoading(false);
            requestInFlightRef.current = false;
        }
    }, [analysisContext]);

    // Check crossing pacebook points
    useEffect(() => {
        const currentPos = getNormalizedCarPos(liveData);
        const lastPos = lastPosRef.current;

        if (currentPos !== undefined && lastPos !== undefined) {
            let crossed = false;
            // The car position goes from 0 to 1 over the lap.
            if (currentPos >= lastPos) {
                // Normal progression (e.g., 0.4 to 0.6)
                crossed = pacebook.some(p => lastPos < p && currentPos >= p);
            } else {
                // Wrapped around the finish line (e.g., 0.99 to 0.01)
                // Check if crossed any point between lastPos and 1.0, or between 0 and currentPos
                crossed = pacebook.some(p => lastPos < p || currentPos >= p);
            }

            if (crossed) {
                fetchGuidance();
            }
        }

        if (currentPos !== undefined) {
            lastPosRef.current = currentPos;
        }
    }, [liveData, pacebook, fetchGuidance]);

    const handleAdd = () => {
        const val = parseFloat(newValue);
        if (!isNaN(val) && val >= 0 && val <= 1 && !pacebook.includes(val)) {
            const newPacebook = [...pacebook, val].sort((a, b) => a - b);
            setPacebook(newPacebook);
            setNewValue('');
        }
    };

    const handleDelete = (val: number) => {
        setPacebook(pacebook.filter(p => p !== val));
    };

    return (
        <Card className={styles.imitationGuidanceChart} style={{ height: '100%', minHeight: 0 }}>
            <Flex direction="column" height="100%" style={{ minHeight: 0, flex: 1 }}>
                <Box p="3" style={{ borderBottom: '1px solid var(--gray-6)', flexShrink: 0 }}>
                    <Text size="3" weight="bold">Auto Guidance Pacebook</Text>
                    <Text size="1" color="gray" style={{ display: 'block', marginTop: '4px' }}>
                        Set normalized car positions (0-1) to automatically request AI guidance.
                    </Text>
                </Box>

                <Flex
                    direction="column"
                    flexGrow="1"
                    style={{
                        overflowY: 'auto',
                        flexBasis: '100%',
                        minHeight: 0,
                        padding: '16px'
                    }}
                >
                    {error && (
                        <Box p="3" mb="3" style={{ borderBottom: '1px solid var(--red-6)', backgroundColor: 'var(--red-2)', flexShrink: 0 }}>
                            <Text size="2" color="red">{error}</Text>
                        </Box>
                    )}

                    <Flex gap="2" mb="4" align="center" style={{ flexShrink: 0 }}>
                        <TextField.Root
                            placeholder="E.g. 0.25"
                            value={newValue}
                            onChange={(e) => setNewValue(e.target.value)}
                            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                                if (e.key === 'Enter') handleAdd();
                            }}
                        />
                        <Button onClick={handleAdd}>Add Position</Button>
                    </Flex>

                    <Table.Root variant="surface">
                        <Table.Header>
                            <Table.Row>
                                <Table.ColumnHeaderCell>Normalized Position</Table.ColumnHeaderCell>
                                <Table.ColumnHeaderCell justify="end">Action</Table.ColumnHeaderCell>
                            </Table.Row>
                        </Table.Header>
                        <Table.Body>
                            {pacebook.length === 0 ? (
                                <Table.Row>
                                    <Table.Cell colSpan={2}>
                                        <Text color="gray">No positions set.</Text>
                                    </Table.Cell>
                                </Table.Row>
                            ) : (
                                pacebook.map((pos) => (
                                    <Table.Row key={pos}>
                                        <Table.Cell>{pos.toFixed(4)}</Table.Cell>
                                        <Table.Cell justify="end">
                                            <IconButton size="1" variant="ghost" color="red" onClick={() => handleDelete(pos)}>
                                                <Cross2Icon />
                                            </IconButton>
                                        </Table.Cell>
                                    </Table.Row>
                                ))
                            )}
                        </Table.Body>
                    </Table.Root>

                    <Box mt="4" style={{ flexShrink: 0 }}>
                        <Text size="2" color="gray">
                            Current Position: {getNormalizedCarPos(liveData)?.toFixed(4) ?? 'Unknown'}
                        </Text>
                        {loading && (
                            <Text size="2" color="blue" style={{ display: 'block', marginTop: '8px' }}>
                                Requesting guidance...
                            </Text>
                        )}
                    </Box>
                </Flex>
            </Flex>
        </Card>
    );
};

export default ImitationGuidanceChart;
