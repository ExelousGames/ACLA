import React, { useContext, useState, useMemo, useEffect } from 'react';
import { Card, Text, Box, Grid, TextField, Button } from '@radix-ui/themes';
import { MagnifyingGlassIcon } from '@radix-ui/react-icons';
import { AnalysisContext } from '../../analysis-context';
import { VisualizationProps } from '../VisualizationRegistry';

const TelemetryOverview: React.FC<VisualizationProps> = ({ id, data, config, width = '100%', height = 200 }) => {
    const analysisContext = useContext(AnalysisContext);
    const [searchTerm, setSearchTerm] = useState('');
    const [history, setHistory] = useState<any[]>([]);
    const [historyIndex, setHistoryIndex] = useState(0);

    const telemetryData = data || analysisContext.liveData;

    useEffect(() => {
        if (telemetryData) {
            const keys = Object.keys(telemetryData);
            // Store data if it has more than just the 'available' key, or a single key that isn't 'available'
            if (keys.length > 1 || (keys.length === 1 && keys[0] !== 'available')) {
                setHistory(prev => {
                    if (prev.length > 0 && JSON.stringify(prev[0]) === JSON.stringify(telemetryData)) {
                        return prev;
                    }
                    return [telemetryData, ...prev].slice(0, 3);
                });
                setHistoryIndex(0);
            }
        }
    }, [telemetryData]);

    const dataToDisplay = useMemo(() => {
        if (telemetryData) {
            const keys = Object.keys(telemetryData);
            if (keys.length > 1 || (keys.length === 1 && keys[0] !== 'available')) {
                return telemetryData;
            }
        }
        return history[historyIndex] || telemetryData;
    }, [telemetryData, history, historyIndex]);

    const isDataUnavailable = !telemetryData || Object.keys(telemetryData).length === 0 || (Object.keys(telemetryData).length === 1 && Object.keys(telemetryData)[0] === 'available');

    const filteredData = useMemo(() => {
        if (!dataToDisplay) return [];

        const entries = Object.entries(dataToDisplay);
        if (!searchTerm) return entries;

        const lowerSearch = searchTerm.toLowerCase();
        return entries
            .filter(([key]) => key.toLowerCase().includes(lowerSearch))
            .sort(([keyA], [keyB]) => {
                const aStartsWith = keyA.toLowerCase().startsWith(lowerSearch);
                const bStartsWith = keyB.toLowerCase().startsWith(lowerSearch);
                if (aStartsWith && !bStartsWith) return -1;
                if (!aStartsWith && bStartsWith) return 1;
                return keyA.localeCompare(keyB);
            });
    }, [dataToDisplay, searchTerm]);

    return (
        <Card style={{ width, height, padding: '16px', display: 'flex', flexDirection: 'column' }}>
            <Box style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <Text size="3" weight="bold">Telemetry Overview</Text>
                <Box style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                    {isDataUnavailable && history.length > 1 && (
                        <Box style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <Button size="1" variant="soft" disabled={historyIndex === history.length - 1} onClick={() => setHistoryIndex(i => Math.min(i + 1, history.length - 1))}>
                                Prev Data
                            </Button>
                            <Text size="1" color="gray">
                                {historyIndex === 0 ? 'Last saved' : `${historyIndex} steps ago`}
                            </Text>
                            <Button size="1" variant="soft" disabled={historyIndex === 0} onClick={() => setHistoryIndex(i => Math.max(i - 1, 0))}>
                                Next Data
                            </Button>
                        </Box>
                    )}
                    <TextField.Root placeholder="Search features..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)}>
                        <TextField.Slot>
                            <MagnifyingGlassIcon height="16" width="16" />
                        </TextField.Slot>
                    </TextField.Root>
                </Box>
            </Box>
            {dataToDisplay && Object.keys(dataToDisplay).length > 0 ? (
                <Box style={{ flex: 1, overflowY: 'auto' }}>
                    {filteredData.length > 0 ? (
                        <Grid columns="2" gap="3" style={{ paddingRight: '8px' }}>
                            {filteredData.map(([key, value]) => {
                                let displayValue = value;
                                if (typeof value === 'number') {
                                    displayValue = Number.isInteger(value) ? value : value.toFixed(3);
                                } else if (typeof value === 'boolean') {
                                    displayValue = value ? 'Yes' : 'No';
                                } else if (typeof value === 'object' && value !== null) {
                                    displayValue = JSON.stringify(value);
                                }

                                return (
                                    <Box key={key}>
                                        <Text size="2" color="gray">{key}</Text>
                                        <Text size="3" weight="bold" as="div" style={{ wordBreak: 'break-word' }}>
                                            {String(displayValue)}
                                        </Text>
                                    </Box>
                                );
                            })}
                        </Grid>
                    ) : (
                        <Box style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                            <Text color="gray">No matching telemetry features found</Text>
                        </Box>
                    )}
                </Box>
            ) : (
                <Box style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
                    <Text color="gray">No telemetry data available</Text>
                </Box>
            )}
        </Card>
    );
};

export default TelemetryOverview;
