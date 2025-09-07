import React, { useContext } from 'react';
import { Card, Text, Box, Grid } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';

const TelemetryOverview: React.FC<VisualizationProps> = ({ id, data, config, width = '100%', height = 200 }) => {
    const analysisContext = useContext(AnalysisContext);

    const telemetryData = data || analysisContext.recordedSessionData;

    // Calculate basic statistics
    const stats = React.useMemo(() => {
        if (!telemetryData || telemetryData.length === 0) return null;

        const speeds = telemetryData.map((d: any) => d.speed || 0).filter((s: number) => s > 0);
        const laps = telemetryData.filter((d: any) => d.lapCount);

        return {
            avgSpeed: speeds.length > 0 ? (speeds.reduce((a: number, b: number) => a + b, 0) / speeds.length).toFixed(1) : '0',
            maxSpeed: speeds.length > 0 ? Math.max(...speeds).toFixed(1) : '0',
            totalLaps: laps.length,
            dataPoints: telemetryData.length
        };
    }, [telemetryData]);

    return (
        <Card style={{ width, height, padding: '16px' }}>
            <Text size="3" weight="bold" style={{ marginBottom: '12px' }}>Telemetry Overview</Text>
            {stats ? (
                <Grid columns="2" gap="3" style={{ height: 'calc(100% - 40px)' }}>
                    <Box>
                        <Text size="2" color="gray">Average Speed</Text>
                        <Text size="4" weight="bold">{stats.avgSpeed} km/h</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Max Speed</Text>
                        <Text size="4" weight="bold">{stats.maxSpeed} km/h</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Total Laps</Text>
                        <Text size="4" weight="bold">{stats.totalLaps}</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Data Points</Text>
                        <Text size="4" weight="bold">{stats.dataPoints}</Text>
                    </Box>
                </Grid>
            ) : (
                <Box style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100% - 40px)' }}>
                    <Text color="gray">No telemetry data available</Text>
                </Box>
            )}
        </Card>
    );
};

export default TelemetryOverview;
