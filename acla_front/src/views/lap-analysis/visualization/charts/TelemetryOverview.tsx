import React, { useContext } from 'react';
import { Card, Text, Box, Grid } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';

const TelemetryOverview: React.FC<VisualizationProps> = ({ id, data, config, width = '100%', height = 200 }) => {
    const analysisContext = useContext(AnalysisContext);

    const telemetryData = data || analysisContext.liveData;

    // Calculate current statistics from live data
    const stats = React.useMemo(() => {
        if (!telemetryData || Object.keys(telemetryData).length === 0) return null;

        // Extract current values from flattened live data structure
        const currentSpeed = telemetryData.Physics_speed_kmh || 0;
        const currentSteer = Math.abs(telemetryData.Physics_steer_angle || 0);
        const currentGas = telemetryData.Physics_gas || 0;
        const currentBrake = telemetryData.Physics_brake || 0;
        const currentLap = telemetryData.Graphics_completed_lap || 0;

        return {
            currentSpeed: currentSpeed.toFixed(1),
            currentSteer: currentSteer.toFixed(3),
            currentGas: currentGas.toFixed(2),
            currentBrake: currentBrake.toFixed(2),
            currentLap: currentLap,
            rpm: telemetryData.Physics_rpm || 0,
            gear: telemetryData.Physics_gear || 0,
            fuel: telemetryData.Physics_fuel || 0
        };
    }, [telemetryData]);

    return (
        <Card style={{ width, height, padding: '16px' }}>
            <Text size="3" weight="bold" style={{ marginBottom: '12px' }}>Telemetry Overview</Text>
            {stats ? (
                <Grid columns="2" gap="2" style={{ height: 'calc(100% - 40px)' }}>
                    <Box>
                        <Text size="2" color="gray">Current Speed</Text>
                        <Text size="4" weight="bold">{stats.currentSpeed} km/h</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Current RPM</Text>
                        <Text size="4" weight="bold">{stats.rpm}</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Steering Input</Text>
                        <Text size="4" weight="bold">{(parseFloat(stats.currentSteer) * 100).toFixed(0)}%</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Gas Pedal</Text>
                        <Text size="4" weight="bold">{(parseFloat(stats.currentGas) * 100).toFixed(0)}%</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Brake Pedal</Text>
                        <Text size="4" weight="bold">{(parseFloat(stats.currentBrake) * 100).toFixed(0)}%</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Current Gear</Text>
                        <Text size="4" weight="bold">{stats.gear}</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Fuel Level</Text>
                        <Text size="4" weight="bold">{stats.fuel}L</Text>
                    </Box>
                    <Box>
                        <Text size="2" color="gray">Completed Laps</Text>
                        <Text size="4" weight="bold">{stats.currentLap}</Text>
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
