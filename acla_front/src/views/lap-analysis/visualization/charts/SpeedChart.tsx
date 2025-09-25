import React, { useContext, useEffect, useState } from 'react';
import { Card, Text, Box } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';

const SpeedChart: React.FC<VisualizationProps> = ({ id, data, config, width = '100%', height = 300 }) => {
    const analysisContext = useContext(AnalysisContext);
    const [speedData, setSpeedData] = useState<any[]>([]);

    useEffect(() => {
        if (data) {
            // Use provided data if available
            setSpeedData(data.filter((d: any) => d.speed));
        } else if (analysisContext.recordedSessionDataFilePath) {
            // Load data from file
            analysisContext.readRecordedSessionData()
                .then(sessionData => {
                    setSpeedData(sessionData.filter((d: any) => d.speed));
                })
                .catch(error => {
                    console.error('Error loading speed data:', error);
                    setSpeedData([]);
                });
        } else {
            setSpeedData([]);
        }
    }, [data, analysisContext.recordedSessionDataFilePath]);

    const maxSpeed = speedData?.reduce((max: number, current: any) => Math.max(max, current.speed || 0), 0) || 0;

    return (
        <Card style={{ width, height, padding: '16px' }}>
            <Text size="3" weight="bold" style={{ marginBottom: '12px' }}>Speed Analysis</Text>
            {speedData && speedData.length > 0 ? (
                <Box style={{ height: 'calc(100% - 40px)', position: 'relative' }}>
                    <svg width="100%" height="100%" viewBox="0 0 400 200">
                        {/* Simple line chart representation */}
                        <polyline
                            fill="none"
                            stroke="#3b82f6"
                            strokeWidth="2"
                            points={speedData.map((d: any, i: number) =>
                                `${(i / speedData.length) * 380 + 10},${190 - (d.speed / maxSpeed) * 170}`
                            ).join(' ')}
                        />
                        {/* Y-axis labels */}
                        <text x="5" y="25" fontSize="10" fill="#666">
                            {maxSpeed.toFixed(0)} km/h
                        </text>
                        <text x="5" y="195" fontSize="10" fill="#666">0</text>
                        {/* X-axis label */}
                        <text x="200" y="215" fontSize="10" fill="#666" textAnchor="middle">Time</text>
                    </svg>
                </Box>
            ) : (
                <Box style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100% - 40px)' }}>
                    <Text color="gray">No speed data available</Text>
                </Box>
            )}
        </Card>
    );
};

export default SpeedChart;
