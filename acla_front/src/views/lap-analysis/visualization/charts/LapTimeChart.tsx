import React, { useContext, useEffect, useState } from 'react';
import { Card, Text, Box } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';

const LapTimeChart: React.FC<VisualizationProps> = ({ id, data, config, width = '100%', height = 250 }) => {
    const analysisContext = useContext(AnalysisContext);
    const [lapData, setLapData] = useState<any[]>([]);

    useEffect(() => {
        if (data) {
            // Use provided data if available
            setLapData(data.filter((d: any) => d.lapTime));
        } else if (analysisContext.recordedSessionDataFilePath) {
            // Load data from file
            analysisContext.readRecordedSessionData()
                .then(sessionData => {
                    setLapData(sessionData.filter((d: any) => d.lapTime));
                })
                .catch(error => {
                    console.error('Error loading lap time data:', error);
                    setLapData([]);
                });
        } else {
            setLapData([]);
        }
    }, [data, analysisContext.recordedSessionDataFilePath]);

    const maxLapTime = lapData?.reduce((max: number, current: any) => Math.max(max, current.lapTime || 0), 0) || 0;
    const minLapTime = lapData?.reduce((min: number, current: any) => Math.min(min, current.lapTime || Infinity), Infinity) || 0;

    const formatTime = (seconds: number) => {
        const minutes = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(3);
        return `${minutes}:${secs.padStart(6, '0')}`;
    };

    return (
        <Card style={{ width, height, padding: '16px' }}>
            <Text size="3" weight="bold" style={{ marginBottom: '12px' }}>Lap Time Analysis</Text>
            {lapData && lapData.length > 0 ? (
                <Box style={{ height: 'calc(100% - 40px)', position: 'relative' }}>
                    <svg width="100%" height="100%" viewBox="0 0 400 200">
                        {/* Bar chart representation */}
                        {lapData.map((d: any, i: number) => {
                            const barHeight = ((d.lapTime - minLapTime) / (maxLapTime - minLapTime)) * 160;
                            const barWidth = (380 / lapData.length) - 2;
                            const x = (i * (380 / lapData.length)) + 10;
                            const y = 180 - barHeight;

                            return (
                                <rect
                                    key={i}
                                    x={x}
                                    y={y}
                                    width={barWidth}
                                    height={barHeight}
                                    fill="#10b981"
                                    opacity={0.8}
                                />
                            );
                        })}
                        {/* Y-axis labels */}
                        <text x="5" y="25" fontSize="10" fill="#666">
                            {formatTime(maxLapTime)}
                        </text>
                        <text x="5" y="185" fontSize="10" fill="#666">
                            {formatTime(minLapTime)}
                        </text>
                        {/* X-axis label */}
                        <text x="200" y="215" fontSize="10" fill="#666" textAnchor="middle">Laps</text>
                    </svg>
                </Box>
            ) : (
                <Box style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100% - 40px)' }}>
                    <Text color="gray">No lap time data available</Text>
                </Box>
            )}
        </Card>
    );
};

export default LapTimeChart;
