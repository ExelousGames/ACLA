import React, { useContext } from 'react';
import { Card, Text, Box } from '@radix-ui/themes';
import { AnalysisContext } from '../../session-analysis';
import { VisualizationProps } from '../VisualizationRegistry';
import SessionAnalysisMap from '../../sessionAnalysis/sessionAnalysisMap';

const MapVisualization: React.FC<VisualizationProps> = ({ id, data, config, width = '100%', height = 300 }) => {
    const analysisContext = useContext(AnalysisContext);

    return (
        <Card style={{ width, height, padding: '0', overflow: 'hidden' }}>
            <Box style={{ height: '100%', position: 'relative' }}>
                <SessionAnalysisMap />
            </Box>
        </Card>
    );
};

export default MapVisualization;
