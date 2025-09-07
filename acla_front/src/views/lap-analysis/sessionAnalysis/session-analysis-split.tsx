import React, { useContext } from 'react';
import { Box, Flex, Card, Separator } from '@radix-ui/themes';
import AiChat from '../ai-chat/ai-chat';
import { AnalysisContext } from '../session-analysis';
import DynamicVisualizationManager from '../visualization/DynamicVisualizationManager';
import '../visualization'; // Initialize visualizations
import './session-analysis-split.css';

const SessionAnalysisSplit: React.FC = () => {
    const analysisContext = useContext(AnalysisContext);

    const handleVisualizationLayoutChange = (instances: any[]) => {
        analysisContext.setActiveVisualizations(instances);
    };

    return (
        <div className="session-analysis-split-container">
            <Flex gap="3" style={{ height: '100%' }}>

                {/* Dynamic Visualization Section - takes up 60% of total width */}
                <Box className="visualization-section" style={{ flex: '0 0 60%' }}>
                    <Card style={{ height: '100%' }}>
                        <DynamicVisualizationManager
                            onLayoutChange={handleVisualizationLayoutChange}
                        />
                    </Card>
                </Box>

                <Separator orientation="vertical" />

                {/* AI Chat Section - takes up 40% of total width */}
                <Box className="chat-section" style={{ flex: '0 0 calc(40% - 24px)' }}>
                    <AiChat
                        sessionId={analysisContext.sessionSelected?.SessionId}
                        title={`AI Analysis - ${analysisContext.sessionSelected?.session_name || 'Session'}`}
                    />
                </Box>
            </Flex>
        </div>
    );
};

export default SessionAnalysisSplit;
