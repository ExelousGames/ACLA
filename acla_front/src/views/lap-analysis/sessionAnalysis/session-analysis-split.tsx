import React, { useContext } from 'react';
import { Box, Flex, Card, Separator } from '@radix-ui/themes';
import SessionAnalysisMap from './sessionAnalysisMap';
import AiChat from '../../../components/ai-chat/ai-chat';
import { AnalysisContext } from '../session-analysis';
import './session-analysis-split.css';

const SessionAnalysisSplit: React.FC = () => {
    const analysisContext = useContext(AnalysisContext);

    return (
        <div className="session-analysis-split-container">
            <Flex gap="3" style={{ height: '100%' }}>
                {/* Map Section - takes up 60% of the width */}
                <Box className="map-section" style={{ flex: '0 0 60%' }}>
                    <Card style={{ height: '100%' }}>
                        <SessionAnalysisMap />
                    </Card>
                </Box>

                <Separator orientation="vertical" />

                {/* AI Chat Section - takes up 40% of the width */}
                <Box className="chat-section" style={{ flex: '0 0 calc(40% - 16px)' }}>
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
