import React, { useContext, useState } from 'react';
import AiChat from '../ai-chat/ai-chat';
import { AnalysisContext } from '../analysis-context';
import DynamicVisualizationManager from '../visualization/DynamicVisualizationManager';
import '../visualization/VisualizationRegistry'; // Initialize visualizations
import './session-analysis-split.css';

type SplitTab = 'visualizations' | 'chat';

const SessionAnalysisSplit: React.FC = () => {
    const analysisContext = useContext(AnalysisContext);
    const [activeTab, setActiveTab] = useState<SplitTab>('chat');

    const handleVisualizationLayoutChange = (instances: any[]) => {
        analysisContext.setActiveVisualizations(instances);
    };

    const sessionLabel = analysisContext.sessionSelected?.session_name || 'Session';

    return (
        <div className="sas-container">
            <div className="sas-tablist" role="tablist" aria-label="Session view">
                <button
                    type="button"
                    role="tab"
                    aria-selected={activeTab === 'chat'}
                    className={`sas-tab ${activeTab === 'chat' ? 'sas-tab--active' : ''}`}
                    onClick={() => setActiveTab('chat')}
                >
                    <span className="sas-tab__dot" />
                    AI Assistant
                </button>
                <button
                    type="button"
                    role="tab"
                    aria-selected={activeTab === 'visualizations'}
                    className={`sas-tab ${activeTab === 'visualizations' ? 'sas-tab--active' : ''}`}
                    onClick={() => setActiveTab('visualizations')}
                >
                    <span className="sas-tab__dot" />
                    Visualizations
                </button>
                <span className="sas-tablist__spacer" />
                <span className="sas-tablist__meta">{sessionLabel}</span>
            </div>

            <div className="sas-panel" role="tabpanel">
                {activeTab === 'visualizations' ? (
                    <DynamicVisualizationManager
                        onLayoutChange={handleVisualizationLayoutChange}
                    />
                ) : (
                    <AiChat
                        sessionId={analysisContext.sessionSelected?.SessionId}
                        title={`AI Assistant — ${sessionLabel}`}
                    />
                )}
            </div>
        </div>
    );
};

export default SessionAnalysisSplit;
