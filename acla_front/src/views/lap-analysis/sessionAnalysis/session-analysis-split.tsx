import React, { useContext } from 'react';
import { AnalysisContext } from '../analysis-context';
import DynamicVisualizationManager from '../visualization/DynamicVisualizationManager';
import '../visualization/VisualizationRegistry'; // Initialize visualizations
import './session-analysis-split.css';
import { AI_TOOL_COMPONENT_NAMES } from 'contexts/AiToolComponentRefContext';

const SessionAnalysisSplit: React.FC = () => {
    const analysisContext = useContext(AnalysisContext);

    const handleVisualizationLayoutChange = (instances: any[]) => {
        analysisContext.setActiveVisualizations(instances);
    };

    return (
        <div className="sas-container">
            <div className="sas-panel" role="tabpanel">
                <DynamicVisualizationManager
                    name={AI_TOOL_COMPONENT_NAMES.RECORDED_VISUALIZATION_MANAGER}
                    onLayoutChange={handleVisualizationLayoutChange}
                />
            </div>
        </div>
    );
};

export default SessionAnalysisSplit;
