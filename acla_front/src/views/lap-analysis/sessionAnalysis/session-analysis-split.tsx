import React, { useContext } from 'react';
import { AnalysisContext } from '../analysis-context';
import DynamicVisualizationManager from '../visualization/DynamicVisualizationManager';
import '../visualization/VisualizationRegistry'; // Initialize visualizations
import './session-analysis-split.css';

const SessionAnalysisSplit: React.FC = () => {
    const analysisContext = useContext(AnalysisContext);

    const handleVisualizationLayoutChange = (instances: any[]) => {
        analysisContext.setActiveVisualizations(instances);
    };

    return (
        <div className="sas-container">
            <div className="sas-panel" role="tabpanel">
                <DynamicVisualizationManager
                    onLayoutChange={handleVisualizationLayoutChange}
                />
            </div>
        </div>
    );
};

export default SessionAnalysisSplit;
