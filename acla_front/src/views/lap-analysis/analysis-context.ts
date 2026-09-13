import { createContext, Dispatch, SetStateAction } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import {
    RecordedAiAnalysisState,
    RecordedPlaybackSummary,
    createEmptyRecordedPlaybackSummary,
    createIdleRecordedAiAnalysis,
} from './recorded-session-analysis';

export interface AnalysisContextType {
    activeTab: string;
    mapSelected: string | null;
    sessionSelected: RacingSessionDetailedInfoDto | null;
    activeVisualizations: VisualizationInstance[];
    latestGuidanceMessage: string | null;
    recordedAiAnalysis: RecordedAiAnalysisState;
    recordedPlaybackSummary: RecordedPlaybackSummary;
    setMap: (map: string | null) => void;
    setSession: Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>;
    setRecordedPlaybackSummary: Dispatch<SetStateAction<RecordedPlaybackSummary>>;
    runRecordedAiAnalysis: (options?: { force?: boolean }) => Promise<RecordedAiAnalysisState>;
    setActiveTab: Dispatch<SetStateAction<string>>;
    setActiveVisualizations: Dispatch<SetStateAction<VisualizationInstance[]>>;
    sendGuidanceToChat: (message: string) => void;
}

export const AnalysisContext = createContext<AnalysisContextType>({
    activeTab: 'mapLists',
    mapSelected: null,
    sessionSelected: null,
    activeVisualizations: [],
    latestGuidanceMessage: null,
    recordedAiAnalysis: createIdleRecordedAiAnalysis(),
    recordedPlaybackSummary: createEmptyRecordedPlaybackSummary(),
    setMap: () => console.warn('No provider for AnalysisContext'),
    setSession: (() => console.warn('No provider for AnalysisContext')) as Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>,
    setRecordedPlaybackSummary: (() => console.warn('No provider for AnalysisContext')) as Dispatch<SetStateAction<RecordedPlaybackSummary>>,
    runRecordedAiAnalysis: async () => {
        console.warn('No provider for AnalysisContext');
        return createIdleRecordedAiAnalysis();
    },
    setActiveTab: (() => console.warn('No provider for AnalysisContext')) as Dispatch<SetStateAction<string>>,
    setActiveVisualizations: (() => console.warn('No provider for AnalysisContext')) as Dispatch<SetStateAction<VisualizationInstance[]>>,
    sendGuidanceToChat: () => console.warn('No provider for AnalysisContext'),
});
