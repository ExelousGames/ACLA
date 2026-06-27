import { createContext, Dispatch, SetStateAction } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import { SessionIntelligence } from './session-intelligence/SessionIntelligence';
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
    liveData: any;
    TelemetryDataLiveStatus: ACC_STATUS | null;
    recordedSessionDataFilePath: string | null;
    recordedTelemetryDataCount: number;
    recordedSessioStaticsData: any;
    activeVisualizations: VisualizationInstance[];
    latestGuidanceMessage: string | null;
    sessionIntelligence: SessionIntelligence | null;
    recordedAiAnalysis: RecordedAiAnalysisState;
    recordedPlaybackSummary: RecordedPlaybackSummary;
    setMap: (map: string | null) => void;
    setSession: Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>;
    setLiveSessionData: (data: {}) => void;
    setRecordedSessionStaticsData: (data: {}) => void;
    setRecordedSessionDataFilePath: (filePath: string | null) => void;
    setRecordedPlaybackSummary: Dispatch<SetStateAction<RecordedPlaybackSummary>>;
    runRecordedAiAnalysis: (options?: { force?: boolean }) => Promise<RecordedAiAnalysisState>;
    setActiveTab: Dispatch<SetStateAction<string>>;
    writeRecordedLiveSessionData: (data: any) => Promise<void>;
    readRecordedSessionData: (onProgress?: (read: number, total: number | null, bytesRead?: number, totalBytes?: number) => void) => Promise<any[]>;
    finalizeRecordingWrites: () => Promise<void>;
    clearRecordingSession: () => void;
    setActiveVisualizations: Dispatch<SetStateAction<VisualizationInstance[]>>;
    sendGuidanceToChat: (message: string) => void;
}

export const AnalysisContext = createContext<AnalysisContextType>({
    activeTab: 'mapLists',
    mapSelected: '',
    sessionSelected: {} as RacingSessionDetailedInfoDto,
    liveData: {} as any,
    TelemetryDataLiveStatus: null,
    recordedSessionDataFilePath: null,
    recordedTelemetryDataCount: 0,
    recordedSessioStaticsData: {} as any,
    activeVisualizations: [],
    latestGuidanceMessage: null,
    sessionIntelligence: null,
    recordedAiAnalysis: createIdleRecordedAiAnalysis(),
    recordedPlaybackSummary: createEmptyRecordedPlaybackSummary(),
    setMap: () => {
        console.warn('No provider for AnalysisContext');
    },
    setSession: ((value: RacingSessionDetailedInfoDto | null) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>,
    setLiveSessionData: () => {
        console.warn('No provider for AnalysisContext');
    },
    setRecordedSessionStaticsData: () => {
        console.warn('No provider for AnalysisContext');
    },
    setRecordedSessionDataFilePath: () => {
        console.warn('No provider for AnalysisContext');
    },
    setRecordedPlaybackSummary: ((value: RecordedPlaybackSummary) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<RecordedPlaybackSummary>>,
    runRecordedAiAnalysis: async () => {
        console.warn('No provider for AnalysisContext');
        return createIdleRecordedAiAnalysis();
    },
    setActiveTab: ((value: string) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<string>>,
    writeRecordedLiveSessionData: async () => {
        console.warn('No provider for AnalysisContext');
    },
    readRecordedSessionData: async () => {
        console.warn('No provider for AnalysisContext');
        return [];
    },
    finalizeRecordingWrites: async () => {
        console.warn('No provider for AnalysisContext');
    },
    clearRecordingSession: () => {
        console.warn('No provider for AnalysisContext');
    },
    setActiveVisualizations: ((value: VisualizationInstance[]) => {
        console.warn('No provider for AnalysisContext');
    }) as Dispatch<SetStateAction<VisualizationInstance[]>>,
    sendGuidanceToChat: () => {
        console.warn('No provider for AnalysisContext');
    }
});
