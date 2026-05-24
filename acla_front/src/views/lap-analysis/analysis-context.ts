import { createContext, Dispatch, SetStateAction } from 'react';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { VisualizationInstance } from './visualization/VisualizationRegistry';
import { SessionIntelligence } from './session-intelligence/SessionIntelligence';

export interface AnalysisContextType {
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
    setMap: (map: string | null) => void;
    setSession: Dispatch<SetStateAction<RacingSessionDetailedInfoDto | null>>;
    setLiveSessionData: (data: {}) => void;
    setRecordedSessionStaticsData: (data: {}) => void;
    setRecordedSessionDataFilePath: (filePath: string | null) => void;
    writeRecordedLiveSessionData: (data: any) => Promise<void>;
    readRecordedSessionData: (onProgress?: (read: number, total: number | null, bytesRead?: number, totalBytes?: number) => void) => Promise<any[]>;
    finalizeRecordingWrites: () => Promise<void>;
    clearRecordingSession: () => void;
    setActiveVisualizations: Dispatch<SetStateAction<VisualizationInstance[]>>;
    sendGuidanceToChat: (message: string) => void;
}

export const AnalysisContext = createContext<AnalysisContextType>({
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
