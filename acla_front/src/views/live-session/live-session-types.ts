import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { RecordingEvent, RecordingState } from 'views/lap-analysis/recording-state';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import type {
    AppendLiveSessionAnalysisResultPageInput,
    AppendLiveSessionAnalysisResultPageResult,
    LiveSessionAnalysisResultPage,
} from './live-session-analysis-results';

export type LiveTelemetry = Record<string, any>;

export interface LiveSessionStaticData {
    track?: string;
    car_model?: string;
    [key: string]: unknown;
}

export interface LiveRecordingMetadata {
    sessionName: string;
    mapName: string;
    carName: string;
    gameRecordedFrom: 'acc' | 'ac' | 'iracing';
}

export const PERSISTED_LIVE_SESSION_DRAFT_VERSION = 1 as const;

export interface PersistedLiveSessionDraft {
    version: typeof PERSISTED_LIVE_SESSION_DRAFT_VERSION;
    ownerEmail: string;
    sessionGame: DesktopGame;
    recordingMetadata: LiveRecordingMetadata;
    telemetryFilePath: string;
    recordedSampleCount: number;
    lastRuntimeState: RecordingState;
    updatedAt: string;
}

export type LiveSessionRestorationStatus =
    | 'idle'
    | 'restoring'
    | 'not-found'
    | 'restored'
    | 'error';

export interface LocalTelemetryFileValidation {
    exists: boolean;
    readable: boolean;
    hasData: boolean;
    size: number;
    error?: string;
}

export interface LiveVisualizationInstance {
    name: string;
    id: string;
    type: 'telemetry-overview' | 'event-log' | 'analysis-results' | 'baseline-collection';
    height: number;
    data?: unknown;
    config?: Record<string, unknown>;
}

export interface LiveSessionRecorderControl {
    openUploadFlow: () => void;
}

export interface LiveSessionRuntime {
    sessionGame: DesktopGame | null;
    currentTelemetry: LiveTelemetry;
    telemetryStatus: ACC_STATUS | null;
    staticData: LiveSessionStaticData;
    recordingState: RecordingState;
    recordingMetadata: LiveRecordingMetadata | null;
    recordingFileKey: string | null;
    recordedSampleCount: number;
    restorationStatus: LiveSessionRestorationStatus;
    restorationError: string | null;
    recordingFileValidation: LocalTelemetryFileValidation | null;
    sessionIntelligence: SessionIntelligence;
    recorderControl: LiveSessionRecorderControl | null;
    analysisResultPages: LiveSessionAnalysisResultPage[];
    activeAnalysisResultPageId: string | null;
    startLiveSession: (game: DesktopGame) => void;
    endLiveSession: () => void;
    setCurrentTelemetry: (data: LiveTelemetry) => void;
    setStaticData: (data: LiveSessionStaticData) => void;
    setRecordingMetadata: (metadata: LiveRecordingMetadata | null) => void;
    transitionRecordingState: (event: RecordingEvent) => void;
    appendTelemetrySample: (data: LiveTelemetry) => Promise<void>;
    readRecordedTelemetry: (
        onProgress?: (read: number, total: number | null, bytesRead?: number, totalBytes?: number) => void,
    ) => Promise<LiveTelemetry[]>;
    finalizeRecordingWrites: () => Promise<void>;
    clearRecordingSession: () => void;
    clearPersistedDraft: () => void;
    registerRecorderControl: (control: LiveSessionRecorderControl | null) => void;
    appendAnalysisResultPage: (
        input: AppendLiveSessionAnalysisResultPageInput,
    ) => AppendLiveSessionAnalysisResultPageResult;
    selectAnalysisResultPage: (pageId: string) => boolean;
    updateActiveAnalysisResultPage: (data: unknown) => boolean;
}
