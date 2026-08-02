import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { RecordingEvent, RecordingState } from 'views/lap-analysis/recording-state';
import { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';

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
}

export interface LiveVisualizationInstance {
    id: string;
    type: 'telemetry-overview' | 'event-log' | 'analysis-results';
    height: number;
    data?: unknown;
}

export interface LiveSessionRuntime {
    currentTelemetry: LiveTelemetry;
    telemetryStatus: ACC_STATUS | null;
    staticData: LiveSessionStaticData;
    recordingState: RecordingState;
    recordingMetadata: LiveRecordingMetadata | null;
    recordingFileKey: string | null;
    recordedSampleCount: number;
    sessionIntelligence: SessionIntelligence;
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
}
