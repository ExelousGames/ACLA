import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { RecordingEvent, RecordingState, StopReason } from 'views/lap-analysis/recording-state';
import type {
    CornerLookahead,
} from 'views/lap-analysis/session-intelligence/types';
import type { LiveSessionType } from 'views/lap-analysis/session-intelligence/live-performance-analyst';
import type {
    AppendLiveSessionAnalysisResultPageInput,
    AppendLiveSessionAnalysisResultPageResult,
    LiveSessionAnalysisResultPage,
} from './live-session-analysis-results';

export type TelemetryJsonValue =
    | string
    | number
    | boolean
    | null
    | TelemetryJsonValue[]
    | { [key: string]: TelemetryJsonValue };

export type StandardTelemetrySample = Record<string, TelemetryJsonValue>;
export type LiveTelemetry = StandardTelemetrySample;

export type RecordingStartFailureType =
    | 'malformed-recording-game'
    | 'unknown-recording-game'
    | 'unsupported-recording-game';

export type RecordingStartResult =
    | { ok: true; game: DesktopGame; filePath: string; startedAt: number }
    | { ok: false; error: { type: RecordingStartFailureType; message: string } };

export interface RecordingStopResult {
    game: DesktopGame;
    filePath?: string;
    writtenSamples?: number;
    error?: string;
}

export interface RecordingViewUpdate {
    type: 'frame';
    game: DesktopGame;
    sample: StandardTelemetrySample;
    sequence: number;
    committedSequence: number;
    committedCount: number;
}

export type RecordedFileReadEvent =
    | { type: 'format'; readId: string; format: 'standard-flat'; game: DesktopGame }
    | { type: 'chunk'; readId: string; rows: StandardTelemetrySample[] }
    | { type: 'progress'; readId: string; rowsRead: number; bytesRead: number; totalBytes: number }
    | { type: 'complete'; readId: string; format: 'standard-flat'; game: DesktopGame; rowCount: number; totalBytes: number }
    | { type: 'error'; readId: string; message: string; row?: number; byteOffset?: number };

export interface LiveSessionStaticData {
    Static_track?: string;
    Static_car_model?: string;
    [key: string]: unknown;
}

export interface LiveRecordingMetadata {
    sessionName: string;
    mapName: string;
    carName: string;
    gameRecordedFrom: DesktopGame;
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
    type: 'live-trajectory-map' | 'telemetry-overview' | 'event-log' | 'analysis-results' | 'baseline-collection';
    height: number;
    data?: unknown;
    config?: Record<string, unknown>;
}

export interface LiveSessionRecorderControl {
    openUploadFlow: () => void;
}

export interface LiveSessionSnapshot {
    status: 'ready' | 'empty';
    track: string;
    car: string;
    current_lap: number;
    completed_laps: number;
    normalized_position: number;
    sample_count: number;
    live_session_type: LiveSessionType;
    completed_lap_count: number;
}

export interface LiveSessionRuntime {
    sessionGame: DesktopGame | null;
    currentTelemetry: LiveTelemetry;
    currentTelemetrySampleIndex: number;
    telemetryStatus: ACC_STATUS | null;
    staticData: LiveSessionStaticData;
    recordingState: RecordingState;
    recordingMetadata: LiveRecordingMetadata | null;
    recordingFileKey: string | null;
    recordingActive: boolean;
    recordingGame: DesktopGame | null;
    recordedSampleCount: number;
    restorationStatus: LiveSessionRestorationStatus;
    restorationError: string | null;
    recordingFileValidation: LocalTelemetryFileValidation | null;
    recorderControl: LiveSessionRecorderControl | null;
    analysisResultPages: LiveSessionAnalysisResultPage[];
    activeAnalysisResultPageId: string | null;
    getNextCorner: () => CornerLookahead | null;
    getLiveSessionSnapshot: () => LiveSessionSnapshot;
    startLiveSession: (game: DesktopGame) => void;
    endLiveSession: () => void;
    setRecordingMetadata: (metadata: LiveRecordingMetadata | null) => void;
    transitionRecordingState: (event: RecordingEvent) => void;
    startRecordingSession: (game: DesktopGame) => Promise<RecordingStartResult>;
    stopRecordingSession: (reason?: StopReason) => Promise<RecordingStopResult | null>;
    streamRecordedTelemetry: (
        onChunk: (rows: StandardTelemetrySample[]) => void | Promise<void>,
        onProgress?: (rowsRead: number, totalRows: number | null, bytesRead: number, totalBytes: number) => void,
    ) => Promise<{ rowCount: number; totalBytes: number }>;
    clearRecordingSession: () => void;
    clearPersistedDraft: () => void;
    registerRecorderControl: (control: LiveSessionRecorderControl | null) => void;
    appendAnalysisResultPage: (
        input: AppendLiveSessionAnalysisResultPageInput,
    ) => AppendLiveSessionAnalysisResultPageResult;
    selectAnalysisResultPage: (pageId: string) => boolean;
    updateActiveAnalysisResultPage: (data: unknown) => boolean;
}
