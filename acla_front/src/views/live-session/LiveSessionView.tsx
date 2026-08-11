import React, { useContext, useRef } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import {
    AI_TOOL_COMPONENT_NAMES,
    NamedAiToolComponentHandle,
    useRegisterAiToolComponentRef,
} from 'contexts/AiToolComponentRefContext';
import { LiveSessionContext } from './LiveSessionContext';
import LiveSessionGameStatus, { LIVE_SESSION_GAME_LABELS } from './LiveSessionGameStatus';
import LiveTelemetryWorkspace from './LiveTelemetryWorkspace';
import { RecordingState } from 'views/lap-analysis/recording-state';
import type { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import type { LiveSessionAnalysisResultPage } from './live-session-analysis-results';
import './live-session.css';

export const LIVE_SESSION_RECORDER_HOST_ID = 'live-session-recorder-host';

export interface LiveSessionHandle extends NamedAiToolComponentHandle {
    getRecordingState(): RecordingState;
    getSessionIntelligence(): SessionIntelligence;
    getCurrentTelemetry(): Record<string, any>;
    queryTelemetryMetric(args: Record<string, any>): any;
    getTelemetryForScope(scope: any): Record<string, any>[];
    getEventLog(args: Record<string, any>): any[];
    getNextCorner(): any;
    getLiveSessionSnapshot(): LiveSessionSnapshot;
    getLiveSectionHistory(limit: number): any[];
    getLiveSectionTelemetry(args: Record<string, any>): any;
    recordLiveSectionClassification(args: Record<string, any>): any;
    getLatestAnalysisResultPage(): LiveSessionAnalysisResultPage | null;
}

type LiveSessionSnapshot = ReturnType<SessionIntelligence['getLiveSessionSnapshot']>;

const EMPTY_LIVE_SNAPSHOT: LiveSessionSnapshot = {
    status: 'empty',
    track: '',
    car: '',
    current_lap: 0,
    completed_laps: 0,
    normalized_position: 0,
    sample_count: 0,
    live_session_type: 'unknown',
    baseline_ready: false,
    baseline_collection_started: false,
    baseline_progress_percent: 0,
    baseline_lap: null,
    completed_lap_count: 0,
    section_count: 0,
};

const getLiveSnapshot = (
    sessionIntelligence: Partial<Pick<SessionIntelligence, 'getLiveSessionSnapshot'>> | null,
): LiveSessionSnapshot => (
    typeof sessionIntelligence?.getLiveSessionSnapshot === 'function'
        ? sessionIntelligence.getLiveSessionSnapshot()
        : EMPTY_LIVE_SNAPSHOT
);

const LimitedLiveWorkspace = ({ game }: { game: Exclude<DesktopGame, 'acc'> }) => (
    <div
        className="live-session-limited-workspace"
        data-testid="limited-live-workspace"
        role="region"
        aria-label={`${LIVE_SESSION_GAME_LABELS[game]} limited live workspace`}
    >
        <span className="live-session-limited-workspace__eyebrow">Limited workspace</span>
        <h2>{LIVE_SESSION_GAME_LABELS[game]}</h2>
        <p>
            This session keeps the selected game locked, but ACC telemetry and recording controls
            are not available for this simulator yet.
        </p>
    </div>
);

const LiveSessionView = ({ name }: { name: string }) => {
    const liveSession = useContext(LiveSessionContext);
    const liveSessionRef = useRef(liveSession);
    liveSessionRef.current = liveSession;
    const componentRef = useRef<LiveSessionHandle | null>(null);

    if (componentRef.current === null) {
        componentRef.current = {
            getComponentName: () => name,
            getRecordingState: () => liveSessionRef.current.recordingState,
            getSessionIntelligence: () => liveSessionRef.current.sessionIntelligence,
            getCurrentTelemetry: () => liveSessionRef.current.currentTelemetry,
            queryTelemetryMetric: (args) => liveSessionRef.current.sessionIntelligence.query(args as any),
            getTelemetryForScope: (scope) => liveSessionRef.current.sessionIntelligence.getRowsForScope(scope),
            getEventLog: (args) => liveSessionRef.current.sessionIntelligence.findEvents(args as any),
            getNextCorner: () => liveSessionRef.current.sessionIntelligence.getNextCorner(),
            getLiveSessionSnapshot: () => getLiveSnapshot(liveSessionRef.current.sessionIntelligence),
            getLiveSectionHistory: (limit) => liveSessionRef.current.sessionIntelligence.getSectionHistory(limit),
            getLiveSectionTelemetry: (args) => liveSessionRef.current.sessionIntelligence.getSectionTelemetryWindow({
                section_id: args.section_id || args.sectionId,
                section_name: args.section_name || args.sectionName,
                lap: args.lap,
            }),
            recordLiveSectionClassification: (args) => liveSessionRef.current.sessionIntelligence.recordSectionClassification(args),
            getLatestAnalysisResultPage: () => {
                const pages = liveSessionRef.current.analysisResultPages;
                return pages[pages.length - 1] ?? null;
            },
        };
    }
    useRegisterAiToolComponentRef(name, componentRef.current);


    const { restorationError, sessionGame } = liveSession;

    return (
        <section className="live-session-view" aria-label="Live Session">
            {sessionGame === null ? (
                <div className="live-session-waiting" data-testid="live-session-gate">
                    <LiveSessionGameStatus />
                    <div className="live-session-waiting__copy">
                        <span className="live-session-waiting__eyebrow">Live session gate</span>
                        <h2>Choose the simulator for this session</h2>
                        <p>
                            Start a new session when your simulator is detected. The selected game
                            will stay fixed until you upload or discard the session.
                        </p>
                    </div>
                </div>
            ) : (
                <>
                    <LiveSessionGameStatus />
                    {restorationError && (
                        <div className="live-session-recovery-error" role="alert">
                            {restorationError}
                        </div>
                    )}
                    <div className="live-session-view__workspace">
                        {sessionGame === 'acc'
                            ? <LiveTelemetryWorkspace name={AI_TOOL_COMPONENT_NAMES.LIVE_VISUALIZATION_MANAGER} />
                            : <LimitedLiveWorkspace game={sessionGame} />}
                    </div>
                    <div
                        id={LIVE_SESSION_RECORDER_HOST_ID}
                        className="live-session-view__recorder"
                        data-testid="live-session-recorder-host"
                    />
                </>
            )}
        </section>
    );
};

export default LiveSessionView;
