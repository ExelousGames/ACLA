import React, { useContext, useMemo, useRef } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import {
    AiChatScreenHandle,
    LIVE_SCREEN_TOOL_NAMES,
    SCREEN_VISUALIZATION_TOOL_NAMES,
    createAiChatScreenToolHandlers,
    toAiChatJsonValue,
    useAiChatScreenRegistration,
} from 'contexts/AiChatScreenContext';
import { LiveSessionContext } from './LiveSessionContext';
import LiveSessionGameStatus, { LIVE_SESSION_GAME_LABELS } from './LiveSessionGameStatus';
import LiveTelemetryWorkspace from './LiveTelemetryWorkspace';
import { RecordingState } from 'views/lap-analysis/recording-state';
import type { SessionIntelligence } from 'views/lap-analysis/session-intelligence/SessionIntelligence';
import './live-session.css';

export const LIVE_SESSION_RECORDER_HOST_ID = 'live-session-recorder-host';

const LIVE_SESSION_TOOL_HANDLERS = createAiChatScreenToolHandlers([
    ...LIVE_SCREEN_TOOL_NAMES,
    ...SCREEN_VISUALIZATION_TOOL_NAMES,
]);

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

const getLiveStatus = (recordingState: string, restorationError: string | null) => {
    if (restorationError) return { label: 'Needs attention', tone: 'error' as const };
    if (recordingState === RecordingState.RECORDING) return { label: 'Recording', tone: 'success' as const };
    if (recordingState === RecordingState.HOLDING) return { label: 'Paused', tone: 'warning' as const };
    if (recordingState === RecordingState.UPLOAD_READY) return { label: 'Ready to upload', tone: 'info' as const };
    return { label: 'Ready', tone: 'neutral' as const };
};

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

const LiveSessionView = () => {
    const liveSession = useContext(LiveSessionContext);
    const liveSessionRef = useRef(liveSession);
    liveSessionRef.current = liveSession;
    const componentRef = useRef<AiChatScreenHandle | null>(null);

    if (componentRef.current === null) {
        componentRef.current = {
            getAiContext: () => {
                const current = liveSessionRef.current;
                const snapshot = getLiveSnapshot(current.sessionIntelligence);
                const track = snapshot.track || current.recordingMetadata?.mapName || current.staticData.track || null;
                const car = snapshot.car || current.recordingMetadata?.carName || current.staticData.car_model || null;

                return {
                    screen_kind: 'live_session',
                    simulator: current.sessionGame,
                    recording_state: current.recordingState,
                    recording_name: current.recordingMetadata?.sessionName || null,
                    track,
                    car,
                    current_lap: snapshot.current_lap || null,
                    completed_laps: snapshot.completed_laps || 0,
                    normalized_position: snapshot.normalized_position || 0,
                    sample_count: snapshot.sample_count || current.recordedSampleCount,
                    latest_telemetry_present: Object.keys(current.currentTelemetry).length > 0,
                    latest_telemetry_key_count: Object.keys(current.currentTelemetry).length,
                    telemetry_status: current.telemetryStatus,
                    session_intelligence: toAiChatJsonValue(snapshot),
                    live_todo: toAiChatJsonValue(current.liveRangeTodoListSnapshot),
                    controls: {
                        live_todo_available: Boolean(current.liveRangeTodoListHandle),
                        recorder_available: Boolean(current.recorderControl),
                    },
                    visualization_capabilities: {
                        telemetry: true,
                        events: true,
                        sections: true,
                        live_todo: true,
                    },
                };
            },
            getToolHandlers: () => LIVE_SESSION_TOOL_HANDLERS,
        };
    }

    const liveSnapshot = getLiveSnapshot(liveSession.sessionIntelligence);
    const track = liveSnapshot.track || liveSession.recordingMetadata?.mapName || liveSession.staticData.track || '—';
    const car = liveSnapshot.car || liveSession.recordingMetadata?.carName || liveSession.staticData.car_model || '—';
    const registration = useMemo(() => ({
        screenId: 'live-session',
        assistantMode: 'live' as const,
        pillLabel: 'Live Session',
        componentRef,
        getPillInfo: () => ({
            title: liveSession.recordingMetadata?.sessionName || 'Live Session',
            description: 'Current simulator, recording, telemetry, coaching, and visualization workspace.',
            status: getLiveStatus(liveSession.recordingState, liveSession.restorationError),
            facts: [
                { label: 'Simulator', value: liveSession.sessionGame?.toUpperCase() || 'Not selected' },
                { label: 'Track', value: String(track) },
                { label: 'Car', value: String(car) },
                { label: 'Lap', value: liveSnapshot.current_lap ? String(liveSnapshot.current_lap) : '—' },
                { label: 'Samples', value: (liveSnapshot.sample_count || liveSession.recordedSampleCount).toLocaleString() },
            ],
        }),
    }), [
        car,
        liveSession.recordedSampleCount,
        liveSession.recordingMetadata?.sessionName,
        liveSession.recordingState,
        liveSession.restorationError,
        liveSession.sessionGame,
        liveSnapshot.current_lap,
        liveSnapshot.sample_count,
        track,
    ]);
    useAiChatScreenRegistration(registration);

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
                            ? <LiveTelemetryWorkspace />
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
