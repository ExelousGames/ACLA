import React, { useContext } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { LiveSessionContext } from './LiveSessionContext';
import LiveSessionGameStatus, { LIVE_SESSION_GAME_LABELS } from './LiveSessionGameStatus';
import LiveTelemetryWorkspace from './LiveTelemetryWorkspace';
import './live-session.css';

export const LIVE_SESSION_RECORDER_HOST_ID = 'live-session-recorder-host';

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
    const { restorationError, sessionGame } = useContext(LiveSessionContext);

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
