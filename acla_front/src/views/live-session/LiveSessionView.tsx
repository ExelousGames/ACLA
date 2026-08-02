import React from 'react';
import LiveTelemetryWorkspace from './LiveTelemetryWorkspace';
import './live-session.css';

export const LIVE_SESSION_RECORDER_HOST_ID = 'live-session-recorder-host';

const LiveSessionView = () => (
    <section className="live-session-view" aria-label="Live Session">
        <div className="live-session-view__workspace">
            <LiveTelemetryWorkspace />
        </div>
        <div id={LIVE_SESSION_RECORDER_HOST_ID} className="live-session-view__recorder" data-testid="live-session-recorder-host" />
    </section>
);

export default LiveSessionView;
