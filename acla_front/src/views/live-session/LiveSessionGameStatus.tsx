import React, { useContext } from 'react';
import { useDesktopGame } from 'contexts/DesktopGameContext';
import type {
    DesktopGame,
    DesktopGameContextValue,
} from 'contexts/DesktopGameContext';
import { LiveSessionContext } from './LiveSessionContext';

type StatusVisualState = 'checking' | 'ready' | 'limited' | 'idle' | 'unsupported' | 'error';

interface StatusPresentation {
    title: string;
    detail: string;
    visualState: StatusVisualState;
}

export const LIVE_SESSION_GAME_LABELS: Record<DesktopGame, string> = {
    acc: 'Assetto Corsa Competizione',
    ac: 'Assetto Corsa',
    iracing: 'iRacing',
};

const detectedGamePresentation: Record<DesktopGame, StatusPresentation> = {
    acc: {
        title: 'Assetto Corsa Competizione detected',
        detail: 'Full live telemetry is available.',
        visualState: 'ready',
    },
    ac: {
        title: 'Assetto Corsa detected',
        detail: 'A limited live workspace is available.',
        visualState: 'limited',
    },
    iracing: {
        title: 'iRacing detected',
        detail: 'A limited live workspace is available.',
        visualState: 'limited',
    },
};

const getDetectorPresentation = ({
    detectedGame,
    detectionStatus,
    error,
}: DesktopGameContextValue): StatusPresentation => {
    switch (detectionStatus) {
        case 'checking':
            return {
                title: 'Scanning for simulator...',
                detail: 'Detection updates automatically.',
                visualState: 'checking',
            };
        case 'detected':
            return detectedGame
                ? detectedGamePresentation[detectedGame]
                : {
                    title: 'No simulator detected.',
                    detail: 'Detection continues automatically.',
                    visualState: 'idle',
                };
        case 'not-detected':
            return {
                title: 'No simulator detected.',
                detail: 'Detection continues automatically.',
                visualState: 'idle',
            };
        case 'unsupported':
            return {
                title: 'Simulator detection unavailable',
                detail: 'Simulator detection is unavailable on this system.',
                visualState: 'unsupported',
            };
        case 'error':
            return {
                title: 'Simulator detection failed',
                detail: error ?? 'An unknown detector error occurred.',
                visualState: 'error',
            };
    }
};

const getSessionPresentation = (game: DesktopGame): StatusPresentation => ({
    title: `${LIVE_SESSION_GAME_LABELS[game]} session`,
    detail: game === 'acc'
        ? 'Game locked for this live session.'
        : 'Game locked. ACC recording controls are unavailable in this limited workspace.',
    visualState: game === 'acc' ? 'ready' : 'limited',
});

const LiveSessionGameStatus = () => {
    const detection = useDesktopGame();
    const liveSession = useContext(LiveSessionContext);
    const presentation = liveSession.sessionGame
        ? getSessionPresentation(liveSession.sessionGame)
        : getDetectorPresentation(detection);
    const canStart = detection.detectionStatus === 'detected' && detection.detectedGame !== null;

    const handleAction = () => {
        if (liveSession.sessionGame) {
            liveSession.recorderControl?.openUploadFlow();
            return;
        }
        if (canStart && detection.detectedGame) {
            liveSession.startLiveSession(detection.detectedGame);
        }
    };

    return (
        <div
            className={`live-session-game-status live-session-game-status--${presentation.visualState}`}
            data-state={presentation.visualState}
            data-session-game={liveSession.sessionGame ?? undefined}
        >
            <div
                className="live-session-game-status__message"
                role="status"
                aria-live="polite"
                aria-atomic="true"
            >
                <span className="live-session-game-status__dot" aria-hidden="true" />
                <div className="live-session-game-status__copy">
                    <span className="live-session-game-status__title">{presentation.title}</span>
                    <span className="live-session-game-status__detail">{presentation.detail}</span>
                </div>
            </div>
            <button
                type="button"
                className="live-session-game-status__action"
                onClick={handleAction}
                disabled={!liveSession.sessionGame && !canStart}
            >
                {liveSession.sessionGame ? 'New Session' : 'Start New Session'}
            </button>
        </div>
    );
};

export default LiveSessionGameStatus;
