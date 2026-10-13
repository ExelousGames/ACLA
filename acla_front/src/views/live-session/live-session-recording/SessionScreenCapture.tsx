import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { captureGameScreen, ScreenCaptureSource } from '../track-vision/screen-capture';

type CaptureState = 'idle' | 'starting' | 'active';
type SessionScreenCapture = {
    stream: MediaStream | null;
    state: CaptureState;
    sources: ScreenCaptureSource[];
    sourceId: string;
    error: string;
    selectSource: (id: string) => void;
    refreshSources: () => Promise<void>;
    start: () => Promise<void>;
    stop: () => void;
};

const ScreenCaptureContext = createContext<SessionScreenCapture | null>(null);
export const useSessionScreenCapture = () => useContext(ScreenCaptureContext);

export function SessionScreenCaptureProvider({ children }: { children: React.ReactNode }) {
    const streamRef = useRef<MediaStream | null>(null);
    const captureVersion = useRef(0);
    const sourcesVersion = useRef(0);
    const starting = useRef(false);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [state, setState] = useState<CaptureState>('idle');
    const [sources, setSources] = useState<ScreenCaptureSource[]>([]);
    const [sourceId, selectSource] = useState('');
    const [error, setError] = useState('');

    const release = useCallback(() => {
        captureVersion.current++;
        starting.current = false;
        streamRef.current?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
        streamRef.current = null;
    }, []);

    const stop = useCallback(() => {
        release();
        setStream(null);
        setState('idle');
    }, [release]);

    useEffect(() => () => {
        sourcesVersion.current++;
        release();
    }, [release]);

    const refreshSources = useCallback(async () => {
        if (!window.screenCapture) return;
        const version = ++sourcesVersion.current;
        setError('');
        try {
            const available = await window.screenCapture.listSources();
            if (version !== sourcesVersion.current) return;
            setSources(available);
            selectSource((current) => available.some(({ id }) => id === current) ? current : '');
            if (!available.length) setError('No capture sources found. Open the simulator and refresh.');
        } catch (reason) {
            if (version === sourcesVersion.current) setError(reason instanceof Error ? reason.message : 'Unable to list capture sources.');
        }
    }, []);

    const start = useCallback(async () => {
        if (!sourceId || starting.current || streamRef.current) return;
        const version = ++captureVersion.current;
        starting.current = true;
        setState('starting');
        setError('');
        try {
            const captured = await captureGameScreen(sourceId);
            if (version !== captureVersion.current) {
                captured.getTracks().forEach((track) => track.stop());
                return;
            }
            streamRef.current = captured;
            captured.getVideoTracks().forEach((track) => { track.onended = stop; });
            starting.current = false;
            setStream(captured);
            setState('active');
        } catch (reason) {
            if (version !== captureVersion.current) return;
            stop();
            setError(reason instanceof Error && reason.name === 'NotAllowedError'
                ? 'Screen sharing was cancelled or denied. Share again and select your game window.'
                : reason instanceof Error ? reason.message : 'Screen capture failed.');
        }
    }, [sourceId, stop]);

    const value = useMemo(() => ({ stream, state, sources, sourceId, error, selectSource, refreshSources, start, stop }),
        [stream, state, sources, sourceId, error, refreshSources, start, stop]);
    return <ScreenCaptureContext.Provider value={value}>{children}</ScreenCaptureContext.Provider>;
}

export function SessionScreenCaptureControls({ startDisabled = false }: { startDisabled?: boolean }) {
    const capture = useSessionScreenCapture();
    if (!window.screenCapture || !capture) return null;
    const { state, sourceId, sources, error, selectSource, refreshSources, start, stop } = capture;
    return <div className="live-recording-bar__capture" role="group" aria-label="Session video capture">
        <span className="live-recording-bar__capture-label">Video</span>
        <select aria-label="Game window or screen" value={sourceId} disabled={startDisabled || state !== 'idle'}
            onChange={(event) => selectSource(event.target.value)}>
            <option value="">Choose a window or screen</option>
            {sources.map(({ id, name }) => <option key={id} value={id}>{name}</option>)}
        </select>
        <button type="button" disabled={startDisabled || state !== 'idle'} onClick={() => void refreshSources()}>Refresh sources</button>
        {state === 'idle'
            ? <button type="button" disabled={startDisabled || !sourceId} onClick={() => void start()}>Share game screen</button>
            : <button type="button" onClick={stop}>{state === 'starting' ? 'Cancel capture' : 'Stop capture'}</button>}
        {state !== 'idle' && <span className="live-recording-bar__capture-label">{state === 'starting' ? 'Starting…' : 'Capturing'}</span>}
        {error && <span className="live-recording-bar__capture-error" role="alert">{error}</span>}
    </div>;
}
