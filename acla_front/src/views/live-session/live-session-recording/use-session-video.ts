import { useCallback, useEffect, useRef, useState } from 'react';
import { recordSessionVideo, SessionVideoClip } from './session-video';

export function useSessionVideo(stream: MediaStream | null, filePath: string | null, recording: boolean) {
    const active = useRef<ReturnType<typeof recordSessionVideo> | null>(null);
    const pending = useRef<Promise<void>>(Promise.resolve());
    // A mounted recorder belongs to one live session. Existing/restored recordings
    // have already passed the only point at which video can be included.
    const startHandled = useRef(Boolean(filePath) || recording);
    const [clips, setClips] = useState<SessionVideoClip[]>([]);
    const [error, setError] = useState('');
    const [saving, setSaving] = useState(false);
    const report = useCallback((reason: unknown) => {
        setSaving(false);
        setError(reason instanceof Error ? reason.message : String(reason));
    }, []);
    const refresh = useCallback(async () => {
        const files = filePath && window.sessionVideo ? await window.sessionVideo.list(filePath) : [];
        if (files.length > 1) {
            setClips([]);
            throw new Error('Only one video is allowed per session. Multiple local videos were found.');
        }
        setClips(files);
        return files;
    }, [filePath]);
    const stop = useCallback(async () => {
        const current = active.current;
        active.current = null;
        if (current) {
            pending.current = current.stop();
            // Cleanup callers may only observe the error through the UI.
            void pending.current.catch(report);
        }
        try { await pending.current; }
        finally { if (!active.current) setSaving(false); }
    }, [report]);
    useEffect(() => { void refresh().catch(report); }, [refresh, report]);
    useEffect(() => {
        const canStart = recording && !startHandled.current;
        if (recording || filePath) startHandled.current = true;
        if (canStart && stream && filePath) {
            try {
                setError('');
                active.current = recordSessionVideo(stream, filePath, report);
                setSaving(true);
            } catch (error) { report(error); }
        }
        return () => { void stop().catch(report).then(refresh).catch(report); };
    }, [stream, filePath, recording, stop, refresh, report]);
    return { clips, error, saving, stop, refresh };
}
