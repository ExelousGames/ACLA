import type { RecordedFileReadEvent, StandardTelemetrySample } from 'views/live-session/live-session-types';

export function readLocalTelemetry(filePath: string, signal: AbortSignal, onProgress: (rows: number) => void): Promise<StandardTelemetrySample[]> {
    const api = window.electronAPI;
    return new Promise((resolve, reject) => {
        const rows: StandardTelemetrySample[] = [];
        const queued: RecordedFileReadEvent[] = [];
        let readId: string | null = null;
        let settled = false;
        let removeListener = () => {};
        const cleanup = () => {
            removeListener();
            signal.removeEventListener('abort', abort);
        };
        const fail = (error: Error) => {
            if (settled) return;
            settled = true;
            cleanup();
            if (readId) void api.cancelRecordedFileRead(readId).catch(() => undefined);
            reject(error);
        };
        const abort = () => fail(new Error('Local telemetry loading was cancelled.'));
        const handleEvent = (event: RecordedFileReadEvent) => {
            if (settled) return;
            if (!readId) { queued.push(event); return; }
            if (event.readId !== readId) return;
            if (event.type === 'chunk') {
                for (const row of event.rows) rows.push(row);
                onProgress(rows.length);
            } else if (event.type === 'error') {
                fail(new Error(event.message));
            } else if (event.type === 'complete') {
                settled = true;
                cleanup();
                resolve(rows);
            }
        };
        if (signal.aborted) { abort(); return; }
        removeListener = api.onRecordedFileReadEvent(handleEvent);
        signal.addEventListener('abort', abort, { once: true });
        void api.startRecordedFileRead({ filePath, game: 'iracing', purpose: 'consume' }).then((result) => {
            readId = result.readId;
            if (settled) {
                void api.cancelRecordedFileRead(readId).catch(() => undefined);
                return;
            }
            for (const event of queued.splice(0)) handleEvent(event);
        }).catch((error) => fail(error instanceof Error ? error : new Error(String(error))));
    });
}
