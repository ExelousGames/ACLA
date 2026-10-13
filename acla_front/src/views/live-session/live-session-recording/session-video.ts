export interface SessionVideoClip {
    filePath: string;
    name: string;
    size: number;
    mimeType: string;
    startedAt: number;
}

declare global {
    interface Window {
        sessionVideo?: {
            create(telemetryPath: string): Promise<{ filePath: string; startedAt: number }>;
            append(filePath: string, bytes: Uint8Array): Promise<void>;
            finish(filePath: string): Promise<void>;
            list(telemetryPath: string): Promise<SessionVideoClip[]>;
            read(filePath: string, offset: number): Promise<{ data: string; bytesRead: number }>;
            delete(filePath: string): Promise<void>;
            reveal(filePath: string): Promise<void>;
        };
    }
}

// Each session recorder produces a complete MP4 from the initial telemetry start.
// Persist chunks immediately, and wait for the final dataavailable event before uploading.
export function recordSessionVideo(stream: MediaStream, telemetryPath: string, onError: (error: Error) => void) {
    const bridge = window.sessionVideo;
    if (!bridge || typeof MediaRecorder === 'undefined') throw new Error('Local video recording is unavailable.');
    const mimeType = ['video/mp4;codecs=avc1', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mimeType) throw new Error('MP4 recording is unavailable.');
    const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 2500000 });
    let writes = Promise.resolve();
    let stopped = false;
    let pendingBytes = 0;
    let failure: Error | null = null;
    let resolveDone: () => void;
    const done = new Promise<void>(resolve => { resolveDone = resolve; });
    const fail = (reason: unknown) => {
        if (!failure) {
            failure = reason instanceof Error ? reason : new Error(String(reason));
            onError(failure);
        }
        stopped = true;
        if (recorder.state !== 'inactive') recorder.stop();
    };
    const ready = bridge.create(telemetryPath).then(({ filePath }) => {
        recorder.ondataavailable = ({ data }) => {
            if (!data.size || failure) return;
            pendingBytes += data.size;
            if (pendingBytes > 32 * 1024 * 1024) {
                fail(new Error('Video recording stopped because local storage could not keep up.'));
                return;
            }
            writes = writes.then(async () => {
                for (let offset = 0; offset < data.size; offset += 512 * 1024) {
                    await bridge.append(filePath, new Uint8Array(await data.slice(offset, offset + 512 * 1024).arrayBuffer()));
                }
            }).catch(fail).finally(() => { pendingBytes -= data.size; });
        };
        recorder.onerror = () => fail(new Error('Video recording was interrupted. The saved portion is kept locally.'));
        recorder.onstop = () => {
            void writes.then(() => bridge.finish(filePath)).catch(fail).finally(() => resolveDone());
        };
        if (stopped) {
            return bridge.finish(filePath).finally(() => resolveDone());
        }
        try { recorder.start(1000); }
        catch (error) {
            fail(error);
            return bridge.finish(filePath).finally(() => resolveDone());
        }
    }).catch(error => { fail(error); resolveDone(); });
    return {
        async stop() {
            stopped = true;
            await ready;
            if (recorder.state !== 'inactive') recorder.stop();
            await done;
            if (failure) throw failure;
        },
    };
}
