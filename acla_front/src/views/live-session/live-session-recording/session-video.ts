import type { Output } from 'mediabunny';

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
            finish(filePath: string, videoStartSeconds?: number): Promise<void>;
            list(telemetryPath: string): Promise<SessionVideoClip[]>;
            read(filePath: string, offset: number): Promise<{ data: string; bytesRead: number }>;
            delete(filePath: string): Promise<void>;
            reveal(filePath: string): Promise<void>;
        };
    }
}

const CHUNK_BYTES = 512 * 1024;

export function recordSessionVideo(stream: MediaStream, telemetryPath: string, telemetryStartedAt: number,
    onError: (error: Error) => void) {
    const bridge = window.sessionVideo;
    const track = stream.getVideoTracks()[0];
    if (!bridge || !track || typeof MediaStreamTrackProcessor === 'undefined' || typeof VideoEncoder === 'undefined') {
        throw new Error('Local video recording is unavailable.');
    }
    if (!Number.isFinite(telemetryStartedAt)) throw new Error('The telemetry recording start time is unavailable.');

    let stopped = false;
    let failure: Error | undefined;
    let reader: ReadableStreamDefaultReader<VideoFrame> | undefined;
    let cancellation: Promise<void> | undefined;
    const fail = (reason: unknown) => {
        if (!failure) {
            failure = reason instanceof Error ? reason : new Error(String(reason));
            onError(failure);
        }
    };
    const stopReading = () => {
        if (reader) cancellation ??= reader.cancel().catch(fail);
        return cancellation;
    };

    const done = (async () => {
        let filePath: string | undefined;
        let output: Output | undefined;
        let finalized = false;
        let videoStartSeconds: number | undefined;
        try {
            const { AppendOnlyStreamTarget, Mp4OutputFormat, Output, VideoSample, VideoSampleSource } = await import('mediabunny');
            if (stopped) return;
            const { filePath: videoPath } = await bridge.create(telemetryPath);
            filePath = videoPath;
            if (stopped) return;

            const target = new AppendOnlyStreamTarget(new WritableStream<Uint8Array>({
                async write(bytes) {
                    for (let position = 0; position < bytes.byteLength; position += CHUNK_BYTES) {
                        await bridge.append(videoPath, bytes.subarray(position, position + CHUNK_BYTES));
                    }
                },
            }));
            const source = new VideoSampleSource({ codec: 'avc', bitrate: 2500000, keyFrameInterval: 1,
                sizeChangeBehavior: 'contain', latencyMode: 'quality' });
            output = new Output({ format: new Mp4OutputFormat({ fastStart: 'fragmented', minimumFragmentDuration: 1 }), target });
            output.addVideoTrack(source);
            // Keep a short queue while the encoder initializes or a fragment is written.
            reader = new MediaStreamTrackProcessor({ track, maxBufferSize: 8 }).readable.getReader();

            let firstTimestamp: number | undefined;
            let lastTimestamp: number | undefined;
            const fallbackDuration = 1 / (track.getSettings().frameRate || 30);
            while (!stopped) {
                const next = await reader.read();
                if (next.done) break;

                const frame = next.value;
                let sample: InstanceType<typeof VideoSample> | undefined;
                try {
                    const timestamp = frame.timestamp;
                    if (stopped || !Number.isFinite(timestamp) || (lastTimestamp !== undefined && timestamp <= lastTimestamp)) continue;
                    if (firstTimestamp === undefined) {
                        // Desktop frames expose a stream-relative timestamp. Anchor the first accepted frame
                        // at delivery against the same epoch used by the telemetry reader.
                        videoStartSeconds = (performance.timeOrigin + performance.now() - telemetryStartedAt) / 1000;
                        firstTimestamp = timestamp;
                        await output.start();
                    }
                    const duration = frame.duration && frame.duration > 0 ? frame.duration / 1e6 : fallbackDuration;
                    sample = new VideoSample(frame, { timestamp: (timestamp - firstTimestamp) / 1e6, duration });
                    await source.add(sample);
                    lastTimestamp = timestamp;
                } finally {
                    if (sample) sample.close();
                    else frame.close();
                }
            }

            if (lastTimestamp !== undefined) {
                await output.finalize();
                finalized = true;
            }
        } catch (error) {
            fail(error);
        } finally {
            await stopReading();
            reader?.releaseLock();
            if (output && !finalized) await output.cancel().catch(fail);
            if (filePath) {
                const finish = finalized ? bridge.finish(filePath, videoStartSeconds) : bridge.finish(filePath);
                await finish.catch(fail);
            }
        }
    })();

    return {
        async stop() {
            stopped = true;
            await stopReading();
            await done;
            if (failure) throw failure;
        },
    };
}
