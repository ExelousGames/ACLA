import { liveTelemetryStore, type LiveTelemetrySnapshot } from '../live-telemetry-store';
import { capturePolicyCameraImage, type PolicyCameraImage } from './policy-camera';

export interface SessionScreenFrame {
    capturedAt: number;
    telemetry: LiveTelemetrySnapshot;
    camera: PolicyCameraImage;
    processingMs: number;
    publishedAt: number;
    captureIntervalMs?: number;
}

/** Follow the recorder's video without taking ownership of its media tracks. */
export class SessionScreenFrameReader {
    private stream: MediaStream | null = null;
    private latest: SessionScreenFrame | null = null;
    private detach?: () => void;

    read(stream: MediaStream | null): SessionScreenFrame | null {
        const track = stream?.getVideoTracks()[0];
        if (!track || track.readyState !== 'live') stream = null;
        if (this.stream !== stream) {
            this.stop();
            if (stream) this.attach(stream);
        }
        return this.latest;
    }

    stop() {
        this.detach?.();
        this.detach = undefined;
        this.stream = null;
        this.latest = null;
    }

    private attach(stream: MediaStream) {
        this.stream = stream;
        const video = document.createElement('video');
        const canvas = document.createElement('canvas');
        video.muted = true;
        video.playsInline = true;
        video.srcObject = stream;
        let active = true;
        let callback: number;
        const capture = () => {
            if (!active) return;
            if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
                const capturedAt = Date.now();
                const telemetry = liveTelemetryStore.getSnapshot();
                const started = performance.now();
                const captureIntervalMs = this.latest ? capturedAt - this.latest.capturedAt : undefined;
                try {
                    const camera = capturePolicyCameraImage(video, canvas);
                    this.latest = { capturedAt, telemetry, camera, captureIntervalMs,
                        publishedAt: Date.now(), processingMs: performance.now() - started };
                } catch {
                    this.latest = null;
                }
            }
            callback = video.requestVideoFrameCallback(capture);
        };
        this.detach = () => {
            active = false;
            if (callback !== undefined) video.cancelVideoFrameCallback(callback);
            video.pause();
            video.srcObject = null;
        };
        void video.play().then(() => {
            if (active) callback = video.requestVideoFrameCallback(capture);
        }).catch(() => {
            if (active) this.detach?.();
        });
    }
}
