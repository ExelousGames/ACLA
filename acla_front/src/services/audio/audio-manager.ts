import { MediaPlaybackAdapter, PcmPlaybackAdapter, type PlaybackAdapter } from './playback-adapters';

export type AudioType = 'voice' | 'music' | 'alert';
export type PlaybackOutcome =
    | { status: 'completed' | 'replaced' | 'rejected' | 'cancelled' }
    | { status: 'failed'; error: Error };

export interface PlaybackOptions {
    type: AudioType;
    /** Arbitration is independent within each type; newer ties win. */
    priority: number;
    volume?: number;
    /** Fires on each transition from idle to audible. */
    onStart?: () => void;
    onIdle?: () => void;
    /** Exactly one terminal notification, including discarded requests. */
    onComplete?: (outcome: PlaybackOutcome) => void;
}

export interface PlayOptions extends PlaybackOptions { url: string }
export interface StreamOptions extends PlaybackOptions {
    format: 'wav' | 'pcm16';
    /** Mono PCM16 sample rate; defaults to 24000 Hz. */
    sampleRate?: number;
}
export interface PlaybackHandle {
    stop(): void;
    readonly finished: Promise<PlaybackOutcome>;
    readonly outcome: PlaybackOutcome | undefined;
    isActive(): boolean;
}
export interface AudioStreamHandle extends PlaybackHandle {
    /** False means this handle is terminal, finished, or has the wrong format. */
    enqueueBase64Wav(base64: string): boolean;
    enqueuePcm16(pcm16: ArrayBuffer): boolean;
    /** Close input; natural completion waits for all accepted chunks. */
    finish(): void;
}

// Notifications run after state transitions. Callers cannot interrupt arbitration
// midway through resource disposal, or turn their own callback errors into failures.
const notify = (callback: () => void): void => {
    void Promise.resolve().then(callback).catch((error) => console.error('[audio] notification failed', error));
};

class PlaybackRequest implements AudioStreamHandle {
    outcome: PlaybackOutcome | undefined;
    readonly finished: Promise<PlaybackOutcome>;
    private resolve!: (outcome: PlaybackOutcome) => void;
    private adapter: PlaybackAdapter | null = null;
    private options: PlaybackOptions | null;
    private admitted = false;
    private closed = false;
    private busy = false;
    audible = false;
    readonly type: AudioType;
    readonly priority: number;
    readonly volume: number;

    constructor(
        private readonly manager: AudioManager,
        options: PlaybackOptions,
        private readonly format: 'url' | 'wav' | 'pcm16',
        private readonly sampleRate = 24000,
    ) {
        this.options = options;
        this.type = options.type;
        this.priority = options.priority;
        this.volume = options.volume ?? 0.9;
        this.finished = new Promise((resolve) => { this.resolve = resolve; });
        if (!Number.isFinite(this.priority) || !Number.isFinite(this.volume)
            || this.volume < 0 || this.volume > 1 || !Number.isFinite(sampleRate) || sampleRate <= 0) {
            this.fail(new Error('Invalid audio priority, volume, or sample rate.'));
        }
    }

    private begin(): boolean {
        if (this.outcome || this.closed) return false;
        if (!this.admitted) {
            if (!this.manager.admit(this)) return false;
            this.admitted = true;
            const events = {
                audible: (value: boolean) => this.setAudible(value),
                idle: () => {
                    if (this.outcome) return;
                    this.busy = false;
                    if (this.closed) this.terminate({ status: 'completed' });
                },
                failed: (error: unknown) => this.fail(error),
            };
            this.adapter = this.format === 'pcm16'
                ? new PcmPlaybackAdapter(this.manager.volumeFor(this), this.sampleRate, events)
                : new MediaPlaybackAdapter(this.manager.volumeFor(this), events);
        }
        this.busy = true;
        return true;
    }

    play(url: string): void {
        if (!this.begin()) return;
        this.closed = true;
        (this.adapter as MediaPlaybackAdapter).enqueue({ url });
    }

    enqueueBase64Wav(base64: string): boolean {
        if (this.format !== 'wav' || !base64 || !this.begin()) return false;
        (this.adapter as MediaPlaybackAdapter).enqueue({ wav: base64 });
        return !this.outcome;
    }

    enqueuePcm16(pcm16: ArrayBuffer): boolean {
        if (this.format !== 'pcm16' || !pcm16.byteLength || !this.begin()) return false;
        (this.adapter as PcmPlaybackAdapter).enqueue(pcm16);
        return !this.outcome;
    }

    finish(): void {
        if (this.outcome) return;
        this.closed = true;
        if (!this.busy) this.terminate({ status: 'completed' });
    }

    private setAudible(value: boolean): void {
        value = value && this.volume > 0;
        if (this.outcome || this.audible === value) return;
        this.audible = value;
        this.manager.updateMusicVolume();
        const callback = value ? this.options?.onStart : this.options?.onIdle;
        if (callback) notify(() => { if (!this.outcome) callback(); });
    }

    updateVolume(): void { this.adapter?.setVolume(this.manager.volumeFor(this)); }
    isActive(): boolean { return !this.outcome && this.busy; }
    stop(): void { this.terminate({ status: 'cancelled' }); }
    private fail(error: unknown): void {
        this.terminate({ status: 'failed', error: error instanceof Error ? error : new Error(String(error)) });
    }

    terminate(outcome: PlaybackOutcome): void {
        if (this.outcome) return;
        this.outcome = outcome;
        this.busy = this.audible = false;
        this.adapter?.stop();
        this.adapter = null;
        this.manager.release(this);
        const callback = this.options?.onComplete;
        this.options = null;
        this.resolve(outcome);
        if (callback) notify(() => callback(outcome));
    }
}

/** Mediates temporary playback only. Assets and replay decisions stay with callers. */
class AudioManager {
    private readonly admitted = new Map<AudioType, PlaybackRequest>();

    play({ url, ...options }: PlayOptions): PlaybackHandle {
        const request = new PlaybackRequest(this, options, 'url');
        request.play(url);
        return request;
    }

    createStream(options: StreamOptions): AudioStreamHandle {
        return new PlaybackRequest(this, options, options.format, options.sampleRate);
    }

    admit(request: PlaybackRequest): boolean {
        const previous = this.admitted.get(request.type);
        if (previous && previous.priority > request.priority) {
            request.terminate({ status: 'rejected' });
            return false;
        }
        previous?.terminate({ status: 'replaced' });
        this.admitted.set(request.type, request);
        return true;
    }

    release(request: PlaybackRequest): void {
        if (this.admitted.get(request.type) === request) this.admitted.delete(request.type);
        this.updateMusicVolume();
    }

    volumeFor(request: PlaybackRequest): number {
        return request.volume * (request.type === 'music' && this.admitted.get('voice')?.audible ? 0.2 : 1);
    }

    updateMusicVolume(): void { this.admitted.get('music')?.updateVolume(); }
}

export const audioManager: Pick<AudioManager, 'play' | 'createStream'> = new AudioManager();
