/** Playback requests own their callbacks; reusable players may retain their audio graph. */
export interface PlaybackAdapter {
    setVolume(volume: number): void;
    stop(): void;
}

export interface AdapterEvents {
    audible(value: boolean): void;
    level?: (value: number, positionMs: number) => void;
    idle(): void;
    failed(error: unknown): void;
}

export class MediaPlaybackResources {
    audio: HTMLAudioElement | null = null;
    context: AudioContext | null = null;
    analyser: AnalyserNode | null = null;
    samples: Float32Array | null = null;
    private source: MediaElementAudioSourceNode | null = null;
    private resuming: Promise<void> | null = null;
    private disposed = false;

    prepare(metered: boolean): HTMLAudioElement {
        if (this.disposed) throw new Error('The audio player has been disposed.');
        if (!this.audio) this.audio = new Audio();
        if (metered && !this.context) {
            this.context = new AudioContext();
            this.source = this.context.createMediaElementSource(this.audio);
            this.analyser = this.context.createAnalyser();
            this.analyser.fftSize = 256;
            this.source.connect(this.analyser);
            this.analyser.connect(this.context.destination);
            this.samples = new Float32Array(this.analyser.fftSize);
        }
        return this.audio;
    }

    resume(): Promise<void> {
        if (this.resuming) return this.resuming;
        if (this.context?.state !== 'suspended') return Promise.resolve();
        this.resuming = this.context.resume().finally(() => { this.resuming = null; });
        return this.resuming;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.source?.disconnect();
        this.analyser?.disconnect();
        if (this.context) void this.context.close().catch(() => undefined);
        this.audio = this.context = this.analyser = this.source = this.samples = null;
    }
}

export class MediaPlaybackAdapter implements PlaybackAdapter {
    private audio: HTMLAudioElement | null = null;
    private objectUrl: string | null = null;
    private pending: Array<{ url: string } | { wav: string }> = [];
    private stopped = false;
    private meterTimer?: ReturnType<typeof setInterval>;
    private generation = 0;
    private resources: MediaPlaybackResources;

    constructor(private volume: number, private readonly events: AdapterEvents,
        private readonly sharedResources?: MediaPlaybackResources) {
        this.resources = sharedResources ?? new MediaPlaybackResources();
    }

    enqueue(chunk: { url: string } | { wav: string }): void {
        if (this.stopped) return;
        this.pending.push(chunk);
        if (!this.audio) this.next();
    }

    private next(): void {
        if (this.stopped) return;
        const chunk = this.pending.shift();
        if (!chunk) {
            this.events.audible(false);
            this.events.idle();
            return;
        }
        try {
            let url: string;
            if ('wav' in chunk) {
                const binary = atob(chunk.wav);
                const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
                url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
                this.objectUrl = url;
            } else {
                url = chunk.url;
            }
            const audio = this.resources.prepare(Boolean(this.events.level));
            this.audio = audio;
            const generation = ++this.generation;
            const current = () => !this.stopped && this.audio === audio && this.generation === generation;
            audio.src = url;
            audio.volume = this.volume;
            if (this.events.level) {
                const { context, analyser, samples } = this.resources;
                this.meterTimer = setInterval(() => {
                    if (!current() || context!.state !== 'running' || audio.paused) return;
                    analyser!.getFloatTimeDomainData(samples!);
                    let peak = 0;
                    for (let i = 0; i < samples!.length; i++) peak = Math.max(peak, Math.abs(samples![i]));
                    this.events.level?.(Math.min(1, peak), audio.currentTime * 1000);
                }, 16);
            }
            void this.resources.resume().catch((error) => {
                if (current()) this.events.failed(error);
            });
            audio.onplaying = () => { if (current()) this.events.audible(true); };
            audio.onwaiting = audio.onpause = () => {
                if (current()) { this.events.audible(false); this.events.level?.(0, audio.currentTime * 1000); }
            };
            audio.onended = () => {
                if (!current()) return;
                this.release();
                this.events.audible(false);
                this.next();
            };
            audio.onerror = () => {
                if (current()) this.events.failed(new Error('The audio could not be played.'));
            };
            void audio.play().then(() => {
                if (current()) this.events.audible(true);
            }, (error) => {
                if (current()) this.events.failed(error);
            });
        } catch (error) {
            this.events.failed(error);
        }
    }

    private release(): void {
        this.generation++;
        if (this.meterTimer !== undefined) clearInterval(this.meterTimer);
        this.meterTimer = undefined;
        const audio = this.audio;
        this.audio = null;
        if (audio) {
            audio.onended = audio.onerror = audio.onplaying = audio.onwaiting = audio.onpause = null;
            try { audio.pause(); } catch { /* Already detached. */ }
            audio.removeAttribute('src');
            try { audio.load(); } catch { /* Already detached. */ }
        }
        if (!this.sharedResources) {
            // Ordinary URL/WAV playback keeps its existing per-chunk lifetime.
            this.resources.dispose();
            this.resources = new MediaPlaybackResources();
        }
        if (this.objectUrl !== null) URL.revokeObjectURL(this.objectUrl);
        this.objectUrl = null;
    }

    setVolume(volume: number): void {
        this.volume = volume;
        if (this.audio) this.audio.volume = volume;
    }

    stop(): void {
        this.stopped = true;
        this.pending.length = 0;
        this.release();
    }
}

export class PcmPlaybackAdapter implements PlaybackAdapter {
    private context: AudioContext | null = null;
    private gain: GainNode | null = null;
    private nodes = new Set<AudioBufferSourceNode>();
    private pending: ArrayBuffer[] = [];
    private ready = false;
    private stopped = false;
    private nextTime = 0;

    constructor(
        private volume: number,
        private readonly sampleRate: number,
        private readonly events: AdapterEvents,
    ) {}

    enqueue(pcm16: ArrayBuffer): void {
        if (this.stopped || !pcm16.byteLength) return;
        try {
            if (pcm16.byteLength % 2) throw new Error('PCM16 chunks must contain complete samples.');
            // The caller may reuse its input buffer after enqueue returns.
            this.pending.push(pcm16.slice(0));
            if (!this.context) {
                const context = new AudioContext({ sampleRate: this.sampleRate });
                this.context = context;
                this.gain = context.createGain();
                this.gain.gain.value = this.volume;
                this.gain.connect(context.destination);
                context.onstatechange = () => {
                    if (!this.stopped) this.events.audible(context.state === 'running' && this.nodes.size > 0);
                };
                if (context.state === 'suspended') {
                    void context.resume().then(() => {
                        if (this.stopped) return;
                        this.ready = true;
                        this.flush();
                    }, (error) => { if (!this.stopped) this.events.failed(error); });
                    return;
                }
                this.ready = true;
            }
            if (this.ready) this.flush();
        } catch (error) {
            this.events.failed(error);
        }
    }

    private flush(): void {
        if (this.stopped || !this.context || !this.gain) return;
        try {
            while (this.pending.length) {
                const bytes = new DataView(this.pending.shift()!);
                const samples = new Float32Array(bytes.byteLength / 2);
                for (let i = 0; i < samples.length; i++) {
                    const value = bytes.getInt16(i * 2, true);
                    samples[i] = value < 0 ? value / 0x8000 : value / 0x7fff;
                }
                const buffer = this.context.createBuffer(1, samples.length, this.sampleRate);
                buffer.copyToChannel(samples, 0);
                const node = this.context.createBufferSource();
                this.nodes.add(node);
                node.buffer = buffer;
                node.connect(this.gain);
                node.onended = () => {
                    if (this.stopped || !this.nodes.delete(node)) return;
                    node.onended = null;
                    node.disconnect();
                    node.buffer = null;
                    if (!this.nodes.size) {
                        this.events.audible(false);
                        this.events.idle();
                    }
                };
                const startAt = Math.max(this.context.currentTime, this.nextTime);
                node.start(startAt);
                this.nextTime = startAt + buffer.duration;
            }
            this.events.audible(this.context.state === 'running' && this.nodes.size > 0);
        } catch (error) {
            this.events.failed(error);
        }
    }

    setVolume(volume: number): void {
        this.volume = volume;
        if (this.gain) this.gain.gain.value = volume;
    }

    stop(): void {
        this.stopped = true;
        this.pending.length = 0;
        this.nodes.forEach((node) => {
            node.onended = null;
            try { node.stop(); } catch { /* May already have ended. */ }
            node.disconnect();
            node.buffer = null;
        });
        this.nodes.clear();
        this.gain?.disconnect();
        this.gain = null;
        const context = this.context;
        this.context = null;
        if (context) {
            context.onstatechange = null;
            try { void context.close().catch(() => undefined); } catch { /* Already closed. */ }
        }
    }
}
