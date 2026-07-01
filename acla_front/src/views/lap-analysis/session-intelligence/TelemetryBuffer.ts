import { TelemetrySample } from './types';

const DEFAULT_CAPACITY = 6000; // ~5 min at 20Hz

export class TelemetryBuffer {
    private buf: TelemetrySample[];
    private head: number = 0;   // next write position
    private count: number = 0;  // total samples ever pushed (never resets)
    private capacity: number;

    constructor(capacity: number = DEFAULT_CAPACITY) {
        this.capacity = capacity;
        this.buf = new Array(capacity);
    }

    push(sample: TelemetrySample): number {
        this.buf[this.head % this.capacity] = sample;
        this.head++;
        this.count++;
        return this.count - 1; // global sample index
    }

    // Total samples pushed since session start (monotonically increasing index).
    get length(): number {
        return this.count;
    }

    // How many samples are currently stored (up to capacity).
    get size(): number {
        return Math.min(this.count, this.capacity);
    }

    // Return sample at global index. Returns null if outside the live window.
    get(globalIdx: number): TelemetrySample | null {
        if (globalIdx < 0 || globalIdx >= this.count) return null;
        const oldest = this.count - this.size;
        if (globalIdx < oldest) return null;
        return this.buf[globalIdx % this.capacity] ?? null;
    }

    // Slice by global index range [from, to). Clamps to available window.
    slice(from: number, to: number): TelemetrySample[] {
        const oldest = this.count - this.size;
        const start = Math.max(from, oldest);
        const end = Math.min(to, this.count);
        const result: TelemetrySample[] = [];
        for (let i = start; i < end; i++) {
            const s = this.buf[i % this.capacity];
            if (s !== undefined) result.push(s);
        }
        return result;
    }

    // Last n samples.
    last(n: number): TelemetrySample[] {
        const from = Math.max(0, this.count - n);
        return this.slice(from, this.count);
    }

    // Samples from the last `ms` milliseconds using Physics_timestamp or arrival order.
    // Falls back to sample-count estimate at 20Hz if no timestamp field.
    sliceByTime(ms: number): TelemetrySample[] {
        const recent = this.last(this.size);
        if (recent.length === 0) return [];

        const lastSample = recent[recent.length - 1];
        const tsField = lastSample['Physics_timestamp'] ?? lastSample['timestamp'];
        if (tsField != null) {
            const cutoff = Number(tsField) - ms;
            const idx = recent.findIndex(s => {
                const t = s['Physics_timestamp'] ?? s['timestamp'];
                return t != null && Number(t) >= cutoff;
            });
            return idx === -1 ? [] : recent.slice(idx);
        }

        // Fallback: estimate at 20Hz
        const samples = Math.ceil(ms / 50);
        return recent.slice(Math.max(0, recent.length - samples));
    }

    reset(): void {
        this.buf = new Array(this.capacity);
        this.head = 0;
        this.count = 0;
    }
}
