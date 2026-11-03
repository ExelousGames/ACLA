import type { ControlSource } from './driving-control';

export const DRIVING_MODEL_TIMING_STAGES = [
    { id: 'memory', label: 'Observation history assembly' },
    { id: 'learning', label: 'Training CPU preparation / submission' },
    { id: 'learningWait', label: 'Training GPU completion wait' },
    { id: 'inference', label: 'Driving policy inference' },
    { id: 'control', label: 'AI control output' },
] as const;

type DrivingModelTimingStage = typeof DRIVING_MODEL_TIMING_STAGES[number]['id'];
export interface DrivingModelTimingSnapshot {
    capturedAt: number;
    source: ControlSource;
    training: boolean;
    /** Delayed plan samples submitted this cycle, excluding the current transition/demonstration. */
    replayCount: number;
    /** Populated history bins in this cycle's input, excluding the current observation. */
    historyLength?: number;
    stages: Partial<Record<DrivingModelTimingStage, number>>;
    currentStage: DrivingModelTimingStage | null;
    stageElapsedMs?: number;
    totalMs: number;
    finishedAt?: number;
    state: 'processing' | 'complete' | 'incomplete';
}

/** Elapsed CPU work and async waits, without adding GPU synchronization. */
export class DrivingModelTimer {
    replayCount = 0;
    historyLength?: number;
    private started = performance.now();
    private stageStarted = this.started;
    private stage: DrivingModelTimingStage | null = null;
    private stages: DrivingModelTimingSnapshot['stages'] = {};
    private finished?: number;
    private finishedAt?: number;
    private state: DrivingModelTimingSnapshot['state'] = 'processing';

    constructor(private capturedAt: number, private source: ControlSource, private training: boolean) {}

    start(stage: DrivingModelTimingStage) {
        if (this.finished !== undefined) return;
        this.end();
        this.stage = stage;
        this.stageStarted = performance.now();
    }

    end() {
        if (this.stage) this.stages[this.stage] = performance.now() - this.stageStarted;
        this.stage = null;
    }

    finish(complete: boolean) {
        if (this.finished !== undefined) return;
        this.end();
        this.finished = performance.now();
        this.finishedAt = Date.now();
        this.state = complete ? 'complete' : 'incomplete';
    }

    snapshot(): DrivingModelTimingSnapshot {
        const now = performance.now();
        return { capturedAt: this.capturedAt, source: this.source, training: this.training,
            replayCount: this.replayCount, historyLength: this.historyLength,
            stages: { ...this.stages }, currentStage: this.stage,
            stageElapsedMs: this.stage ? now - this.stageStarted : undefined,
            totalMs: (this.finished ?? now) - this.started, finishedAt: this.finishedAt, state: this.state };
    }
}
