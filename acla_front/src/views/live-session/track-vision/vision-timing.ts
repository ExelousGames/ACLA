export const VISION_TIMING_STAGES = [
    { id: 'capture', label: 'Screen frame copy' },
    { id: 'camera', label: 'Camera RGB preparation' },
    { id: 'references', label: 'Map / reference lookup' },
    { id: 'segment', label: 'Segmentation GPU submission' },
    { id: 'depth', label: 'Depth GPU submission' },
    { id: 'depthReadback', label: 'Shared GPU completion / depth download / CPU conversion' },
    { id: 'segmentReadback', label: 'Final mask readback / CPU assembly' },
    { id: 'position', label: 'Position model' },
    { id: 'preview', label: 'Preview frame copy' },
    { id: 'track', label: 'Track reconstruction' },
    { id: 'scene', label: 'Scene reconstruction' },
    { id: 'birdsEye', label: 'Bird’s-eye projection / history' },
] as const;

/** Wall-clock phases; runMs is the ORT call, not GPU kernel execution time. */
export interface VisionModelTiming {
    queueMs: number;
    inputMs: number;
    runMs: number;
    outputMs: number;
}

export interface DepthReadbackTiming {
    queueMs: number;
    /** Shared queue wait before depth download, including pending segmentation/depth work and callback scheduling. */
    gpuWaitMs: number;
    /** Tensor download, including mapping/copy and promise scheduling. */
    downloadMs: number;
    conversionMs: number;
    totalMs: number;
}

export interface SegmentReadbackTiming {
    /** Staging allocation and final mask/metadata copy submission. */
    submissionMs: number;
    /** Mapping includes pending GPU work, transfer and promise scheduling. */
    mapMs: number;
    /** CPU instance metadata assembly and byte-mask copying. */
    assemblyMs: number;
    totalMs: number;
}

export type VisionTimingStage = typeof VISION_TIMING_STAGES[number]['id'];
export interface VisionTimingSnapshot {
    capturedAt: number;
    captureIntervalMs?: number;
    stages: Partial<Record<VisionTimingStage, number>>;
    currentStage: VisionTimingStage | null;
    stageElapsedMs?: number;
    totalMs: number;
    finishedAt?: number;
    state: 'processing' | 'complete' | 'incomplete';
}

/** Measures elapsed work, including async waits, without changing the capture clock. */
export class VisionFrameTimer {
    private started = performance.now();
    private stageStarted = this.started;
    private stage: VisionTimingStage | null = null;
    private stages: VisionTimingSnapshot['stages'] = {};
    private finished?: number;
    private finishedAt?: number;
    private state: VisionTimingSnapshot['state'] = 'processing';

    constructor(public capturedAt: number, private previousCapturedAt?: number) {}

    start(stage: VisionTimingStage) {
        const now = performance.now();
        if (this.stage) this.stages[this.stage] = now - this.stageStarted;
        this.stage = stage;
        this.stageStarted = now;
    }

    finish(complete: boolean) {
        this.finished = performance.now();
        if (this.stage) this.stages[this.stage] = this.finished - this.stageStarted;
        this.stage = null;
        this.finishedAt = Date.now();
        this.state = complete ? 'complete' : 'incomplete';
    }

    snapshot(): VisionTimingSnapshot {
        const now = performance.now();
        return { capturedAt: this.capturedAt,
            captureIntervalMs: this.previousCapturedAt === undefined ? undefined : this.capturedAt - this.previousCapturedAt,
            stages: { ...this.stages },
            currentStage: this.stage, stageElapsedMs: this.stage ? now - this.stageStarted : undefined,
            totalMs: (this.finished ?? now) - this.started, finishedAt: this.finishedAt, state: this.state };
    }
}
