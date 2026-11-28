import { DEFAULT_DRIVING_CONFIG, emptyRecordedSamplingWeights, RECORDED_SAMPLING_CATEGORIES, UNLABELED_RECORDED_SAMPLING_CATEGORY,
    validateRecordedBatchLabelCount, type RecordedSamplingCategory, type RecordedSamplingWeights } from './driving-model-config';
import type { DrivingPolicy, PlannedActionTarget, RecordedPlanTarget } from './driving-policy';
import type { DrivingPolicyInput } from './driving-memory';
import type { PolicyCameraImage } from 'views/live-session/live-session-recording/policy-camera';

type ObservedAction = Omit<PlannedActionTarget, 'step'>;
type PendingPlan = { features: DrivingPolicyInput; appliedAt: number; nextStep: number;
    targets: PlannedActionTarget[]; pool: SamplePool; clipId?: string };
type ArchivedCamera = { width: number; height: number; pixels: Blob };
type CameraReads = Map<ArchivedCamera, Promise<PolicyCameraImage>>;
type RetainedPlan = RecordedPlanTarget | ((reads: CameraReads) => Promise<RecordedPlanTarget>);
type SamplePool = { plans: RecordedPlanTarget[]; seen: number; capacity: number; clips: Map<string, Map<number, RetainedPlan>> };
type PoolEntry = [RecordedSamplingCategory, RetainedPlan[][]];
export const RECORDED_PLAN_WINDOW = 256;
const RECORDED_PLAN_READ_CHUNK = 8;

function readTrainingBytes(blob: Blob): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as ArrayBuffer);
        reader.onerror = () => reject(reader.error ?? new Error('Could not read recorded training data.'));
        reader.readAsArrayBuffer(blob);
    });
}

function restoreCamera(camera: ArchivedCamera, reads: CameraReads): Promise<PolicyCameraImage> {
    const cached = reads.get(camera);
    if (cached) return cached;
    const read = readTrainingBytes(camera.pixels).then(bytes =>
        ({ width: camera.width, height: camera.height, rgb: new Uint8Array(bytes) }));
    reads.set(camera, read);
    return read;
}

function labelQuotas(categories: RecordedSamplingCategory[], weights: RecordedSamplingWeights, slots: number) {
    const quotas = emptyRecordedSamplingWeights();
    let totalWeight = categories.reduce((sum, category) => sum + weights[category], 0);
    const byWeight = [...categories].sort((a, b) => weights[a] - weights[b]);
    // Reserve one for labels below one slot, then rescale the remaining weights.
    for (const category of byWeight) {
        const weight = weights[category];
        if (weight / totalWeight * slots >= 1) break;
        quotas[category] = 1;
        totalWeight -= weight;
        slots--;
    }
    for (const category of categories) {
        if (!quotas[category]) quotas[category] = Math.max(1, weights[category] / totalWeight * slots);
    }
    return quotas;
}

/** Align plan labels in time, then keep every eligible label represented in each optimizer batch. */
export class DrivingPlanTraining {
    private pending: PendingPlan[] = [];
    private pools = new Map<RecordedSamplingCategory, SamplePool>();
    private archivedCameras = new WeakMap<PolicyCameraImage, ArchivedCamera>();
    private sampling?: { available: PoolEntry[]; quotas: RecordedSamplingWeights };
    private credit = emptyRecordedSamplingWeights();
    private counts = emptyRecordedSamplingWeights();
    private batches = 0;

    constructor(private weights: RecordedSamplingWeights = {
        ...emptyRecordedSamplingWeights(), [UNLABELED_RECORDED_SAMPLING_CATEGORY]: 100,
    },
        private batchSize = DEFAULT_DRIVING_CONFIG.recordedBatchSize,
        private batchCount = DEFAULT_DRIVING_CONFIG.recordedBatchCount,
        private random = Math.random) {
        const categories = RECORDED_SAMPLING_CATEGORIES.filter((category) => weights[category] > 0);
        const slots = Math.max(batchSize, categories.length);
        const quotas = labelQuotas(categories, weights, slots);
        for (const category of categories) {
            const capacity = Math.ceil(RECORDED_PLAN_WINDOW * quotas[category] / slots);
            this.pools.set(category, { plans: [], seen: 0, capacity, clips: new Map() });
        }
    }

    get sampled() { return { ...this.counts }; }
    get completedBatches() { return this.batches; }

    record(policy: DrivingPolicy, features: DrivingPolicyInput | undefined, target: ObservedAction,
        appliedAt: number, category: RecordedSamplingCategory = UNLABELED_RECORDED_SAMPLING_CATEGORY, clipId?: string) {
        for (const plan of this.pending) {
            while (plan.nextStep < policy.planTimes.length) {
                const due = plan.appliedAt + policy.planTimes[plan.nextStep] * 1000;
                const delay = appliedAt - due;
                // Frame and slot timestamps reach the same time through different arithmetic.
                const roundoff = 4 * Number.EPSILON * Math.max(Math.abs(appliedAt), Math.abs(due));
                if (delay < -roundoff) break;
                if (delay <= policy.configuration.learning.planTargetToleranceMs + roundoff) {
                    plan.targets.push({ step: plan.nextStep, ...target });
                }
                plan.nextStep++;
            }
        }
        const pool = this.pools.get(category);
        if (features && pool) {
            const cameras = features.cameras.slice();
            // Recorded plans already own the consecutive history images used for training.
            if (features.recordedHistory) cameras.fill(null, 0, cameras.length - 1);
            this.pending.push({ features: { ...features, features: features.features.slice(), cameras },
                appliedAt, nextStep: 1, targets: [{ step: 0, ...target }], pool, clipId });
        }
        this.completePlans(policy.planTimes.length);
    }

    completeRange() { this.completePlans(0); }

    private completePlans(slots: number) {
        while (this.pending.length) {
            const { features, targets, nextStep, appliedAt, pool, clipId } = this.pending[0];
            if (nextStep < slots) break;
            if (clipId !== undefined) {
                let clip = pool.clips.get(clipId);
                if (!clip) pool.clips.set(clipId, clip = new Map());
                if (!clip.has(appliedAt)) clip.set(appliedAt, this.archivePlan(features, targets));
            } else {
                // Calls without clip identity still use the bounded frame reservoir.
                const index = pool.seen < pool.capacity ? pool.seen : Math.floor(this.random() * (pool.seen + 1));
                if (index < pool.capacity) pool.plans[index] = { input: features, targets };
                pool.seen++;
            }
            this.pending.shift();
        }
    }

    private archivePlan({ features, cameras, recordedHistory, recurrent }: DrivingPolicyInput, targets: PlannedActionTarget[]): RetainedPlan {
        const archive = (image: PolicyCameraImage) => {
            let camera = this.archivedCameras.get(image);
            if (!camera) {
                // Share consecutive history images and let the browser page stored pixels to disk.
                camera = { width: image.width, height: image.height, pixels: new Blob([image.rgb]) };
                this.archivedCameras.set(image, camera);
            }
            return camera;
        };
        const archivedFeatures = new Blob([Float64Array.from(features)]);
        const archivedCameras = cameras.map(image => image && archive(image));
        const archivedHistory = recordedHistory?.map(({ camera, ...frame }) => ({ ...frame, camera: archive(camera) }));
        return async (reads: CameraReads) => {
            const [featureBytes, cameras, recordedHistory] = await Promise.all([
                readTrainingBytes(archivedFeatures),
                Promise.all(archivedCameras.map(camera => camera && restoreCamera(camera, reads))),
                archivedHistory && Promise.all(archivedHistory.map(async ({ camera, ...frame }) =>
                    ({ ...frame, camera: await restoreCamera(camera, reads) }))),
            ]);
            return { input: { features: Array.from(new Float64Array(featureBytes)), cameras, recurrent,
                ...(recordedHistory && { recordedHistory }) }, targets };
        };
    }

    private shuffle<T>(items: T[]): T[] {
        for (let i = items.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [items[i], items[j]] = [items[j], items[i]];
        }
        return items;
    }

    private prepareSampling() {
        const available = Array.from(this.pools).filter(([, pool]) => pool.plans.length > 0 || pool.clips.size > 0);
        validateRecordedBatchLabelCount(this.batchSize, available.length);
        const rotations = available.map(([category, { plans, clips }]): PoolEntry => {
            const rotation = Array.from(clips.values(), origins => this.shuffle(Array.from(origins.values())));
            if (plans.length) rotation.push(this.shuffle(plans));
            return [category, this.shuffle(rotation)];
        });
        const quotas = labelQuotas(rotations.map(([category]) => category), this.weights, this.batchSize);
        return { available: rotations, quotas };
    }

    async finishAsync(policy: DrivingPolicy, active = () => true, maxBatches = Infinity) {
        if (!active() || this.batches >= this.batchCount) return;
        this.completeRange();
        const { available, quotas } = this.sampling ??= this.prepareSampling();
        if (!available.length) return;
        const batches = Math.min(this.batchCount - this.batches, maxBatches);
        for (let i = 0; i < batches && active(); i++) {
            const credit = { ...this.credit }, counts = { ...this.counts };
            const batch: RetainedPlan[] = [];
            const draw = ([category, clips]: PoolEntry) => {
                const index = counts[category]++;
                // Use one turn per clip before revisiting it with its next shuffled plan start.
                const clip = clips[index % clips.length];
                batch.push(clip[Math.floor(index / clips.length) % clip.length]);
            };
            for (const entry of available) {
                const [category] = entry;
                draw(entry);
                credit[category] += quotas[category] - 1;
            }
            // Allocate unreserved slots by weighted quota and carry unused fractions forward.
            while (batch.length < this.batchSize) {
                const selected = available.reduce((best, entry) => credit[entry[0]] > credit[best[0]] ? entry : best);
                draw(selected);
                credit[selected[0]]--;
            }
            const reads: CameraReads = new Map();
            const plans: RecordedPlanTarget[] = [];
            for (let offset = 0; offset < batch.length && active(); offset += RECORDED_PLAN_READ_CHUNK) {
                const chunk = batch.slice(offset, offset + RECORDED_PLAN_READ_CHUNK);
                plans.push(...await Promise.all(chunk.map(plan => typeof plan === 'function' ? plan(reads) : plan)));
            }
            if (!active() || !await policy.learnRecordedBatchAsync(plans, active)) break;
            this.credit = credit;
            this.counts = counts;
            this.batches++;
        }
    }
}
