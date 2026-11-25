import { DrivingPolicyInput, drivingMemoryInputs } from './driving-memory';
import { DRIVING_ANALOG_AXES, DrivingModelConfig, drivingExplorationFrequencies, drivingExplorationSigmas, drivingFeatureSizes,
    isReservedPositionFeature } from './driving-model-config';

/** Fixed observation features keep exploration independent of actor weight updates.
 * Retain the saved group widths and scaling when reserved position slots are zeroed.
 * The bias keeps exploration alive at rest; bounded features limit its variance. */
function explorationFeatures(input: DrivingPolicyInput, config: DrivingModelConfig): number[] {
    const sizes = drivingFeatureSizes(config), groups: number[][] = [];
    let offset = drivingMemoryInputs(config);
    for (const name of config.observation.features) {
        const size = sizes[name];
        if (size && !['game', 'car', 'map'].includes(name)) {
            const group = isReservedPositionFeature(name) ? new Array<number>(size).fill(0)
                : input.features.slice(offset, offset + size).map((value) => Math.max(-1, Math.min(1, value)));
            groups.push(group);
        }
        offset += size;
    }
    return [1, ...groups.flatMap((group) => group.map((value) => value / Math.sqrt(group.length * groups.length)))];
}

/** gSDE-inspired feature-to-action weights with fixed configured scales.
 * Signed pedal and steering each have their own Gaussian column. The columns
 * share driving features and persist across overlapping plans. Smoothly rotating
 * between independent columns preserves their marginal Gaussian variance.
 * https://proceedings.mlr.press/v164/raffin22a.html */
export class DrivingExploration {
    private startedAt?: number;
    private updatedAt = -Infinity;
    private samples = Array.from({ length: DRIVING_ANALOG_AXES + 1 }, () => new Map<number, number[]>());

    constructor(private config: DrivingModelConfig, private random: () => number) {}

    reset() {
        this.startedAt = undefined;
        this.updatedAt = -Infinity;
        this.samples.forEach((samples) => samples.clear());
    }

    plan(input: DrivingPolicyInput, now: number) {
        if (now < this.updatedAt || now - this.updatedAt > this.config.runtime.maxTransitionSeconds * 1000) this.reset();
        if (this.startedAt === undefined) this.startedAt = now;
        this.updatedAt = now;
        const elapsed = now - this.startedAt;
        const intervals = drivingExplorationFrequencies(this.config).map((frequency) => 1000 / frequency);
        const features = explorationFeatures(input, this.config);
        const variance = features.reduce((sum, value) => sum + value * value, 0);
        const variances = drivingExplorationSigmas(this.config).map((sigma) => sigma * sigma * variance);
        this.samples.forEach((samples, axis) => {
            const oldest = Math.floor(elapsed / intervals[axis]);
            samples.forEach((_, key) => { if (key < oldest) samples.delete(key); });
        });
        return this.config.actions.planTimesSeconds.map((seconds) => {
            const times = intervals.map((interval) => (elapsed + seconds * 1000) / interval);
            const axes = times.slice(0, DRIVING_ANALOG_AXES).map((time, axis) => {
                // Snap rounding error at shared plan boundaries, not elapsed time.
                const nearest = Math.round(time);
                if (Math.abs(time - nearest) < 1e-9) time = nearest;
                const bucket = Math.floor(time), fraction = time - bucket;
                const angle = fraction * fraction * (3 - 2 * fraction) * Math.PI / 2;
                const a = Math.cos(angle), b = Math.sin(angle);
                const left = this.sample(axis, bucket, features.length);
                const right = this.sample(axis, bucket + 1, features.length);
                return features.reduce((sum, feature, i) => sum
                    + feature * (a * left[i] + b * right[i]), 0);
            });
            const shift = this.sample(DRIVING_ANALOG_AXES, Math.floor(times[DRIVING_ANALOG_AXES] + 1e-9), 1)[0];
            return { axes, variances: [...variances], shift };
        });
    }

    private sample(axis: number, bucket: number, size: number): number[] {
        let sample = this.samples[axis].get(bucket);
        if (!sample) {
            sample = Array.from({ length: size }, () => axis === DRIVING_ANALOG_AXES ? this.random()
                : Math.sqrt(-2 * Math.log(Math.max(1e-9, this.random()))) * Math.cos(2 * Math.PI * this.random()));
            this.samples[axis].set(bucket, sample);
        }
        return sample;
    }
}
