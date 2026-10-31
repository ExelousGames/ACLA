import { DrivingObservation } from './driving-observation';
import { DEFAULT_DRIVING_CONFIG, drivingFeatureSizes, drivingMemorySlots } from './driving-model-config';
import type { PolicyCameraImage } from 'views/live-session/track-vision/policy-camera';

export interface DrivingPolicyInput {
    features: number[];
    /** Historical images in bin order, followed by the current full-resolution image. */
    cameras: (PolicyCameraImage | null)[];
}

export const drivingMemorySlotInputs = (config = DEFAULT_DRIVING_CONFIG) => {
    const sizes = drivingFeatureSizes(config);
    return 2 + config.memory.features.reduce((sum, name) => sum + sizes[name], 0);
};
export const drivingMemoryInputs = (config = DEFAULT_DRIVING_CONFIG) =>
    drivingMemorySlots(config) * drivingMemorySlotInputs(config);

/** A timestamped history shared by inference, the critic and human demonstrations.
 * Defaults retain the complete observation; saved models keep their declared feature layout. */
export class DrivingMemory {
    private frames: { capturedAt: number; features: number[]; camera: PolicyCameraImage }[] = [];
    private cameraCopies = new WeakMap<PolicyCameraImage, PolicyCameraImage>();
    private last?: DrivingObservation;

    constructor(private config = DEFAULT_DRIVING_CONFIG) {}

    reset() {
        this.frames = [];
        this.last = undefined;
        this.cameraCopies = new WeakMap();
    }

    encode(observation: DrivingObservation): DrivingPolicyInput {
        const previous = this.last;
        if (previous && (observation.capturedAt < previous.capturedAt
            || observation.capturedAt - previous.capturedAt > this.config.runtime.maxTransitionSeconds * 1000
            || observation.session !== previous.session || observation.reference !== previous.reference
            || JSON.stringify(observation.context) !== JSON.stringify(previous.context))) this.reset();

        this.frames = this.frames.filter((frame) => observation.capturedAt - frame.capturedAt <= this.config.memory.windowMs);
        const slotInputs = drivingMemorySlotInputs(this.config);
        const history = new Array(drivingMemoryInputs(this.config)).fill(0);
        const cameras: (PolicyCameraImage | null)[] = new Array(drivingMemorySlots(this.config)).fill(null);
        // Newest sample per elapsed-time bin; missing bins stay explicitly absent.
        for (let i = this.frames.length - 1; i >= 0; i--) {
            const frame = this.frames[i];
            const age = observation.capturedAt - frame.capturedAt;
            if (age <= 0) continue;
            const offset = (Math.ceil(age / this.config.memory.intervalMs) - 1) * slotInputs;
            if (history[offset]) continue;
            history[offset] = 1;
            history[offset + 1] = age / this.config.memory.windowMs;
            frame.features.forEach((value, j) => { history[offset + 2 + j] = value; });
            if (this.config.memory.features.includes('camera')) cameras[offset / slotInputs] = frame.camera;
        }
        // Own one immutable copy per capture, shared by every history/plan that needs it.
        let camera = this.cameraCopies.get(observation.camera);
        if (!camera) {
            camera = { ...observation.camera, rgb: observation.camera.rgb.slice() };
            this.cameraCopies.set(observation.camera, camera);
        }
        if (!this.last || observation.capturedAt > this.last.capturedAt) {
            const features = observation.features;
            const sizes = drivingFeatureSizes(this.config);
            const groups: Record<string, number[]> = {};
            let offset = 0;
            this.config.observation.features.forEach((name) => {
                groups[name] = features.slice(offset, offset + sizes[name]);
                offset += sizes[name];
            });
            this.frames.push({ capturedAt: observation.capturedAt, camera,
                features: this.config.memory.features.flatMap((name) => groups[name]) });
            this.last = observation;
        }
        return { features: [...history, ...observation.features], cameras: [...cameras, camera] };
    }
}
