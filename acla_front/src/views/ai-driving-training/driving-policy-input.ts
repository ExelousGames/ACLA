import { drivingMemoryInputs, drivingMemorySlotInputs, DrivingPolicyInput } from './driving-memory';
import { DrivingModelConfig, drivingCameraInputs, drivingFeatureSizes, drivingMemorySlots, drivingObservationInputs, drivingPolicyInputs } from './driving-model-config';
import type { PolicyCameraImage } from 'views/live-session/track-vision/policy-camera';

/** Assemble present observations oldest first, in one shared feature layout.
 * Missing history bins are skipped; features omitted from history are zeroed. */
export function assembleDrivingInput(input: DrivingPolicyInput, config: DrivingModelConfig) {
    const bins = drivingMemorySlots(config);
    if (!input?.features || input.features.length !== drivingObservationInputs(config) + drivingMemoryInputs(config)
        || !input.features.every(Number.isFinite) || input.cameras?.length !== bins + 1
        || !input.cameras[bins]) throw new Error('Invalid policy observation.');
    const sizes = drivingFeatureSizes(config), inputs = drivingPolicyInputs(config), features: number[] = [];
    const cameras: { offset: number; image: PolicyCameraImage }[] = [];
    const append = (names: string[], cursor: number, image: PolicyCameraImage | null, age: number) => {
        const groups: Record<string, number[]> = {};
        for (const name of names) {
            groups[name] = input.features.slice(cursor, cursor + sizes[name]);
            cursor += sizes[name];
        }
        features.push(1, age);
        for (const name of config.observation.features) {
            if (name === 'camera') {
                if (image) {
                    if (image.width !== config.observation.cameraWidth || image.height !== config.observation.cameraHeight
                        || !(image.rgb instanceof Uint8Array) || image.rgb.length !== image.width * image.height * 3) {
                        throw new Error('Invalid policy camera observation.');
                    }
                    cameras.push({ offset: features.length, image });
                }
                features.push(...new Array(drivingCameraInputs(config)).fill(0));
            } else features.push(...(groups[name] ?? new Array(sizes[name]).fill(0)));
        }
    };
    for (let bin = bins - 1; bin >= 0; bin--) {
        const offset = bin * drivingMemorySlotInputs(config), present = input.features[offset], age = input.features[offset + 1];
        if ((present !== 0 && present !== 1) || age < 0 || age > 1
            || Boolean(input.cameras[bin]) !== Boolean(present && config.memory.features.includes('camera'))) {
            throw new Error('Invalid policy camera history.');
        }
        if (present) append(config.memory.features, offset + 2, input.cameras[bin], age);
    }
    append(config.observation.features, drivingMemoryInputs(config), input.cameras[bins], 0);
    return { features, cameras, steps: features.length / inputs };
}
