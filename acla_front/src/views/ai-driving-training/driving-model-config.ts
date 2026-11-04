import defaults from './driving-model-defaults.json';
import { POLICY_CAMERA_HEIGHT, POLICY_CAMERA_WIDTH } from 'views/live-session/track-vision/policy-camera';

export type DrivingModelConfig = typeof defaults;
export const DEFAULT_DRIVING_CONFIG: DrivingModelConfig = defaults;
export const cloneConfig = (config = DEFAULT_DRIVING_CONFIG): DrivingModelConfig => JSON.parse(JSON.stringify(config));

/** Include the oldest partial bin when the window is not an interval multiple. */
export const drivingMemorySlots = (config = DEFAULT_DRIVING_CONFIG): number =>
    Math.ceil(config.memory.windowMs / config.memory.intervalMs);

export function drivingFeatureEncoding(config = DEFAULT_DRIVING_CONFIG): string {
    const observation = config.observation;
    return JSON.stringify([Object.keys(observation).sort().map((key) =>
        [key, observation[key as keyof typeof observation]]), config.reward.referenceTiming, config.reward.aggressionMaxRadius]);
}

/** Scalar feature sizes; camera bytes travel separately until learned convolution.
 * Feature names, order and preprocessing are part of the saved model, not a version number. */
export function drivingFeatureSizes(config = DEFAULT_DRIVING_CONFIG): Record<string, number> {
    const o = config.observation;
    return { localVelocity: 3, speed: 1, game: o.categoryBits, car: o.categoryBits, map: o.categoryBits,
        position: 2, controls: 3, referenceOffset: 3, gear: 1, rpm: 1, yawRate: 1, gForce: 3,
        camera: 0, aggression: 1, referenceSpeed: 1, targetSpeed: 1 };
}

export function drivingCameraInputs(config = DEFAULT_DRIVING_CONFIG): number {
    return config.network.cameraProjectionUnits;
}

/** Row-major (y, x, channel) grid retained by the learned linear projection. */
export function drivingCameraPooledInputs(config = DEFAULT_DRIVING_CONFIG): number {
    const layers = drivingCameraLayers(config), last = layers[layers.length - 1];
    return config.network.cameraPooling.width * config.network.cameraPooling.height * last.channels;
}

/** Overlapping convolutions with symmetric zero padding, including odd input sizes. */
export function drivingCameraLayers(config = DEFAULT_DRIVING_CONFIG) {
    let width = config.observation.cameraWidth, height = config.observation.cameraHeight, channels = 3;
    return config.network.cameraLayers.map((layer) => {
        const inputWidth = width, inputHeight = height, inputChannels = channels;
        width = Math.floor((width + 2 * layer.padding - layer.kernelSize) / layer.stride) + 1;
        height = Math.floor((height + 2 * layer.padding - layer.kernelSize) / layer.stride) + 1;
        channels = layer.channels;
        return { ...layer, inputWidth, inputHeight, inputChannels, width, height,
            columns: layer.kernelSize ** 2 * inputChannels + 1 };
    });
}

/** A single LSTM timestep, including presence and normalized age. */
export function drivingPolicyInputs(config = DEFAULT_DRIVING_CONFIG): number {
    const sizes: Record<string, number> = { ...drivingFeatureSizes(config), camera: drivingCameraInputs(config) };
    return 2 + config.observation.features.reduce((sum, name) => sum + sizes[name], 0);
}

export function drivingObservationInputs(config = DEFAULT_DRIVING_CONFIG): number {
    const sizes = drivingFeatureSizes(config);
    return config.observation.features.reduce((sum, name) => sum + sizes[name], 0);
}

/** Reject missing/unknown settings rather than silently substituting today's defaults. */
export function validateDrivingConfig(value: unknown): asserts value is DrivingModelConfig {
    const invalid = () => { throw new Error('Invalid or unsupported self-driving model configuration.'); };
    const shape = (actual: any, expected: any, path = ''): void => {
        if (Array.isArray(expected)) {
            if (!Array.isArray(actual) || !actual.length) return invalid();
            if (path === 'observation.features' || path === 'memory.features') {
                if (actual.some((name) => !defaults.observation.features.includes(name)) || new Set(actual).size !== actual.length) invalid();
            } else if (path === 'network.cameraLayers') {
                if (actual.length !== expected.length) return invalid();
                actual.forEach((layer, i) => shape(layer, expected[i], `${path}.${i}`));
            } else if (path === 'actions.planTimesSeconds') {
                if (actual.length > 32 || actual[0] !== 0 || actual.some((n, i) => typeof n !== 'number'
                    || !Number.isFinite(n) || n < 0 || n > 30 || (i > 0 && n <= actual[i - 1]))) invalid();
            } else if (path === 'network.initialActionBias') {
                if (actual.length !== expected.length || actual.some((n) => typeof n !== 'number' || !Number.isFinite(n))) invalid();
            } else if (JSON.stringify(actual) !== JSON.stringify(expected)) invalid();
        } else if (expected && typeof expected === 'object') {
            if (!actual || typeof actual !== 'object' || Array.isArray(actual)
                || Object.keys(actual).length !== Object.keys(expected).length) return invalid();
            Object.keys(expected).forEach((key) => shape(actual[key], expected[key], path ? `${path}.${key}` : key));
        } else if (typeof expected === 'number') {
            if (typeof actual !== 'number' || !Number.isFinite(actual) || actual < 0 || actual > 1e6) invalid();
        } else if (actual !== expected) invalid();
    };
    shape(value, defaults);
    const c = value as DrivingModelConfig, o = c.observation, l = c.learning;
    const integer = (n: number, min: number, max: number) => Number.isSafeInteger(n) && n >= min && n <= max;
    if (c.network.cameraLayers.some((layer, i) => layer.kernelSize !== defaults.network.cameraLayers[i].kernelSize
        || layer.stride !== defaults.network.cameraLayers[i].stride
        || layer.padding !== defaults.network.cameraLayers[i].padding || !integer(layer.channels, 1, 64))) return invalid();
    if (c.network.cameraPooling.width !== defaults.network.cameraPooling.width
        || c.network.cameraPooling.height !== defaults.network.cameraPooling.height
        || !integer(c.network.cameraProjectionUnits, 1, 256)) return invalid();
    if (!integer(c.network.hiddenUnits, 1, 256) || !integer(o.categoryBits, 1, 32)
        || o.cameraWidth !== POLICY_CAMERA_WIDTH || o.cameraHeight !== POLICY_CAMERA_HEIGHT
        || 4 * c.network.hiddenUnits * (drivingPolicyInputs(c) + c.network.hiddenUnits + 1) > 2000000
        || !o.features.includes('camera')
        || ![o.velocityScale, o.speedScale, o.referenceOffsetScale, o.gearScale, o.rpmScale, o.yawRateScale, o.gForceScale,
            l.explorationSigma, l.discountIntervalSeconds, l.weightLimit, l.advantageLimit, l.policyGradientLimit,
            l.demonstrationGradientLimit, c.reward.proximityScale, c.reward.totalLimit,
            c.reward.speedTolerance, c.reward.accelerationTolerance,
            c.reward.lineDistanceTolerance, c.reward.lineSmoothingMs].every((n) => n > 0)
        || c.aggression > 100 || l.discount > 1 || l.explorationSigma < 1e-6 || l.actionEpsilon < 1e-9 || l.actionEpsilon >= 0.5
        || !integer(c.memory.intervalMs, 1, 30000) || !integer(c.memory.windowMs, 1, 30000)
        || c.memory.intervalMs > c.memory.windowMs || !integer(drivingMemorySlots(c), 1, 100)
        || c.memory.features.some((name) => !o.features.includes(name))
        || c.network.initialActionBias.some((n) => Math.abs(n) > l.weightLimit)
        || c.network.initialOutputWeightScale > l.weightLimit
        || !integer(c.actions.shiftPulseMs, 1, 1000) || c.actions.shiftCooldownMs < c.actions.shiftPulseMs
        || !integer(c.runtime.saveEverySteps, 1, 1000000)
        || ![c.runtime.telemetryMaxAgeMs, c.runtime.visionMaxAgeMs, c.runtime.maxTransitionSeconds,
            c.runtime.controlIntervalMs, c.runtime.manualControlIntervalMs].every((n) => n > 0)
        || c.runtime.controlIntervalMs > 250 || c.runtime.manualControlIntervalMs > 250
        || c.runtime.maxPositionJump <= 0 || c.runtime.maxPositionJump > 0.5 || c.reward.progressLimit > 0.5) invalid();
}
