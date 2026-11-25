import recordedTrainingLabels from 'data/recorded-training-labels.json';
import defaults from './driving-model-defaults.json';
import { POLICY_CAMERA_HEIGHT, POLICY_CAMERA_WIDTH } from 'views/live-session/live-session-recording/policy-camera';

export type DrivingModelConfig = typeof defaults;
export type RecordedSamplingWeights = DrivingModelConfig['recordedSamplingWeights'];
export type RecordedSamplingCategory = keyof RecordedSamplingWeights;
export const UNLABELED_RECORDED_SAMPLING_CATEGORY = 'Unlabeled';
export const RECORDED_SAMPLING_CATEGORIES: readonly RecordedSamplingCategory[] =
    [...recordedTrainingLabels, UNLABELED_RECORDED_SAMPLING_CATEGORY] as RecordedSamplingCategory[];
export const DEFAULT_DRIVING_CONFIG: DrivingModelConfig = defaults;
export const cloneConfig = (config = DEFAULT_DRIVING_CONFIG): DrivingModelConfig => JSON.parse(JSON.stringify(config));

export const emptyRecordedSamplingWeights = (): RecordedSamplingWeights =>
    Object.fromEntries(RECORDED_SAMPLING_CATEGORIES.map((label) => [label, 0])) as RecordedSamplingWeights;

function validSamplingPercentages(values: unknown[]): values is number[] {
    return values.every((value): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 100)
        && Math.abs(values.reduce((sum, value) => sum + value, 0) - 100) < 1e-9;
}

export function validRecordedSamplingWeights(weights: Record<RecordedSamplingCategory, number | ''>): weights is RecordedSamplingWeights {
    return validSamplingPercentages(RECORDED_SAMPLING_CATEGORIES.map((label) => weights[label]));
}

function migrateRecordedSamplingWeights(value: unknown): RecordedSamplingWeights | undefined {
    if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== 3) return undefined;
    const legacy = value as Record<string, unknown>;
    const percentages = [legacy.general, legacy.cornering, legacy.shifting];
    if (!validSamplingPercentages(percentages)) return undefined;
    const [general, cornering, shifting] = percentages;
    const generalLabels = RECORDED_SAMPLING_CATEGORIES.filter((label) => label !== 'Cornering' && label !== 'Shifting'
        && label !== UNLABELED_RECORDED_SAMPLING_CATEGORY);
    // Split the former General share proportionally to the new defaults. Keep
    // a readable percentage in each field and put rounding residue in Unlabeled.
    const share = Number((general / (generalLabels.length + 2)).toPrecision(3));
    const weights = { ...emptyRecordedSamplingWeights(), Cornering: cornering, Shifting: shifting,
        Unlabeled: Number((general - share * generalLabels.length).toPrecision(15)) };
    for (const label of generalLabels) weights[label] = share;
    return weights;
}

export const DRIVING_ANALOG_AXES = 2;
export const DRIVING_GRU_GATES = 3;
export const DRIVING_DEMONSTRATION_BACKLOG = 512;
export const MAX_EXPLORATION_GEAR = 10;

/** Match the signed pedal and steering output order. */
export const drivingExplorationSigmas = (config: DrivingModelConfig): [number, number] =>
    [config.learning.pedalExplorationSigma, config.learning.steeringExplorationSigma];

/** Match the signed pedal, steering and categorical gear sampling order. */
export const drivingExplorationFrequencies = (config: DrivingModelConfig): [number, number, number] =>
    [config.learning.pedalExplorationUpdateHz, config.learning.steeringExplorationUpdateHz, config.learning.gearExplorationUpdateHz];

export function readDrivingConfig(value: unknown): DrivingModelConfig {
    const outputs = (value as Partial<DrivingModelConfig> | null)?.actions?.outputs;
    if (Array.isArray(outputs) && outputs.includes('throttle-tanh')) {
        throw new Error('This checkpoint uses separate throttle and brake outputs. Create a new model file to train the signed pedal policy.');
    }
    if (value && typeof value === 'object' && !Array.isArray(value)) {
        const weights = 'recordedSamplingWeights' in value ? migrateRecordedSamplingWeights(value.recordedSamplingWeights)
            : defaults.recordedSamplingWeights;
        if (weights) value = { ...value, recordedSamplingWeights: { ...weights } };
    }
    const learning = (value as { learning?: Record<string, unknown> } | null)?.learning;
    if (learning && typeof learning === 'object' && !Array.isArray(learning)
        && (!('throttleExplorationSpeedLimitKmh' in learning) || !('maxExplorationGear' in learning))) {
        value = { ...value as object, learning: {
            throttleExplorationSpeedLimitKmh: defaults.learning.throttleExplorationSpeedLimitKmh,
            maxExplorationGear: defaults.learning.maxExplorationGear, ...learning } };
    }
    // Speed and pedal rewards changed without changing the SAC model's input or weight layout.
    const reward = (value as { reward?: Record<string, unknown> } | null)?.reward;
    if (reward && typeof reward === 'object' && !('throttleChangeWeight' in reward) && !('brakeChangeWeight' in reward)) {
        const { accelerationTolerance, timingWeight, ...settings } = reward;
        if (typeof accelerationTolerance === 'number' && accelerationTolerance > 0 && accelerationTolerance <= 1e6
            && typeof timingWeight === 'number' && timingWeight >= 0 && timingWeight <= 1e6) {
            value = { ...value as object, reward: { ...settings, timingWeight: timingWeight / 100,
                throttleChangeWeight: defaults.reward.throttleChangeWeight, brakeChangeWeight: defaults.reward.brakeChangeWeight } };
        }
    }
    validateDrivingConfig(value);
    return value;
}

/** Include the oldest partial bin when the window is not an interval multiple. */
export const drivingMemorySlots = (config = DEFAULT_DRIVING_CONFIG): number =>
    Math.ceil(config.memory.windowMs / config.memory.intervalMs);

export function drivingFeatureEncoding(config = DEFAULT_DRIVING_CONFIG): string {
    const observation = config.observation;
    return JSON.stringify([Object.keys(observation).sort().map((key) =>
        [key, observation[key as keyof typeof observation]]), config.reward.referenceTiming]);
}

const reservedPositionFeatures = new Set(['position', 'referenceSpeed', 'targetSpeed']);

/** These saved feature slots keep their dimensions for older checkpoints. */
export function isReservedPositionFeature(name: string): boolean {
    return reservedPositionFeatures.has(name);
}

/** Scalar feature sizes; camera bytes travel separately until learned convolution.
 * Feature names, order and preprocessing are part of the saved model, not a version number. */
export function drivingFeatureSizes(config = DEFAULT_DRIVING_CONFIG): Record<string, number> {
    const o = config.observation;
    return { localVelocity: 3, speed: 1, game: o.categoryBits, car: o.categoryBits, map: o.categoryBits,
        position: 2, controls: 3, gear: 1, rpm: 1, yawRate: 1, gForce: 3,
        camera: 0, centerline: o.centerlineStepsPerSide * 2 * 3, aggression: 1, referenceSpeed: 1, targetSpeed: 1 };
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

/** A single GRU timestep, including presence and normalized age. */
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
                if (actual.length > 64 || actual[0] !== 0 || actual.some((n, i) => typeof n !== 'number'
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
        } else if (typeof expected === 'boolean') {
            if (typeof actual !== 'boolean') invalid();
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
        || !integer(c.network.cameraProjectionUnits, 1, 256)
        || c.network.layerNormEpsilon < 1e-9 || c.network.layerNormEpsilon > 0.01) return invalid();
    if (!validRecordedSamplingWeights(c.recordedSamplingWeights)
        || !integer(c.network.hiddenUnits, 1, 256) || !integer(o.categoryBits, 1, 32)
        || o.cameraWidth !== POLICY_CAMERA_WIDTH || o.cameraHeight !== POLICY_CAMERA_HEIGHT
        || DRIVING_GRU_GATES * c.network.hiddenUnits * (drivingPolicyInputs(c) + c.network.hiddenUnits + 1) > 2000000
        || !o.features.includes('camera') || !o.features.includes('centerline')
        || o.centerlineStepsPerSide !== defaults.observation.centerlineStepsPerSide
        || ![o.velocityScale, o.speedScale, o.gearScale, o.rpmScale, o.yawRateScale, o.gForceScale, o.centerlineScale,
            l.discountIntervalSeconds, l.weightLimit, l.criticGradientLimit, l.policyGradientLimit,
            l.demonstrationGradientLimit, l.gradientNormLimit, c.reward.totalLimit,
            c.reward.speedTolerance].every((n) => n > 0)
        || c.aggression > 100 || l.discount > 1 || drivingExplorationSigmas(c).some((sigma) => sigma < 1e-6)
        || l.entropyTemperature <= 0 || l.targetTau <= 0 || l.targetTau > 1
        || !integer(l.replayCapacity, 1, 2048) || !integer(l.replayBatchSize, 1, 64) || l.replayBatchSize > l.replayCapacity
        || l.actionEpsilon < 1e-9 || l.actionEpsilon >= 0.5
        || l.shiftTemperature < 0.001 || l.shiftTemperature > 1
        || !integer(l.maxExplorationGear, 0, MAX_EXPLORATION_GEAR)
        || !integer(l.throttleExplorationSpeedLimitKmh, 0, defaults.learning.throttleExplorationSpeedLimitKmh)
        || drivingExplorationFrequencies(c).some((frequency) => frequency < 0.1 || frequency > 10)
        || [c.reward.steeringChangeWeight, c.reward.throttleChangeWeight, c.reward.brakeChangeWeight].some((weight) => weight > 1)
        || l.sequenceSeconds <= 0 || l.sequenceSeconds > 30
        || !integer(c.memory.intervalMs, 1, 30000) || !integer(c.memory.windowMs, 1, 30000)
        || c.memory.intervalMs > c.memory.windowMs || !integer(drivingMemorySlots(c), 1, 100)
        || c.memory.features.some((name) => !o.features.includes(name))
        || c.network.initialActionBias.some((n) => Math.abs(n) > l.weightLimit)
        || c.network.initialOutputWeightScale > l.weightLimit
        || !integer(c.actions.shiftPulseMs, 1, 1000)
        || c.actions.planAgreementTolerance > 1
        || !integer(c.runtime.saveEverySteps, 1, 1000000)
        || ![c.runtime.telemetryMaxAgeMs, c.runtime.visionMaxAgeMs, c.runtime.maxTransitionSeconds,
            c.runtime.controlIntervalMs, c.runtime.manualControlIntervalMs].every((n) => n > 0)
        || c.runtime.controlIntervalMs > 250 || c.runtime.manualControlIntervalMs > 250
        || c.runtime.maxPositionJump <= 0 || c.runtime.maxPositionJump > 0.5 || c.reward.progressLimit > 0.5) invalid();
}
