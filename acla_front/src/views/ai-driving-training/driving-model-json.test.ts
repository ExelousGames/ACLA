import { mockDrivingGpu } from './driving-gpu-test-model';
import { cloneConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingObservationInputs, drivingPolicyInputs, validateDrivingConfig } from './driving-model-config';
import { DrivingMemory, drivingMemoryInputs } from './driving-memory';
import { DrivingPolicy, emptyTrainingDocument, readTrainingDocument } from './driving-policy';
import { drivingReward, readDrivingObservation, NEUTRAL } from './driving-observation';
import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingControlOutput } from './driving-control-output';
import { trainingFixture } from './training-fixtures';


beforeEach(() => { mockDrivingGpu(); });
afterEach(() => { jest.restoreAllMocks(); });

it('keeps a widescreen spatial grid and projects it to 256 visual LSTM inputs', () => {
    const config = cloneConfig();
    expect(() => validateDrivingConfig(config)).not.toThrow();
    expect(drivingCameraLayers(config).map(({ width, height, channels }) => [width, height, channels]))
        .toEqual([[192, 108, 16], [96, 54, 32], [48, 27, 64], [24, 14, 64]]);
    expect(drivingCameraPooledInputs(config)).toBe(8 * 4 * 64);
    expect(drivingCameraInputs(config)).toBe(256);
    expect(drivingPolicyInputs(config)).toBe(drivingObservationInputs(config) + 256 + 2);
    const { observation } = trainingFixture();
    const saved = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    expect(saved.model.cameraProjectionWeights).toHaveLength(256);
    expect(saved.model.cameraProjectionWeights.every((row) => row.length === 2049)).toBe(true);
});

it('rebuilds the whole observation, memory, network and plan from a non-default JSON document', async () => {
    const config = cloneConfig();
    config.observation.features.reverse();
    config.observation.velocityScale = 50;
    config.observation.categoryBits = 8;
    config.network.cameraLayers[2].channels = 8;
    config.memory.windowMs = 600;
    config.memory.intervalMs = 300;
    config.memory.features = ['speed', 'localVelocity', 'camera'];
    config.network.hiddenUnits = 7;
    config.actions.planTimesSeconds = [0, 0.3, 1.2];
    config.aggression = 75;
    config.reward.progressWeight = 200;
    const fixture = trainingFixture();
    const observation = readDrivingObservation(fixture.live, fixture.vision, fixture.referenceLap, config.aggression, 1000, config).observation!;
    expect(observation.features).toHaveLength(drivingObservationInputs(config));
    expect(observation.features.slice(-3)).toEqual([0.02, 0.6, 0]);
    const memory = new DrivingMemory(config);
    memory.encode(observation);
    const features = memory.encode({ ...observation, capturedAt: 1300 });
    expect(features.features).toHaveLength(observation.features.length + 2 * (2 + 1 + 3));
    expect(features.features.slice(0, 6)).toEqual([1, 0.5, 0.3, 0.02, 0.6, 0]);
    const original = new DrivingPolicy(observation.context, observation.features.length, () => 0.6, config);
    const json = JSON.stringify(original.checkpoint());
    const restored = DrivingPolicy.restore(JSON.parse(json));
    expect(restored.matches(observation)).toBe(true);
    expect(restored.checkpoint().model.lstmWeights).toHaveLength(4 * 7);
    expect(restored.checkpoint().model.outputWeights).toHaveLength(16);
    expect(restored.planTimes).toEqual([0, 0.3, 1.2]);
    expect(restored.checkpoint()).toEqual(JSON.parse(json));
    expect(restored.checkpoint()).not.toHaveProperty('version');
});

it('detects different feature meaning or ordering even when the vector length is unchanged', () => {
    const fixture = trainingFixture();
    const policy = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length);
    for (const change of ['order', 'scale'] as const) {
        const config = cloneConfig();
        if (change === 'order') config.observation.features.reverse();
        else config.observation.yawRateScale = 10;
        const observation = readDrivingObservation(fixture.live, fixture.vision, fixture.referenceLap, 50, 1000, config).observation!;
        expect(observation.features).toHaveLength(fixture.observation.features.length);
        expect(policy.matches(observation)).toBe(false);
        expect(policy.inputMismatches(observation)).toContain('Input encoding differs from the model JSON. Read live inputs using this model’s configuration.');
    }
});

it('uses the JSON plan times and target tolerance for delayed training', async () => {
    const { observation } = trainingFixture();
    const config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.3, 1.2];
    config.learning.planTargetToleranceMs = 20;
    const policy = DrivingPolicy.restore(new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint());
    const learn = jest.spyOn(policy, 'learnPlannedActionsAsync').mockResolvedValue(undefined);
    const training = new DrivingPlanTraining();
    const features = new DrivingMemory(config).encode(observation);
    const shifts = { up: true, down: true };
    await training.recordAsync(policy, features, NEUTRAL, shifts, 1000);
    await training.recordAsync(policy, features, NEUTRAL, shifts, 1310);
    expect(learn).toHaveBeenCalledWith(features, [{ step: 1, action: NEUTRAL, shifts }]);
    learn.mockClear();
    await training.recordAsync(policy, features, NEUTRAL, shifts, 2221);
    expect(learn).not.toHaveBeenCalled();
});

it('uses configured reward coefficients and freshness instead of current defaults', () => {
    const fixture = trainingFixture();
    const config = cloneConfig();
    config.reward.timingWeight *= 3;
    config.runtime.visionMaxAgeMs = 50;
    const before = fixture.observation;
    const after = { ...before, capturedAt: 1200, measuredAt: 1200, normalizedPosition: 0.255, speed: 106, controls: NEUTRAL };
    const normal = drivingReward(before, after);
    const changed = drivingReward(before, after, config);
    expect(normal.timing).not.toBe(0);
    expect(changed.timing).toBeCloseTo(normal.timing * 3);
    expect(readDrivingObservation(fixture.live, fixture.vision, fixture.referenceLap, 50, 1100, config).error).toMatch(/fresh image/);
    expect(readDrivingObservation(fixture.live, fixture.vision, fixture.referenceLap, 50, 1100).observation).toBeDefined();
});

it('supports a configuration-only JSON and isolates exported values from the running model', () => {
    const document = emptyTrainingDocument();
    document.configuration.aggression = 80;
    expect(readTrainingDocument(JSON.parse(JSON.stringify(document)))).toEqual(document);
    const { observation } = trainingFixture();
    const policy = new DrivingPolicy(observation.context, observation.features.length);
    const exported = policy.checkpoint();
    exported.configuration.network.hiddenUnits = 99;
    exported.model.context.car = 'modified';
    exported.model.lstmWeights[0][0] = 4;
    expect(policy.configuration.network.hiddenUnits).toBe(128);
    expect(policy.context.car).toEqual(observation.context.car);
    expect(policy.checkpoint().model.lstmWeights[0][0]).not.toBe(4);
});

it('rejects old pedal-target models even though the number of inputs is unchanged', () => {
    const legacy: any = emptyTrainingDocument();
    legacy.configuration.observation.features.splice(-2, 2, 'targetPedals');
    expect(() => readTrainingDocument(legacy)).toThrow(/unsupported/);
    const oldReward: any = emptyTrainingDocument();
    delete oldReward.configuration.reward.timingWeight;
    oldReward.configuration.reward.speedWeight = 1;
    oldReward.configuration.reward.pedalWeight = 100;
    expect(() => readTrainingDocument(oldReward)).toThrow(/unsupported/);
});

it('uses a neutral combined pedal output in new models and rejects separate-pedal model layouts', () => {
    const document = emptyTrainingDocument();
    expect(document.configuration.actions.outputs).toEqual(['throttle-brake-tanh', 'steering-tanh', 'hold-logit', 'up-logit', 'down-logit']);
    expect(document.configuration.network.initialActionBias).toEqual([0, 0, 1, 0, 0]);
    document.configuration.actions.outputs = ['throttle-sigmoid', 'brake-sigmoid', 'steering-tanh', 'hold-logit', 'up-logit', 'down-logit'];
    document.configuration.network.initialActionBias = [-2, -3, 0, 1, 0, 0];
    const original = JSON.stringify(document);
    expect(() => readTrainingDocument(document)).toThrow(/configuration/);
    expect(JSON.stringify(document)).toBe(original);
});

it('requires the Live Phrases reward encoding and rejects earlier aggression calculations without migration', () => {
    const document: any = emptyTrainingDocument();
    expect(document.configuration.reward.referenceTiming).toBe('live-phrase-speed-profile');
    delete document.configuration.reward.referenceTiming;
    expect(() => readTrainingDocument(document)).toThrow(/configuration/);
    for (const encoding of ['speed-window', 'pedal-event-speed-profile']) {
        document.configuration.reward.referenceTiming = encoding;
        expect(() => readTrainingDocument(document)).toThrow(/configuration/);
    }
});

it('omits instability settings and rejects model JSON that still includes them', () => {
    const document = emptyTrainingDocument();
    for (const key of ['yawWindowMs', 'yawDeadband', 'yawPenalty']) {
        expect(document.configuration.reward).not.toHaveProperty(key);
    }
    Object.assign(document.configuration.reward, { yawWindowMs: 1000, yawDeadband: 0.05, yawPenalty: 0.4 });
    expect(() => readTrainingDocument(document)).toThrow(/configuration/);
});

it.each([
    (c: any) => { c.observation.features.push('referenceOffset'); },
    (c: any) => { c.memory.features.push('referenceOffset'); },
    (c: any) => { c.observation.referenceFrame = 'car-local-right-forward-up'; },
    (c: any) => { c.observation.referenceOffsetScale = 20; },
    (c: any) => { c.reward.proximityScale = 3; },
    (c: any) => { c.reward.lineWeight = c.reward.progressWeight; delete c.reward.progressWeight; },
    (c: any) => { c.reward.lineDistanceTolerance = 10; },
    (c: any) => { c.reward.lineSmoothingMs = 500; },
])('rejects removed reference-error model settings without migration %#', (change) => {
    const document = emptyTrainingDocument();
    expect(document.configuration.observation.features).not.toContain('referenceOffset');
    expect(document.configuration.memory.features).not.toContain('referenceOffset');
    change(document.configuration);
    const original = JSON.stringify(document);
    expect(() => readTrainingDocument(document)).toThrow(/configuration/);
    expect(JSON.stringify(document)).toBe(original);
});

it.each([
    (c: any) => { delete c.learning.discount; },
    (c: any) => { c.network.activation = 'relu'; },
    (c: any) => { c.observation.categoryEncoding = 'unknown'; },
    (c: any) => { c.observation.features[0] = 'unknown'; },
    (c: any) => { c.observation.features[0] = c.observation.features[1]; },
    (c: any) => { c.observation.cameraWidth = 8; },
    (c: any) => { c.observation.gForceScale = 0; },
    (c: any) => { c.network.type = 'lstm'; },
    (c: any) => { c.network.cameraLayers.pop(); },
    (c: any) => { c.network.cameraLayers[0].kernelSize = 4; },
    (c: any) => { c.network.cameraActivation = 'relu'; },
    (c: any) => { c.network.cameraLayers[0].channels = 0; },
    (c: any) => { c.network.cameraLayers[0].channels = 1.5; },
    (c: any) => { c.network.cameraLayers[0].stride = 5; },
    (c: any) => { c.network.cameraLayers[0].channels = 65; },
    (c: any) => { c.network.cameraLayers[0].padding = 0; },
    (c: any) => { c.network.cameraPooling.width = 1; },
    (c: any) => { c.network.cameraPooling.height = 1; },
    (c: any) => { c.network.cameraPooling.type = 'global-average'; },
    (c: any) => { c.network.cameraProjectionUnits = 0; },
    (c: any) => { c.network.cameraProjectionUnits = 257; },
    (c: any) => { c.network.cameraProjectionUnits = 1.5; },
    (c: any) => { c.observation.cameraResize = 'stretch'; },
    (c: any) => { c.observation.cameraEncoding = 'rgb-bytes-convolution'; },
    (c: any) => { c.network.hiddenUnits = 0; },
    (c: any) => { c.memory.windowMs = 0; },
    (c: any) => { c.memory.intervalMs = 0; },
    (c: any) => { c.memory.intervalMs = c.memory.windowMs + 1; },
    (c: any) => { c.memory.windowMs = 30000; c.memory.intervalMs = 299; },
    (c: any) => { c.actions.planTimesSeconds = [0, 1, 0.5]; },
    (c: any) => { c.actions.outputs.reverse(); },
    (c: any) => { c.learning.explorationSigma = 0; },
    (c: any) => { c.reward.speedTolerance = 0; },
    (c: any) => { c.reward.accelerationTolerance = 0; },
    (c: any) => { c.reward.progressWeight = -1; },
    (c: any) => { c.learning.explorationSigma = Number.MIN_VALUE; },
    (c: any) => { c.learning.actorRate = NaN; },
    (c: any) => { c.aggression = 101; },
    (c: any) => { c.extra = true; },
])('rejects missing, unsupported or inconsistent configuration %#', (change) => {
    const document = emptyTrainingDocument();
    change(document.configuration);
    expect(() => readTrainingDocument(document)).toThrow(/configuration/);
});

it('rejects weights that do not match the declared layout without relying on a version', () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.configuration.network.cameraLayers[2].channels = 8;
    expect(() => validateDrivingConfig(checkpoint.configuration)).not.toThrow();
    expect(() => DrivingPolicy.restore(checkpoint)).toThrow(/checkpoint/);
    expect(() => DrivingPolicy.restore({ version: 13, ...checkpoint.model })).toThrow(/checkpoint/);
});

it('initializes weights within a small JSON weight limit', () => {
    const { observation } = trainingFixture();
    const config = cloneConfig();
    config.learning.weightLimit = 0.0001;
    config.network.initialActionBias.fill(0);
    config.network.initialOutputWeightScale = 0.00001;
    const policy = new DrivingPolicy(observation.context, observation.features.length, () => 0.9, config);
    expect(() => DrivingPolicy.restore(policy.checkpoint())).not.toThrow();
});

it('runs the control loop with saved memory dimensions, shift pulse and cooldown', async () => {
    jest.useFakeTimers();
    const config = cloneConfig();
    config.memory.windowMs = 400;
    config.memory.intervalMs = 100;
    config.actions.shiftPulseMs = 50;
    config.actions.shiftCooldownMs = 300;
    config.runtime.controlIntervalMs = 50;
    let now = 1000;
    let observation = trainingFixture(now).observation;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.outputWeights.slice(2, 5).forEach((row, i) => { row.fill(0); row[config.network.hiddenUnits] = i === 1 ? 4 : 0; });
    const policy = DrivingPolicy.restore(checkpoint);
    const decide = jest.spyOn(policy, 'decideAsync');
    const initialize = jest.spyOn(policy, 'initializeGpu');
    const controller = { connect: jest.fn(), disconnect: jest.fn(), apply: jest.fn().mockResolvedValue(undefined) };
    const loop = new DrivingTrainingLoop(new DrivingControlOutput(controller), () => ({ observation }), () => undefined, () => now);
    const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
    try {
        await loop.start(policy, false); await flush();
        expect(loop.state.action.shiftUp).toBe(true);
        expect(decide.mock.calls[0][0].features).toHaveLength(observation.features.length + drivingMemoryInputs(config));
        now = 1050;
        jest.advanceTimersByTime(50); await flush();
        expect(loop.state.action.shiftUp).toBe(false);
        now = 1200;
        observation = { ...observation, capturedAt: now, measuredAt: now };
        jest.advanceTimersByTime(150); await flush();
        expect(loop.state.action.shiftUp).toBe(false);
        now = 1400;
        observation = { ...observation, capturedAt: now, measuredAt: now };
        jest.advanceTimersByTime(200); await flush();
        expect(loop.state.action.shiftUp).toBe(true);
    } finally {
        await loop.stop();
        initialize.mockRestore();
        jest.useRealTimers();
    }
});
