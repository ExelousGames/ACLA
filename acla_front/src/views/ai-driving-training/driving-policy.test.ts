import { DrivingPolicy } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingMemory } from './driving-memory';
import { trainingFixture } from './training-fixtures';
import { cloneConfig } from './driving-model-config';
import { readDrivingObservation } from './driving-observation';

beforeEach(() => { mockDrivingGpu(); });
afterEach(() => { jest.restoreAllMocks(); });

function setup(draw = 0.7) {
    const { observation } = trainingFixture(), config = cloneConfig();
    config.network.hiddenUnits = 3;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    return { observation, input: new DrivingMemory(config).encode(observation),
        policy: new DrivingPolicy(observation.context, observation.features.length, () => draw, config) };
}
const shifts = { up: true, down: false };
const action = { throttle: 0.7, brake: 0.1, steering: -0.3, shiftUp: true };

it('requires GPU initialization for every prediction and learning entry point', async () => {
    const { policy, input } = setup();
    await expect(policy.decideAsync(input, false)).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnAsync({} as any, 1, input, 0.2, [])).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnDemonstrationAsync(input, action, shifts)).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnPlannedActionsAsync(input, [])).rejects.toThrow(/Initialize.*GPU/);
    const train = jest.fn(async () => undefined);
    await expect(policy.trainBatchAsync(train)).rejects.toThrow(/Initialize.*GPU/);
    expect(train).not.toHaveBeenCalled();
    expect(policy.steps).toBe(0);
    for (const retired of ['decide', 'learn', 'learnDemonstration', 'learnPlannedActions', 'lstm', 'cameraEncoder']) {
        expect(policy).not.toHaveProperty(retired);
    }
});

it('uploads initialized weights once and releases the CPU seed after GPU initialization', async () => {
    const { policy } = setup(), seed = policy.checkpoint();
    await Promise.all([policy.initializeGpu(), policy.initializeGpu()]);
    expect(DrivingGpuModel.create).toHaveBeenCalledTimes(1);
    expect(DrivingGpuModel.create).toHaveBeenCalledWith(policy.configuration, seed.model);
    expect(policy).toHaveProperty('initialWeights', undefined);
    expect(() => policy.checkpoint()).toThrow(/awaited/);
    expect(await policy.checkpointAsync()).toEqual(seed);
});

it('creates and restores 31 action slots spanning three seconds at 100 ms intervals', async () => {
    const { policy, input } = setup();
    const checkpoint = policy.checkpoint();
    expect(checkpoint.configuration.actions.planTimesSeconds).toEqual(Array.from({ length: 31 }, (_, step) => step / 10));
    expect(checkpoint.model.outputWeights).toHaveLength(156); // 31 × five action outputs + critic.
    const restored = DrivingPolicy.restore(JSON.parse(JSON.stringify(checkpoint)));
    await restored.initializeGpu();
    const decision = await restored.decideAsync(input, false);
    expect(decision.plan.map(({ timeSeconds }) => timeSeconds)).toEqual(Array.from({ length: 31 }, (_, step) => step / 10));
    expect(Number.isFinite(decision.value)).toBe(true);
    restored.releaseDecision(decision);
    restored.dispose();
    policy.dispose();
});

it('fails without a GPU and permits an explicit initialization retry', async () => {
    const { policy, input } = setup();
    jest.mocked(DrivingGpuModel.create).mockRejectedValueOnce(new Error('Hardware GPU unavailable'));
    await expect(policy.initializeGpu()).rejects.toThrow('Hardware GPU unavailable');
    await expect(policy.decideAsync(input, false)).rejects.toThrow(/Initialize/);
    await policy.initializeGpu();
    await expect(policy.decideAsync(input, false)).resolves.toHaveProperty('gpu');
});

it.each([0.1, 0.7, 0.95])('decodes GPU outputs, samples controls and preserves execution masks (draw=%s)', async (draw) => {
    const { policy, input } = setup(draw);
    await policy.initializeGpu();
    const decision = await policy.decideAsync(input, true, { up: false, down: false });
    expect(decision.input).toBe(input);
    expect(decision.plan.map((step) => step.timeSeconds)).toEqual(policy.planTimes);
    expect(decision.plan[0].shiftProbabilities).toEqual([1, 0, 0]);
    expect(decision.plan[0].shifts).toEqual({ up: false, down: false });
    expect(decision.plan[0].shift).toBe(0);
    decision.plan.forEach(({ means, latent }) => expect(latent).not.toEqual(means));
    decision.plan.forEach(({ action: result }) => {
        expect(result.throttle).toBeGreaterThanOrEqual(0);
        expect(result.throttle).toBeLessThanOrEqual(1);
        expect(result.brake).toBeGreaterThanOrEqual(0);
        expect(result.brake).toBeLessThanOrEqual(1);
        expect(result.throttle * result.brake).toBe(0);
        expect(Math.abs(result.steering)).toBeLessThanOrEqual(1);
        expect(result.shiftUp && result.shiftDown).toBe(false);
    });
    expect(decision).not.toHaveProperty('sequence');
    policy.releaseDecision(decision);
    expect(decision.gpu.released).toBe(true);
});

it.each([-100, -0.7, 0, 0.4, 100])('decodes a signed pedal output for every plan slot (latent=%s)', async (pedalLatent) => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const prediction = await gpu.forward(input);
    prediction.outputs = [...policy.planTimes.flatMap(() => [pedalLatent, Math.atanh(-0.3), 0, 4, -1]), 0.75];
    jest.mocked(gpu.forward).mockResolvedValueOnce(prediction);
    const decision = await policy.decideAsync(input, false);
    expect(decision.value).toBe(0.75);
    for (const { action: result, means, latent } of decision.plan) {
        const pedal = Math.tanh(pedalLatent);
        expect(result.throttle).toBe(Math.max(0, pedal));
        expect(result.brake).toBe(Math.max(0, -pedal));
        expect(result.throttle - result.brake).toBe(pedal);
        expect(result.steering).toBeCloseTo(-0.3);
        expect(result.shiftUp).toBe(true);
        expect(result.shiftDown).toBe(false);
        expect(means).toHaveLength(2);
        expect(latent).toHaveLength(2);
    }
    policy.releaseDecision(decision);
});

it('sends raw demonstration and delayed targets straight to GPU learning without inference readback', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    await policy.learnDemonstrationAsync(input, action, shifts);
    const targets = [{ step: 6, action, shifts }];
    await policy.learnPlannedActionsAsync(input, targets);
    expect(gpu.learnTargets).toHaveBeenNthCalledWith(1, input, [{ step: 0, action, shifts }]);
    expect(gpu.learnTargets).toHaveBeenNthCalledWith(2, input, targets);
    expect(gpu.forward).not.toHaveBeenCalled();
    expect(gpu.weights).not.toHaveBeenCalled();
    expect(policy.steps).toBe(1);
});

it.each([false, true])('sends reward and every executed plan slot to GPU actor/critic learning (terminal=%s)', async (terminal) => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const decision = await policy.decideAsync(input, true, shifts);
    const credits = [0, 2, 6].map((step) => ({ step, weight: 1 / 3, shift: decision.plan[step].shift, shifts: decision.plan[step].shifts }));
    await policy.learnAsync(decision, 0.8, input, 0.3, credits, terminal);
    expect(gpu.learnTransition).toHaveBeenCalledWith(decision.gpu, input, {
        reward: 0.8, elapsedSeconds: 0.3, terminal,
        actions: credits.map((credit) => ({ ...credit, latent: decision.plan[credit.step].latent })),
    });
    expect(gpu.forward).toHaveBeenCalledTimes(1);
    expect(gpu.weights).not.toHaveBeenCalled();
    expect(policy.steps).toBe(1);
});

it('rejects invalid targets and transitions before submitting GPU learning', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    for (const step of [-1, 0.5, policy.planTimes.length]) {
        await expect(policy.learnPlannedActionsAsync(input, [{ step, action, shifts }])).rejects.toThrow(/plan step/);
    }
    for (const invalid of [{ ...action, throttle: NaN }, { ...action, brake: 2 }, { ...action, steering: -2 }, { ...action, shiftDown: true }]) {
        await expect(policy.learnDemonstrationAsync(input, invalid, shifts)).rejects.toThrow(/demonstration/);
    }
    const decision = await policy.decideAsync(input, false);
    for (const [reward, elapsed] of [[NaN, 0.2], [1, 0], [1, -1], [1, Infinity]]) {
        await expect(policy.learnAsync(decision, reward, input, elapsed, [])).rejects.toThrow(/transition/);
    }
    await policy.learnPlannedActionsAsync(input, []);
    expect(gpu.learnTargets).not.toHaveBeenCalled();
    expect(gpu.learnTransition).not.toHaveBeenCalled();
    expect(policy.steps).toBe(0);
});

it('waits once for GPU batch completion before allowing a checkpoint readback', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    let finish!: () => void;
    const submitted = jest.fn(() => {
        expect(gpu.learnTargets).toHaveBeenCalledTimes(1);
        expect(gpu.finishTraining).not.toHaveBeenCalled();
    });
    jest.mocked(gpu.finishTraining).mockImplementation(() => new Promise<void>((resolve) => { finish = resolve; }));
    const batch = policy.trainBatchAsync(async () => { await policy.learnDemonstrationAsync(input, action, shifts); }, submitted);
    const saved = policy.checkpointAsync();
    for (let i = 0; i < 10; i++) await Promise.resolve();
    expect(gpu.finishTraining).toHaveBeenCalledTimes(1);
    expect(submitted).toHaveBeenCalledTimes(1);
    expect(gpu.weights).not.toHaveBeenCalled();
    finish(); await batch;
    expect((await saved).model.steps).toBe(1);
    expect(gpu.weights).toHaveBeenCalledTimes(1);
});

it('does not report submission or enter the GPU wait when batch preparation fails', async () => {
    const { policy } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const submitted = jest.fn();
    await expect(policy.trainBatchAsync(async () => { throw new Error('Submission failed'); }, submitted))
        .rejects.toThrow('Submission failed');
    expect(submitted).not.toHaveBeenCalled();
    expect(gpu.finishTraining).not.toHaveBeenCalled();
    await expect(policy.checkpointAsync()).resolves.toHaveProperty('model');
});

it('propagates GPU failures and never switches to CPU computation', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    jest.mocked(gpu.forward).mockRejectedValue(new Error('GPU lost'));
    await expect(policy.decideAsync(input, false)).rejects.toThrow('GPU lost');
    jest.mocked(gpu.learnTargets).mockImplementation(() => { throw new Error('GPU lost'); });
    await expect(policy.learnDemonstrationAsync(input, action, shifts)).rejects.toThrow('GPU lost');
    expect(policy.steps).toBe(0);
    policy.dispose();
    await expect(policy.decideAsync(input, false)).rejects.toThrow(/released/);
    await expect(policy.initializeGpu()).rejects.toThrow(/released/);
});

it('disposes a GPU that finishes initialization after the policy was disposed', async () => {
    const { policy } = setup();
    let finish!: (gpu: DrivingGpuModel) => void;
    jest.mocked(DrivingGpuModel.create).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const initializing = policy.initializeGpu();
    policy.dispose();
    const gpu = { dispose: jest.fn() } as unknown as DrivingGpuModel;
    finish(gpu);
    await expect(initializing).rejects.toThrow(/released/);
    expect(gpu.dispose).toHaveBeenCalledTimes(1);
});

it('matches checkpoint context regardless of key order and isolates exported seed weights', () => {
    const { policy, observation } = setup(), saved = policy.checkpoint();
    const { game, car, map } = saved.model.context;
    saved.model.context = { map, car, game };
    const restored = DrivingPolicy.restore(saved);
    expect(restored.matches(observation)).toBe(true);
    expect(restored.checkpoint()).toEqual(saved);
    saved.model.lstmWeights[0][0] = 4;
    expect(restored.checkpoint().model.lstmWeights[0][0]).not.toBe(4);
});

it.each([
    { game: 'iracing', car: 'GT3', map: 'monza' },
    { game: 'acc', car: 'MX5', map: 'monza' },
    { game: 'acc', car: 'GT3', map: 'spa' },
    { game: 'iracing', car: 'MX5', map: 'spa' },
] as const)('accepts live inputs for $game / $car / $map after restoring an existing checkpoint', (context) => {
    const { policy } = setup(), fixture = trainingFixture();
    const saved = policy.checkpoint();
    saved.model.steps = 10;
    const restored = DrivingPolicy.restore(JSON.parse(JSON.stringify(saved)));
    const live = { ...fixture.live, game: context.game, currentTelemetry: { ...fixture.live.currentTelemetry,
        Static_car_model: context.car, Static_track: context.map } };
    const lap = { ...fixture.referenceLap, game: context.game, car: context.car, track: context.map };
    const result = readDrivingObservation(live, { ...fixture.vision, telemetry: live }, lap, 50, 1000, restored.configuration);
    expect(result.error).toBeUndefined();
    expect(result.observation!.context).toEqual(context);
    expect(result.observation!.features).not.toEqual(fixture.observation.features);
    expect(restored.inputMismatches(result.observation!)).toEqual([]);
    expect(restored.matches(result.observation!)).toBe(true);
    expect(restored.checkpoint()).toEqual(saved);
});

it('rejects incompatible input layouts and invalid checkpoint dimensions', () => {
    const { policy, observation } = setup();
    expect(policy.matches({ ...observation, features: observation.features.slice(1) })).toBe(false);
    expect(policy.matches({ ...observation, featureEncoding: 'different' })).toBe(false);
    const saved = policy.checkpoint();
    saved.model.outputWeights.pop();
    expect(() => DrivingPolicy.restore(saved)).toThrow(/checkpoint/);
});
