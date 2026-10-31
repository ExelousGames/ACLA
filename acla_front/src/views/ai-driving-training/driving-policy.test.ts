import { DrivingPolicy } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingMemory } from './driving-memory';
import { trainingFixture } from './training-fixtures';
import { cloneConfig } from './driving-model-config';

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
    await expect(policy.learnAsync({} as any, 1, input, 0.2)).rejects.toThrow(/Initialize.*GPU/);
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
    decision.plan.forEach(({ action: result }) => {
        expect(result.throttle).toBeGreaterThanOrEqual(0);
        expect(result.throttle).toBeLessThanOrEqual(1);
        expect(Math.abs(result.steering)).toBeLessThanOrEqual(1);
        expect(result.shiftUp && result.shiftDown).toBe(false);
    });
    expect(decision).not.toHaveProperty('sequence');
    policy.releaseDecision(decision);
    expect(decision.gpu.released).toBe(true);
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

it.each([false, true])('sends reward, elapsed time and sampled action to GPU actor/critic learning (terminal=%s)', async (terminal) => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const decision = await policy.decideAsync(input, true, shifts);
    await policy.learnAsync(decision, 0.8, input, 0.3, terminal);
    expect(gpu.learnTransition).toHaveBeenCalledWith(decision.gpu, input, {
        reward: 0.8, elapsedSeconds: 0.3, terminal,
        latent: decision.plan[0].latent, shift: decision.plan[0].shift, shifts,
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
        await expect(policy.learnAsync(decision, reward, input, elapsed)).rejects.toThrow(/transition/);
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
    jest.mocked(gpu.finishTraining).mockImplementation(() => new Promise<void>((resolve) => { finish = resolve; }));
    const batch = policy.trainBatchAsync(async () => { await policy.learnDemonstrationAsync(input, action, shifts); });
    const saved = policy.checkpointAsync();
    for (let i = 0; i < 10; i++) await Promise.resolve();
    expect(gpu.finishTraining).toHaveBeenCalledTimes(1);
    expect(gpu.weights).not.toHaveBeenCalled();
    finish(); await batch;
    expect((await saved).model.steps).toBe(1);
    expect(gpu.weights).toHaveBeenCalledTimes(1);
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

it.each(['game', 'car', 'map'] as const)('rejects a different %s and invalid checkpoint dimensions', (key) => {
    const { policy, observation } = setup();
    expect(policy.matches({ ...observation, context: { ...observation.context, [key]: 'different' } })).toBe(false);
    const saved = policy.checkpoint();
    saved.model.outputWeights.pop();
    expect(() => DrivingPolicy.restore(saved)).toThrow(/checkpoint/);
});
