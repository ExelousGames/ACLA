import { DrivingPolicy } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingMemory } from './driving-memory';
import { circuitMapFixture, trainingFixture } from './training-fixtures';
import { cloneConfig } from './driving-model-config';
import { readDrivingObservation } from './driving-observation';

beforeEach(() => { mockDrivingGpu(); });
afterEach(() => { jest.restoreAllMocks(); });

function setup(draw: number | (() => number) = 0.7, learning: Partial<ReturnType<typeof cloneConfig>['learning']> = {}) {
    const { observation } = trainingFixture(), config = cloneConfig();
    Object.assign(config.learning, learning);
    config.network.hiddenUnits = 3;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    return { observation, input: new DrivingMemory(config).encode(observation),
        policy: new DrivingPolicy(observation.context, observation.features.length, typeof draw === 'number' ? () => draw : draw, config) };
}
const shifts = { up: true, down: false };
const action = { pedal: -0.1, steering: -0.3, shiftUp: true };
const explorationFrequencies = (frequency: number) => ({ pedalExplorationUpdateHz: frequency,
    steeringExplorationUpdateHz: frequency, gearExplorationUpdateHz: frequency });

it('requires GPU initialization for every prediction and learning entry point', async () => {
    const { policy, input } = setup();
    await expect(policy.decideAsync(input, false)).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnAsync({} as any, 1, input, 0.2, [])).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnDemonstrationAsync(input, action, shifts)).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnDemonstratedPlanAsync(input, [])).rejects.toThrow(/Initialize.*GPU/);
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

it('initializes camera inspection on demand without changing model weights or training steps', async () => {
    const { policy, observation } = setup(), seed = policy.checkpoint();
    const inspection = await policy.inspectCameraAsync(observation.camera);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.inspectCamera).toHaveBeenCalledWith(observation.camera);
    expect(inspection.layers).toHaveLength(policy.configuration.network.cameraLayers.length);
    expect(await policy.checkpointAsync()).toEqual(seed);
    policy.dispose();
    await expect(policy.inspectCameraAsync(observation.camera)).rejects.toThrow(/released/);
});

it('creates and restores 51 action slots spanning five seconds at 100 ms intervals', async () => {
    const { policy, input } = setup();
    const checkpoint = policy.checkpoint();
    expect(checkpoint.configuration.actions.planTimesSeconds).toEqual(Array.from({ length: 51 }, (_, step) => step / 10));
    expect(checkpoint.model.outputWeights).toHaveLength(256); // 51 × five action outputs + value.
    const restored = DrivingPolicy.restore(JSON.parse(JSON.stringify(checkpoint)));
    await restored.initializeGpu();
    const decision = await restored.decideAsync(input, false);
    expect(decision.plan.map(({ timeSeconds }) => timeSeconds)).toEqual(Array.from({ length: 51 }, (_, step) => step / 10));
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
        expect(Math.abs(result.pedal)).toBeLessThanOrEqual(1);
        expect(Math.abs(result.steering)).toBeLessThanOrEqual(1);
        expect(result.shiftUp && result.shiftDown).toBe(false);
    });
    expect(decision).not.toHaveProperty('sequence');
    policy.releaseDecision(decision);
    expect(decision.gpu.released).toBe(true);
});

it.each([-100, -0.7, 0, 0.4, 100])(
    'decodes the signed pedal output for every plan slot (latent=%s)', async (pedalLatent) => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const prediction = await gpu.forward(input);
    prediction.outputs = [...policy.planTimes.flatMap(() => [pedalLatent, Math.atanh(-0.3), 0, 4, -1]), 0.75];
    jest.mocked(gpu.forward).mockResolvedValueOnce(prediction);
    const decision = await policy.decideAsync(input, false);
    expect(decision.value).toBe(0.75);
    for (const { action: result, means, latent } of decision.plan) {
        expect(result.pedal).toBe(Math.tanh(pedalLatent));
        expect(result.steering).toBeCloseTo(-0.3);
        expect(result.shiftUp).toBe(true);
        expect(result.shiftDown).toBe(false);
        expect(means).toEqual([pedalLatent, Math.atanh(-0.3)]);
        expect(latent).toEqual(means);
    }
    policy.releaseDecision(decision);
});

it('explores each control independently while keeping fresh-model movements small and shifts rare', async () => {
    let seed = 42;
    let now = 1000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const { policy, input } = setup(random);
    await policy.initializeGpu();
    // Keep the exploration test independent of the number of initialized critic weights.
    seed = 42;
    const pedal: number[] = [], steering: number[] = [], choices = [0, 0, 0];
    for (let i = 0; i < 300; i++) {
        const decision = await policy.decideAsync(input, true);
        for (const { action, shift } of decision.plan) {
            pedal.push(action.pedal);
            steering.push(action.steering);
            choices[shift]++;
        }
        policy.releaseDecision(decision);
        now += 5100;
    }
    for (const axis of [pedal, steering]) {
        const rms = Math.sqrt(axis.reduce((sum, value) => sum + value * value, 0) / axis.length);
        const changes = axis.slice(1).map((value, i) => value - axis[i]);
        const movement = Math.sqrt(changes.reduce((sum, value) => sum + value * value, 0) / changes.length);
        expect(rms).toBeGreaterThan(0.04);
        expect(rms).toBeLessThan(0.06);
        expect(movement).toBeLessThan(0.09);
        expect(axis.some((value) => value > 0)).toBe(true);
        expect(axis.some((value) => value < 0)).toBe(true);
    }
    const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
    const pedalMean = average(pedal), steeringMean = average(steering);
    const covariance = average(pedal.map((value, i) => (value - pedalMean) * (steering[i] - steeringMean)));
    const variance = (values: number[], mean: number) => average(values.map((value) => (value - mean) ** 2));
    expect(Math.abs(covariance / Math.sqrt(variance(pedal, pedalMean) * variance(steering, steeringMean)))).toBeLessThan(0.05);
    expect((choices[1] + choices[2]) / pedal.length).toBeLessThan(0.05);
    expect(choices[1]).toBeGreaterThan(0);
    expect(choices[2]).toBeGreaterThan(0);
    policy.dispose();
});

it.each([0.1, 2.5, 10])('shares smooth exploration at %s Hz across overlapping plans', async (frequency) => {
    const clock = jest.spyOn(Date, 'now').mockReturnValue(1000);
    let seed = 42;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const { policy, input } = setup(random, explorationFrequencies(frequency));
    await policy.initializeGpu();
    const first = await policy.decideAsync(input, true);
    clock.mockReturnValue(1100);
    const replacement = await policy.decideAsync(input, true);
    replacement.plan.slice(0, -1).forEach((step, i) => {
        step.latent.forEach((value, axis) => expect(value).toBeCloseTo(first.plan[i + 1].latent[axis], 10));
        expect(step.shift).toBe(first.plan[i + 1].shift);
        expect(step.variances).toEqual(first.plan[i + 1].variances);
    });
    [first, replacement].forEach((decision) => policy.releaseDecision(decision));
    policy.dispose();
});

it.each(['pedalExplorationUpdateHz', 'steeringExplorationUpdateHz', 'gearExplorationUpdateHz'] as const)
('respects the selected control refresh frequency: %s', async (key) => {
    jest.spyOn(Date, 'now').mockReturnValue(1000);
    const baseline = setup(0.98, explorationFrequencies(0.1));
    const adjusted = setup(0.98, { ...explorationFrequencies(0.1), [key]: 2.5 });
    await baseline.policy.initializeGpu();
    await adjusted.policy.initializeGpu();
    const first = await baseline.policy.decideAsync(baseline.input, true);
    const changed = await adjusted.policy.decideAsync(adjusted.input, true);
    const changedControl = ['pedalExplorationUpdateHz', 'steeringExplorationUpdateHz', 'gearExplorationUpdateHz'].indexOf(key);
    changed.plan[2].latent.forEach((value, axis) => {
        if (axis === changedControl) expect(value).not.toBe(first.plan[2].latent[axis]);
        else expect(value).toBe(first.plan[2].latent[axis]);
    });
    baseline.policy.releaseDecision(first); adjusted.policy.releaseDecision(changed);
    baseline.policy.dispose(); adjusted.policy.dispose();
});

it('updates predictions and shift availability while retaining the current exploration sample', async () => {
    const clock = jest.spyOn(Date, 'now').mockReturnValue(1000);
    const { policy, input } = setup(0.98, explorationFrequencies(1));
    const seed = policy.checkpoint();
    await policy.initializeGpu();
    const first = await policy.decideAsync(input, true);
    expect(first.plan[0].shift).toBe(1);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const outputs = seed.model.outputWeights.map((row) => row[policy.configuration.network.hiddenUnits]);
    outputs.splice(0, 2, 0.4, -0.3);
    jest.mocked(gpu.forward).mockResolvedValueOnce({ pass: { released: false } as any, outputs, logStds: seed.model.sac.logStds });
    clock.mockReturnValue(1100);
    const next = await policy.decideAsync(input, true, { up: false, down: false });
    expect(next.plan[0].means).toEqual([0.4, -0.3]);
    next.plan[0].latent.forEach((value, axis) => expect(value - next.plan[0].means[axis])
        .toBeCloseTo(first.plan[1].latent[axis] - first.plan[1].means[axis], 10));
    expect(next.plan[0].action.pedal).not.toBe(first.plan[0].action.pedal);
    expect(next.plan[0].shift).toBe(0);
    expect(next.plan[0].shiftProbabilities).toEqual([1, 0, 0]);
    policy.releaseDecision(first); policy.releaseDecision(next); policy.dispose();
});

it('uses no random exploration during evaluation and starts fresh when training resumes', async () => {
    const clock = jest.spyOn(Date, 'now').mockReturnValue(1000);
    const random = jest.fn(() => 0.98);
    const { policy, input } = setup(random, explorationFrequencies(1));
    await policy.initializeGpu();
    const first = await policy.decideAsync(input, true);
    random.mockClear().mockReturnValue(0.5);
    clock.mockReturnValue(1100);
    const evaluation = await policy.decideAsync(input, false);
    evaluation.plan.forEach((step) => expect(step.latent).toEqual(step.means));
    expect(random).not.toHaveBeenCalled();
    clock.mockReturnValue(1200);
    const resumed = await policy.decideAsync(input, true);
    expect(resumed.plan[0].latent).not.toEqual(first.plan[0].latent);
    expect(resumed.plan[0].shift).toBe(0);
    [first, evaluation, resumed].forEach((decision) => policy.releaseDecision(decision));
    policy.dispose();
});

it.each(['pedalExplorationSigma', 'steeringExplorationSigma'] as const)
('changes only the selected control exploration across the action plan: %s', async (key) => {
    const baseline = setup(0.99), adjusted = setup(0.99, { [key]: 0.2 });
    await baseline.policy.initializeGpu();
    await adjusted.policy.initializeGpu();
    const original = await baseline.policy.decideAsync(baseline.input, true);
    const changed = await adjusted.policy.decideAsync(adjusted.input, true);
    const changedAxis = ['pedalExplorationSigma', 'steeringExplorationSigma'].indexOf(key);
    changed.plan.forEach((step, index) => {
        const before = original.plan[index];
        expect(step.means).toEqual(before.means);
        step.latent.forEach((value, axis) => {
            const scale = axis === changedAxis ? 4 : 1;
            expect(value - step.means[axis]).toBeCloseTo((before.latent[axis] - before.means[axis]) * scale, 10);
        });
        expect(step.shift).toBe(before.shift);
    });
    const evaluation = await adjusted.policy.decideAsync(adjusted.input, false);
    evaluation.plan.forEach((step) => expect(step.latent).toEqual(step.means));
    baseline.policy.releaseDecision(original);
    adjusted.policy.releaseDecision(changed);
    adjusted.policy.releaseDecision(evaluation);
    baseline.policy.dispose(); adjusted.policy.dispose();
});

it('does not restore old exploration when inference finishes after an episode reset', async () => {
    const random = jest.fn(() => 0.7);
    const { policy, input } = setup(random);
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const prediction = await gpu.forward(input);
    let finish!: (value: typeof prediction) => void;
    jest.mocked(gpu.forward).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    random.mockClear();
    const pending = policy.decideAsync(input, true);
    policy.resetExploration();
    finish(prediction);
    const stale = await pending;
    expect(random).not.toHaveBeenCalled();
    stale.plan.forEach((step) => expect(step.latent).toEqual(step.means));
    const fresh = await policy.decideAsync(input, true);
    expect(random).toHaveBeenCalled();
    policy.releaseDecision(stale); policy.releaseDecision(fresh); policy.dispose();
});

it('learns the executed control after weights change without using old latents or variances', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const decision = await policy.decideAsync(input, true);
    const original = [...decision.plan[0].variances!];
    policy.releaseDecision(decision);
    const next = { ...input, features: input.features.map(() => 0) };
    const changed = await policy.decideAsync(next, true);
    expect(changed.plan[0].variances).not.toEqual(original);
    const executed = { pedal: -0.2, steering: -0.1, shiftUp: false, shiftDown: false };
    await policy.learnAsync(decision, 1, next, 0.2, [{ step: 0, weight: 1, shift: 0, shifts, action: executed }]);
    expect(gpu.replay).toHaveBeenCalledWith(input);
    expect(gpu.learnTransition).toHaveBeenCalledWith(expect.anything(), next,
        expect.objectContaining({ actions: [expect.objectContaining({ action: executed })] }));
    expect(jest.mocked(gpu.learnTransition).mock.calls[0][2].actions[0]).not.toHaveProperty('latent');
    expect(jest.mocked(gpu.learnTransition).mock.calls[0][2].actions[0]).not.toHaveProperty('variances');
    policy.releaseDecision(changed); policy.dispose();
});

it.each([[0.5, 0], [0.98, 1], [0.999, 2]])('still explores each gear choice with a small shift probability (draw=%s)', async (draw, choice) => {
    const { policy, input } = setup(draw);
    await policy.initializeGpu();
    const decision = await policy.decideAsync(input, true);
    for (const step of decision.plan) {
        expect(step.shiftProbabilities[0]).toBeGreaterThan(0.96);
        expect(step.shift).toBe(choice);
    }
    policy.releaseDecision(decision);
    policy.dispose();
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

it('trains all available outputs of a demonstrated future together and counts the origin once', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const targets = policy.planTimes.map((_, step) => ({ step, action: {
        pedal: step / 50, steering: -step / 50, shiftUp: step === 3, shiftDown: step === 15,
    }, shifts: { up: true, down: true } }));
    await policy.learnDemonstratedPlanAsync(input, targets);
    expect(gpu.learnTargets).toHaveBeenCalledTimes(1);
    expect(gpu.learnTargets).toHaveBeenLastCalledWith(input, targets);
    expect(policy.steps).toBe(1);
    const available = targets.filter(({ step }) => [0, 2, 9].includes(step));
    await policy.learnDemonstratedPlanAsync(input, available);
    expect(gpu.learnTargets).toHaveBeenCalledTimes(2);
    expect(gpu.learnTargets).toHaveBeenLastCalledWith(input, available);
    expect(policy.steps).toBe(2);
    await policy.learnDemonstratedPlanAsync(input, []);
    expect(gpu.learnTargets).toHaveBeenCalledTimes(2);
    expect(policy.steps).toBe(2);
    expect(gpu.forward).not.toHaveBeenCalled();
    expect(gpu.weights).not.toHaveBeenCalled();
});

it.each([false, true])('sends reward and every executed plan slot to GPU actor/critic learning (terminal=%s)', async (terminal) => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const decision = await policy.decideAsync(input, true, shifts);
    const credits = [0, 2, 6].map((step) => ({ step, weight: 1 / 3, shift: decision.plan[step].shift, shifts: decision.plan[step].shifts }));
    await policy.learnAsync(decision, 0.8, input, 0.3, credits, terminal);
    expect(gpu.replay).toHaveBeenCalledWith(decision.input);
    expect(gpu.learnTransition).toHaveBeenCalledWith(jest.mocked(gpu.replay).mock.results[0].value, input, expect.objectContaining({
        reward: 0.8, elapsedSeconds: 0.3, terminal, nextShifts: { up: true, down: true },
        actions: credits.map((credit) => ({ ...credit, action: decision.plan[credit.step].action })),
    }));
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
    for (const invalid of [{ ...action, pedal: NaN }, { ...action, pedal: -2 }, { ...action, steering: -2 }, { ...action, shiftDown: true }]) {
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
    const result = readDrivingObservation(live, { ...fixture.vision, telemetry: live }, lap, 50, 1000, restored.configuration, circuitMapFixture(lap));
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

it('uses learned SAC standard deviations while preserving deterministic controls', async () => {
    const { policy, input } = setup(0.7);
    const checkpoint = policy.checkpoint();
    checkpoint.model.sac.logStds.fill(Math.log(0.2));
    const restored = DrivingPolicy.restore(checkpoint);
    await restored.initializeGpu();
    const sampled = await restored.decideAsync(input, true);
    const fixed = await restored.decideAsync(input, false);
    expect(sampled.plan[0].latent).not.toEqual(sampled.plan[0].means);
    expect(sampled.plan[0].variances![0]).toBeGreaterThanOrEqual(0.04);
    expect(fixed.plan[0].latent).toEqual(fixed.plan[0].means);
    expect(fixed.plan[0].variances![0]).toBeCloseTo(0.04);
    restored.releaseDecision(sampled); restored.releaseDecision(fixed); restored.dispose();
});

it('keeps exploration controls effective on a trained SAC checkpoint without changing weights', async () => {
    jest.spyOn(Math, 'random').mockReturnValue(0.7);
    const { policy, input } = setup();
    const checkpoint = policy.checkpoint();
    checkpoint.model.sac.logStds.fill(Math.log(0.1));
    const baseline = DrivingPolicy.restore(checkpoint);
    checkpoint.configuration.learning.pedalExplorationSigma *= 2;
    const changed = DrivingPolicy.restore(checkpoint);
    expect(changed.checkpoint().model).toEqual(baseline.checkpoint().model);
    await baseline.initializeGpu(); await changed.initializeGpu();
    const first = await baseline.decideAsync(input, true), second = await changed.decideAsync(input, true);
    expect(second.plan[0].latent[0] - second.plan[0].means[0]).toBeCloseTo(2 * (first.plan[0].latent[0] - first.plan[0].means[0]));
    expect(second.plan[0].latent.slice(1)).toEqual(first.plan[0].latent.slice(1));
    baseline.releaseDecision(first); changed.releaseDecision(second);
    baseline.dispose(); changed.dispose();
});

it('bounds replay, retains adjacent states and executed actions, and discards cancelled additions', async () => {
    const { policy, input } = setup(0, { replayCapacity: 2, replayBatchSize: 2 });
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const decision = await policy.decideAsync(input, false);
    policy.releaseDecision(decision);
    const next = { ...input, features: input.features.map(() => 0.2) };
    const endpoint = { ...input, features: input.features.map(() => 0.9) };
    const executed = { pedal: -0.1, steering: 0.2, shiftUp: false, shiftDown: false };
    const interval = (reward: number) => ({ startedAt: 1000, endedAt: 1200, reward, input: next, next: input,
        samples: [{ decision, rewardShare: 1, credits: [{ step: 0, weight: 1, shift: 0 as const, shifts, action: executed }] }] });
    await policy.learnSequenceAsync([interval(1)], endpoint);
    expect(gpu.replay).toHaveBeenLastCalledWith(next);
    expect(gpu.learnTransition).toHaveBeenLastCalledWith(expect.anything(), input, expect.objectContaining({
        reward: 1, elapsedSeconds: 0.2, actions: [expect.objectContaining({ action: executed })],
    }));
    executed.pedal = 0.9;
    await policy.learnSequenceAsync([interval(2)], endpoint);
    const calls = jest.mocked(gpu.learnTransition).mock.calls;
    expect(calls[calls.length - 2][2].actions[0].action.pedal).toBe(-0.1);
    await policy.learnSequenceAsync([interval(3)], endpoint);
    expect(calls.slice(-2).map((call: Parameters<DrivingGpuModel['learnTransition']>) => call[2].reward)).toEqual([2, 3]);
    const before = await policy.checkpointAsync();
    let checks = 0;
    await policy.learnSequenceAsync([interval(99)], endpoint, () => ++checks <= 2);
    expect(await policy.checkpointAsync()).toEqual(before);
    await policy.learnSequenceAsync([interval(4)], endpoint);
    expect(calls.slice(-2).map((call: Parameters<DrivingGpuModel['learnTransition']>) => call[2].reward)).toEqual([3, 4]);
    policy.dispose();
});
