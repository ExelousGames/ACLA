import { DrivingPolicy, readTrainingDocument } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingMemory } from './driving-memory';
import { circuitMapFixture, trainingFixture } from './training-fixtures';
import { cloneConfig, DRIVING_GEAR_COUNT, DRIVING_MAX_GEAR, DRIVING_MIN_GEAR, drivingGearIndex } from './driving-model-config';
import { readDrivingObservation } from './driving-observation';
import { directedActionCandidates } from './driving-directed-exploration';

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
const gears = { max: 2 };
const reverseOnly = [1, ...Array(DRIVING_GEAR_COUNT - 1).fill(0)];
const action = { pedal: -0.1, steering: -0.3, gear: 2 };
const explorationFrequencies = (frequency: number) => ({ pedalExplorationUpdateHz: frequency,
    steeringExplorationUpdateHz: frequency, gearExplorationUpdateHz: frequency });

it('requires GPU initialization for every prediction and learning entry point', async () => {
    const { policy, input } = setup();
    await expect(policy.decideAsync(input, false)).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnAsync({} as any, 1, input, 0.2, [])).rejects.toThrow(/Initialize.*GPU/);
    await expect(policy.learnDemonstrationAsync(input, action)).rejects.toThrow(/Initialize.*GPU/);
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
    expect(checkpoint.model.outputWeights).toHaveLength(51 * (2 + DRIVING_GEAR_COUNT) + 1);
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
    const decision = await policy.decideAsync(input, true, { max: DRIVING_MIN_GEAR });
    expect(decision.input).toBe(input);
    expect(decision.plan.map((step) => step.timeSeconds)).toEqual(policy.planTimes);
    expect(decision.plan[0].gearProbabilities).toEqual(reverseOnly);
    expect(decision.plan[0].gears).toEqual({ max: DRIVING_MIN_GEAR });
    expect(decision.plan[0].action.gear).toBe(DRIVING_MIN_GEAR);
    decision.plan.forEach(({ means, latent }) => expect(latent).not.toEqual(means));
    decision.plan.forEach(({ action: result }) => {
        expect(Math.abs(result.pedal)).toBeLessThanOrEqual(1);
        expect(Math.abs(result.steering)).toBeLessThanOrEqual(1);
        expect(result.gear).toBe(DRIVING_MIN_GEAR);
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
    const gearLogits = Array.from({ length: DRIVING_GEAR_COUNT }, (_, index) => index === drivingGearIndex(DRIVING_MAX_GEAR) ? 4 : 0);
    prediction.outputs = [...policy.planTimes.flatMap(() => [pedalLatent, Math.atanh(-0.3), ...gearLogits]), 0.75];
    jest.mocked(gpu.forward).mockResolvedValueOnce(prediction);
    const decision = await policy.decideAsync(input, false);
    expect(decision.value).toBe(0.75);
    for (const { action: result, means, latent } of decision.plan) {
        expect(result.pedal).toBe(Math.tanh(pedalLatent));
        expect(result.steering).toBeCloseTo(-0.3);
        expect(result.gear).toBe(DRIVING_MAX_GEAR);
        expect(means).toEqual([pedalLatent, Math.atanh(-0.3)]);
        expect(latent).toEqual(means);
    }
    policy.releaseDecision(decision);
});

it.each([-1, 0])('predicts and learns target gear %s under a first-gear exploration cap', async (gear) => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const prediction = await gpu.forward(input);
    const logits = Array(DRIVING_GEAR_COUNT).fill(0);
    logits[drivingGearIndex(gear)] = 4;
    logits[drivingGearIndex(DRIVING_MAX_GEAR)] = 8;
    prediction.outputs = [...policy.planTimes.flatMap(() => [0, 0, ...logits]), 0];
    jest.mocked(gpu.forward).mockResolvedValueOnce(prediction);
    const decision = await policy.decideAsync(input, false, { max: 1 });
    expect(decision.plan.every((step) => step.action.gear === gear)).toBe(true);
    expect(decision.plan[0].gearProbabilities[drivingGearIndex(DRIVING_MAX_GEAR)]).toBe(0);
    await policy.learnDemonstrationAsync(input, decision.plan[0].action);
    expect(gpu.learnTargets).toHaveBeenCalledWith(input, [{ step: 0, action: { pedal: 0, steering: 0, gear } }], true, 'standard');
    policy.releaseDecision(decision);
});

it('explores each control independently while keeping fresh-model movements small and favoring first gear', async () => {
    let seed = 42;
    let now = 1000;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const { policy, input } = setup(random);
    await policy.initializeGpu();
    // Keep the exploration test independent of the number of initialized critic weights.
    seed = 42;
    const pedal: number[] = [], steering: number[] = [], choices = Array(DRIVING_GEAR_COUNT).fill(0);
    for (let i = 0; i < 300; i++) {
        const decision = await policy.decideAsync(input, true);
        for (const { action } of decision.plan) {
            pedal.push(action.pedal);
            steering.push(action.steering);
            choices[drivingGearIndex(action.gear)]++;
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
    expect(choices[drivingGearIndex(1)] / pedal.length).toBeGreaterThan(0.8);
    expect(choices.every((count) => count > 0)).toBe(true);
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
        expect(step.action.gear).toBe(first.plan[i + 1].action.gear);
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

it('updates predictions and gear availability while retaining the current exploration sample', async () => {
    const clock = jest.spyOn(Date, 'now').mockReturnValue(1000);
    const { policy, input } = setup(0.98, explorationFrequencies(1));
    const seed = policy.checkpoint();
    await policy.initializeGpu();
    const first = await policy.decideAsync(input, true);
    expect(first.plan[0].action.gear).toBe(9);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const outputs = seed.model.outputWeights.map((row) => row[policy.configuration.network.hiddenUnits]);
    outputs.splice(0, 2, 0.4, -0.3);
    jest.mocked(gpu.forward).mockResolvedValueOnce({ pass: { released: false } as any, outputs, logStds: seed.model.sac.logStds });
    clock.mockReturnValue(1100);
    const next = await policy.decideAsync(input, true, { max: 1 });
    expect(next.plan[0].means).toEqual([0.4, -0.3]);
    next.plan[0].latent.forEach((value, axis) => expect(value - next.plan[0].means[axis])
        .toBeCloseTo(first.plan[1].latent[axis] - first.plan[1].means[axis], 10));
    expect(next.plan[0].action.pedal).not.toBe(first.plan[0].action.pedal);
    expect(next.plan[0].action.gear).toBe(1);
    expect(next.plan[0].gearProbabilities.slice(0, drivingGearIndex(1) + 1).every((value) => value > 0)).toBe(true);
    expect(next.plan[0].gearProbabilities.slice(drivingGearIndex(2)).every((value) => value === 0)).toBe(true);
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
    expect(resumed.plan[0].action.gear).toBe(1);
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
        expect(step.action.gear).toBe(before.action.gear);
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

it('uses actor exploration at every plan step with restored critic weights', async () => {
    jest.spyOn(Date, 'now').mockReturnValue(1000);
    jest.spyOn(Math, 'random').mockReturnValue(0.98);
    const { policy, input } = setup(0.98);
    const checkpoint = policy.checkpoint();
    checkpoint.model.sac.criticWeights.forEach((row) => row.fill(1));
    const restored = DrivingPolicy.restore(checkpoint);
    await policy.initializeGpu(); await restored.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[1].value;
    const available = { max: 2 };
    const fresh = await policy.decideAsync(input, true, available);
    const loaded = await restored.decideAsync(input, true, available);
    expect(gpu.forward).toHaveBeenLastCalledWith(input);
    expect(loaded.plan).toEqual(fresh.plan);
    loaded.plan.forEach(({ latent, means }) => expect(latent).not.toEqual(means));
    expect(loaded.plan[0].gearProbabilities[drivingGearIndex(3)]).toBe(0);
    policy.releaseDecision(fresh); restored.releaseDecision(loaded);
    policy.dispose(); restored.dispose();
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
    const executed = { pedal: -0.2, steering: -0.1, gear: 2 };
    await policy.learnAsync(decision, 1, next, 0.2, [{ step: 0, weight: 1, gear: executed.gear, gears, action: executed }]);
    expect(gpu.replay).toHaveBeenCalledWith(input);
    expect(gpu.learnTransition).toHaveBeenCalledWith(expect.anything(), next,
        expect.objectContaining({ actions: [expect.objectContaining({ action: executed })] }));
    expect(jest.mocked(gpu.learnTransition).mock.calls[0][2].actions[0]).not.toHaveProperty('latent');
    expect(jest.mocked(gpu.learnTransition).mock.calls[0][2].actions[0]).not.toHaveProperty('variances');
    policy.releaseDecision(changed); policy.dispose();
});

it.each([[0.001, -1], [0.02, 0], [0.5, 1], [0.9, 4], [0.999, 10]])('samples target gear numbers from the categorical distribution (draw=%s)', async (draw, choice) => {
    const { policy, input } = setup(draw);
    await policy.initializeGpu();
    const decision = await policy.decideAsync(input, true);
    for (const step of decision.plan) {
        expect(step.gearProbabilities[drivingGearIndex(1)]).toBeGreaterThan(0.8);
        expect(step.action.gear).toBe(choice);
    }
    policy.releaseDecision(decision);
    policy.dispose();
});

it('sends raw demonstration and delayed targets straight to GPU learning without inference readback', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    await policy.learnDemonstrationAsync(input, action);
    const targets = [{ step: 6, action }];
    await policy.learnPlannedActionsAsync(input, targets);
    expect(gpu.learnTargets).toHaveBeenNthCalledWith(1, input, [{ step: 0, action }], true, 'standard');
    expect(gpu.learnTargets).toHaveBeenNthCalledWith(2, input, targets, false, 'standard');
    expect(gpu.forward).not.toHaveBeenCalled();
    expect(gpu.weights).not.toHaveBeenCalled();
    expect(policy.steps).toBe(1);
});

it('sends both recorded target types to the recorded loss, including neutral steering', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const neutralSteering = { ...action, steering: 0 };
    const future = [{ step: 2, action }];
    await policy.learnDemonstrationAsync(input, neutralSteering, 'recorded');
    await policy.learnPlannedActionsAsync(input, future, 'recorded');
    expect(gpu.learnTargets).toHaveBeenNthCalledWith(1, input,
        [{ step: 0, action: neutralSteering }], true, 'recorded');
    expect(gpu.learnTargets).toHaveBeenNthCalledWith(2, input, future, false, 'recorded');
    expect(policy.steps).toBe(1);
});

it('counts a grouped recorded plan once without replacing a past recurrent snapshot', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const targets = [0, 1, 8].map((step) => ({ step, action }));
    await policy.learnRecordedPlanAsync(input, targets);
    expect(gpu.learnTargets).toHaveBeenCalledTimes(1);
    expect(gpu.learnTargets).toHaveBeenCalledWith(input, targets, false, 'recorded');
    expect(policy.steps).toBe(1);
});

it('commits recorded optimizer batches together and rolls back interrupted sample counts', async () => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const sample = { input, targets: [{ step: 0, action }] };
    const batch = [sample, sample];

    expect(await policy.learnRecordedBatchAsync(batch, () => policy.steps < 1)).toBe(false);
    expect(gpu.endSequence).toHaveBeenLastCalledWith(false);
    const cancelled = await policy.checkpointAsync();
    expect(cancelled.model.steps).toBe(0);

    jest.mocked(gpu.endSequence).mockImplementationOnce(() => { throw new Error('GPU submission failed'); });
    await expect(policy.learnRecordedBatchAsync(batch, () => true)).rejects.toThrow('GPU submission failed');
    expect(policy.steps).toBe(0);

    expect(await policy.learnRecordedBatchAsync(batch, () => true)).toBe(true);
    expect(gpu.beginRecordedSequence).toHaveBeenCalledTimes(3);
    expect(gpu.endSequence).toHaveBeenLastCalledWith(true);
    const committed = await policy.checkpointAsync();
    expect(committed.model.steps).toBe(2);
    expect(committed.model.gruWeights[0][0]).not.toBe(cancelled.model.gruWeights[0][0]);
});

it('ignores obsolete fitted heads when restoring the native weights and step count', async () => {
    const { policy } = setup();
    policy.steps = 37;
    const checkpoint = policy.checkpoint();
    const oldCheckpoint = { ...checkpoint, model: { ...checkpoint.model,
        recordedControlHeads: [{ featureEncoding: 'vehicle-motion-v1' }],
        recordedShiftHeads: [{ featureEncoding: 'vehicle-shift-v1' }],
    } };
    expect(readTrainingDocument(oldCheckpoint)).toEqual(checkpoint);
    const restored = DrivingPolicy.restore(oldCheckpoint);
    await restored.initializeGpu();
    expect(await restored.checkpointAsync()).toEqual(checkpoint);
    restored.dispose();
});

it.each([false, true])('sends reward and every executed plan slot to GPU actor/critic learning (terminal=%s)', async (terminal) => {
    const { policy, input } = setup();
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const decision = await policy.decideAsync(input, true, gears);
    const credits = [0, 2, 6].map((step) => ({ step, weight: 1 / 3, gear: decision.plan[step].action.gear, gears: decision.plan[step].gears }));
    await policy.learnAsync(decision, 0.8, input, 0.3, credits, terminal);
    expect(gpu.replay).toHaveBeenCalledWith(decision.input);
    expect(gpu.learnTransition).toHaveBeenCalledWith(jest.mocked(gpu.replay).mock.results[0].value, input, expect.objectContaining({
        reward: 0.8, elapsedSeconds: 0.3, terminal, nextGears: { max: DRIVING_MAX_GEAR },
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
        await expect(policy.learnPlannedActionsAsync(input, [{ step, action }])).rejects.toThrow(/plan step/);
    }
    for (const invalid of [{ ...action, pedal: NaN }, { ...action, pedal: -2 }, { ...action, steering: -2 }, { ...action, gear: DRIVING_MIN_GEAR - 1 }, { ...action, gear: 2.5 }, { ...action, gear: DRIVING_MAX_GEAR + 1 }]) {
        await expect(policy.learnDemonstrationAsync(input, invalid)).rejects.toThrow(/demonstration/);
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
    const batch = policy.trainBatchAsync(async () => { await policy.learnDemonstrationAsync(input, action); }, submitted);
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
    await expect(policy.learnDemonstrationAsync(input, action)).rejects.toThrow('GPU lost');
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
    saved.model.gruWeights[0][0] = 4;
    expect(restored.checkpoint().model.gruWeights[0][0]).not.toBe(4);
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

it('scores the configured steering and pedal grid in every allowed gear after live critic training', async () => {
    const { policy, input } = setup(0.5, { replayBatchSize: 1, directedSteeringMin: -0.8, directedSteeringMax: 0.4,
        directedSteeringSamples: 3, directedPedalMin: -0.75, directedPedalMax: 0.8, directedPedalSamples: 4 });
    await policy.initializeGpu();
    const gpu: DrivingGpuModel = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const allowed = { max: 7 };
    const options = { throttleLimited: true };
    const cold = await policy.decideAsync(input, true, allowed, options);
    expect(cold.directedAction).toBeUndefined();
    expect(gpu.scoreActions).not.toHaveBeenCalled();
    const action = cold.plan[0].action;
    await policy.learnSequenceAsync([{ startedAt: 1000, endedAt: 1200, input, next: input, reward: 1,
        samples: [{ decision: cold, rewardShare: 1, credits: [{ step: 0, weight: 1, gear: action.gear, gears: allowed, action }] }] }], input);
    policy.releaseDecision(cold);

    const best = { gear: 7, steering: 0.4, pedal: -0.75 };
    jest.mocked(gpu.scoreActions).mockImplementationOnce(async (_pass, [, ...alternatives]) => {
        expect(alternatives).toHaveLength(9 * 3 * 4);
        expect(Array.from(new Set(alternatives.map(({ gear }) => gear)))).toEqual([-1, 0, 1, 2, 3, 4, 5, 6, 7]);
        expect(alternatives.every(({ pedal, steering }) => pedal >= -0.75 && pedal <= 0 && steering >= -0.8 && steering <= 0.4)).toBe(true);
        expect(alternatives).toContainEqual({ gear: -1, steering: -0.8, pedal: 0 });
        return [[1, 1], ...alternatives.map(({ gear, steering, pedal }): [number, number] =>
            gear === best.gear && steering === best.steering && pedal === best.pedal ? [0.7, 1.2] : [0, 0])];
    });
    const trial = await policy.decideAsync(input, true, allowed, options);
    expect(trial.directedAction).toEqual(best);
    policy.releaseDecision(trial);

    const evaluation = await policy.decideAsync(input, false, allowed, options);
    expect(evaluation.directedAction).toBeUndefined();
    expect(gpu.scoreActions).toHaveBeenCalledTimes(1);
    policy.releaseDecision(evaluation);
    policy.dispose();
});

it.each([[true, false], [false, true], [false, false]])(
    'applies live exploration independently (random=%s, directed=%s)', async (randomEnabled, directedEnabled) => {
        const draw = jest.fn(() => 0.99);
        const { policy, input } = setup(draw, { replayBatchSize: 1,
            randomExplorationEnabled: randomEnabled, directedExplorationEnabled: directedEnabled });
        await policy.initializeGpu();
        const gpu: DrivingGpuModel = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
        const options = { throttleLimited: false };
        const evaluation = await policy.decideAsync(input, false, gears, options);
        const action = evaluation.plan[0].action;
        await policy.learnSequenceAsync([{ startedAt: 1000, endedAt: 1200, input, next: input, reward: 1,
            samples: [{ decision: evaluation, rewardShare: 1, credits: [{ step: 0, weight: 1, gear: action.gear, gears, action }] }] }], input);
        policy.releaseDecision(evaluation);
        expect(policy.steps).toBeGreaterThan(0);
        expect(policy.canExploreDirectly).toBe(directedEnabled);

        const baseline = await policy.decideAsync(input, false, gears, options);
        const best = { gear: 2, steering: 1, pedal: 1 };
        jest.mocked(gpu.scoreActions).mockImplementationOnce(async (_pass, candidates) => {
            expect(candidates[0]).toEqual(baseline.plan[0].action);
            return candidates.map(({ gear, steering, pedal }) =>
                gear === best.gear && steering === best.steering && pedal === best.pedal ? [1, 1] : [0, 0]);
        });
        draw.mockClear();
        const trial = await policy.decideAsync(input, true, gears, options);
        if (randomEnabled) {
            expect(draw).toHaveBeenCalled();
            expect(trial.plan[0].latent).not.toEqual(baseline.plan[0].latent);
        } else {
            expect(draw).not.toHaveBeenCalled();
            expect(trial.plan).toEqual(baseline.plan);
        }
        expect(trial.directedAction).toEqual(directedEnabled ? best : undefined);
        expect(gpu.scoreActions).toHaveBeenCalledTimes(Number(directedEnabled));
        policy.releaseDecision(baseline);
        policy.releaseDecision(trial);
        policy.dispose();
    },
);

it('covers the full ranges at maximum density and skips a pedal range above the throttle limit', () => {
    const settings = { ...cloneConfig().learning, directedSteeringSamples: 15, directedPedalSamples: 15 };
    const baseline = { gear: 4, steering: 0.01, pedal: 0 };
    const candidates = directedActionCandidates(baseline, DRIVING_MAX_GEAR, 1, settings);
    expect(candidates).toHaveLength(1 + DRIVING_GEAR_COUNT * 15 * 15);
    for (let gear = DRIVING_MIN_GEAR; gear <= DRIVING_MAX_GEAR; gear++) {
        expect(candidates).toContainEqual({ gear, steering: -1, pedal: -1 });
        expect(candidates).toContainEqual({ gear, steering: 1, pedal: 1 });
    }
    const limited = { ...baseline, gear: 1 };
    expect(directedActionCandidates(limited, 1, 0.2, { ...settings, directedPedalMin: 0.5 })).toEqual([limited]);
    expect(directedActionCandidates(limited, 1, 1, { ...settings, directedSteeringMin: 0.4, directedSteeringMax: 0.4,
        directedPedalMin: 0.5, directedPedalMax: 0.5 })).toEqual([limited,
        ...[-1, 0, 1].map((gear) => ({ gear, steering: 0.4, pedal: 0.5 }))]);
});

it('bounds replay, retains adjacent states and executed actions, and discards cancelled additions', async () => {
    const { policy, input } = setup(0, { replayCapacity: 2, replayBatchSize: 2 });
    await policy.initializeGpu();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const decision = await policy.decideAsync(input, false);
    policy.releaseDecision(decision);
    const next = { ...input, features: input.features.map(() => 0.2) };
    const endpoint = { ...input, features: input.features.map(() => 0.9) };
    const executed = { pedal: -0.1, steering: 0.2, gear: 2 };
    const interval = (reward: number) => ({ startedAt: 1000, endedAt: 1200, reward, input: next, next: input,
        samples: [{ decision, rewardShare: 1, credits: [{ step: 0, weight: 1, gear: executed.gear, gears, action: executed }] }] });
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
