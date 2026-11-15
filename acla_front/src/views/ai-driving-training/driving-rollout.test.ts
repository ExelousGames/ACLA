import { drivingSequenceTargets, DrivingRolloutInterval } from './driving-rollout';
import { cloneConfig } from './driving-model-config';
import { DrivingPolicy, PolicyDecision } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingControlOutput } from './driving-control-output';
import { NEUTRAL, ObservationResult } from './driving-observation';
import { ManualControls } from './driving-control';
import { trainingFixture } from './training-fixtures';

const flush = async () => { for (let i = 0; i < 100; i++) await Promise.resolve(); };
beforeEach(() => { jest.useFakeTimers(); mockDrivingGpu(); });
afterEach(() => { jest.restoreAllMocks(); jest.useRealTimers(); });

it.each([0, 0.5, 1])('propagates later penalties backward using actual intervals (discount=%s)', (discount) => {
    const learning = { ...cloneConfig().learning, discount, discountIntervalSeconds: 1 };
    const decision = {} as PolicyDecision;
    const samples = [{ decision, credits: [], rewardShare: 1 }];
    const sequence: DrivingRolloutInterval[] = [
        { startedAt: 1000, endedAt: 1500, reward: 1, samples },
        { startedAt: 1500, endedAt: 3000, reward: 2, samples },
        { startedAt: 3000, endedAt: 4000, reward: -5, samples },
    ];
    const targets = drivingSequenceTargets(sequence, learning);
    expect(targets.map(({ elapsedSeconds }) => elapsedSeconds)).toEqual([3, 2.5, 1]);
    expect(targets[2].reward).toBe(-5);
    expect(targets[1].reward).toBeCloseTo(2 - discount ** 1.5 * 5);
    expect(targets[0].reward).toBeCloseTo(1 + discount ** 0.5 * (2 - discount ** 1.5 * 5));
    expect(drivingSequenceTargets([], learning)).toEqual([]);
});

it('splits immediate reward by acknowledged duration while retaining later consequences', () => {
    const decision = {} as PolicyDecision, learning = { ...cloneConfig().learning, discount: 1 };
    const samples = [0.25, 0.75].map((rewardShare) => ({ decision, credits: [], rewardShare }));
    const targets = drivingSequenceTargets([
        { startedAt: 0, endedAt: 1000, reward: 4, samples },
        { startedAt: 1000, endedAt: 2000, reward: -3, samples: samples.slice(0, 1) },
    ], learning);
    expect(targets.slice(0, 2).map(({ reward }) => reward)).toEqual([-2, 0]);
});

function setup(sequenceSeconds = 4) {
    const fixture = trainingFixture(), config = cloneConfig();
    config.learning.sequenceSeconds = sequenceSeconds;
    config.network.hiddenUnits = 2;
    config.network.cameraProjectionUnits = 2;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 1; });
    const policy = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length, () => 0.1, config);
    let now = 1000, input: ObservationResult = { observation: fixture.observation };
    const controller = { apply: jest.fn().mockResolvedValue(undefined), connect: jest.fn(), disconnect: jest.fn() };
    const manual = jest.fn<Promise<ManualControls | null>, []>().mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    const loop = new DrivingTrainingLoop(new DrivingControlOutput(controller), () => input, jest.fn(), () => now, manual);
    const frame = async (at: number, offTrack = false) => {
        now = at;
        input = { observation: { ...fixture.observation, capturedAt: at, measuredAt: at,
            normalizedPosition: 0.25 + (at - 1000) / 1e6, offTrack } };
        jest.advanceTimersByTime(100); await flush();
    };
    return { policy, loop, controller, manual, frame, invalid: async () => {
        input = { error: 'Camera disconnected.' }; jest.advanceTimersByTime(100); await flush();
    } };
}

it.each([3, 4, 5])('keeps weights fixed for %s seconds, then learns backward returns and frees old GPU samples', async (seconds) => {
    const run = setup(seconds), learn = jest.spyOn(run.policy, 'learnAsync'), decide = jest.spyOn(run.policy, 'decideAsync');
    await run.loop.start(run.policy, true); await flush();
    const before = await run.policy.checkpointAsync();
    const rewards: number[] = [];
    for (let second = 1; second <= seconds; second++) {
        await run.frame(1000 + second * 1000, second === seconds);
        rewards.push(run.loop.state.reward!.total);
        if (second < seconds) {
            expect(learn).not.toHaveBeenCalled();
            expect(await run.policy.checkpointAsync()).toEqual(before);
        }
        const decisions = await Promise.all(decide.mock.results.map(({ value }) => value));
        expect(decisions.filter(({ gpu }) => !gpu.released).length).toBeLessThanOrEqual(2);
    }
    expect(learn).toHaveBeenCalledTimes(seconds);
    let expected = 0;
    const gamma = 0.99 ** 5;
    for (let i = rewards.length - 1; i >= 0; i--) {
        expected = rewards[i] + gamma * expected;
        expect(learn.mock.calls[i][1]).toBeCloseTo(expected);
        expect(learn.mock.calls[i][3]).toBe(seconds - i);
        for (const credit of learn.mock.calls[i][4]) {
            expect(credit.appliedAt).toBeGreaterThanOrEqual(1000);
            expect(run.controller.apply).toHaveBeenCalledWith(credit.action);
        }
    }
    expect(run.policy.steps).toBe(seconds);
    expect((await run.policy.checkpointAsync()).model.lstmWeights).not.toEqual(before.model.lstmWeights);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.beginSequence).toHaveBeenCalledTimes(1);
    expect(gpu.endSequence).toHaveBeenCalledWith(true);
    expect(gpu.replay).toHaveBeenCalled();
    await run.loop.stop();
});

it.each(['stop', 'invalid', 'gap'] as const)('discards a partial sequence at %s', async (boundary) => {
    const run = setup(), learn = jest.spyOn(run.policy, 'learnSequenceAsync');
    await run.loop.start(run.policy, true); await flush();
    await run.frame(2000); await run.frame(3000);
    if (boundary === 'stop') { await run.loop.stop(); await run.loop.start(run.policy, true); await flush(); }
    if (boundary === 'invalid') await run.invalid();
    const restart = boundary === 'gap' ? 5000 : 4000;
    await run.frame(restart);
    await run.frame(restart + 1000); await run.frame(restart + 2000);
    expect(learn).not.toHaveBeenCalled();
    expect(run.policy.steps).toBe(0);
    await run.loop.stop();
});

it('never trains in evaluation even after a full sequence duration', async () => {
    const run = setup(), learn = jest.spyOn(run.policy, 'learnSequenceAsync');
    await run.loop.start(run.policy, false); await flush();
    const before = await run.policy.checkpointAsync();
    for (let at = 2000; at <= 7000; at += 1000) await run.frame(at);
    expect(learn).not.toHaveBeenCalled();
    expect(await run.policy.checkpointAsync()).toEqual(before);
    await run.loop.stop();
});

it('rolls back a stopped batch and keeps its checkpoint consistent', async () => {
    const run = setup(3);
    await run.loop.start(run.policy, true); await flush();
    const before = await run.policy.checkpointAsync();
    const learn = run.policy.learnAsync.bind(run.policy);
    let finish!: () => void;
    jest.spyOn(run.policy, 'learnAsync').mockImplementationOnce(async (...args) => {
        await learn(...args);
        await new Promise<void>((resolve) => { finish = resolve; });
    });
    await run.frame(2000); await run.frame(3000); await run.frame(4000);
    expect(run.policy.steps).toBe(1);
    let saved = false;
    const checkpoint = run.policy.checkpointAsync().then((value) => { saved = true; return value; });
    await flush();
    expect(saved).toBe(false);
    await run.loop.stop();
    finish(); await flush();
    expect(await checkpoint).toEqual(before);
    expect(run.policy.steps).toBe(0);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.endSequence).toHaveBeenCalledWith(false);
    await run.loop.stop();
});

it.each(['ui', 'controller'] as const)('updates partial AI and final human samples on %s handoffs', async (control) => {
    const run = setup(), learn = jest.spyOn(run.policy, 'learnSequenceAsync');
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstrationAsync');
    const decide = jest.spyOn(run.policy, 'decideAsync');
    await run.loop.start(run.policy, true); await flush();
    const before = await run.policy.checkpointAsync();
    await run.frame(2000); await run.frame(3000);
    expect(learn).not.toHaveBeenCalled();
    const endpoint = decide.mock.calls[decide.mock.calls.length - 1][0];
    const action = { throttle: 0.5, brake: 0, steering: 0.2 };
    run.manual.mockResolvedValue({ action, toggleControl: control === 'controller' });
    if (control === 'ui') run.loop.setControlSource('manual');
    await run.frame(3100);
    expect(run.loop.state.source).toBe('manual');
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn.mock.calls[0][0].map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([[1000, 2000], [2000, 3000]]);
    expect(learn.mock.calls[0][1]).toBe(endpoint);
    expect(run.policy.steps).toBe(2);
    expect((await run.policy.checkpointAsync()).model.lstmWeights).not.toEqual(before.model.lstmWeights);

    run.manual.mockResolvedValue({ action, toggleControl: false });
    await run.frame(4000); await run.frame(4500);
    const beforeReturn = await run.policy.checkpointAsync();
    const demonstrations = demonstrate.mock.calls.length;
    run.manual.mockResolvedValue({ action, toggleControl: control === 'controller' });
    if (control === 'ui') run.loop.setControlSource('ai');
    await run.frame(4600);
    expect(run.loop.state.source).toBe('ai');
    expect(demonstrate).toHaveBeenCalledTimes(demonstrations + 1);
    expect(demonstrate).toHaveBeenLastCalledWith(expect.anything(), expect.objectContaining(action), { up: true, down: true });
    expect((await run.policy.checkpointAsync()).model.lstmWeights).not.toEqual(beforeReturn.model.lstmWeights);
    expect(run.loop.state.steps).toBe(run.policy.steps);
    // The next rollout starts under AI ownership, with no human reward interval.
    run.manual.mockResolvedValue({ action, toggleControl: false });
    for (const at of [5000, 6000, 7000, 8000, 9000]) await run.frame(at);
    expect(learn).toHaveBeenCalledTimes(2);
    expect(learn.mock.calls[1][0][0].startedAt).toBeGreaterThanOrEqual(4600);
    await run.loop.stop();
});

it.each(['partial', 'full'] as const)('finishes a pending %s AI batch before human learning without blocking controls', async (batch) => {
    const run = setup(), demonstrate = jest.spyOn(run.policy, 'learnDemonstrationAsync');
    await run.loop.start(run.policy, true); await flush();
    const before = await run.policy.checkpointAsync();
    const learn = run.policy.learnAsync.bind(run.policy);
    let finish!: () => void;
    const learning = jest.spyOn(run.policy, 'learnAsync').mockImplementationOnce(async (...args) => {
        await learn(...args);
        await new Promise<void>((resolve) => { finish = resolve; });
    });
    await run.frame(2000); await run.frame(3000);
    if (batch === 'full') { await run.frame(4000); await run.frame(5000); }
    run.loop.setControlSource('manual'); await flush();
    const action = { throttle: 0, brake: 0.7, steering: -0.2 };
    run.manual.mockResolvedValue({ action, toggleControl: false });
    await run.frame(5100); await run.frame(5200);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(action));
    expect(demonstrate).not.toHaveBeenCalled();
    let saved = false;
    const checkpoint = run.policy.checkpointAsync().then((value) => { saved = true; return value; });
    await flush(); expect(saved).toBe(false);
    finish(); await flush();
    expect((await checkpoint).model.lstmWeights).not.toEqual(before.model.lstmWeights);
    expect(learning).toHaveBeenCalledTimes(batch === 'full' ? 4 : 2);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.endSequence).toHaveBeenCalledWith(true);
    expect(run.loop.state.steps).toBe(run.policy.steps);
    await run.frame(5400); await run.frame(5600);
    expect(demonstrate).toHaveBeenCalledTimes(1);
    await run.loop.stop();
});

it('flushes a partial rollout once across rapid repeated handoffs', async () => {
    const run = setup(), learn = jest.spyOn(run.policy, 'learnSequenceAsync');
    await run.loop.start(run.policy, true); await flush();
    await run.frame(2000); await run.frame(3000);
    for (const source of ['manual', 'ai', 'manual', 'ai'] as const) run.loop.setControlSource(source);
    await flush();
    expect(learn).toHaveBeenCalledTimes(1);
    expect(run.policy.steps).toBe(2);
    expect(run.loop.state).toMatchObject({ source: 'ai', steps: 2, plan: null });
    await run.frame(4000);
    expect(run.loop.state.plan).not.toBeNull();
    await run.loop.stop();
});

it('finishes pending and final human learning before resuming AI inference', async () => {
    const run = setup(), decide = jest.spyOn(run.policy, 'decideAsync');
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstrationAsync');
    const planned = jest.spyOn(run.policy, 'learnPlannedActionsAsync');
    await run.loop.start(run.policy, true); await flush();
    run.loop.setControlSource('manual'); await flush();
    const first = { throttle: 0.6, brake: 0, steering: 0.1 };
    run.manual.mockResolvedValue({ action: first, toggleControl: false });
    await run.frame(2000);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    let finish!: () => void;
    jest.mocked(gpu.finishTraining).mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    const last = { throttle: 0, brake: 0.4, steering: -0.2 };
    run.manual.mockResolvedValue({ action: last, toggleControl: false });
    await run.frame(3000);
    expect(demonstrate).toHaveBeenCalledTimes(1);
    run.loop.setControlSource('ai');
    await run.frame(3100);
    expect(decide).toHaveBeenCalledTimes(1);
    expect(demonstrate).toHaveBeenCalledTimes(1);
    finish(); await flush();
    expect(demonstrate).toHaveBeenCalledTimes(2);
    expect(demonstrate.mock.calls.map(([, action]) => action)).toEqual([
        expect.objectContaining(first), expect.objectContaining(last),
    ]);
    expect(planned).toHaveBeenCalledWith(demonstrate.mock.calls[0][0], [
        expect.objectContaining({ step: 10, action: expect.objectContaining(last) }),
    ]);
    expect(run.loop.state).toMatchObject({ source: 'ai', steps: 2, plan: null });
    await run.frame(3200);
    expect(decide).toHaveBeenCalledTimes(2);
    expect(run.loop.state.plan).not.toBeNull();
    await run.loop.stop();
});

it('preserves a partial rollout while stale inference finishes during takeover', async () => {
    const run = setup(), learn = jest.spyOn(run.policy, 'learnSequenceAsync');
    await run.loop.start(run.policy, true); await flush();
    await run.frame(2000);
    const decide = run.policy.decideAsync.bind(run.policy);
    let finish!: () => void;
    jest.spyOn(run.policy, 'decideAsync').mockImplementationOnce(async (...args) => {
        const decision = await decide(...args);
        await new Promise<void>((resolve) => { finish = resolve; });
        return decision;
    });
    await run.frame(3000);
    run.loop.setControlSource('manual'); await flush();
    expect(learn).not.toHaveBeenCalled();
    finish(); await flush();
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn.mock.calls[0][0]).toHaveLength(2);
    expect(run.loop.state).toMatchObject({ source: 'manual', steps: 2, plan: null });
    await run.loop.stop();
});

it('reports an in-flight training failure after handing control to the human', async () => {
    const run = setup();
    await run.loop.start(run.policy, true); await flush();
    let fail!: () => void;
    jest.spyOn(run.policy, 'learnSequenceAsync').mockImplementationOnce(() => new Promise<void>((resolve, reject) => {
        fail = () => reject(new Error('GPU training failed.'));
    }));
    for (const at of [2000, 3000, 4000, 5000]) await run.frame(at);
    run.loop.setControlSource('manual'); await flush();
    fail(); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', status: 'GPU training failed.' });
});

it('keeps weights fixed across evaluation handoffs', async () => {
    const run = setup();
    await run.loop.start(run.policy, false); await flush();
    const before = await run.policy.checkpointAsync();
    await run.frame(2000); await run.frame(3000);
    run.loop.setControlSource('manual');
    await run.frame(4000); await run.frame(5000);
    run.loop.setControlSource('ai'); await flush();
    expect(await run.policy.checkpointAsync()).toEqual(before);
    await run.loop.stop();
});
