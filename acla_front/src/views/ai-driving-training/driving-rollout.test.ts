import { drivingSequenceTargets, DrivingRolloutInterval } from './driving-rollout';
import { cloneConfig } from './driving-model-config';
import { DrivingPolicy, PolicyDecision } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingControlOutput } from './driving-control-output';
import { drivingControllerAction, DrivingObservation, NEUTRAL, ObservationResult } from './driving-observation';
import { ManualControls } from './driving-control';
import { trainingFixture } from './training-fixtures';

const flush = async () => { for (let i = 0; i < 100; i++) await Promise.resolve(); };
beforeEach(() => { jest.useFakeTimers(); mockDrivingGpu(); });
afterEach(() => { jest.restoreAllMocks(); jest.useRealTimers(); });

it.each([0, 0.5, 1])('discounts the later impact by elapsed telemetry time (discount=%s)', (discount) => {
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
    expect(targets[1].reward).toBeCloseTo(2 - 5 * discount ** 1.5);
    expect(targets[0].reward).toBeCloseTo(1 + 2 * discount ** 0.5 - 5 * discount ** 2);
    expect(drivingSequenceTargets([], learning)).toEqual([]);
});

it('keeps the full multi-step return when identical controls have multiple acknowledgements', () => {
    const decision = {} as PolicyDecision, learning = { ...cloneConfig().learning, discount: 1 };
    const samples = [0.25, 0.75].map((rewardShare) => ({ decision, credits: [], rewardShare }));
    const sequence = [
        { startedAt: 0, endedAt: 1000, reward: 4, samples },
        { startedAt: 1000, endedAt: 2000, reward: -3, samples: samples.slice(0, 1) },
    ];
    const targets = drivingSequenceTargets(sequence, learning);
    const [single] = drivingSequenceTargets([
        { ...sequence[0], samples: [{ ...samples[0], rewardShare: 1 }] }, sequence[1],
    ], learning);
    expect(targets.slice(0, 2).map(({ reward, elapsedSeconds }) => ({ reward, elapsedSeconds })))
        .toEqual(Array(2).fill({ reward: single.reward, elapsedSeconds: single.elapsedSeconds }));
    expect(single.reward).toBe(1);
    expect(single.elapsedSeconds).toBe(2);
    expect(targets[2].reward).toBe(-3);
});

it('includes the terminal reward and retains the original input and executed controls', () => {
    const oldInput = { features: [0], cameras: [] }, input = { features: [1], cameras: [] };
    const next = { features: [2], cameras: [] }, terminal = { features: [3], cameras: [] };
    const decision = { input: oldInput, plan: [{ latent: [0, 0, 0] }] } as unknown as PolicyDecision;
    const action = { pedal: 0.4, steering: 0.3 };
    const credits = [{ step: 0, weight: 1, action, shift: 0 as const, shifts: { up: true, down: true } }];
    const sequence: DrivingRolloutInterval[] = [
        { startedAt: 1000, endedAt: 1200, input, next,
            reward: 2, samples: [{ decision, credits, rewardShare: 1, startedAt: 1100 }] },
        { startedAt: 1200, endedAt: 1400, input: next, next: terminal, nextShifts: { up: false, down: true },
            terminal: true, reward: -5, samples: [] },
        { startedAt: 1400, endedAt: 1600, reward: -10, samples: [] },
    ];
    // A later update/inference can change the live policy without changing the stored action.
    decision.plan[0].latent = [0.8, 0.5, -0.4];
    const [target] = drivingSequenceTargets(sequence, { ...cloneConfig().learning, discount: 0.5 });
    expect(target).toMatchObject({ reward: -0.5, elapsedSeconds: 0.4, terminal: true, nextShifts: { up: false, down: true } });
    expect(target.input).toBe(input);
    expect(target.next).toBe(terminal);
    expect(target.credits[0].action).toBe(action);
});

it('stops accumulating at a missing interval and bootstraps from the last contiguous observation', () => {
    const next = { features: [1], cameras: [] }, finalInput = { features: [2], cameras: [] };
    const samples = [{ decision: {} as PolicyDecision, credits: [], rewardShare: 1 }];
    const targets = drivingSequenceTargets([
        { startedAt: 1000, endedAt: 1200, reward: 1, next, samples },
        { startedAt: 1400, endedAt: 1600, reward: -5, samples },
    ], cloneConfig().learning, finalInput);
    expect(targets[0]).toMatchObject({ reward: 1, elapsedSeconds: 0.2, next, terminal: false });
    expect(targets[1]).toMatchObject({ reward: -5, elapsedSeconds: 0.2, next: finalInput, terminal: false });
});

function setup(sequenceSeconds = 4, planTimesSeconds?: number[]) {
    const fixture = trainingFixture(), config = cloneConfig();
    config.learning.sequenceSeconds = sequenceSeconds;
    if (planTimesSeconds) config.actions.planTimesSeconds = planTimesSeconds;
    config.network.hiddenUnits = 2;
    config.network.cameraProjectionUnits = 2;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 1; });
    const policy = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length, () => 0.1, config);
    let now = 1000, input: ObservationResult = { observation: fixture.observation };
    const controller = { apply: jest.fn().mockResolvedValue(undefined), connect: jest.fn(), disconnect: jest.fn() };
    const manual = jest.fn<Promise<ManualControls | null>, []>().mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    const loop = new DrivingTrainingLoop(new DrivingControlOutput(controller), () => input, jest.fn(), () => now, manual);
    const frame = async (at: number, offTrack = false, motion?: Pick<DrivingObservation, 'localVelocity' | 'speed'>) => {
        now = at;
        input = { observation: { ...fixture.observation, capturedAt: at, measuredAt: at,
            normalizedPosition: 0.25 + (at - 1000) / 1e6, offTrack, ...motion } };
        jest.advanceTimersByTime(100); await flush();
    };
    return { policy, loop, controller, manual, frame, invalid: async () => {
        input = { error: 'Camera disconnected.' }; jest.advanceTimersByTime(100); await flush();
    } };
}

it.each([3, 4, 5])('waits for the %s-second return and frees GPU samples', async (seconds) => {
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
    expect(learn).toHaveBeenCalledTimes(1);
    const { discount, discountIntervalSeconds } = run.policy.configuration.learning;
    expect(learn.mock.calls[0][1]).toBeCloseTo(rewards.reduce((sum, reward, index) =>
        sum + reward * discount ** (index / discountIntervalSeconds), 0));
    expect(learn.mock.calls[0][3]).toBe(seconds);
    for (const credit of learn.mock.calls[0][4]) {
        expect(credit.appliedAt).toBeGreaterThanOrEqual(1000);
        expect(run.controller.apply).toHaveBeenCalledWith(drivingControllerAction(credit.action!));
    }
    expect(run.policy.steps).toBe(1);
    expect((await run.policy.checkpointAsync()).model.lstmWeights).not.toEqual(before.model.lstmWeights);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.beginSequence).toHaveBeenCalledTimes(1);
    expect(gpu.endSequence).toHaveBeenCalledWith(true);
    expect(gpu.replay).toHaveBeenCalled();
    await run.loop.stop();
});

it('feeds a later impact into the critic target for earlier driving and keeps the AI in control', async () => {
    const run = setup(0.6), decide = jest.spyOn(run.policy, 'decideAsync');
    await run.loop.start(run.policy, true); await flush();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    const rewards: number[] = [];
    for (const at of [1200, 1400]) {
        await run.frame(at);
        rewards.push(run.loop.state.reward!.total);
    }
    expect(gpu.learnTransition).not.toHaveBeenCalled();

    await run.frame(1600, false, { speed: 0, localVelocity: [0, 0, 0] });
    const impact = run.loop.state.reward!;
    expect(impact.impact).toBeLessThan(0);
    const { discount } = run.policy.configuration.learning;
    expect(gpu.learnTransition).toHaveBeenCalledTimes(1);
    const [, next, target] = jest.mocked(gpu.learnTransition).mock.calls[0];
    expect(next).toBe(decide.mock.calls[decide.mock.calls.length - 1][0]);
    expect(target).toMatchObject({ elapsedSeconds: 0.6, terminal: false,
        actions: [expect.objectContaining({ appliedAt: 1000, step: 0 })] });
    expect(target.reward).toBeCloseTo(rewards[0] + discount * rewards[1] + discount ** 2 * impact.total);
    const applied = run.controller.apply.mock.calls.length;
    await run.frame(1800, false, { speed: 0, localVelocity: [0, 0, 0] });
    expect(run.controller.apply.mock.calls.length).toBeGreaterThan(applied);
    expect(run.loop.state).toMatchObject({ mode: 'training', source: 'ai' });
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
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    const decide = jest.spyOn(run.policy, 'decideAsync');
    await run.loop.start(run.policy, true); await flush();
    const before = await run.policy.checkpointAsync();
    await run.frame(2000); await run.frame(3000);
    expect(learn).not.toHaveBeenCalled();
    const endpoint = decide.mock.calls[decide.mock.calls.length - 1][0];
    const action = { pedal: 0.5, steering: 0.2 };
    run.manual.mockResolvedValue({ action: drivingControllerAction(action), toggleControl: control === 'controller' });
    if (control === 'ui') run.loop.setControlSource('manual');
    await run.frame(3100);
    expect(run.loop.state.source).toBe('manual');
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn.mock.calls[0][0].map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([[1000, 2000], [2000, 3000]]);
    expect(learn.mock.calls[0][1]).toBe(endpoint);
    expect(run.policy.steps).toBe(2);
    expect((await run.policy.checkpointAsync()).model.lstmWeights).not.toEqual(before.model.lstmWeights);

    run.manual.mockResolvedValue({ action: drivingControllerAction(action), toggleControl: false });
    await run.frame(4000); await run.frame(4500);
    const beforeReturn = await run.policy.checkpointAsync();
    const demonstrations = demonstrate.mock.calls.length;
    run.manual.mockResolvedValue({ action: drivingControllerAction(action), toggleControl: control === 'controller' });
    if (control === 'ui') run.loop.setControlSource('ai');
    await run.frame(4600);
    expect(run.loop.state.source).toBe('ai');
    expect(demonstrate.mock.calls.length).toBeGreaterThan(demonstrations);
    expect(demonstrate).toHaveBeenLastCalledWith(expect.anything(), expect.arrayContaining([
        expect.objectContaining({ step: 0, action: expect.objectContaining(action), shifts: { up: true, down: true } }),
    ]));
    expect((await run.policy.checkpointAsync()).model.lstmWeights).not.toEqual(beforeReturn.model.lstmWeights);
    expect(run.loop.state.steps).toBe(run.policy.steps);
    // The next rollout starts under AI ownership, with no human reward interval.
    run.manual.mockResolvedValue({ action: drivingControllerAction(action), toggleControl: false });
    for (const at of [5000, 6000, 7000, 8000, 9000]) await run.frame(at);
    expect(learn).toHaveBeenCalledTimes(2);
    expect(learn.mock.calls[1][0][0].startedAt).toBeGreaterThanOrEqual(4600);
    await run.loop.stop();
});

it.each(['partial', 'full'] as const)('finishes a pending %s AI batch before human learning without blocking controls', async (batch) => {
    const run = setup(), demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
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
    const action = { pedal: -0.7, steering: -0.2 };
    run.manual.mockResolvedValue({ action: drivingControllerAction(action), toggleControl: false });
    await run.frame(5100); await run.frame(5200);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(drivingControllerAction(action)));
    expect(demonstrate).not.toHaveBeenCalled();
    let saved = false;
    const checkpoint = run.policy.checkpointAsync().then((value) => { saved = true; return value; });
    await flush(); expect(saved).toBe(false);
    finish(); await flush();
    expect((await checkpoint).model.lstmWeights).not.toEqual(before.model.lstmWeights);
    expect(learning).toHaveBeenCalledTimes(batch === 'full' ? 5 : 2);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.endSequence).toHaveBeenCalledWith(true);
    expect(run.loop.state.steps).toBe(run.policy.steps);
    await run.frame(5400); await run.frame(5600);
    run.loop.setControlSource('ai'); await flush();
    expect(demonstrate).toHaveBeenCalled();
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
    const run = setup(4, [0, 0.1, 0.2]), decide = jest.spyOn(run.policy, 'decideAsync');
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    run.loop.setControlSource('manual'); await flush();
    const first = { pedal: 0.6, steering: 0.1 };
    run.manual.mockResolvedValue({ action: drivingControllerAction(first), toggleControl: false });
    await run.frame(2000);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    let finish!: () => void;
    jest.mocked(gpu.finishTraining).mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    const last = { pedal: -0.4, steering: -0.2 };
    run.manual.mockResolvedValue({ action: drivingControllerAction(last), toggleControl: false });
    await run.frame(2200); await run.frame(2300);
    expect(demonstrate).toHaveBeenCalledTimes(1);
    run.loop.setControlSource('ai');
    await run.frame(2400);
    expect(decide).toHaveBeenCalledTimes(1);
    expect(demonstrate).toHaveBeenCalledTimes(1);
    finish(); await flush();
    expect(demonstrate).toHaveBeenCalledTimes(3);
    expect(demonstrate.mock.calls.map(([, targets]) => targets[0].action)).toEqual([
        expect.objectContaining(first), expect.objectContaining(last), expect.objectContaining(last),
    ]);
    expect(demonstrate.mock.calls[0][1]).toContainEqual(
        expect.objectContaining({ step: 2, action: expect.objectContaining(last) }));
    expect(run.loop.state).toMatchObject({ source: 'ai', steps: 3, plan: null });
    await run.frame(2500);
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
