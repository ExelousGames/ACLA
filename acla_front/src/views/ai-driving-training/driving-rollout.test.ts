import { drivingSequenceTargets, DrivingRolloutInterval } from './driving-rollout';
import { cloneConfig } from './driving-model-config';
import { DrivingPolicy, PolicyDecision } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingControlOutput } from './driving-control-output';
import { ObservationResult } from './driving-observation';
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
    const loop = new DrivingTrainingLoop(new DrivingControlOutput(controller), () => input, jest.fn(), () => now);
    const frame = async (at: number, offTrack = false) => {
        now = at;
        input = { observation: { ...fixture.observation, capturedAt: at, measuredAt: at,
            normalizedPosition: 0.25 + (at - 1000) / 1e6, offTrack } };
        jest.advanceTimersByTime(100); await flush();
    };
    return { policy, loop, controller, frame, invalid: async () => {
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

it.each(['stop', 'handoff', 'invalid', 'gap'] as const)('discards a partial sequence at %s', async (boundary) => {
    const run = setup(), learn = jest.spyOn(run.policy, 'learnSequenceAsync');
    await run.loop.start(run.policy, true); await flush();
    await run.frame(2000); await run.frame(3000);
    if (boundary === 'stop') { await run.loop.stop(); await run.loop.start(run.policy, true); await flush(); }
    if (boundary === 'handoff') { run.loop.setControlSource('manual'); run.loop.setControlSource('ai'); }
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

it('rolls back an interrupted batch and keeps its checkpoint consistent', async () => {
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
    run.loop.setControlSource('manual');
    finish(); await flush();
    expect(await checkpoint).toEqual(before);
    expect(run.policy.steps).toBe(0);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.endSequence).toHaveBeenCalledWith(false);
    await run.loop.stop();
});
