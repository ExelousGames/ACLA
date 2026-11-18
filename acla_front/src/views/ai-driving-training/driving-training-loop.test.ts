import { mockDrivingGpu } from './driving-gpu-test-model';
import { DrivingGpuModel } from './driving-gpu-model';
import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingPolicy, PlanStepCredit, PolicyDecision } from './driving-policy';
import { drivingControllerAction, NEUTRAL, ObservationResult, readDrivingObservation } from './driving-observation';
import { circuitMapFixture, trainingFixture } from './training-fixtures';
import { normalizedAggression, referenceSpeedTargets } from './driving-reference';
import { ManualControls } from './driving-control';
import { DrivingControlOutput } from './driving-control-output';
import { DrivingMemory, DrivingPolicyInput, drivingMemoryInputs, drivingMemorySlotInputs } from './driving-memory';
import { cloneConfig } from './driving-model-config';
import { DrivingImpactMonitor } from './driving-impact-monitor';

// Keep these control/ownership tests on a short update cadence. The production
// collection window and multi-step replay targets are covered in driving-rollout.test.ts.
jest.mock('./driving-model-defaults.json', () => {
    const config = jest.requireActual('./driving-model-defaults.json');
    return { ...config, learning: { ...config.learning, sequenceSeconds: 0.001 },
        network: { ...config.network, hiddenUnits: 3, cameraProjectionUnits: 8,
            cameraLayers: config.network.cameraLayers.map((layer: object) => ({ ...layer, channels: 2 })) } };
});

const flush = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };
beforeEach(() => {
    jest.useFakeTimers();
    mockDrivingGpu();
    jest.spyOn(DrivingPolicy.prototype, 'initializeGpu');
});
afterEach(() => { jest.restoreAllMocks(); jest.useRealTimers(); });

/** Observe acknowledged reward attribution before SAC builds full Bellman targets
 * and randomly resamples historical transitions. Keep real training running so
 * ownership, cancellation and weight checks remain covered. */
function collectedTransitions(policy: DrivingPolicy) {
    const transitions = jest.fn<void, [PolicyDecision, number, DrivingPolicyInput, number, PlanStepCredit[]]>();
    const learn = (jest.isMockFunction(policy.learnSequenceAsync)
        ? jest.mocked(policy.learnSequenceAsync).getMockImplementation()! : policy.learnSequenceAsync).bind(policy);
    jest.spyOn(policy, 'learnSequenceAsync').mockImplementation(async (sequence, next, active) => {
        for (const interval of sequence) {
            for (const sample of interval.samples) {
                transitions(sample.decision, interval.reward * sample.rewardShare, interval.next ?? next,
                    (interval.endedAt - interval.startedAt) / 1000, sample.credits);
            }
        }
        await learn(sequence, next, active);
    });
    return transitions;
}

it.each([true, false])('scores off-track penalties and on-track rewards once per fresh observation and includes them in learning (training=%s)', async (train) => {
    const run = setup();
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200, offTrack: true } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    const reward = run.loop.state.reward!;
    expect(reward.offTrack).toBe(-1);
    expect(reward.onTrack).toBe(0);
    expect(reward.total).toBeCloseTo(-1.08);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    expect(learn).toHaveBeenCalledTimes(train ? 1 : 0);
    expect(learn.mock.calls.slice(-1)).toEqual(train ? [[expect.anything(), reward.total, expect.anything(), 0.2, expect.any(Array)]] : []);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.episodeReward).toBe(reward.total);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400, normalizedPosition: 0.252, offTrack: false } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    const onTrackReward = run.loop.state.reward!;
    expect(onTrackReward.offTrack).toBe(0);
    expect(onTrackReward.onTrack).toBeCloseTo(0.2);
    expect(onTrackReward.total).toBeCloseTo(0.141);
    expect(run.loop.state.episodeReward).toBeCloseTo(reward.total + onTrackReward.total);
    expect(learn).toHaveBeenCalledTimes(train ? 2 : 0);
    expect(learn.mock.calls.slice(-1)).toEqual(train ? [[expect.anything(), onTrackReward.total, expect.anything(), 0.2, expect.any(Array)]] : []);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.episodeReward).toBeCloseTo(reward.total + onTrackReward.total);
    expect(learn).toHaveBeenCalledTimes(train ? 2 : 0);
    await run.loop.stop();
});

it('requires GPU initialization before applying AI controls and reports initialization failures', async () => {
    const run = setup();
    jest.spyOn(run.policy, 'initializeGpu').mockRejectedValue(new Error('GPU unavailable'));
    await expect(run.loop.start(run.policy, true)).rejects.toThrow('GPU unavailable');
    expect(run.loop.state.mode).toBe('idle');
    expect(run.controller.apply).not.toHaveBeenCalled();
});

it.each([[true, 30, 0, 0, 0], [false, 30, 0, 0, 0], [true, 30, 23, 0, 0], [false, 30, 23, 0, 0],
    [true, 10 / 3.6, 0, 0, 0], [false, 10 / 3.6, 0, 0, 0], [true, 30, 24, -6, 0.4], [false, 30, 36, 6, 0.4]] as const)(
    'scores an impact once and passes its penalty to AI learning (training=%s, from=%s, forward=%s, right=%s, steering=%s)', async (train, initial, forward, right, steering) => {
    const run = setup();
    const learn = collectedTransitions(run.policy);
    run.input({ observation: { ...run.fixture.observation, speed: initial * 3.6,
        localVelocity: [0, initial, 0], controls: { ...run.fixture.observation.controls, steering } } }, 1000);
    await run.loop.start(run.policy, train); await flush();
    const afterImpact = { ...run.fixture.observation, capturedAt: 1200, speed: Math.hypot(right, forward) * 3.6,
        localVelocity: [right, forward, 0] as [number, number, number], controls: { ...run.fixture.observation.controls, steering } };
    run.input({ observation: afterImpact }, 1200);
    jest.advanceTimersByTime(100); await flush();
    const reward = run.loop.state.reward!;
    if (initial === 30 && forward === 0) expect(reward.impact).toBe(-5);
    else expect(reward.impact).toBeLessThan(-0.4);
    expect(reward.impactEstimate.detected).toBe(true);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    expect(learn).toHaveBeenCalledTimes(train ? 1 : 0);
    if (train) expect(learn).toHaveBeenLastCalledWith(expect.anything(), reward.total, expect.anything(), 0.2, expect.any(Array));
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.episodeReward).toBe(reward.total);
    run.input({ observation: { ...afterImpact, capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.impact).toBe(0);
    expect(run.loop.state.episodeReward).toBeCloseTo(reward.total + run.loop.state.reward!.total);
    await run.loop.stop();
});

it.each([[true, 20], [false, 20], [true, 80], [false, 80]] as const)(
    'scores a telemetry impact between slow AI images once (training=%s, speed=%s km/h)', async (train, kph) => {
    const monitor = new DrivingImpactMonitor();
    const run = setup(false, (previous, current) => monitor.between(previous, current));
    const speed = kph / 3.6;
    const initial = { ...run.fixture.observation, speed: kph, localVelocity: [0, speed, 0] as [number, number, number] };
    const observe = (at: number, forward = speed, right = 0) => monitor.observe({ ...run.fixture.live, receivedAt: at,
        currentTelemetry: { ...run.fixture.live.currentTelemetry, Physics_local_velocity_x: right, Physics_local_velocity_y: forward } });
    observe(1000);
    run.input({ observation: initial }, 1000);
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, train); await flush();
    observe(1100, speed * 0.6, speed * 0.3);
    for (const at of [1200, 1400, 1600, 1800]) observe(at);
    run.input({ observation: { ...initial, capturedAt: 1800 } }, 1800);
    jest.advanceTimersByTime(100); await flush();
    const reward = run.loop.state.reward!;
    expect(reward.impactEstimate.detected).toBe(true);
    expect(reward.impact).toBeLessThan(0);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    expect(learn).toHaveBeenCalledTimes(train ? 1 : 0);
    if (train) expect(learn).toHaveBeenLastCalledWith(expect.anything(), reward.total, expect.anything(), 0.8, expect.any(Array));
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.episodeReward).toBe(reward.total);
    observe(2000);
    run.input({ observation: { ...initial, capturedAt: 2000 } }, 2000);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.impact).toBe(0);
    await run.loop.stop();
});

it('cancels a start while the GPU is initializing', async () => {
    const run = setup();
    let ready!: () => void;
    jest.spyOn(run.policy, 'initializeGpu').mockImplementation(() => new Promise<void>((resolve) => { ready = resolve; }));
    const starting = run.loop.start(run.policy, true);
    await run.loop.stop();
    ready(); await starting; await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.controller.apply.mock.calls.every(([action]) => action === NEUTRAL)).toBe(true);
});

it.each([true, false])('waits for fresh inputs after GPU startup without taking human control (training=%s)', async (train) => {
    const run = setup(true);
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    const initialize = jest.mocked(run.policy.initializeGpu).getMockImplementation()!;
    jest.mocked(run.policy.initializeGpu).mockImplementationOnce(async () => {
        await initialize.call(run.policy);
        run.input({ error: 'Waiting for fresh, running live telemetry.', waitingForFreshInputs: true }, 2000);
    });
    const starting = run.loop.start(run.policy, train); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'manual', source: 'manual' });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 2100 } }, 2100);
    jest.advanceTimersByTime(50); await starting; await flush();
    expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source: 'manual' });
    await run.loop.stop();
});

it('cancels startup while waiting for fresh inputs and never starts when they arrive later', async () => {
    const run = setup();
    run.input({ error: 'Waiting for fresh, running live telemetry.', waitingForFreshInputs: true }, 2000);
    const starting = run.loop.start(run.policy, true); await flush();
    expect(run.controller.apply).not.toHaveBeenCalled();
    await run.loop.stop();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 2100 } }, 2100);
    jest.advanceTimersByTime(50); await starting; await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', source: null });
    expect(run.controller.apply.mock.calls.every(([action]) => action === NEUTRAL)).toBe(true);
});

it('times out startup when fresh inputs never arrive and permits another start', async () => {
    const run = setup();
    run.input({ error: 'Waiting for fresh, running live telemetry.', waitingForFreshInputs: true }, 2000);
    const starting = run.loop.start(run.policy, false);
    const failed = expect(starting).rejects.toThrow('Waiting for fresh, running live telemetry.');
    await flush();
    for (let elapsed = 0; elapsed < 5000; elapsed += 50) {
        jest.advanceTimersByTime(50); await flush();
    }
    await failed;
    expect(run.loop.state.mode).toBe('idle');
    expect(run.controller.apply).not.toHaveBeenCalled();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 7100 } }, 7100);
    await run.loop.start(run.policy, false); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'evaluation', source: 'ai' });
    await run.loop.stop();
});

it.each(['stop', 'handoff'] as const)('discards an asynchronous GPU decision after %s', async (boundary) => {
    const run = setup();
    await run.policy.initializeGpu();
    const decision = await run.policy.decideAsync(new DrivingMemory().encode(run.fixture.observation), false);
    let ready!: () => void;
    jest.spyOn(run.policy, 'decideAsync').mockImplementation(() => new Promise((resolve) => { ready = () => resolve(decision); }));
    const release = jest.spyOn(run.policy, 'releaseDecision');
    await run.loop.start(run.policy, false); await flush();
    jest.advanceTimersByTime(80); await flush();
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ state: 'processing', currentStage: 'inference', stageElapsedMs: 80 });
    if (boundary === 'stop') await run.loop.stop();
    else run.loop.setControlSource('manual');
    const interrupted = run.loop.getModelTimingSnapshot();
    expect(interrupted).toMatchObject({ state: 'incomplete', currentStage: null, stages: { inference: 80 } });
    const count = run.controller.apply.mock.calls.length;
    ready(); await flush();
    expect(run.loop.getModelTimingSnapshot()).toEqual(interrupted);
    expect(run.controller.apply.mock.calls).toHaveLength(count);
    expect(release).toHaveBeenCalledWith(decision);
    expect(run.loop.state.plan).toBeNull();
    await run.loop.stop();
});

it('keeps the training sample alive until pending GPU backprop finishes after Stop', async () => {
    const run = setup();
    await run.loop.start(run.policy, true); await flush();
    let finish!: () => void;
    const learn = jest.spyOn(run.policy, 'learnAsync').mockImplementation(() => new Promise<void>((resolve) => { finish = resolve; }));
    const sequences = jest.spyOn(run.policy, 'learnSequenceAsync');
    const release = jest.spyOn(run.policy, 'releaseDecision');
    const plans = jest.spyOn(run.policy, 'learnPlannedActionsAsync');
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    const sample = sequences.mock.calls[0][0][0].samples[0].decision;
    expect(learn.mock.calls[0][0].input).toBe(sample.input);
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ state: 'processing', currentStage: 'learning' });
    await run.loop.stop();
    const interrupted = run.loop.getModelTimingSnapshot();
    expect(interrupted).toMatchObject({ state: 'incomplete', currentStage: null });
    expect(release).not.toHaveBeenCalledWith(sample);
    finish(); await flush();
    expect(release).toHaveBeenCalledWith(sample);
    expect(plans).not.toHaveBeenCalled();
    expect(run.loop.getModelTimingSnapshot()).toEqual(interrupted);
    expect(run.loop.state).toMatchObject({ mode: 'idle', action: NEUTRAL, plan: null });
});

it.each([true, false])('publishes five-second plans, keeps inferring and refreshes expired plans without imitation (training=%s)', async (train) => {
    const run = setup();
    const decide = jest.spyOn(run.policy, 'decideAsync');
    const planned = jest.spyOn(run.policy, 'learnPlannedActionsAsync');
    run.loop.start(run.policy, train); await flush();
    const plan = run.loop.state.plan!;
    expect(plan.map(({ timeSeconds }) => timeSeconds)).toEqual(Array.from({ length: 51 }, (_, step) => step / 10));
    expect(run.controller.apply).toHaveBeenLastCalledWith(drivingControllerAction(plan[0].action));
    run.input({ observation: run.fixture.observation }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.plan).toBe(plan);
    expect(decide).toHaveBeenCalledTimes(1);
    for (let capturedAt = 1200; capturedAt <= 6400; capturedAt += 200) {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    expect(run.loop.state.plan).not.toBe(plan);
    expect(decide).toHaveBeenCalledTimes(28);
    expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
    expect(planned).not.toHaveBeenCalled();
    await run.loop.stop();
    expect(run.loop.state.plan).toBeNull();
});

it.each(['stop', 'handoff', 'invalid', 'gap'] as const)('discards unfinished plan training after %s', async (boundary) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const planned = jest.spyOn(run.policy, 'learnPlannedActionsAsync');
    run.loop.start(run.policy, true); await flush();
    for (const capturedAt of [1200, 1400]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    expect(planned).not.toHaveBeenCalled();
    if (boundary === 'stop') await run.loop.stop();
    if (boundary === 'handoff') { run.loop.setControlSource('manual'); run.loop.setControlSource('ai'); }
    if (boundary === 'invalid') {
        run.input({ error: 'Camera disconnected.' }, 1500);
        jest.advanceTimersByTime(100); await flush();
    }
    expect(run.loop.state.plan).toEqual(boundary === 'gap' ? expect.any(Array) : null);
    const restartAt = boundary === 'gap' ? 3000 : 1800;
    run.input({ observation: { ...run.fixture.observation, capturedAt: restartAt } }, restartAt);
    if (boundary === 'stop') run.loop.start(run.policy, true);
    jest.advanceTimersByTime(200); await flush();
    for (const capturedAt of [restartAt + 200, restartAt + 400]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    expect(planned).not.toHaveBeenCalled();
    await run.loop.stop();
});

it.each([true, false])('uses later acknowledged human actions to teach the five-second plan only in training (training=%s)', async (train) => {
    const run = setup(true);
    const planned = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    const learn = collectedTransitions(run.policy);
    const original = new DrivingMemory().encode(run.fixture.observation);
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    run.loop.start(run.policy, train); await flush();
    const later = { throttle: 0.8, brake: 0, steering: -0.6 };
    for (let capturedAt = 1100; capturedAt <= 6000; capturedAt += 100) {
        run.manual.mockResolvedValue({ action: capturedAt === 6000 ? later : NEUTRAL, toggleControl: false });
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(100); await flush();
    }
    expect(planned).not.toHaveBeenCalled();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 6000 } }, 6051);
    jest.advanceTimersByTime(60); await flush();
    expect(planned).toHaveBeenCalledTimes(train ? 1 : 0);
    if (train) {
        expect(planned.mock.calls[0][0]).toEqual(original);
        expect(planned.mock.calls[0][1]).toHaveLength(51);
        expect(planned.mock.calls[0][1][50]).toEqual({ step: 50,
            action: { pedal: later.throttle, steering: later.steering, shiftUp: false, shiftDown: false }, shifts: { up: true, down: true } });
    }
    expect(learn).not.toHaveBeenCalled(); // Human driving never supplies an on-policy reward gradient.
    expect(run.loop.state.plan).toBeNull();
    await run.loop.stop();
});

function setup(withManual = false, readImpact?: ConstructorParameters<typeof DrivingTrainingLoop>[5]) {
    const fixture = trainingFixture();
    let input: ObservationResult = { observation: fixture.observation };
    let now = 1000;
    const controller = { connect: jest.fn(), disconnect: jest.fn(), apply: jest.fn().mockResolvedValue(undefined) };
    const publish = jest.fn();
    // Force an exploratory upshift so pulse and handoff tests exercise pressed paddles.
    const policy = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length, () => 0.98);
    const manual = jest.fn<Promise<ManualControls | null>, []>().mockResolvedValue(null);
    const loop = new DrivingTrainingLoop(new DrivingControlOutput(controller), () => input, publish, () => now, withManual ? manual : undefined, readImpact);
    return { fixture, controller, policy, loop, publish, manual,
        input: (next: ObservationResult, time: number, measuredAt = next.observation?.capturedAt) => {
            input = next.observation ? { ...next, observation: { ...next.observation, measuredAt: measuredAt! } } : next;
            now = time;
        } };
}

it('resets exploration on starts, control handoffs, missing inputs and stops', async () => {
    const run = setup();
    const reset = jest.spyOn(run.policy, 'resetExploration');
    await run.loop.start(run.policy, true); await flush();
    expect(reset).toHaveBeenCalled();
    reset.mockClear();
    run.loop.setControlSource('manual'); await flush();
    expect(reset).toHaveBeenCalled();
    reset.mockClear();
    run.loop.setControlSource('ai'); await flush();
    expect(reset).toHaveBeenCalled();
    reset.mockClear();
    run.input({ error: 'Camera unavailable' }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(reset).toHaveBeenCalled();
    reset.mockClear();
    await run.loop.stop();
    expect(reset).toHaveBeenCalled();
    reset.mockClear();
    run.input({ observation: run.fixture.observation }, 1400);
    await run.loop.start(run.policy, true); await flush();
    expect(reset).toHaveBeenCalled();
    await run.loop.stop();
});

it.each(['learning', 'learningWait', 'inference'] as const)('collects overlapping consequence windows during pending %s', async (stage) => {
    const run = setup();
    run.policy.configuration.learning.sequenceSeconds = 0.4;
    const sequences = jest.spyOn(run.policy, 'learnSequenceAsync');
    const decide = jest.spyOn(run.policy, 'decideAsync');
    await run.loop.start(run.policy, true); await flush();
    const firstDecision = await decide.mock.results[0].value;
    let finish!: () => void;
    if (stage === 'learning') {
        const learn = run.policy.learnAsync.bind(run.policy);
        jest.spyOn(run.policy, 'learnAsync').mockImplementationOnce(async (...args) => {
            await learn(...args);
            await new Promise<void>((resolve) => { finish = resolve; });
        });
    } else if (stage === 'learningWait') {
        const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
        jest.mocked(gpu.finishTraining).mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    } else {
        const infer = run.policy.decideAsync.bind(run.policy);
        decide.mockImplementationOnce(async (...args) => {
            const decision = await infer(...args);
            await new Promise<void>((resolve) => { finish = resolve; });
            return decision;
        });
    }
    const frame = async (capturedAt: number) => {
        run.input({ observation: { ...run.fixture.observation, capturedAt,
            normalizedPosition: 0.25 + (capturedAt - 1000) / 1e6, offTrack: true } }, capturedAt);
        jest.advanceTimersByTime(100); await flush();
    };
    for (const at of [1200, 1400, 1600, 1800]) {
        await frame(at);
        expect(run.loop.state.reward?.offTrack).toBe(-1);
    }
    expect(sequences).toHaveBeenCalledTimes(stage === 'inference' ? 0 : 1);
    const firstBatch = sequences.mock.calls[0]?.[0];
    expect(firstBatch?.map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual(
        stage === 'inference' ? undefined : [[1000, 1200], [1200, 1400]]);
    const collectedReward = run.loop.state.episodeReward;
    finish(); await flush();
    // Even without a new camera frame, the completed windows are available.
    for (let tick = 0; tick < 2; tick++) { jest.advanceTimersByTime(20); await flush(); }
    expect(sequences).toHaveBeenCalledTimes(3);
    expect(sequences.mock.calls.map(([sequence]) => sequence.map(({ startedAt, endedAt }) => [startedAt, endedAt]))).toEqual([
        [[1000, 1200], [1200, 1400]], [[1200, 1400], [1400, 1600]], [[1400, 1600], [1600, 1800]],
    ]);
    expect(run.loop.state.episodeReward).toBe(collectedReward); // Repeated polls never duplicate reward.
    expect(sequences.mock.calls[1][0][0].samples[0].decision).toBe(firstDecision);
    expect(sequences.mock.calls.every(([sequence]) => sequence[1].samples.length === 0)).toBe(true);
    expect(sequences.mock.calls[0][1]).not.toBe(sequences.mock.calls[1][1]);
    for (const [sequence] of sequences.mock.calls) {
        // An old plan may still execute, but SAC observes the current state pair.
        expect(sequence[0].input).toBeDefined();
        expect(sequence[0].next).toBe(sequence[1].input);
        expect(sequence[0].next).not.toBe(sequence[1].next);
        expect(sequence[0].nextShifts).toEqual({ up: true, down: true });
        expect(sequence[0].terminal).toBe(false);
    }
    expect(sequences.mock.calls[1][0][0].input).not.toBe(firstDecision.input);
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    expect(gpu.replay).toHaveBeenCalled(); // Collected actions replay against each batch's frozen weights.
    await run.loop.stop();
});

it('keeps scoring fresh consequences during evaluation inference without changing weights', async () => {
    const run = setup();
    const learn = jest.spyOn(run.policy, 'learnSequenceAsync');
    await run.loop.start(run.policy, false); await flush();
    const before = await run.policy.checkpointAsync();
    const infer = run.policy.decideAsync.bind(run.policy);
    let finish!: () => void;
    jest.spyOn(run.policy, 'decideAsync').mockImplementationOnce(async (...args) => {
        const decision = await infer(...args);
        await new Promise<void>((resolve) => { finish = resolve; });
        return decision;
    });
    for (const at of [1200, 1400, 1600, 1800]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt: at, offTrack: true } }, at);
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state.reward?.offTrack).toBe(-1);
    }
    expect(run.loop.state.episodeReward).toBeCloseTo(-4.32);
    expect(learn).not.toHaveBeenCalled();
    finish(); await flush();
    expect(await run.policy.checkpointAsync()).toEqual(before);
    await run.loop.stop();
});

it('uses telemetry time for each consequence horizon despite changing camera timestamp offsets', async () => {
    const run = setup();
    run.policy.configuration.learning.sequenceSeconds = 3;
    const sequences = jest.spyOn(run.policy, 'learnSequenceAsync');
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, true); await flush();
    for (const [now, capturedAt, measuredAt] of [[2000, 1900, 1900], [3000, 3000, 2900], [4000, 4000, 3900]]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, now, measuredAt);
        jest.advanceTimersByTime(100); await flush();
    }
    // The camera has advanced three seconds, but telemetry only 2.9 seconds.
    expect(sequences).not.toHaveBeenCalled();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 4050 } }, 4100, 4000);
    jest.advanceTimersByTime(100); await flush();
    expect(sequences).toHaveBeenCalledTimes(1);
    expect(sequences.mock.calls[0][0].map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([
        [1000, 1900], [1900, 2900], [2900, 3900], [3900, 4000],
    ]);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn.mock.calls[0][3]).toBe(0.9);
    expect(learn.mock.calls[0][4].map(({ appliedAt }) => appliedAt)).toEqual([1000]);
    await run.loop.stop();
});

it('keeps collecting across a long GPU wait and provides full recent windows without a false input gap', async () => {
    const run = setup();
    run.policy.configuration.learning.sequenceSeconds = 0.4;
    run.policy.configuration.reward.timingWeight = 0;
    run.policy.configuration.reward.steeringChangeWeight = 0;
    run.policy.configuration.reward.throttleChangeWeight = 0;
    run.policy.configuration.reward.brakeChangeWeight = 0;
    const batches = jest.spyOn(run.policy, 'learnSequenceAsync');
    await run.loop.start(run.policy, true); await flush();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    let finish!: () => void;
    jest.mocked(gpu.finishTraining).mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    for (let at = 1200; at <= 3400; at += 200) {
        run.input({ observation: { ...run.fixture.observation, capturedAt: at, offTrack: true } }, at);
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state.reward?.offTrack).toBe(-1);
    }
    expect(batches).toHaveBeenCalledTimes(1);
    expect(run.loop.state.episodeReward).toBe(-12);
    finish(); await flush();
    for (let tick = 0; tick < 3; tick++) { jest.advanceTimersByTime(20); await flush(); }
    expect(batches.mock.calls.map(([sequence]) => [sequence[0].startedAt, sequence[sequence.length - 1].endedAt])).toEqual([
        [1000, 1400], [2600, 3000], [2800, 3200], [3000, 3400],
    ]);
    expect(run.loop.state.mode).toBe('training');
    expect(run.loop.state.episodeReward).toBe(-12);
    await run.loop.stop();
});

function shortHumanPlan() {
    const run = setup(true), config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.1];
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1, config);
    return run;
}

it.each([true, false])('reuses learned weights in a new game, car and map after stopping (training=%s)', async (train) => {
    const run = setup();
    await run.loop.start(run.policy, true); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBeGreaterThan(0);
    await run.loop.stop();
    const saved = await run.policy.checkpointAsync();
    run.policy.dispose();
    run.policy = DrivingPolicy.restore(JSON.parse(JSON.stringify(saved)));
    const fixture = trainingFixture(2000);
    const live = { ...fixture.live, game: 'iracing' as const, currentTelemetry: { ...fixture.live.currentTelemetry,
        Static_car_model: 'MX5', Static_track: 'spa' } };
    const lap = { ...fixture.referenceLap, game: 'iracing', car: 'MX5', track: 'spa' };
    const input = readDrivingObservation(live, { ...fixture.vision, telemetry: live }, lap, 50, 2000, run.policy.configuration, circuitMapFixture(lap));
    expect(input.error).toBeUndefined();
    run.input(input, 2000);
    await run.loop.start(run.policy, train); await flush();
    expect(run.loop.state.mode).toBe(train ? 'training' : 'evaluation');
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(run.loop.state.reward).toBeNull();
    expect(await run.policy.checkpointAsync()).toEqual(saved);
    run.input({ observation: { ...input.observation!, capturedAt: 2200 } }, 2200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(saved.model.steps + (train ? 1 : 0));
    expect(run.loop.state.reward).not.toBeNull();
    await run.loop.stop();
    const continued = await run.policy.checkpointAsync();
    expect(continued.model.lstmWeights).toEqual(train ? expect.not.arrayContaining(saved.model.lstmWeights) : saved.model.lstmWeights);
    const reopened = DrivingPolicy.restore(JSON.parse(JSON.stringify(continued)));
    expect(reopened.matches(run.fixture.observation)).toBe(true);
    expect(reopened.matches(input.observation!)).toBe(true);
    expect(reopened.steps).toBe(run.policy.steps);
});

async function plannedDecision(run: ReturnType<typeof setup>, direction = 1) {
    await run.policy.initializeGpu();
    const decision = await run.policy.decideAsync(new DrivingMemory(run.policy.configuration).encode(run.fixture.observation), false);
    decision.plan.forEach((slot, step) => {
        slot.action = { pedal: Math.min(0.98, 0.2 + step * 0.02), steering: direction * Math.min(0.98, (step + 1) * 0.02), shiftUp: false, shiftDown: false };
        slot.latent = [Math.atanh(slot.action.pedal), Math.atanh(slot.action.steering)];
        slot.shift = 0;
        // These deterministic fixtures edit controls directly. Keep the policy
        // intent consistent with those edits for remaining-plan comparisons.
        Object.defineProperty(slot, 'means', { get: () => [Math.atanh(slot.action.pedal), Math.atanh(slot.action.steering)] });
        Object.defineProperty(slot, 'shiftProbabilities', { get: () => [0, 1, 2].map((choice) => Number(choice === slot.shift)) });
    });
    return decision;
}

it.each([true, false])('uses current speed to limit extra throttle on an active plan during training (training=%s)', async (train) => {
    const run = setup(), { observation } = run.fixture;
    run.policy.configuration.learning.throttleExplorationSpeedLimitKmh = 120;
    const decision = await plannedDecision(run);
    decision.plan = decision.plan.map((point, step) => ({ ...point,
        means: [Math.atanh([0.1, 0.2, -0.1][step] ?? 0.2), point.means[1]] }));
    decision.plan[3].action.pedal = -0.4;
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    await run.loop.start(run.policy, train); await flush();
    expect(run.loop.state.action.throttle).toBeCloseTo(0.2);

    const updateSpeed = async (speed: number, time: number) => {
        run.input({ observation, telemetry: { ...observation, speed, measuredAt: time } }, time);
        jest.advanceTimersByTime(20); await flush();
    };
    await updateSpeed(120, 1050);
    expect(run.loop.state.action.throttle).toBeCloseTo(train ? 0.15 : 0.21);
    expect(run.loop.state.action.steering).toBeCloseTo(0.03);
    expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
    await updateSpeed(130, 1200);
    expect(run.loop.state.action.throttle).toBeCloseTo(train ? 0 : 0.24);
    expect(run.loop.state.action.brake).toBe(0);
    await updateSpeed(130, 1300);
    expect(run.loop.state.action.brake).toBeCloseTo(0.4);
    await updateSpeed(119, 1400);
    expect(run.loop.state.action.throttle).toBeCloseTo(0.28);
    expect(decide).toHaveBeenCalledTimes(1);
    await run.loop.stop();
});

it.each([true, false])('checks current gear before executing planned upshifts (training=%s)', async (train) => {
    const run = setup(), observation = { ...run.fixture.observation, gear: 2 };
    run.policy.configuration.learning.maxExplorationGear = 3;
    const decision = await plannedDecision(run);
    [1, 1, 2, 1].forEach((shift, step) => {
        const point = decision.plan[step];
        point.shift = shift as 1 | 2;
        point.action = { ...point.action, shiftUp: shift === 1, shiftDown: shift === 2 };
    });
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    run.input({ observation }, 1000);
    await run.loop.start(run.policy, train); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: true }));

    const updateGear = async (gear: number, time: number) => {
        run.input({ observation, telemetry: { ...observation, gear, measuredAt: time } }, time);
        jest.advanceTimersByTime(20); await flush();
    };
    await updateGear(3, 1100);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: !train, shiftDown: false }));
    await updateGear(4, 1200);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: false, shiftDown: true }));
    await updateGear(2, 1300);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: true, shiftDown: false }));
    expect(decide).toHaveBeenCalledTimes(1);
    await run.loop.stop();
});

it.each([true, false])('keeps an agreeing plan clock and credits its original actions while inference continues (training=%s)', async (train) => {
    const run = setup(), first = await plannedDecision(run), confirmation = await plannedDecision(run);
    confirmation.plan.forEach((point, step) => {
        point.action = { ...first.plan[Math.min(step + 2, first.plan.length - 1)].action };
    });
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first).mockResolvedValueOnce(confirmation);
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, train); await flush();
    const published = run.loop.state.plan;
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(20); await flush();
    expect(decide).toHaveBeenCalledTimes(2);
    expect(run.loop.state.plan).toBe(published);
    expect(confirmation.gpu.released).toBe(true);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1300);
    jest.advanceTimersByTime(20); await flush();
    expect(run.loop.state.action).toEqual(drivingControllerAction(first.plan[3].action));
    expect(learn).toHaveBeenCalledTimes(train ? 1 : 0);
    expect(learn.mock.calls.every(([decision]) => decision === first)).toBe(true);
    await run.loop.stop();
});

it.each([true, false].flatMap((train) => [-1, 1].map((direction) => ({ train, direction }))))(
    'preserves full steering across speed changes during pending inference (training=$train, direction=$direction)', async ({ train, direction }) => {
        const run = setup();
        const decision = await plannedDecision(run);
        const requested = { throttle: 0.7, brake: 0, steering: direction, shiftUp: false, shiftDown: false };
        decision.plan.forEach((slot) => { slot.action = { pedal: requested.throttle, steering: direction, shiftUp: false, shiftDown: false }; });
        let finish!: (value: typeof decision) => void;
        const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision)
            .mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
        run.input({ observation: { ...run.fixture.observation, speed: 0 } }, 1000);
        await run.loop.start(run.policy, train); await flush();
        expect(run.controller.apply).toHaveBeenLastCalledWith(requested);
        run.input({ observation: { ...run.fixture.observation, capturedAt: 1100, speed: 100 } }, 1100);
        jest.advanceTimersByTime(20); await flush();
        expect(decide).toHaveBeenCalledTimes(2);
        expect(run.loop.state.action).toEqual(requested);
        for (const [at, speed] of [[1120, 200], [1140, 300], [1160, 50], [1180, 0]]) {
            // Telemetry changes while the camera frame and inference remain pending.
            run.input({ observation: { ...run.fixture.observation, capturedAt: 1100, speed } }, at);
            jest.advanceTimersByTime(20); await flush();
            expect(run.loop.state.action).toEqual(requested);
            expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
        }
        expect(decision.plan.every(({ action }) => action.steering === direction)).toBe(true);
        run.input({ observation: { ...run.fixture.observation, capturedAt: 1100, speed: 200 } }, 1200);
        jest.advanceTimersByTime(20); await flush();
        finish(decision); await flush();
        expect(run.controller.apply).toHaveBeenLastCalledWith(requested);
        await run.loop.stop();
    });

it('records uncapped controls for reward credit while retaining the sampled policy actions', async () => {
    const run = setup();
    const decision = await plannedDecision(run);
    decision.plan.forEach((slot) => {
        slot.action.steering = 0.8;
        slot.latent[1] = Math.atanh(0.8);
    });
    jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    const learn = collectedTransitions(run.policy);
    run.input({ observation: { ...run.fixture.observation, speed: 200 } }, 1000);
    await run.loop.start(run.policy, true); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100, speed: 200 } }, 1100);
    jest.advanceTimersByTime(20); await flush();
    expect(learn).toHaveBeenCalled();
    expect(learn.mock.calls[0][4].every(({ action }) => action?.steering === 0.8)).toBe(true);
    expect(decision.plan[0].action.steering).toBe(0.8);
    expect(decision.plan[0].latent[1]).toBe(Math.atanh(0.8));
    await run.loop.stop();
});

it.each([true, false])('interpolates due slots, skips missed slots and holds the final controls (training=%s)', async (train) => {
    const run = setup();
    const decision = await plannedDecision(run);
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    await run.loop.start(run.policy, train); await flush();
    for (const [at, position] of [[1099, 0.99], [1100, 1], [1200, 2], [1380, 3.8], [4000, 30], [4500, 35], [6000, 50], [6500, 50]]) {
        run.input({ observation: run.fixture.observation }, at);
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state.action.throttle).toBeCloseTo(Math.min(0.98, 0.2 + position * 0.02));
        expect(run.loop.state.action.steering).toBeCloseTo(Math.min(0.98, (position + 1) * 0.02));
        expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
    }
    expect(decide).toHaveBeenCalledTimes(1);
    expect(run.controller.apply.mock.calls.some(([action]) => action.steering === decision.plan[4].action.steering)).toBe(false);
    await run.loop.stop();
    expect(decision.gpu.released).toBe(true);
});

it.each([true, false])('interpolates signed pedal and steering every 20 ms with manual polling=%s', async (withManual) => {
    const run = setup(withManual);
    const decision = await plannedDecision(run);
    decision.plan[0].action = { pedal: 0.8, steering: -0.8 };
    decision.plan[1].action = { pedal: -0.8, steering: 0.8, shiftUp: true };
    decision.plan[1].shift = 1;
    jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    await run.loop.start(run.policy, false); await flush();
    for (let elapsed = 20; elapsed <= 100; elapsed += 20) {
        run.input({ observation: run.fixture.observation }, 1000 + elapsed);
        jest.advanceTimersByTime(20); await flush();
        const action = run.loop.state.action, progress = elapsed / 100;
        const pedal = 0.8 - 1.6 * progress;
        expect(action.throttle).toBeCloseTo(Math.max(0, pedal));
        expect(action.brake).toBeCloseTo(Math.max(0, -pedal));
        expect(action.throttle * action.brake).toBe(0);
        expect(action.steering).toBeCloseTo(-0.8 + 1.6 * progress);
        expect(action.shiftUp).toBe(elapsed === 100);
        expect(action.shiftDown).toBe(false);
        expect(run.controller.apply).toHaveBeenLastCalledWith(action);
    }
    expect(run.controller.apply).toHaveBeenCalledTimes(6);
    expect(run.loop.state.plan?.[0].action).toEqual(decision.plan[0].action);
    await run.loop.stop();
});

it('uses configured knot times and never repeats a shift while interpolating within its slot', async () => {
    const run = setup();
    run.policy.configuration.actions.planTimesSeconds = [0, 0.2, 1];
    const decision = await plannedDecision(run);
    decision.plan[0].action.steering = -1;
    decision.plan[1].action = { pedal: 1, steering: 1, shiftUp: true };
    decision.plan[1].shift = 1;
    decision.plan[2].action.steering = -1;
    jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    await run.loop.start(run.policy, false); await flush();
    for (const [at, steering, shiftUp] of [[1100, 0, false], [1200, 1, true], [1300, 0.75, false],
        [1600, 0, false], [1800, -0.5, false], [2200, -1, false]] as const) {
        run.input({ observation: run.fixture.observation }, at);
        jest.advanceTimersByTime(20); await flush();
        expect(run.loop.state.action.steering).toBeCloseTo(steering);
        expect(run.loop.state.action.shiftUp).toBe(shiftUp);
    }
    await run.loop.stop();
});

it('credits both analog endpoints by their acknowledged contribution without crediting a future shift', async () => {
    const run = setup();
    const decision = await plannedDecision(run);
    decision.plan[1].shift = 1; decision.plan[1].action.shiftUp = true;
    jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, true); await flush();
    for (const at of [1020, 1040, 1060, 1080]) {
        run.input({ observation: run.fixture.observation }, at);
        jest.advanceTimersByTime(20); await flush();
    }
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(20); await flush();
    expect(learn).toHaveBeenCalledTimes(5);
    expect(learn.mock.calls.map(([sample]) => sample)).toEqual(Array(5).fill(decision));
    expect(learn.mock.calls.map(([, , , elapsed]) => elapsed)).toEqual(Array(5).fill(0.1));
    expect(learn.mock.calls.reduce((sum, [, reward]) => sum + reward, 0)).toBeCloseTo(run.loop.state.reward!.total);
    const credits = learn.mock.calls.flatMap(([, , , , segment]) => segment);
    const weight = (step: number, key: 'weight' | 'shiftWeight') => credits.filter((credit) => credit.step === step)
        .reduce((sum, credit) => sum + credit[key]!, 0) / 5; // Equal 20 ms segments.
    expect(weight(0, 'weight')).toBeCloseTo(0.6);
    expect(weight(1, 'weight')).toBeCloseTo(0.4);
    expect(weight(0, 'shiftWeight')).toBeCloseTo(1);
    expect(weight(1, 'shiftWeight')).toBe(0);
    expect(credits.filter(({ step }) => step === 1).every(({ shifts }) => !shifts.up && !shifts.down)).toBe(true);
    expect(credits.find(({ appliedAt }) => appliedAt === 1040)?.action?.steering).toBeCloseTo(0.028);
    await run.loop.stop();
});

it('starts interpolation at the first acknowledgement and credits a delayed command only after acknowledgement', async () => {
    const run = setup();
    const decision = await plannedDecision(run);
    jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    const learn = collectedTransitions(run.policy);
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    await run.loop.start(run.policy, true); await flush();
    run.input({ observation: run.fixture.observation }, 1100);
    acknowledge(); await flush();
    expect(run.loop.state.action).toEqual(drivingControllerAction(decision.plan[0].action));
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.input({ observation: run.fixture.observation }, 1120);
    jest.advanceTimersByTime(20); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ steering: 0.024 }));
    run.input({ observation: run.fixture.observation }, 1160);
    acknowledge(); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(20); await flush();
    expect(learn).toHaveBeenCalledTimes(2);
    expect(learn.mock.calls.map(([, , , elapsed]) => elapsed)).toEqual([0.2, 0.2]);
    const total = run.loop.state.reward!.total;
    expect(learn.mock.calls[0][1]).toBeCloseTo(total * 0.6);
    expect(learn.mock.calls[1][1]).toBeCloseTo(total * 0.4);
    const credits = learn.mock.calls.flatMap(([, , , , segment]) => segment);
    expect(credits.map(({ appliedAt }) => appliedAt)).toEqual([1100, 1160, 1160]);
    expect(learn.mock.calls[0][4].map(({ step, weight }) => [step, weight])).toEqual([[0, 1]]);
    // The second command used the 1120 ms interpolation position but only earns
    // its 40 ms acknowledged share of reward after the write completes at 1160.
    expect(learn.mock.calls[1][4].find(({ step }) => step === 1)?.weight).toBeCloseTo(0.2);
    await run.loop.stop();
});

it.each(['learning', 'inference'] as const)('advances the new plan during learning and the old plan during inference (pending=%s)', async (stage) => {
    const run = setup();
    const first = await plannedDecision(run), second = await plannedDecision(run, -1);
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first);
    await run.loop.start(run.policy, true); await flush();
    let finish!: () => void;
    if (stage === 'learning') {
        jest.spyOn(run.policy, 'learnAsync').mockImplementationOnce((decision) => {
            expect(decision.input).toEqual(first.input);
            expect(run.loop.state.plan?.[0].action).toEqual(second.plan[0].action);
            return new Promise<void>((resolve) => { finish = resolve; });
        });
        decide.mockResolvedValueOnce(second);
    } else decide.mockImplementationOnce(() => new Promise((resolve) => { finish = () => resolve(second); }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    if (stage === 'learning') expect(run.loop.state.plan?.[0].action).toEqual(second.plan[0].action);
    for (const at of [1500, 1550, 2000]) {
        const position = (at - (stage === 'learning' ? 1200 : 1000)) / 100;
        run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, at);
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state.action.throttle).toBeCloseTo(0.2 + position * 0.02);
        expect(run.loop.state.action.steering).toBeCloseTo((stage === 'learning' ? -1 : 1) * (position + 1) * 0.02);
        expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
    }
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 2100);
    finish(); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(drivingControllerAction(second.plan[stage === 'learning' ? 8 : 0].action));
    expect(run.loop.state.plan?.[0].action).toEqual(second.plan[0].action);
    expect(first.gpu.released).toBe(true); // Later consequences replay the saved input with updated weights.
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 2199);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action.steering).toBeCloseTo(stage === 'learning' ? -0.2198 : -0.0398);
    expect(run.loop.state.action.throttle).toBeCloseTo(stage === 'learning' ? 0.3998 : 0.2198);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 2200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(drivingControllerAction(second.plan[stage === 'learning' ? 10 : 1].action));
    await run.loop.stop();
});

it('rewards every executed slot by acknowledged duration and excludes a skipped slot', async () => {
    const run = setup(), config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1, config);
    const decision = await plannedDecision(run);
    jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, true); await flush();
    for (const at of [1100, 1200, 1400, 1500, 1600]) {
        run.input({ observation: run.fixture.observation }, at);
        jest.advanceTimersByTime(100); await flush();
    }
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1800 } }, 1800);
    jest.advanceTimersByTime(100); await flush();
    expect(learn).toHaveBeenCalledTimes(6);
    expect(learn.mock.calls.map(([sample]) => sample)).toEqual(Array(6).fill(decision));
    expect(learn.mock.calls.map(([, , , elapsed]) => elapsed)).toEqual(Array(6).fill(0.8));
    const shares = [0.125, 0.125, 0.25, 0.125, 0.125, 0.25];
    learn.mock.calls.forEach(([, reward], index) => expect(reward).toBeCloseTo(run.loop.state.reward!.total * shares[index]));
    expect(learn.mock.calls.flatMap(([, , , , credits]) => credits.map(({ step, weight }) => [step, weight])))
        .toEqual([[0, 1], [1, 1], [2, 1], [4, 1], [5, 1], [6, 1]]);
    await run.loop.stop();
});

it.each([true, false].flatMap((train) => [20, 100].map((pollMs) => ({ train, pollMs }))))(
    'scores steering and pedal reversals between camera frames (training=$train, poll=$pollMs ms)', async ({ train, pollMs }) => {
    const run = setup(), first = await plannedDecision(run), second = await plannedDecision(run);
    const held = { steering: 0.4, pedal: 0.8 };
    first.plan.forEach((slot, step) => { slot.action = step === 1 ? { steering: -0.4, pedal: -0.6 } : held; });
    second.plan.forEach((slot) => { slot.action = held; });
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first).mockResolvedValue(second);
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, train); await flush();
    for (let at = 1000 + pollMs; at < 1400; at += pollMs) {
        run.input({ observation: run.fixture.observation }, at);
        jest.advanceTimersByTime(pollMs); await flush();
        const steering = at < 1100 ? 0.4 - 0.8 * (at - 1000) / 100
            : at < 1200 ? -0.4 + 0.8 * (at - 1100) / 100 : 0.4;
        expect(run.loop.state.action.steering).toBeCloseTo(steering);
    }
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    const reward = run.loop.state.reward!;
    expect(reward.steeringChange).toBeCloseTo(-0.064);
    const throttlePenalty = pollMs === 20 ? -0.1072 : -0.064;
    const brakePenalty = pollMs === 20 ? -0.0792 : -0.036;
    expect(reward.throttleChange).toBeCloseTo(throttlePenalty);
    expect(reward.brakeChange).toBeCloseTo(brakePenalty);
    expect(reward.total).toBeCloseTo(-0.16 - 0.064 + throttlePenalty + brakePenalty);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    expect(learn).toHaveBeenCalledTimes(train ? 400 / pollMs : 0);
    if (train) {
        expect(learn.mock.calls.reduce((sum, [, value]) => sum + value, 0)).toBeCloseTo(reward.total);
        for (const [, value] of learn.mock.calls) expect(value).toBeCloseTo(reward.total * pollMs / 400);
    }
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward).toBe(reward);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1600 } }, 1600);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.steeringChange).toBe(0);
    expect(run.loop.state.reward!.throttleChange).toBe(0);
    expect(run.loop.state.reward!.brakeChange).toBe(0);
    expect(run.loop.state.action.steering).toBe(0.4);
    expect(decide).toHaveBeenCalledTimes(3);
    await run.loop.stop();
});

it('charges a replacement plan for the change from the last acknowledged outgoing command', async () => {
    const run = setup(), first = await plannedDecision(run), second = await plannedDecision(run);
    first.plan.forEach((slot) => { slot.action = { steering: 0.4, pedal: 0.7 }; });
    second.plan.forEach((slot) => { slot.action = { steering: -0.4, pedal: -0.5 }; });
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first);
    await run.loop.start(run.policy, true); await flush();
    let finish!: () => void;
    decide.mockImplementationOnce(() => new Promise((resolve) => { finish = () => resolve(second); }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.steeringChange).toBe(0);
    expect(run.loop.state.reward!.throttleChange).toBe(0);
    expect(run.loop.state.reward!.brakeChange).toBe(0);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1300);
    jest.advanceTimersByTime(100); await flush();
    finish(); await flush();
    expect(run.loop.state.action.steering).toBe(-0.4);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1500 } }, 1500);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.steeringChange).toBeCloseTo(-0.032);
    expect(run.loop.state.reward!.throttleChange).toBeLessThan(0);
    expect(run.loop.state.reward!.brakeChange).toBeLessThan(0);
    expect(run.loop.state.reward!.throttleChange / run.loop.state.reward!.brakeChange).toBeCloseTo(0.7 ** 2 / 0.5 ** 2);
    await run.loop.stop();
});

it.each([0.001, 0.8])('splits outgoing plan credit during inference including tails after an update (sequence=%s)', async (sequenceSeconds) => {
    const run = setup();
    run.policy.configuration.learning.sequenceSeconds = sequenceSeconds;
    const first = await plannedDecision(run), second = await plannedDecision(run, -1);
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first);
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, true); await flush();
    let finish!: () => void;
    decide.mockImplementationOnce(() => new Promise((resolve) => { finish = () => resolve(second); }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1500);
    jest.advanceTimersByTime(100); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1600);
    finish(); await flush();
    learn.mockClear();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1800, normalizedPosition: 0.252 } }, 1800);
    jest.advanceTimersByTime(100); await flush();
    if (sequenceSeconds === 0.8) {
        expect(learn).toHaveBeenCalledTimes(1); // The tail's own horizon has not matured.
        run.loop.setControlSource('manual'); await flush(); // Close its partial window.
    }
    expect(learn).toHaveBeenCalledTimes(sequenceSeconds === 0.8 ? 4 : 3);
    expect(learn.mock.calls.map(([decision]) => decision)).toEqual(sequenceSeconds === 0.8
        ? [first, first, first, second] : [first, first, second]);
    const total = run.loop.state.reward!.total;
    const tails = learn.mock.calls.slice(-3, -1);
    expect(tails[0][1]).toBeCloseTo(total / 2);
    expect(tails[1][1]).toBeCloseTo(total / 6);
    expect(tails.map(([, , , elapsed]) => elapsed)).toEqual([0.6, 0.6]);
    expect(tails.map(([, , , , credits]) => credits.map(({ step, weight }) => [step, weight])))
        .toEqual([[[2, 1]], [[5, 1]]]);
    const last = learn.mock.calls[learn.mock.calls.length - 1];
    expect(last[1]).toBeCloseTo(total / 3);
    expect(last[4].map(({ step, weight }) => [step, weight])).toEqual([[0, 1]]);
    expect(first.gpu.released).toBe(true);
    await run.loop.stop();
});

it.each([true, false])('executes and credits rapid repeated and alternating AI shifts without cooldown (training=%s)', async (train) => {
    const run = setup(), config = cloneConfig();
    config.actions.planTimesSeconds = [0, 0.02, 0.04, 0.06, 0.08];
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1, config);
    const decision = await plannedDecision(run);
    const choices = [1, 1, 2, 2, 1] as const;
    decision.plan.forEach((slot, step) => {
        slot.shift = choices[step];
        slot.action.shiftUp = slot.shift === 1;
        slot.action.shiftDown = slot.shift === 2;
    });
    jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(decision);
    const learn = collectedTransitions(run.policy);
    await run.loop.start(run.policy, train); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    for (let step = 1; step < choices.length; step++) {
        run.input({ observation: run.fixture.observation }, 1000 + step * 20);
        jest.advanceTimersByTime(20); await flush();
        expect(run.controller.apply.mock.calls.slice(-2)).toEqual([
            [expect.objectContaining({ shiftUp: false, shiftDown: false })], [drivingControllerAction(decision.plan[step].action)],
        ]);
    }
    run.input({ observation: run.fixture.observation }, 1180);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: false });
    expect(run.controller.apply.mock.calls.filter(([action]) => action.shiftUp || action.shiftDown)).toHaveLength(5);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(20); await flush();
    expect(learn).toHaveBeenCalledTimes(train ? 6 : 0);
    if (train) {
        expect(learn.mock.calls.map(([sample]) => sample)).toEqual(Array(6).fill(decision));
        expect(learn.mock.calls.map(([, , , elapsed]) => elapsed)).toEqual(Array(6).fill(0.2));
        expect(learn.mock.calls.reduce((sum, [, reward]) => sum + reward, 0)).toBeCloseTo(run.loop.state.reward!.total);
        const credits = learn.mock.calls.flatMap(([, , , , segment]) => segment);
        expect(Array.from(new Set(credits.map(({ step }) => step)))).toEqual([0, 1, 2, 3, 4]);
        expect(credits.every(({ step, shift, shifts }) => shift === choices[step] && shifts.up && shifts.down)).toBe(true);
    }
    await run.loop.stop();
});

it.each([true, false])('allows a fresh AI decision to shift before the previous pulse ends (training=%s)', async (train) => {
    const run = setup(), config = cloneConfig();
    config.actions.planTimesSeconds = [0];
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1, config);
    const first = await plannedDecision(run), second = await plannedDecision(run);
    first.plan[0].shift = 1; first.plan[0].action.shiftUp = true;
    second.plan[0].shift = 2; second.plan[0].action.shiftDown = true;
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    await run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1020 } }, 1020);
    jest.advanceTimersByTime(20); await flush();
    expect(decide).toHaveBeenCalledTimes(2);
    expect(run.controller.apply.mock.calls.slice(-2)).toEqual([
        [drivingControllerAction({ ...first.plan[0].action, shiftUp: false, shiftDown: false })],
        [drivingControllerAction(second.plan[0].action)],
    ]);
    await run.loop.stop();
});

it('finishes an in-flight plan write and releases its paddle before shifting with the replacement plan', async () => {
    const run = setup();
    const first = await plannedDecision(run), second = await plannedDecision(run, -1);
    first.plan[5].shift = 1; first.plan[5].action.shiftUp = true;
    second.plan[0].shift = 2; second.plan[0].action.shiftDown = true;
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first);
    await run.loop.start(run.policy, true); await flush();
    let ready!: () => void, acknowledge!: () => void;
    decide.mockImplementationOnce(() => new Promise((resolve) => { ready = () => resolve(second); }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1500);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(drivingControllerAction(first.plan[5].action));
    ready(); await flush();
    expect(run.loop.state.plan?.[0].action).toEqual(first.plan[0].action);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1700);
    acknowledge(); await flush();
    expect(run.loop.state.plan?.[0].action).toEqual(second.plan[0].action);
    expect(run.controller.apply.mock.calls.slice(-2)).toEqual([
        [drivingControllerAction({ ...first.plan[5].action, shiftUp: false, shiftDown: false })],
        [drivingControllerAction(second.plan[0].action)],
    ]);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1800);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(drivingControllerAction({ ...second.plan[1].action, shiftUp: false, shiftDown: false }));
    await run.loop.stop();
});

it('does not re-press a released paddle when a new output arrives during its acknowledgement', async () => {
    const run = setup();
    const first = await plannedDecision(run), second = await plannedDecision(run, -1);
    first.plan[5].shift = 1; first.plan[5].action.shiftUp = true;
    const decide = jest.spyOn(run.policy, 'decideAsync').mockResolvedValueOnce(first);
    await run.loop.start(run.policy, true); await flush();
    let ready!: () => void, acknowledge!: () => void;
    decide.mockImplementationOnce(() => new Promise((resolve) => { ready = () => resolve(second); }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1500);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1600);
    jest.advanceTimersByTime(100); await flush();
    ready(); await flush();
    acknowledge(); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(drivingControllerAction(second.plan[0].action));
    expect(run.controller.apply.mock.calls.filter(([action]) => action.shiftUp)).toHaveLength(1);
    await run.loop.stop();
});

it.each([true, false])('profiles model work on fresh AI cycles and retains it while waiting (training=%s)', async (train) => {
    const run = setup();
    let elapsed = 0;
    jest.spyOn(performance, 'now').mockImplementation(() => elapsed);
    const encode = DrivingMemory.prototype.encode;
    jest.spyOn(DrivingMemory.prototype, 'encode').mockImplementation(function (this: DrivingMemory, observation) {
        elapsed += 2;
        return encode.call(this, observation);
    });
    const decide = run.policy.decideAsync.bind(run.policy);
    jest.spyOn(run.policy, 'decideAsync').mockImplementation(async (...args) => { elapsed += 7; return decide(...args); });
    const learn = run.policy.trainBatchAsync.bind(run.policy);
    jest.spyOn(run.policy, 'trainBatchAsync').mockImplementation(async (trainBatch, onSubmitted) => { elapsed += 11; await learn(trainBatch, onSubmitted); });
    run.controller.apply.mockImplementation(async () => { elapsed += 3; });
    expect(run.loop.getModelTimingSnapshot()).toBeUndefined();
    await run.loop.start(run.policy, train); await flush();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    jest.mocked(gpu.finishTraining).mockImplementation(async () => {
        elapsed += 17;
        expect(run.loop.getModelTimingSnapshot()).toMatchObject({ currentStage: 'learningWait', stageElapsedMs: 17,
            stages: { learning: 11 }, replayCount: 0, historyLength: 1 });
    });
    const first = run.loop.getModelTimingSnapshot();
    expect(first).toMatchObject({ capturedAt: 1000, training: train, source: 'ai', state: 'complete', currentStage: null,
        stages: { inference: 7, control: 3 }, totalMs: 10, replayCount: 0, historyLength: 0 });
    expect(first?.stages.memory).toBeUndefined(); // History now belongs to independent collection.
    expect(first?.stages.learning).toBeUndefined();
    expect(first?.stages.learningWait).toBeUndefined();
    jest.advanceTimersByTime(200); await flush();
    expect(run.loop.getModelTimingSnapshot()).toEqual(first);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    const second = run.loop.getModelTimingSnapshot();
    expect(second).toMatchObject({ capturedAt: 1200, state: 'complete', totalMs: train ? 35 : 7, replayCount: 0, historyLength: 1 });
    expect(second?.stages.control).toBe(0); // Agreement keeps the acknowledged plan.
    expect(second?.stages.learning).toBe(train ? 11 : undefined);
    expect(second?.stages.learningWait).toBe(train ? 17 : undefined);
    run.input({ error: 'Waiting for a fresh image.' }, 1300);
    jest.advanceTimersByTime(200); await flush();
    expect(run.loop.getModelTimingSnapshot()).toEqual(second);
    await run.loop.stop();
    expect(run.loop.getModelTimingSnapshot()).toEqual(second);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400 } }, 1400);
    await run.loop.start(run.policy, false); await flush();
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ capturedAt: 1400, training: false, state: 'complete' });
    expect(run.loop.getModelTimingSnapshot()?.stages.learning).toBeUndefined();
    expect(run.loop.getModelTimingSnapshot()?.stages.learningWait).toBeUndefined();
    expect(run.loop.getModelTimingSnapshot()?.historyLength).toBe(0);
    await run.loop.stop();
});

it('profiles human demonstration learning without reporting AI inference or output', async () => {
    const run = shortHumanPlan();
    let elapsed = 0;
    jest.spyOn(performance, 'now').mockImplementation(() => elapsed);
    jest.spyOn(run.policy, 'learnDemonstratedPlanAsync').mockImplementation(async () => { elapsed += 13; });
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    expect(run.loop.getModelTimingSnapshot()).toBeUndefined();
    await run.loop.start(run.policy, true); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    const timing = run.loop.getModelTimingSnapshot();
    expect(timing).toMatchObject({ source: 'manual', training: true, state: 'complete',
        capturedAt: 1000, stages: { learning: 13, learningWait: 0 }, totalMs: 13, replayCount: 0, historyLength: 0 });
    expect(timing?.stages.inference).toBeUndefined();
    expect(timing?.stages.control).toBeUndefined();
    await run.loop.stop();
});

it('profiles the buffered origin history without historical camera inputs or incremental replays', async () => {
    const run = setup(true), config = cloneConfig();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    config.memory.features = ['speed'];
    config.actions.planTimesSeconds = [0, 0.1];
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.7, config);
    const replay = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    for (const [capturedAt, historyLength] of [[1200, 0], [1400, 1], [1600, 2], [1800, 2]]) {
        replay.mockClear();
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
        expect(run.loop.getModelTimingSnapshot()).toMatchObject({ capturedAt: capturedAt - 200,
            state: 'complete', historyLength, replayCount: 0 });
        expect(replay).toHaveBeenCalledTimes(1);
    }
    const last = run.loop.getModelTimingSnapshot();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 2000 } }, 2000);
    jest.advanceTimersByTime(200); await flush();
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ capturedAt: 1800, historyLength: 3, replayCount: 0 });
    expect(last).toMatchObject({ capturedAt: 1600, historyLength: 2, replayCount: 0 });
    await run.loop.stop();
});

it('retains separate CPU and wait timings when GPU completion fails', async () => {
    const run = setup();
    let elapsed = 0;
    jest.spyOn(performance, 'now').mockImplementation(() => elapsed);
    await run.loop.start(run.policy, true); await flush();
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    jest.mocked(gpu.finishTraining).mockImplementation(async () => { elapsed += 19; throw new Error('GPU lost'); });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ state: 'incomplete', currentStage: null,
        stages: { learning: 0, learningWait: 19 }, totalMs: 19, replayCount: 0, historyLength: 1 });
    expect(run.loop.getModelTimingSnapshot()?.stages.inference).toBe(0); // Inference preceded the failed learning wait.
    expect(run.loop.state).toMatchObject({ mode: 'idle', status: 'GPU lost' });
});

it('keeps a failed model cycle incomplete and clears its running stage', async () => {
    const run = setup();
    jest.spyOn(run.policy, 'decideAsync').mockRejectedValue(new Error('GPU lost'));
    await run.loop.start(run.policy, false); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', status: 'GPU lost' });
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ state: 'incomplete', currentStage: null,
        stages: { inference: expect.any(Number) } });
    expect(run.loop.getModelTimingSnapshot()?.stages.control).toBeUndefined();
});

it.each([true, false])('scores progress once per fresh frame without reference-line rewards (training=%s)', async (train) => {
    const run = setup();
    const config = cloneConfig();
    config.reward.progressWeight = 200;
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1, config);
    const learn = collectedTransitions(run.policy);
    run.input({ observation: run.fixture.observation }, 1000);
    run.loop.start(run.policy, train); await flush();
    let episodeReward = 0;
    for (let index = 0; index < 10; index++) {
        const capturedAt = 1200 + index * 200;
        run.input({ observation: { ...run.fixture.observation, capturedAt,
            normalizedPosition: 0.25 + (capturedAt - 1000) / 100000 } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
        expect(run.loop.state.reward?.progress).toBeCloseTo(0.05);
        expect(run.loop.state.reward).not.toHaveProperty('line');
        expect(run.loop.state.reward).not.toHaveProperty('approach');
        const scored = run.loop.state.reward!;
        expect(scored.total).toBeCloseTo(scored.timing + 0.05 + scored.steeringChange + scored.throttleChange + scored.brakeChange);
        episodeReward += run.loop.state.reward!.total;
        expect(run.loop.state.episodeReward).toBeCloseTo(episodeReward);
        expect(learn.mock.calls.slice(-1)).toEqual(train ? [[expect.anything(), run.loop.state.reward!.total,
            expect.objectContaining({ features: expect.any(Array), cameras: expect.any(Array) }), 0.2, expect.any(Array)]] : []);
        const reward = run.loop.state.reward;
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state.reward).toBe(reward); // Repeated images do not count reward again.
    }
    expect(learn).toHaveBeenCalledTimes(train ? 10 : 0);
    await run.loop.stop();
});

it.each(['stop', 'handoff', 'invalid', 'gap'] as const)('does not score progress or steering changes across %s', async (boundary) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    let steering = 0.4;
    const infer = run.policy.decideAsync.bind(run.policy);
    jest.spyOn(run.policy, 'decideAsync').mockImplementation(async (...args) => {
        const decision = await infer(...args);
        decision.plan.forEach((point) => { point.action.steering = steering; });
        return decision;
    });
    run.input({ observation: run.fixture.observation }, 1000);
    run.loop.start(run.policy, false); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200,
        normalizedPosition: 0.252 } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    expect(run.loop.state.reward?.progress).toBeCloseTo(0.025);
    const episodeReward = run.loop.state.episodeReward;
    if (boundary === 'stop') await run.loop.stop();
    if (boundary === 'handoff') { run.loop.setControlSource('manual'); run.loop.setControlSource('ai'); }
    if (boundary === 'invalid') {
        run.input({ error: 'Camera disconnected.' }, 1300);
        jest.advanceTimersByTime(100); await flush();
    }
    const restartAt = boundary === 'gap' ? 2800 : 1400;
    steering = -0.4;
    run.input({ observation: { ...run.fixture.observation, capturedAt: restartAt } }, restartAt);
    if (boundary === 'stop') run.loop.start(run.policy, false);
    jest.advanceTimersByTime(200); await flush();
    expect(run.loop.state.episodeReward).toBe(boundary === 'stop' ? 0 : episodeReward);
    run.input({ observation: { ...run.fixture.observation, capturedAt: restartAt + 200,
        normalizedPosition: 0.252 } }, restartAt + 200);
    jest.advanceTimersByTime(200); await flush();
    expect(run.loop.state.reward?.progress).toBeCloseTo(0.025);
    expect(run.loop.state.reward?.steeringChange).toBe(0);
    await run.loop.stop();
});

it.each([true, false])('learns and reports reward without a yaw reversal penalty (training=%s)', async (train) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const learn = collectedTransitions(run.policy);
    run.loop.start(run.policy, train); await flush();
    for (const [capturedAt, yaw] of [[1200, 0.3], [1400, -0.3], [1600, 0.3]]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt, yawRate: yaw } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
        const reward = run.loop.state.reward!;
        expect(reward).not.toHaveProperty('instability');
        expect(reward.progress).toBe(0);
        expect(reward.timing).toBeCloseTo(-0.08);
        expect(reward.total).toBeCloseTo(-0.08 + reward.steeringChange + reward.throttleChange + reward.brakeChange);
    }
    const reward = run.loop.state.reward!;
    expect(learn).toHaveBeenCalledTimes(train ? 3 : 0);
    expect(learn.mock.calls[learn.mock.calls.length - 1]?.[1]).toBe(train ? reward.total : undefined);
    await run.loop.stop();
});

it.each([true, false])('passes five seconds of history to decisions and critic learning (training=%s)', async (train) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const decide = jest.spyOn(run.policy, 'decideAsync');
    const learn = collectedTransitions(run.policy);
    run.loop.start(run.policy, train); await flush();
    const first = await decide.mock.results[0].value;
    expect(first.input.features.slice(0, drivingMemoryInputs())).toEqual(new Array(drivingMemoryInputs()).fill(0));
    for (let capturedAt = 1200; capturedAt <= 6000; capturedAt += 200) {
        const features = run.fixture.observation.features.slice();
        features[3] = 0.5;
        run.input({ observation: { ...run.fixture.observation, capturedAt, features } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    const current = decide.mock.calls[decide.mock.calls.length - 1][0];
    expect(current.features[16 * drivingMemorySlotInputs() + 1]).toBe(1); // Oldest input is exactly five seconds old.
    expect(current.features.slice(16 * drivingMemorySlotInputs() + 2, 17 * drivingMemorySlotInputs()))
        .toEqual(run.fixture.observation.features);
    expect(learn).toHaveBeenCalledTimes(train ? 25 : 0);
    expect(learn.mock.calls[0]?.[0]).toBe(train ? first : undefined);
    expect(learn.mock.calls[learn.mock.calls.length - 1]?.[2]).toEqual(train ? current : undefined);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 6200 } }, 6200);
    jest.advanceTimersByTime(200); await flush();
    expect(decide.mock.calls[decide.mock.calls.length - 1][0].features[16 * drivingMemorySlotInputs() + 5]).toBe(0.5);
    await run.loop.stop();
});

it.each(['stop', 'handoff', 'invalid', 'gap'] as const)('discards history after %s', async (boundary) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const decide = jest.spyOn(run.policy, 'decideAsync');
    run.manual.mockResolvedValue({ action: { throttle: 0.5, brake: 0, steering: 0 }, toggleControl: false });
    run.loop.start(run.policy, false); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    expect(decide.mock.calls[decide.mock.calls.length - 1][0].features[0]).toBe(1);
    if (boundary === 'stop') await run.loop.stop();
    if (boundary === 'handoff') {
        run.loop.setControlSource('manual');
        run.loop.setControlSource('ai');
    }
    if (boundary === 'invalid') {
        run.input({ error: 'Camera disconnected.' }, 1300);
        jest.advanceTimersByTime(100); await flush();
    }
    const capturedAt = boundary === 'gap' ? 2800 : 1400;
    run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
    if (boundary === 'stop') run.loop.start(run.policy, false);
    jest.advanceTimersByTime(200); await flush();
    expect(decide.mock.calls[decide.mock.calls.length - 1][0].features.slice(0, drivingMemoryInputs()).every((value) => value === 0)).toBe(true);
    await run.loop.stop();
});

it('learns human demonstrations with the history available before that action', async () => {
    const run = shortHumanPlan();
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    const action = { throttle: 0.7, brake: 0, steering: -0.2 };
    run.manual.mockResolvedValue({ action, toggleControl: false });
    run.loop.startManual(); await flush();
    run.loop.start(run.policy, true); await flush();
    for (const capturedAt of [1200, 1400]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    const memory = new DrivingMemory();
    memory.encode(run.fixture.observation);
    expect(demonstrate).toHaveBeenLastCalledWith(memory.encode({ ...run.fixture.observation, capturedAt: 1200 }),
        [{ step: 0, action: expect.objectContaining({ pedal: 0.7, steering: -0.2 }), shifts: { up: true, down: true } }]);
    await run.loop.stop();
});

it('keeps applying current human inputs while demonstration learning is pending', async () => {
    const run = shortHumanPlan();
    const first = { throttle: 0.7, brake: 0, steering: -0.2 };
    run.manual.mockResolvedValue({ action: first, toggleControl: false });
    run.loop.startManual(); await flush();
    await run.loop.start(run.policy, true); await flush();
    let finish!: () => void;
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync')
        .mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    const sampled = { throttle: 0.5, brake: 0, steering: 0.1 };
    run.manual.mockResolvedValue({ action: sampled, toggleControl: false });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(50); await flush();
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn).toHaveBeenLastCalledWith(expect.anything(), [expect.objectContaining({ step: 0, action: expect.objectContaining({ pedal: first.throttle, steering: first.steering }) })]);

    const latest = { throttle: 0, brake: 0.9, steering: 0.8 };
    for (const capturedAt of [1250, 1300, 1350]) {
        run.manual.mockResolvedValue({ action: latest, toggleControl: false });
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        const reads = run.manual.mock.calls.length;
        jest.advanceTimersByTime(50); await flush();
        expect(run.manual).toHaveBeenCalledTimes(reads + 1);
        expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(latest));
        expect(run.loop.state.action).toMatchObject(latest);
        expect(learn).toHaveBeenCalledTimes(1); // Collection continues while model work stays serial.
    }
    finish(); await flush();
    expect(run.loop.state.action).toMatchObject(latest); // Completion cannot restore the old controls.
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(50); await flush();
    expect(learn.mock.calls.some(([, targets]) => targets.some(({ step, action }) => step === 0
        && action.pedal === sampled.throttle && action.steering === sampled.steering))).toBe(true);
    expect(learn.mock.calls.length).toBeGreaterThanOrEqual(2);
    await run.loop.stop();
});

it('applies and releases a human paddle during learning and retains its acknowledged action', async () => {
    const run = shortHumanPlan();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    await run.loop.start(run.policy, true); await flush();
    let finish!: () => void;
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync')
        .mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(50); await flush();
    const pressed = { throttle: 0.4, brake: 0, steering: 0.3, shiftUp: true, shiftDown: false };
    run.manual.mockResolvedValue({ action: pressed, toggleControl: false });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1300);
    jest.advanceTimersByTime(50); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(pressed);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith({ ...pressed, shiftUp: false });
    expect(learn).toHaveBeenCalledTimes(1);
    finish(); await flush();
    expect(run.loop.state.action.shiftUp).toBe(false);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(50); await flush();
    expect(learn.mock.calls.some(([, targets]) => targets.some(({ step, action }) => step === 1 && action.shiftUp))).toBe(true);
    await run.loop.stop();
});

it('collects complete demonstrated futures and individual shifts while GPU completion is blocked', async () => {
    const run = setup(true);
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    await run.loop.start(run.policy, true); await flush();
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    const gpu = await jest.mocked(DrivingGpuModel.create).mock.results[0].value;
    let finish!: () => void;
    jest.mocked(gpu.finishTraining).mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    const actions = new Map<number, { pedal: number; steering: number; shiftUp: boolean; shiftDown: boolean }>();
    const sample = async (at: number) => {
        const action = { throttle: (at % 700) / 700, brake: 0, steering: (at % 900) / 900,
            shiftUp: at === 6300 || at === 6500, shiftDown: at === 6700 };
        actions.set(at, { pedal: action.throttle, steering: action.steering, shiftUp: action.shiftUp, shiftDown: action.shiftDown });
        run.manual.mockResolvedValue({ action, toggleControl: false });
        const features = run.fixture.observation.features.slice();
        features[features.length - 1] = at / 10000;
        run.input({ observation: { ...run.fixture.observation, features, capturedAt: at } }, at);
        jest.advanceTimersByTime(100); await flush();
        expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(action));
    };
    for (let at = 1100; at <= 6100; at += 100) await sample(at);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ currentStage: 'learningWait', capturedAt: 1000 });
    for (let at = 6200; at <= 11100; at += 100) await sample(at);
    expect(learn).toHaveBeenCalledTimes(1);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 11100 } }, 11151);
    jest.advanceTimersByTime(60); await flush();
    const latest = run.loop.state.action;
    finish();
    for (let i = 0; i < 10; i++) await flush();
    const origin = learn.mock.calls.find(([input]) => input.features[input.features.length - 1] === 0.61);
    expect(origin).toBeDefined();
    expect(origin![1]).toEqual(Array.from({ length: 51 }, (_, step) => ({ step,
        action: actions.get(6100 + step * 100), shifts: { up: true, down: true } })));
    expect(origin![1].filter(({ action }) => action.shiftUp).map(({ step }) => step)).toEqual([2, 4]);
    expect(origin![1].filter(({ action }) => action.shiftDown).map(({ step }) => step)).toEqual([6]);
    expect(run.loop.state.action).toEqual(latest);
    await run.loop.stop();
});

it('completes an existing demonstration from controls and telemetry while camera capture is unavailable', async () => {
    const run = shortHumanPlan();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    const action = { throttle: 0.2, brake: 0, steering: -0.3, shiftDown: true, shiftUp: false };
    run.manual.mockResolvedValue({ action, toggleControl: false });
    for (const at of [1100, 1200]) {
        run.input({ error: 'Waiting for a fresh image.', cameraUnavailable: true,
            telemetry: { ...run.fixture.observation, measuredAt: at } }, at);
        jest.advanceTimersByTime(100); await flush();
    }
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn.mock.calls[0]).toEqual([new DrivingMemory().encode(run.fixture.observation), [
        { step: 0, action: { pedal: 0, steering: 0, shiftUp: false, shiftDown: false }, shifts: { up: true, down: true } },
        { step: 1, action: { pedal: 0.2, steering: -0.3, shiftUp: false, shiftDown: true }, shifts: { up: true, down: true } },
    ]]);
    expect(run.loop.state).toMatchObject({ source: 'manual', action: { ...action, shiftDown: false } });
    await run.loop.stop();
});

it.each(['session', 'reference'] as const)('discards buffered futures on a %s error even when telemetry is available', async (boundary) => {
    const run = shortHumanPlan();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    run.input({ error: boundary === 'session' ? 'Live session changed.' : 'Live Map reference changed.',
        telemetry: { ...run.fixture.observation, measuredAt: 1100,
            session: boundary === 'session' ? 'different-session' : run.fixture.observation.session } }, 1100);
    jest.advanceTimersByTime(100); await flush();
    for (const capturedAt of [1200, 1400]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(100); await flush();
    }
    expect(learn).toHaveBeenCalledTimes(1);
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ capturedAt: 1200, historyLength: 0 });
    await run.loop.stop();
});

it.each(['jump', 'gap'] as const)('invalidates collected demonstrations on a telemetry %s between controller polls', async (boundary) => {
    const run = shortHumanPlan();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    run.loop.recordTelemetry(run.fixture.observation);
    const at = boundary === 'jump' ? 1100 : 1601;
    const normalizedPosition = boundary === 'jump' ? 0.5 : run.fixture.observation.normalizedPosition;
    run.input({ observation: run.fixture.observation }, at);
    run.loop.recordTelemetry({ ...run.fixture.observation, measuredAt: at, normalizedPosition });
    for (const capturedAt of [at + 100, at + 300]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt, normalizedPosition } }, capturedAt);
        jest.advanceTimersByTime(100); await flush();
    }
    expect(learn).toHaveBeenCalledTimes(1);
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ capturedAt: at + 100, historyLength: 0 });
    await run.loop.stop();
});

it.each(['jump', 'gap'] as const)('does not add a stale observation after a telemetry %s during a manual acknowledgement', async (boundary) => {
    const run = shortHumanPlan();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    run.loop.recordTelemetry({ ...run.fixture.observation, sampleIndex: 1 });
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(100); await flush();
    const at = boundary === 'jump' ? 1200 : 1601;
    const normalizedPosition = boundary === 'jump' ? 0.5 : run.fixture.observation.normalizedPosition;
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, at);
    run.loop.recordTelemetry({ ...run.fixture.observation, measuredAt: at, normalizedPosition, sampleIndex: 2 });
    acknowledge(); await flush();
    run.input({ error: 'Waiting for a fresh image.', cameraUnavailable: true,
        telemetry: { ...run.fixture.observation, measuredAt: at + 100, normalizedPosition, sampleIndex: 3 } }, at + 100);
    jest.advanceTimersByTime(100); await flush();
    expect(learn).not.toHaveBeenCalled();
    for (const capturedAt of [at + 200, at + 400]) {
        run.input({ observation: { ...run.fixture.observation, capturedAt, normalizedPosition } }, capturedAt);
        jest.advanceTimersByTime(100); await flush();
    }
    expect(learn).toHaveBeenCalledTimes(1);
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ capturedAt: at + 200, historyLength: 0 });
    await run.loop.stop();
});

it.each(['timestamp', 'sample index'] as const)('ignores a stale telemetry %s without rewinding or clearing collected futures', async (stale) => {
    const run = shortHumanPlan();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    run.loop.recordTelemetry({ ...run.fixture.observation, measuredAt: 1100, sampleIndex: 10 });
    run.loop.recordTelemetry({ ...run.fixture.observation,
        measuredAt: stale === 'timestamp' ? 500 : 1150, sampleIndex: stale === 'timestamp' ? 11 : 9,
        normalizedPosition: stale === 'timestamp' ? run.fixture.observation.normalizedPosition : 0.5 });
    run.loop.recordTelemetry({ ...run.fixture.observation, measuredAt: 1200, sampleIndex: 12 });
    run.input({ observation: run.fixture.observation }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(learn).toHaveBeenCalledTimes(1);
    expect(run.loop.getModelTimingSnapshot()).toMatchObject({ capturedAt: 1000 });
    await run.loop.stop();
});

it('invalidates a position jump in distinct telemetry samples received within the same millisecond', async () => {
    const run = shortHumanPlan();
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    const learn = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    await run.loop.start(run.policy, true); await flush();
    run.loop.recordTelemetry({ ...run.fixture.observation, sampleIndex: 1 });
    run.loop.recordTelemetry({ ...run.fixture.observation, normalizedPosition: 0.5, sampleIndex: 2 });
    run.input({ error: 'Waiting for a fresh image.', cameraUnavailable: true,
        telemetry: { ...run.fixture.observation, measuredAt: 1200, normalizedPosition: 0.5, sampleIndex: 3 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(learn).not.toHaveBeenCalled();
    await run.loop.stop();
});

it.each(['inference', 'learning'] as const)('accepts the physical takeover button while AI %s is pending', async (stage) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    let finish!: () => void;
    if (stage === 'inference') {
        await run.policy.initializeGpu();
        const decision = await run.policy.decideAsync(new DrivingMemory().encode(run.fixture.observation), false);
        jest.spyOn(run.policy, 'decideAsync').mockImplementationOnce(() => new Promise((resolve) => { finish = () => resolve(decision); }));
    }
    await run.loop.start(run.policy, true); await flush();
    if (stage === 'learning') {
        jest.spyOn(run.policy, 'learnAsync').mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
        run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
        jest.advanceTimersByTime(50); await flush();
    }
    const action = { throttle: 0, brake: 1, steering: -0.6 };
    run.manual.mockResolvedValue({ action, toggleControl: true });
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ source: 'manual', action });
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(action));
    const writes = run.controller.apply.mock.calls.length;
    finish(); await flush();
    expect(run.controller.apply).toHaveBeenCalledTimes(writes);
    expect(run.loop.state).toMatchObject({ source: 'manual', action, plan: null });
    await run.loop.stop();
});

it.each(['stop', 'disconnect', 'observations'] as const)('invalidates pending human learning on %s without delaying control', async (boundary) => {
    const run = shortHumanPlan();
    const action = { throttle: 0.4, brake: 0, steering: 0.2 };
    run.manual.mockResolvedValue({ action, toggleControl: false });
    run.loop.startManual(); await flush();
    await run.loop.start(run.policy, true); await flush();
    let finish!: () => void;
    const demonstration = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync')
        .mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve; }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(50); await flush();
    if (boundary === 'stop') await run.loop.stop();
    if (boundary === 'disconnect') run.manual.mockRejectedValue(new Error('Mapped controller disconnected.'));
    if (boundary === 'observations') run.input({ error: 'Camera disconnected.' }, 1250);
    jest.advanceTimersByTime(50); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(boundary === 'observations' ? expect.objectContaining(action) : NEUTRAL);
    const state = run.loop.state;
    const writes = run.controller.apply.mock.calls.length;
    finish(); await flush();
    expect(run.loop.state).toBe(state);
    expect(run.controller.apply).toHaveBeenCalledTimes(writes);
    expect(demonstration).toHaveBeenCalledTimes(1);
    await run.loop.stop();
});

it('drives with human input alone and requires an armed AI before returning control', async () => {
    const run = setup(true);
    const manual = { action: { throttle: 0.6, brake: 0, steering: -0.3 }, toggleControl: true };
    run.manual.mockResolvedValue(manual);
    run.input({ error: 'No camera or telemetry.' }, 1000);
    run.loop.startManual(); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'manual', source: 'manual', action: manual.action });
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(manual.action));
    expect(() => run.loop.setControlSource('ai')).toThrow(/Start training/);
    expect(run.policy.steps).toBe(0);
    run.input({ observation: run.fixture.observation }, 1100);
    run.loop.start(run.policy, true); await flush();
    // Starting training must not steal control, even with the toggle button already held.
    expect(run.loop.state).toMatchObject({ mode: 'training', source: 'manual' });
    run.manual.mockResolvedValue({ ...manual, toggleControl: false });
    jest.advanceTimersByTime(50); await flush();
    run.manual.mockResolvedValue(manual);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'training', source: 'ai', action: NEUTRAL });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    await run.loop.stop();
});

it('keeps human control through observation loss and discards the incomplete learning transition', async () => {
    const run = shortHumanPlan();
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    const manual = { action: { throttle: 0.6, brake: 0, steering: 0.1 }, toggleControl: false };
    run.manual.mockResolvedValue(manual);
    run.loop.startManual(); await flush();
    run.loop.start(run.policy, true); await flush();
    run.input({ error: 'Camera disconnected.' }, 1100);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'training', source: 'manual', action: manual.action });
    expect(run.loop.state.status).toContain('learning paused');
    expect(() => run.loop.setControlSource('ai')).toThrow('Camera disconnected');
    expect(demonstrate).not.toHaveBeenCalled();
    run.manual.mockResolvedValue({ ...manual, toggleControl: true, action: { ...manual.action, throttle: 0, brake: 0.8 } });
    jest.advanceTimersByTime(50); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ throttle: 0, brake: 0.8 }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state.source).toBe('manual'); // Recovery does not auto-resume AI.
    expect(demonstrate).not.toHaveBeenCalled();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(50); await flush();
    expect(demonstrate).toHaveBeenCalledTimes(1);
    await run.loop.stop();
});

it('accepts human takeover on the same tick that AI observations disappear', async () => {
    const run = setup(true);
    run.loop.start(run.policy, true); await flush();
    const manual = { action: { throttle: 0, brake: 1, steering: 0.2 }, toggleControl: true };
    run.manual.mockResolvedValue(manual);
    run.input({ error: 'Telemetry disconnected.' }, 1100);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ source: 'manual', mode: 'training', action: manual.action });
    expect(run.policy.steps).toBe(0);
    await run.loop.stop();
});

it('revokes an AI write immediately when a UI takeover races with its acknowledgement', async () => {
    const run = setup(true);
    const manual = { action: { throttle: 0, brake: 1, steering: -0.2 }, toggleControl: false };
    run.manual.mockResolvedValue(manual);
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.loop.start(run.policy, true); await flush();
    run.loop.setControlSource('manual');
    acknowledge(); await flush();
    expect(run.loop.state).toMatchObject({ source: 'manual', action: NEUTRAL });
    jest.advanceTimersByTime(50); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(manual.action));
    expect(run.policy.steps).toBe(0);
    await run.loop.stop();
});

it('releases standalone human control when the controller disappears', async () => {
    const run = setup(true);
    run.manual.mockResolvedValue({ action: { throttle: 1, brake: 0, steering: 1 }, toggleControl: false });
    run.loop.startManual(); await flush();
    run.manual.mockRejectedValue(new Error('Mapped controller disconnected.'));
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', source: null, action: NEUTRAL });
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
});

it.each([true, false])('routes manual control through vJoy between images and only learns demonstrations in training=%s', async (train) => {
    const run = setup(true);
    const learn = collectedTransitions(run.policy);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    const manual = { action: { throttle: 0.8, brake: 0.1, steering: -0.6 }, toggleControl: false };
    const before = (await run.policy.checkpointAsync());
    run.manual.mockResolvedValue(manual);
    run.loop.start(run.policy, train); await flush();
    const tick = async (capturedAt: number, time = capturedAt) => {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, time);
        jest.advanceTimersByTime(100); await flush();
    };
    run.manual.mockResolvedValue({ ...manual, toggleControl: true });
    await tick(1000, 1100); // Takeover must work without waiting for the next image.
    expect(run.loop.state).toMatchObject({ source: 'manual', action: manual.action });
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(manual.action));
    expect(learn).not.toHaveBeenCalled();
    await tick(1200); // First frame wholly under manual ownership.
    await tick(1300);
    expect(learn).not.toHaveBeenCalled();
    expect(demonstrate).not.toHaveBeenCalled(); // The demonstrated future is still being collected.
    run.manual.mockResolvedValue({ ...manual, action: { ...manual.action, steering: 0.4 } });
    await tick(1300, 1350);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ steering: 0.4 }));
    expect(demonstrate).not.toHaveBeenCalled();
    run.loop.setControlSource('ai');
    await tick(1300, 1400);
    expect(demonstrate).toHaveBeenCalledTimes(train ? 2 : 0); // Handoff trains the available partial futures.
    if (train) expect(demonstrate.mock.calls[0]).toEqual([
        expect.objectContaining({ features: new DrivingMemory().encode(run.fixture.observation).features }),
        expect.arrayContaining([expect.objectContaining({ step: 0, action: expect.objectContaining({ pedal: 0.8, steering: -0.6 }) })]),
    ]);
    expect(run.loop.state).toMatchObject({ source: 'ai', action: NEUTRAL });
    await tick(1500);
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(learn).not.toHaveBeenCalled(); // No mixed-driver transition on return either.
    expect((await run.policy.checkpointAsync())).toEqual(train ? expect.not.objectContaining(before) : before);
    await run.loop.stop();
});

it.each([true, false])('collects delayed observations against acknowledged controls across repeated handoffs (training=%s)', async (train) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const learn = collectedTransitions(run.policy);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    run.manual.mockResolvedValue({ action: { throttle: 0.6, brake: 0, steering: 0.1 }, toggleControl: false });
    const frame = async (capturedAt: number, time = capturedAt) => {
        const paired = { ...run.fixture.live, receivedAt: capturedAt, sampleIndex: capturedAt,
            currentTelemetry: { ...run.fixture.live.currentTelemetry, Physics_speed_kmh: 100 + capturedAt / 1000 } };
        const input = readDrivingObservation({ ...paired, receivedAt: time },
            { ...run.fixture.vision, capturedAt, telemetry: paired }, run.fixture.referenceLap, 50, time, undefined, circuitMapFixture(run.fixture.referenceLap));
        expect(input.error).toBeUndefined();
        run.input(input, time);
        jest.advanceTimersByTime(50); await flush();
        return input.observation!;
    };
    const initial = (await run.policy.checkpointAsync());
    run.loop.start(run.policy, train); await flush();
    let time = 2000;
    for (const source of ['manual', 'ai', 'manual', 'ai'] as const) {
        run.input({ observation: { ...run.fixture.observation, capturedAt: time } }, time);
        run.loop.setControlSource(source);
        await flush(); // Finish the outgoing owner's samples before the new transition.
        const steps = run.policy.steps;
        const retainedTransitions = learn.mock.calls.length;
        const replaySize = (newTransitions: number) => Math.min(run.policy.configuration.learning.replayBatchSize,
            run.policy.configuration.learning.replayCapacity, retainedTransitions + newTransitions);
        await frame(time + 100);
        expect(run.policy.steps).toBe(steps); // Never learn across the ownership change.
        if (source === 'manual') {
            for (const [capturedAt, acknowledgedAt] of [[time + 300, time + 1200], [time + 1100, time + 2000],
                [time + 1900, time + 2800], [time + 2900, time + 2900]]) await frame(capturedAt, acknowledgedAt);
            expect(run.policy.steps).toBe(steps); // Collection stays independent of action acknowledgements and GPU timing.
            expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source });
            time += 5000;
            continue;
        }
        await frame(time + 300, time + 1200);
        expect(run.policy.steps).toBe(steps + (train ? replaySize(1) : 0));
        const beforeDelayed = await run.policy.checkpointAsync();
        // An image predating the latest acknowledgement still describes the
        // outgoing plan. Collection must retain it instead of inventing a gap.
        await frame(time + 1100, time + 2000);
        expect(run.loop.state.reward).not.toBeNull();
        expect(learn.mock.calls.slice(-1)[0]?.[3]).toBe(train ? 0.8 : undefined);
        await frame(time + 1900, time + 2800);
        expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source,
            reward: expect.any(Object) });
        expect(await run.policy.checkpointAsync()).toEqual(train ? expect.not.objectContaining(beforeDelayed) : beforeDelayed);
        expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
        expect(run.loop.state.action).not.toEqual(NEUTRAL);
        await frame(time + 2900);
        expect(run.policy.steps).toBe(steps + (train ? [1, 2, 4, 7].reduce((sum, count) => sum + replaySize(count), 0) : 0));
        expect(learn.mock.calls.slice(-7).map(([, , , elapsed]) => elapsed)).toEqual(train ? [0.2, 0.8, 0.8, 0.8, 1, 1, 1] : []);
        time += 5000;
    }
    expect((await run.policy.checkpointAsync())).toEqual(train ? expect.not.objectContaining(initial) : initial);
    expect(learn).toHaveBeenCalledTimes(train ? 14 : 0);
    // Each telemetry gap invalidates the earlier human origins; only the last
    // observation in each human segment reaches the handoff with valid targets.
    expect(demonstrate).toHaveBeenCalledTimes(train ? 2 : 0);
    await run.loop.stop();
});

it.each([1500, 1501])('only skips a fresh transition beyond the 1.5-second limit (gap=%s ms)', async (gap) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    run.loop.start(run.policy, true); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1000 + gap } }, 1000 + gap);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'training', steps: gap === 1500 ? 1 : 0 });
    await run.loop.stop();
});

it.each((['ai', 'manual'] as const).flatMap((source) =>
    ['session', 'reference', 'telemetry', 'capture', 'range', 'camera', 'context', 'features'].map((fault) => ({ source, fault }))))(
    'preserves $fault validation across a long gap under $source control', async ({ source, fault }) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    run.manual.mockResolvedValue({ action: { throttle: 0.5, brake: 0, steering: 0 }, toggleControl: false });
    run.loop.start(run.policy, true); await flush();
    if (source === 'manual') run.loop.setControlSource(source);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(50); await flush();
    const before = (await run.policy.checkpointAsync());
    const next = trainingFixture(2800, 50, fault === 'reference' ? undefined : run.fixture.referenceLap);
    if (fault === 'session') {
        next.live.sessionGeneration++;
        next.vision.telemetry!.sessionGeneration++;
    }
    if (fault === 'telemetry') next.live.receivedAt = 2299;
    if (fault === 'capture') next.vision.capturedAt = 1299;
    if (fault === 'range') next.vision.telemetry = { ...next.vision.telemetry!,
        currentTelemetry: { ...next.vision.telemetry!.currentTelemetry, Physics_gas: 2 } };
    if (fault === 'camera') delete next.vision.camera;
    const input = readDrivingObservation(next.live, next.vision, next.referenceLap, 50, 2800, undefined, circuitMapFixture(next.referenceLap));
    if (fault === 'context') input.observation!.context = { ...input.observation!.context, car: 'different' };
    if (fault === 'features') input.observation!.features[0] = NaN;
    run.input(input, 2800);
    jest.advanceTimersByTime(50); await flush();
    expect((await run.policy.checkpointAsync())).toEqual(before);
    const expectedAction = source === 'ai' ? NEUTRAL : expect.objectContaining({ throttle: 0.5, brake: 0, steering: 0 });
    expect(run.loop.state).toMatchObject({ mode: 'training', source, action: expectedAction });
    const waitingForCamera = source === 'manual' && ['capture', 'camera'].includes(fault);
    expect(run.loop.state.status).toContain(source === 'ai' ? 'retrying' : waitingForCamera ? 'recording controls' : 'learning paused');
    expect(run.controller.apply).toHaveBeenLastCalledWith(expectedAction);
    expect(() => run.loop.setControlSource('ai')).toThrow();
    expect(run.loop.state.status).toMatch(/session changed|reference changed|changed during this episode|fresh, running live telemetry|fresh image|valid ranges|camera pixels|camera observations|matching model inputs|Invalid policy observation/);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 3000 } }, 3000);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'training', source });
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect((await run.policy.checkpointAsync())).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 3100 } }, 3100);
    jest.advanceTimersByTime(50); await flush();
    expect(run.policy.steps).toBe(before.model.steps + (source === 'ai' ? 2 : 0));
    await run.loop.stop();
});

it('releases vJoy when physical input disappears, including while AI owns the output', async () => {
    const run = setup(true);
    run.loop.start(run.policy, true); await flush();
    run.manual.mockRejectedValue(new Error('Mapped controller disconnected.'));
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', action: NEUTRAL });
    expect(run.loop.state.status).toContain('Mapped controller disconnected');
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it('cancels a pending physical input read when Stop is pressed', async () => {
    const run = setup(true);
    let resolve!: (input: ManualControls) => void;
    run.manual.mockImplementation(() => new Promise((done) => { resolve = done; }));
    await run.loop.start(run.policy, true);
    await run.loop.stop();
    resolve({ action: { throttle: 1, brake: 0, steering: 1 }, toggleControl: true });
    await flush();
    expect(run.controller.apply).toHaveBeenCalledTimes(1);
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it('pulses each human paddle pull once until released and pressed again', async () => {
    const run = setup(true);
    const input = { action: { throttle: 0.4, brake: 0, steering: 0, shiftUp: true }, toggleControl: true };
    run.manual.mockResolvedValue(input);
    run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    run.input({ observation: run.fixture.observation }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action.shiftUp).toBe(false);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1600 } }, 1600);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply.mock.calls.filter(([action]) => action.shiftUp)).toHaveLength(1);
    run.manual.mockResolvedValue({ ...input, action: { ...input.action, shiftUp: false } });
    jest.advanceTimersByTime(100); await flush();
    run.manual.mockResolvedValue(input);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    await run.loop.stop();
});

it.each((['manual', 'training', 'evaluation'] as const).flatMap((mode) =>
    (['shiftUp', 'shiftDown'] as const).map((key) => ({ mode, key }))))(
    'accepts rapid repeated and alternating human paddles without cooldown in $mode, starting with $key', async ({ mode, key }) => {
    const run = setup(true);
    const pressed = { throttle: 0.4, brake: 0, steering: 0.002, shiftUp: false, shiftDown: false, [key]: true };
    const released = { ...pressed, shiftUp: false, shiftDown: false };
    run.manual.mockResolvedValue({ action: released, toggleControl: false });
    run.loop.startManual(); await flush();
    if (mode !== 'manual') { await run.loop.start(run.policy, mode === 'training'); await flush(); }
    const interval = mode === 'manual' ? 50 : 20;
    let at = 1000;
    const poll = async (action: typeof pressed) => {
        at += interval;
        run.input({ observation: run.fixture.observation }, at);
        run.manual.mockResolvedValue({ action, toggleControl: false });
        jest.advanceTimersByTime(interval); await flush();
    };
    await poll(pressed);
    expect(run.controller.apply).toHaveBeenLastCalledWith(pressed);
    await poll(released);
    await poll(pressed);
    expect(run.controller.apply.mock.calls.slice(-2)).toEqual([[released], [pressed]]);
    const opposite = { ...released, [key === 'shiftUp' ? 'shiftDown' : 'shiftUp']: true };
    await poll(opposite);
    expect(run.controller.apply.mock.calls.slice(-2)).toEqual([[released], [opposite]]);
    for (let elapsed = 0; elapsed < 200; elapsed += interval) await poll(opposite);
    expect(run.controller.apply).toHaveBeenLastCalledWith(released);
    const edges = run.controller.apply.mock.calls.filter(([action], index, calls) =>
        (action.shiftUp && !calls[index - 1]?.[0].shiftUp) || (action.shiftDown && !calls[index - 1]?.[0].shiftDown));
    expect(edges.map(([action]) => action)).toEqual([pressed, pressed, opposite]);
    expect(run.controller.apply.mock.calls.every(([action]) => !(action.shiftUp && action.shiftDown))).toBe(true);
    await run.loop.stop();
});

it('accepts a human paddle immediately after taking over during an AI shift pulse', async () => {
    const run = setup(true);
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    await run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    const action = { ...NEUTRAL, shiftUp: false, shiftDown: true };
    run.input({ observation: run.fixture.observation }, 1020);
    run.manual.mockResolvedValue({ action, toggleControl: true });
    jest.advanceTimersByTime(20); await flush();
    expect(run.loop.state).toMatchObject({ source: 'manual', action });
    expect(run.controller.apply).toHaveBeenLastCalledWith(action);
    await run.loop.stop();
});

it('learns a human shift acknowledged between images, after the pulse has released', async () => {
    const run = setup(true);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    const input = { action: { throttle: 0.7, brake: 0, steering: 0.2, shiftDown: false }, toggleControl: true };
    run.manual.mockResolvedValue(input);
    run.loop.start(run.policy, true); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(100); await flush();
    run.manual.mockResolvedValue({ ...input, action: { ...input.action, shiftDown: true } });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftDown: true }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(demonstrate).not.toHaveBeenCalled();
    expect(run.loop.state.action.shiftDown).toBe(false);
    run.loop.setControlSource('ai'); await flush();
    expect(demonstrate.mock.calls[0]).toEqual([new DrivingMemory().encode(run.fixture.observation),
        expect.arrayContaining([expect.objectContaining({ step: 1, action: expect.objectContaining({ shiftDown: true }) })])]);
    await run.loop.stop();
});

it('does not learn from a failed manual write or a handover across a session boundary', async () => {
    const run = setup(true);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstratedPlanAsync');
    run.manual.mockResolvedValue({ action: { throttle: 1, brake: 0, steering: 1 }, toggleControl: true });
    run.loop.start(run.policy, true); await flush();
    run.controller.apply.mockRejectedValueOnce(new Error('Write failed.'));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(demonstrate).not.toHaveBeenCalled();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    run.loop.start(run.policy, true); await flush();
    run.manual.mockResolvedValue({ action: { throttle: 1, brake: 0, steering: 1 }, toggleControl: false });
    run.loop.setControlSource('ai');
    run.input({ observation: { ...run.fixture.observation, session: 'new-session', capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'training', source: 'ai', action: NEUTRAL });
    expect(run.loop.state.status).toContain('Live session changed');
    expect(demonstrate).not.toHaveBeenCalled();
    await run.loop.stop();
});

it('waits for the effect of an acknowledged action and learns only once per new captured image', async () => {
    const run = setup();
    const learn = collectedTransitions(run.policy);
    run.loop.start(run.policy, true);
    await flush();
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(0);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200, normalizedPosition: 0.252 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
    const reward = run.loop.state.reward!;
    expect(reward.timing).toBeCloseTo(-0.084);
    expect(reward.progress).toBeGreaterThan(0);
    expect(reward.total).toBeCloseTo(-0.059 + reward.steeringChange + reward.throttleChange + reward.brakeChange);
    const memory = new DrivingMemory();
    memory.encode(run.fixture.observation);
    expect(learn).toHaveBeenLastCalledWith(expect.anything(), reward.total,
        memory.encode({ ...run.fixture.observation, capturedAt: 1200 }), 0.2, expect.any(Array));
    expect(run.loop.state.episodeReward).toBe(reward.total);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
    await run.loop.stop();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.loop.state.mode).toBe('idle');
});

it.each([true, false].flatMap((train) => [0.248, 0.25].map((normalizedPosition) => ({ train, normalizedPosition }))))(
    'includes backward or low-speed progress penalties in learning and episode totals (training=$train, position=$normalizedPosition)',
    async ({ train, normalizedPosition }) => {
    const run = setup();
    const learn = collectedTransitions(run.policy);
    run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200, normalizedPosition,
        speed: normalizedPosition === 0.25 ? 0.5 : run.fixture.observation.speed } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    const reward = run.loop.state.reward!;
    expect(reward.progress).toBeLessThan(0);
    expect(reward.total).toBeCloseTo(reward.progress + reward.timing + reward.steeringChange + reward.throttleChange + reward.brakeChange);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    expect(learn.mock.calls.slice(-1)).toEqual(train ? [[expect.anything(), reward.total,
        expect.objectContaining({ features: expect.any(Array), cameras: expect.any(Array) }), 0.2, expect.any(Array)]] : []);
    await run.loop.stop();
});

it.each([true, false].flatMap((train) => [40, 160].map((speed) => ({ train, speed }))))(
    'accumulates the speed gap across scored observations (training=$train, speed=$speed)', async ({ train, speed }) => {
    const run = setup();
    const observation = { ...run.fixture.observation, speed };
    run.input({ observation }, 1000);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const learn = collectedTransitions(run.policy);
    run.loop.start(run.policy, train); await flush();
    let episodeReward = 0;
    const throttles = [0.3, 0.5, 0.5, 0.3, 0.3, 0.3];
    const corrections = [1, 2, 1, 0, 0, 1];
    const direction = speed < 100 ? 1 : -1;
    let previousGap = Math.abs(speed - 100);
    for (let index = 0; index < throttles.length; index++) {
        const capturedAt = 1200 + index * 200;
        run.input({ observation: { ...observation, capturedAt, normalizedPosition: 0.252 + index * 0.002,
            speed: speed + direction * corrections[index], controls: { ...observation.controls, throttle: throttles[index] } } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
        const reward = run.loop.state.reward!;
        const gap = Math.abs(speed + direction * corrections[index] - (99.2 - index * 0.8));
        expect(reward.timing).toBeCloseTo(-(previousGap + gap) / 200);
        previousGap = gap;
        expect(reward.progress).toBeGreaterThan(0);
        episodeReward += reward.total;
        expect(run.loop.state.episodeReward).toBeCloseTo(episodeReward);
        expect(learn.mock.calls.slice(-1)).toEqual(train ? [[expect.anything(), reward.total,
            expect.objectContaining({ features: expect.any(Array), cameras: expect.any(Array) }), 0.2, expect.any(Array)]] : []);
    }
    expect(learn).toHaveBeenCalledTimes(train ? throttles.length : 0);
    await run.loop.stop();
});

it.each([true, false])('retries invalid inputs and resumes only from fresh data without losing progress (training=%s)', async (train) => {
    const run = setup();
    run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    const before = (await run.policy.checkpointAsync());
    const episodeReward = run.loop.state.episodeReward;
    for (const time of [1300, 1400, 5000]) {
        run.input({ error: 'Telemetry disconnected.' }, time);
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source: 'ai',
            action: NEUTRAL, reward: null, episodeReward, steps: before.model.steps });
        expect(run.loop.state.status).toContain('waiting for valid inputs; retrying: Telemetry disconnected.');
        expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
        expect((await run.policy.checkpointAsync())).toEqual(before);
    }
    // Neither an old image nor telemetry measured before neutralization can resume AI.
    run.input({ observation: { ...run.fixture.observation, capturedAt: 4900 } }, 5100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 5200 } }, 5200, 4900);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.loop.state.status).toContain('retrying');
    expect((await run.policy.checkpointAsync())).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 5300 } }, 5300);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(run.loop.state.status).not.toContain('retrying');
    expect(run.loop.state.episodeReward).toBe(episodeReward);
    expect((await run.policy.checkpointAsync())).toEqual(before); // No learning across the interrupted transition.
    run.input({ observation: { ...run.fixture.observation, capturedAt: 5400 } }, 5400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(before.model.steps + (train ? 2 : 0));
    expect((await run.policy.checkpointAsync())).toEqual(train ? expect.not.objectContaining(before) : before);
    await run.loop.stop();
});

it.each((['ai', 'manual'] as const).flatMap((source) => [true, false].map((train) => ({ source, train }))))(
    'discards a lap-position jump and retries under $source control (training=$train)', async ({ source, train }) => {
    const run = setup(true);
    const manual = { action: { throttle: 0.3, brake: 0, steering: 0.1 }, toggleControl: false };
    run.manual.mockResolvedValue(manual);
    if (source === 'manual') { run.loop.startManual(); await flush(); }
    run.loop.start(run.policy, train); await flush();
    const before = (await run.policy.checkpointAsync());
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1050, normalizedPosition: 0.5 } }, 1050);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source,
        action: source === 'ai' ? NEUTRAL : manual.action, reward: null });
    expect(run.loop.state.status).toContain('Lap position jumped');
    expect((await run.policy.checkpointAsync())).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state.status).not.toContain('Lap position jumped');
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect((await run.policy.checkpointAsync())).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(50); await flush();
    expect(run.policy.steps).toBe(train && source === 'ai' ? 1 : 0);
    expect((await run.policy.checkpointAsync())).toEqual(train && source === 'ai' ? expect.not.objectContaining(before) : before);
    await run.loop.stop();
});

it('does not learn during evaluation, and waits on session changes', async () => {
    const run = setup();
    const before = (await run.policy.checkpointAsync());
    run.loop.start(run.policy, false); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect((await run.policy.checkpointAsync())).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, session: 'different', capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('evaluation');
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect((await run.policy.checkpointAsync())).toEqual(before);
    await run.loop.stop();
});

it.each(['stop', 'manual'] as const)('honors %s while a retry neutralization is awaiting acknowledgement', async (next) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    run.loop.start(run.policy, true); await flush();
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.input({ error: 'Invalid camera pixels.' }, 1100);
    jest.advanceTimersByTime(50); await flush();
    const stopped = next === 'stop' ? run.loop.stop() : undefined;
    if (next === 'manual') run.loop.setControlSource('manual');
    acknowledge(); await flush();
    await stopped;
    const manual = { action: { throttle: 0.2, brake: 0.5, steering: 0.1 }, toggleControl: false };
    run.manual.mockResolvedValue(manual);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    const count = run.controller.apply.mock.calls.length;
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject(next === 'stop'
        ? { mode: 'idle', source: null, action: NEUTRAL }
        : { mode: 'training', source: 'manual', action: manual.action });
    expect(run.controller.apply.mock.calls.slice(count)).toEqual(next === 'stop'
        ? [] : expect.arrayContaining([[expect.objectContaining(manual.action)]]));
    expect(run.controller.apply).toHaveBeenLastCalledWith(next === 'stop' ? NEUTRAL : expect.objectContaining(manual.action));
    expect(run.policy.steps).toBe(0);
    await run.loop.stop();
});

it('does not resume after Stop races with an in-flight controller acknowledgement', async () => {
    const run = setup();
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.loop.start(run.policy, true);
    await flush();
    const stopped = run.loop.stop();
    acknowledge(); await flush();
    await stopped;
    expect(run.loop.state.mode).toBe('idle');
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it.each([true, false])('starts and continues without position-model output (training=%s)', async (train) => {
    const run = setup();
    const { referenceLap } = run.fixture;
    const observationAt = (capturedAt: number) => {
        const { live, vision } = trainingFixture(capturedAt, 50, referenceLap);
        live.currentTelemetry.Graphics_normalized_car_position = 0.25 + (capturedAt - 1000) / 100000;
        delete vision.position;
        return readDrivingObservation(live, vision, referenceLap, 50, capturedAt, undefined, circuitMapFixture(referenceLap));
    };
    const learn = collectedTransitions(run.policy);
    run.input(observationAt(1000), 1000);
    await run.loop.start(run.policy, train); await flush();
    // Holding 108 km/h leaves average gaps of 8.4 and 9.2 km/h over the intervals.
    for (const [capturedAt, timing] of [[1200, -0.084], [1400, -0.092]]) {
        run.input(observationAt(capturedAt), capturedAt);
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state.mode).toBe(train ? 'training' : 'evaluation');
        expect(run.loop.state.reward?.progress).toBeCloseTo(0.025);
        expect(run.loop.state.reward?.timing).toBeCloseTo(timing, 8);
        expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
        expect(run.loop.state.action).not.toEqual(NEUTRAL);
    }
    expect(learn).toHaveBeenCalledTimes(train ? 2 : 0);
    expect(run.policy.steps).toBe(train ? 3 : 0); // The second batch replays both transitions.
    await run.loop.stop();
});

it('stops cleanly on hardware failure', async () => {
    const run = setup();
    run.controller.apply.mockRejectedValue(new Error('Driver disconnected.'));
    run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.loop.state.status).toContain('Driver disconnected');
});

it.each([['shiftUp', 0.98], ['shiftDown', 0.999]] as const)('releases a learned %s pulse without a fresh image and never repeats it within the same plan slot', async (key, draw) => {
    const run = setup(), config = cloneConfig();
    config.actions.planTimesSeconds = [0];
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => draw, config);
    run.loop.start(run.policy, true); await flush();
    const pressed = run.loop.state.action;
    expect(pressed[key]).toBe(true);
    run.input({ observation: run.fixture.observation }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith({ ...pressed, shiftUp: false, shiftDown: false });
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: false });
    run.input({ observation: run.fixture.observation }, 1400);
    jest.advanceTimersByTime(300); await flush();
    expect(run.controller.apply.mock.calls.filter(([action]) => action[key])).toHaveLength(1);
    expect(run.policy.steps).toBe(0);
    await run.loop.stop();
});

it('learns the actual shift and keeps both directions available on every fresh decision', async () => {
    const run = setup();
    const learn = collectedTransitions(run.policy);
    run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100, gear: 4, normalizedPosition: 0.251 } }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(learn.mock.calls[0][0].plan[0].action.shiftUp).toBe(true);
    expect(run.loop.state.action).toMatchObject({ shiftUp: true, shiftDown: false });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400, gear: 4, normalizedPosition: 0.252 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(learn.mock.calls[1][0].plan[0].shiftProbabilities.every((probability) => probability > 0)).toBe(true);
    expect(learn.mock.calls[1][0].plan[0].shifts).toEqual({ up: true, down: true });
    expect(run.loop.state.action).toMatchObject({ shiftUp: true, shiftDown: false });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1500, gear: 4, normalizedPosition: 0.253 } }, 1500);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    expect(run.policy.steps).toBe(6); // Batches of one, two and three replay transitions.
});

it.each([true, false].flatMap((train) => [-1, 0, 1, 2].map((gear) => ({ train, gear }))))(
    'allows consecutive downshifts at gear $gear with training=$train', async ({ train, gear }) => {
    const run = setup();
    const checkpoint = (await run.policy.checkpointAsync());
    checkpoint.model.outputWeights.slice(2, 5).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === 2 ? 4 : 0; });
    run.policy = train
        ? new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.999)
        : DrivingPolicy.restore(checkpoint);
    const learn = collectedTransitions(run.policy);
    run.input({ observation: { ...run.fixture.observation, gear } }, 1000);
    run.loop.start(run.policy, train); await flush();
    expect(run.loop.state.action.shiftDown).toBe(true);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftDown: true, shiftUp: false }));
    run.input({ observation: { ...run.fixture.observation, gear, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: true });
    expect(run.policy.steps).toBe(train ? 1 : 0);
    expect(learn).toHaveBeenCalledTimes(train ? 1 : 0);
    expect(learn.mock.calls[0]?.[0].plan[0].action.shiftDown).toBe(train ? true : undefined);
    expect((await run.policy.checkpointAsync())).toEqual(train ? expect.not.objectContaining(checkpoint) : checkpoint);
    run.input({ observation: { ...run.fixture.observation, gear, capturedAt: 1500 } }, 1500);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftDown: true, shiftUp: false }));
    await run.loop.stop();
});

it('stops if releasing a learned paddle fails, without learning from unacknowledged output', async () => {
    const run = setup();
    run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    run.controller.apply.mockRejectedValueOnce(new Error('Shift release failed.'));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', status: 'Shift release failed.', action: NEUTRAL });
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it('does not send another action when Stop races with a paddle release', async () => {
    const run = setup();
    run.loop.start(run.policy, true); await flush();
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(100);
    await flush();
    const stopped = run.loop.stop();
    acknowledge(); await flush();
    await stopped;
    expect(run.loop.state).toMatchObject({ mode: 'idle', action: NEUTRAL });
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it.each([true, false])('attributes reward to the action’s aggression while the next policy input follows the live slider (training=%s)', async (train) => {
    const run = setup();
    const observation = { ...run.fixture.observation, speed: 100 };
    const withAggression = (aggression: number, capturedAt: number, position: number) => {
        const speedTargets = referenceSpeedTargets(observation.reference, position, aggression);
        return { ...observation, aggression, capturedAt, normalizedPosition: position, speedTargets,
            features: [...observation.features.slice(0, -3), normalizedAggression(aggression), speedTargets.reference / 360, speedTargets.target / 360] };
    };
    run.policy = new DrivingPolicy(observation.context, observation.features.length, () => 0.1);
    const before = (await run.policy.checkpointAsync());
    const learn = collectedTransitions(run.policy);
    const decide = jest.spyOn(run.policy, 'decideAsync');
    run.input({ observation: withAggression(0, 1000, 0.25) }, 1000);
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.loop.start(run.policy, train);
    await flush();
    const sent = run.controller.apply.mock.calls[0][0];
    expect(sent).toEqual(drivingControllerAction((await decide.mock.results[0].value).plan[0].action));
    expect(sent.brake).not.toBe(1); // Targets never replace policy commands.
    run.input({ observation: withAggression(100, 1000, 0.25) }, 1100);
    acknowledge(); await flush();
    jest.advanceTimersByTime(100); await flush();
    expect(learn).not.toHaveBeenCalled();
    const next = withAggression(100, 1200, 0.252);
    run.input({ observation: { ...next, speed: 99.2, controls: { ...NEUTRAL, throttle: 0, brake: 1 } } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.speedComparison).toMatchObject({ aggression: 0 });
    expect(run.loop.state.reward!.speedComparison.target).toBeCloseTo(99.2);
    expect(run.loop.state.reward!.speedComparison.targetAcceleration).toBeCloseTo(-0.8 / 0.72);
    expect(run.loop.state.reward!.timing).toBeCloseTo(0);
    const memory = new DrivingMemory();
    memory.encode(withAggression(0, 1000, 0.25));
    expect(decide).toHaveBeenLastCalledWith(memory.encode(next), train, expect.anything());
    expect(learn).toHaveBeenCalledTimes(train ? 1 : 0);
    expect(learn.mock.calls[0]?.[0].input.features.slice(-3)[0]).toBe(train ? -1 : undefined);
    expect((await run.policy.checkpointAsync())).toEqual(train ? expect.not.objectContaining(before) : before);
    const last = withAggression(100, 1400, 0.254);
    run.input({ observation: last }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.speedComparison.aggression).toBe(100);
    expect(run.loop.state.reward!.speedComparison.target).toBeCloseTo(147.6);
    expect(run.loop.state.reward!.timing).toBeCloseTo(-0.486);
    expect(run.policy.steps).toBe(train ? 3 : 0);
    expect((await run.policy.checkpointAsync())).toEqual(train ? expect.not.objectContaining(before) : before);
    await run.loop.stop();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
});

it('waits for both the image and measured pedals to follow acknowledgement', async () => {
    const run = setup();
    run.loop.start(run.policy, true); await flush();
    const freshImage = { ...run.fixture.observation, capturedAt: 1200, normalizedPosition: 0.252 };
    run.input({ observation: freshImage }, 1200, 1000);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(0);
    run.input({ observation: freshImage }, 1300, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
    await run.loop.stop();
});

it.each([true, false])('waits with neutral controls on a reference change before reusing an image (training=%s)', async (train) => {
    const run = setup();
    run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, reference: trainingFixture().observation.reference } }, 1000);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', action: NEUTRAL });
    expect(run.loop.state.status).toMatch(/reference changed/);
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
    await run.loop.stop();
});

it('blocks starting but waits during an active episode when reference data is unavailable', async () => {
    const run = setup();
    run.input({ error: 'Live Map needs a reference lap with usable raw XYZ coordinates, normalized positions, throttle and brake.' }, 1000);
    await expect(run.loop.start(run.policy, true)).rejects.toThrow(/reference lap/);
    expect(run.controller.apply).not.toHaveBeenCalled();
    run.input({ observation: run.fixture.observation }, 1000);
    run.loop.start(run.policy, true); await flush();
    run.input({ error: 'Live Map has no reference lap selected.' }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('training');
    expect(run.loop.state.status).toContain('retrying');
    expect(run.policy.steps).toBe(0);
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    await run.loop.stop();
});
