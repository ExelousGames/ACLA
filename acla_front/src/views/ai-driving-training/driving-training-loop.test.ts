import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingPolicy } from './driving-policy';
import { NEUTRAL, ObservationResult } from './driving-observation';
import { trainingFixture } from './training-fixtures';
import { normalizedAggression, referencePedalTargets } from './driving-reference';

const flush = async () => { for (let i = 0; i < 6; i++) await Promise.resolve(); };
beforeEach(() => { jest.useFakeTimers(); });
afterEach(() => { jest.useRealTimers(); });

function setup() {
    const fixture = trainingFixture();
    let input: ObservationResult = { observation: fixture.observation };
    let now = 1000;
    const controller = { connect: jest.fn(), disconnect: jest.fn(), apply: jest.fn().mockResolvedValue(undefined) };
    const publish = jest.fn();
    const policy = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length, () => 0.7);
    const loop = new DrivingTrainingLoop(controller, () => input, publish, () => now);
    return { fixture, controller, policy, loop, publish,
        input: (next: ObservationResult, time: number, measuredAt = next.observation?.capturedAt) => {
            input = next.observation ? { observation: { ...next.observation, measuredAt: measuredAt! } } : next;
            now = time;
        } };
}

it('waits for the effect of an acknowledged action and learns only once per new captured image', async () => {
    const run = setup();
    const learn = jest.spyOn(run.policy, 'learn');
    run.loop.start(run.policy, true);
    await flush();
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(0);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200, normalizedPosition: 0.252 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
    const reward = run.loop.state.reward!;
    expect(reward.speed).toBeGreaterThan(0);
    expect(reward.total).toBeCloseTo(reward.speed + reward.line + reward.instability + reward.referencePedal);
    expect(learn).toHaveBeenLastCalledWith(expect.anything(), reward.total, run.fixture.observation.features, 0.2);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
    await run.loop.stop();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.loop.state.mode).toBe('idle');
});

it('releases controls on stale inputs and requires an explicit restart', async () => {
    const run = setup();
    run.loop.start(run.policy, true); await flush();
    run.input({ error: 'Telemetry disconnected.' }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', status: 'Telemetry disconnected.', action: NEUTRAL });
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    run.input({ observation: run.fixture.observation }, 1300);
    const count = run.controller.apply.mock.calls.length;
    jest.advanceTimersByTime(2000); await flush();
    expect(run.controller.apply).toHaveBeenCalledTimes(count);
});

it('does not learn during evaluation, and stops on session changes', async () => {
    const run = setup();
    const before = run.policy.checkpoint();
    run.loop.start(run.policy, false); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.checkpoint()).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, session: 'different', capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
});

it('does not resume after Stop races with an in-flight controller acknowledgement', async () => {
    const run = setup();
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.loop.start(run.policy, true);
    await run.loop.stop();
    acknowledge(); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it.each([true, false])('continues beyond 25 coordinate units from the reference line with training=%s', async (train) => {
    const run = setup();
    run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200, referenceDistance: 30 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe(train ? 'training' : 'evaluation');
    expect(run.loop.state.reward?.line).toBeLessThan(0);
    expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(run.policy.steps).toBe(train ? 1 : 0);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400, referenceDistance: 28 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe(train ? 'training' : 'evaluation');
    expect(run.loop.state.reward?.line).toBeLessThan(0);
    expect(run.policy.steps).toBe(train ? 2 : 0);
    await run.loop.stop();
});

it.each([true, false])('starts beyond 25 coordinate units from the reference line with training=%s', async (train) => {
    const run = setup();
    run.input({ observation: { ...run.fixture.observation, referenceDistance: 30 } }, 1000);
    run.loop.start(run.policy, train); await flush();
    expect(run.loop.state.mode).toBe(train ? 'training' : 'evaluation');
    expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    await run.loop.stop();
});

it('stops cleanly on hardware failure', async () => {
    const run = setup();
    run.controller.apply.mockRejectedValue(new Error('Driver disconnected.'));
    run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.loop.state.status).toContain('Driver disconnected');
});

it.each([['shiftUp', 0.7], ['shiftDown', 0.95]] as const)('releases a learned %s pulse without a fresh image and never repeats it from a heartbeat', async (key, draw) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => draw);
    run.loop.start(run.policy, true); await flush();
    const pressed = run.loop.state.action;
    expect(pressed[key]).toBe(true);
    run.input({ observation: run.fixture.observation }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith({ ...pressed, shiftUp: false, shiftDown: false });
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: false });
    run.input({ observation: run.fixture.observation }, 1700);
    jest.advanceTimersByTime(600); await flush();
    expect(run.controller.apply.mock.calls.filter(([action]) => action[key])).toHaveLength(1);
    expect(run.policy.steps).toBe(0);
});

it('learns the actual shift, masks both directions during cooldown, and allows a later fresh shift', async () => {
    const run = setup();
    const learn = jest.spyOn(run.policy, 'learn');
    run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100, gear: 4, normalizedPosition: 0.251 } }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(learn.mock.calls[0][0].action.shiftUp).toBe(true);
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: false });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400, gear: 4, normalizedPosition: 0.252 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(learn.mock.calls[1][0].shiftProbabilities).toEqual([1, 0, 0]);
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: false });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1500, gear: 4, normalizedPosition: 0.253 } }, 1500);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action.shiftUp).toBe(true);
    expect(run.policy.steps).toBe(3);
});

it.each([true, false].flatMap((train) => [-1, 0, 1, 2].map((gear) => ({ train, gear }))))(
    'allows downshifts at gear $gear with training=$train while respecting cooldown', async ({ train, gear }) => {
    const run = setup();
    const checkpoint = run.policy.checkpoint();
    checkpoint.outputWeights.slice(4).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === 2 ? 4 : 0; });
    run.policy = train
        ? new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.95)
        : DrivingPolicy.restore(checkpoint);
    const learn = jest.spyOn(run.policy, 'learn');
    run.input({ observation: { ...run.fixture.observation, gear } }, 1000);
    run.loop.start(run.policy, train); await flush();
    expect(run.loop.state.action.shiftDown).toBe(true);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftDown: true, shiftUp: false }));
    run.input({ observation: { ...run.fixture.observation, gear, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: false });
    expect(run.policy.steps).toBe(train ? 1 : 0);
    if (train) expect(learn.mock.calls[0][0].action.shiftDown).toBe(true);
    else expect(run.policy.checkpoint()).toEqual(checkpoint);
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
    await run.loop.stop();
    acknowledge(); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', action: NEUTRAL });
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it.each([true, false])('attributes reward to the action’s aggression while the next policy input follows the live slider (training=%s)', async (train) => {
    const run = setup();
    const observation = run.fixture.observation;
    const withAggression = (aggression: number, capturedAt: number, position: number) => {
        const pedals = referencePedalTargets(observation.reference, position, aggression);
        return { ...observation, aggression, capturedAt, normalizedPosition: position, pedals,
            features: [...observation.features.slice(0, -3), normalizedAggression(aggression), pedals.target.throttle, pedals.target.brake] };
    };
    run.policy = new DrivingPolicy(observation.context, observation.features.length, () => 0.1);
    const before = run.policy.checkpoint();
    const learn = jest.spyOn(run.policy, 'learn');
    const decide = jest.spyOn(run.policy, 'decide');
    run.input({ observation: withAggression(0, 1000, 0.25) }, 1000);
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.loop.start(run.policy, train);
    const sent = run.controller.apply.mock.calls[0][0];
    expect(sent).toEqual(decide.mock.results[0].value.action);
    expect(sent.brake).not.toBe(1); // Targets never replace policy commands.
    run.input({ observation: withAggression(100, 1000, 0.25) }, 1100);
    acknowledge(); await flush();
    jest.advanceTimersByTime(100); await flush();
    expect(learn).not.toHaveBeenCalled();
    const next = withAggression(100, 1200, 0.252);
    run.input({ observation: { ...next, controls: { ...NEUTRAL, throttle: 0, brake: 1 } } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.pedalComparison).toMatchObject({ aggression: 0, target: { throttle: 0, brake: 1 } });
    expect(run.loop.state.reward!.referencePedal).toBeCloseTo(0.2 * Math.exp(-2 / 3));
    expect(decide).toHaveBeenLastCalledWith(next.features, train, expect.anything());
    if (train) expect(learn.mock.calls[0][0].features.slice(-3)[0]).toBe(-1);
    else expect(run.policy.checkpoint()).toEqual(before);
    const last = withAggression(100, 1400, 0.254);
    run.input({ observation: { ...last, controls: { ...NEUTRAL, ...last.pedals.target } } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.reward!.pedalComparison.aggression).toBe(100);
    expect(run.loop.state.reward!.pedalComparison.target.brake).toBeCloseTo(0.04);
    expect(run.loop.state.reward!.referencePedal).toBeCloseTo(0.2 * Math.exp(-2 / 3));
    expect(run.policy.steps).toBe(train ? 2 : 0);
    if (!train) expect(run.policy.checkpoint()).toEqual(before);
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

it.each([true, false])('ends the episode and releases controls on a reference change before reusing an image (training=%s)', async (train) => {
    const run = setup();
    run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, reference: trainingFixture().observation.reference } }, 1000);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'idle', action: NEUTRAL });
    expect(run.loop.state.status).toMatch(/reference changed/);
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it('blocks starting and stops an active episode when reference data is unavailable', async () => {
    const run = setup();
    run.input({ error: 'Live Map needs a reference lap with usable raw XYZ coordinates, normalized positions, throttle and brake.' }, 1000);
    expect(() => run.loop.start(run.policy, true)).toThrow(/reference lap/);
    expect(run.controller.apply).not.toHaveBeenCalled();
    run.input({ observation: run.fixture.observation }, 1000);
    run.loop.start(run.policy, true); await flush();
    run.input({ error: 'Live Map reference changed. Waiting for matching Track Vision inputs.' }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.policy.steps).toBe(0);
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
});
