import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingPolicy } from './driving-policy';
import { NEUTRAL, ObservationResult } from './driving-observation';
import { trainingFixture } from './training-fixtures';

const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };
beforeEach(() => { jest.useFakeTimers(); });
afterEach(() => { jest.useRealTimers(); });

function setup() {
    const fixture = trainingFixture();
    let input: ObservationResult = { observation: fixture.observation };
    let now = 1000;
    const controller = { connect: jest.fn(), disconnect: jest.fn(), apply: jest.fn().mockResolvedValue(undefined) };
    const publish = jest.fn();
    const policy = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length);
    const loop = new DrivingTrainingLoop(controller, () => input, publish, () => now);
    return { fixture, controller, policy, loop, publish,
        input: (next: ObservationResult, time: number) => { input = next; now = time; } };
}

it('waits for the effect of an acknowledged action and learns only once per new captured image', async () => {
    const run = setup();
    run.loop.start(run.policy, true);
    await flush();
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(0);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200, normalizedPosition: 0.252 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
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

it('stops cleanly on hardware failure and trains a terminal deviation penalty', async () => {
    const run = setup();
    run.loop.start(run.policy, true); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200, referenceDistance: 30 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.loop.state.reward?.total).toBe(-10);
    expect(run.policy.steps).toBe(1);
    run.input({ observation: run.fixture.observation }, 1300);
    run.controller.apply.mockRejectedValue(new Error('Driver disconnected.'));
    run.loop.start(run.policy, true); await flush();
    expect(run.loop.state.mode).toBe('idle');
    expect(run.loop.state.status).toContain('Driver disconnected');
});
