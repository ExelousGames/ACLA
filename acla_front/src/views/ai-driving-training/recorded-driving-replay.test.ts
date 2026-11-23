import { RecordedDrivingReplay } from './recorded-driving-replay';
import { cloneConfig } from './driving-model-config';
import type { DrivingPolicy, PolicyDecision } from './driving-policy';
import type { DrivingObservation } from './driving-observation';
import type { DrivingPolicyInput } from './driving-memory';
import { parseRecordedDrivingSession, readRecordedDrivingObservation } from './recorded-driving-session';
import { referenceLapFixture, trainingFixture } from './training-fixtures';

function setup(train = true) {
    const configuration = cloneConfig();
    configuration.memory.windowMs = 300;
    configuration.memory.intervalMs = 100;
    configuration.memory.features = ['speed', 'camera'];
    configuration.actions.planTimesSeconds = [0, 0.1, 0.2, 0.3];
    const fake = {
        configuration, planTimes: configuration.actions.planTimesSeconds,
        matches: jest.fn(() => true),
        trainBatchAsync: jest.fn(async (run: () => Promise<void>) => run()),
        learnDemonstrationAsync: jest.fn(async (_input: DrivingPolicyInput, _action: unknown, _shifts: unknown) => undefined),
        learnPlannedActionsAsync: jest.fn(async (_input: DrivingPolicyInput, _targets: unknown) => undefined),
        decideAsync: jest.fn(async (_input: DrivingPolicyInput, _explore: boolean, _shifts: unknown) => ({
            plan: [{ action: { pedal: 0.5, steering: 0.2 } }],
        } as PolicyDecision)),
        releaseDecision: jest.fn(),
    };
    const replay = new RecordedDrivingReplay(fake as unknown as DrivingPolicy, train);
    const lap = referenceLapFixture();
    const frame = (at: number, overrides: Partial<DrivingObservation> = {}): DrivingObservation => ({
        ...trainingFixture(at, 50, lap).observation, ...overrides,
    });
    return { fake, replay, frame };
}

it('learns from earlier held controls with real image history and later action targets', async () => {
    const { fake, replay, frame } = setup();
    const first = frame(1000), second = frame(1100, { controls: { throttle: 0, brake: 0.6, steering: -0.4 } });
    await replay.consume({ observation: first });
    expect(fake.learnDemonstrationAsync).not.toHaveBeenCalled();
    await replay.consume({ observation: second });
    await replay.consume({ observation: frame(1200) });
    expect(fake.learnDemonstrationAsync).toHaveBeenCalledTimes(2);
    const firstInput = fake.learnDemonstrationAsync.mock.calls[0][0];
    const secondInput = fake.learnDemonstrationAsync.mock.calls[1][0];
    expect(firstInput.features.slice(0, 9)).toEqual(new Array(9).fill(0));
    expect(secondInput.features.slice(0, 3)).toEqual([1, 1 / 3, first.speed / fake.configuration.observation.speedScale]);
    expect(firstInput.cameras.slice(0, 3)).toEqual([null, null, null]);
    expect(secondInput.cameras[0]).toEqual(first.camera);
    expect(secondInput.cameras[0]).not.toBe(first.camera);
    expect(fake.learnDemonstrationAsync.mock.calls[0][1]).toEqual({ pedal: first.controls.throttle, steering: first.controls.steering, shiftUp: false, shiftDown: false });
    expect(fake.learnPlannedActionsAsync).toHaveBeenCalledWith(firstInput, [{ step: 1,
        action: { pedal: -0.6, steering: -0.4, shiftUp: false, shiftDown: false }, shifts: { up: true, down: true } }]);
    expect(replay.progress.samples).toBe(2);
});

it('keeps history across a frame gap up to 100 ms beyond the configured limit', async () => {
    const { fake, replay, frame } = setup();
    fake.configuration.memory.windowMs = 2000;
    fake.configuration.memory.intervalMs = 500;
    const first = frame(1000);
    await replay.consume({ observation: first });
    await replay.consume({ observation: frame(2600) });
    await replay.consume({ observation: frame(2700) });

    expect(replay.progress.samples).toBe(2);
    expect(fake.learnDemonstrationAsync.mock.calls[1][0].cameras[3]).toEqual(first.camera);
});

it('keeps pending samples when video frames share telemetry and resumes on the next reading', async () => {
    const { fake, replay } = setup();
    const fixture = trainingFixture();
    const session = parseRecordedDrivingSession([0, 0.16, 0.31, 0.45].map((time) => JSON.stringify({
        ...fixture.live.currentTelemetry, Graphics_analyst_session_time: time,
    })).join('\n'), 'acc');

    for (const time of [0, 0.1, 0.2, 0.3, 0.4]) {
        await replay.consume(readRecordedDrivingObservation(session, time, fixture.observation.camera,
            fixture.referenceLap, 50, fake.configuration, fixture.circuitMap));
    }

    expect(replay.progress).toMatchObject({ samples: 2, skipped: 0 });
    const [[firstInput], [secondInput]] = fake.learnDemonstrationAsync.mock.calls;
    expect(secondInput.features.slice(3, 5)).toEqual([1, 200 / 300]);
    expect(fake.learnPlannedActionsAsync).toHaveBeenCalledWith(firstInput, [expect.objectContaining({ step: 2 })]);
});

it('labels immediate and future recorded targets with brake throughout pedal overlap', async () => {
    const { fake, replay, frame } = setup();
    const pedals = [[0.6, 0], [0.7, 0.2], [0.7, 0.2], [0.8, 0.1], [0.8, 0], [0, 0]];
    for (let i = 0; i < pedals.length; i++) {
        const [throttle, brake] = pedals[i];
        await replay.consume({ observation: frame(1000 + i * 100, { controls: { throttle, brake, steering: 0 } }) });
    }
    expect(fake.learnDemonstrationAsync.mock.calls.map(([, action]) => action)).toEqual(
        [0.6, -0.2, -0.2, -0.1, 0.8].map((pedal) => expect.objectContaining({ pedal })));
    expect(fake.learnPlannedActionsAsync).toHaveBeenCalledWith(fake.learnDemonstrationAsync.mock.calls[0][0], [
        expect.objectContaining({ step: 1, action: expect.objectContaining({ pedal: -0.2 }) }),
    ]);
    expect(fake.learnPlannedActionsAsync).toHaveBeenCalledWith(fake.learnDemonstrationAsync.mock.calls[3][0], [
        expect.objectContaining({ step: 1, action: expect.objectContaining({ pedal: 0.8 }) }),
    ]);

    await replay.consume({ error: 'Missing recorded frame.' });
    await replay.consume({ observation: frame(2000, { controls: { throttle: 0.6, brake: 0.1, steering: 0 } }) });
    await replay.consume({ observation: frame(2100) });
    expect(fake.learnDemonstrationAsync.mock.calls[5][1]).toMatchObject({ pedal: -0.1 });
});

it('labels each consecutive gear transition and masks ambiguous multi-gear jumps', async () => {
    const { fake, replay, frame } = setup();
    const gears = [3, 4, 3, 6, 6];
    for (let index = 0; index < gears.length; index++) await replay.consume({ observation: frame(1000 + index * 100, { gear: gears[index] }) });
    expect(fake.learnDemonstrationAsync.mock.calls.map(([, action, shifts]) => ({ action, shifts }))).toEqual([
        { action: expect.objectContaining({ shiftUp: true, shiftDown: false }), shifts: { up: true, down: true } },
        { action: expect.objectContaining({ shiftUp: false, shiftDown: true }), shifts: { up: true, down: true } },
        { action: expect.objectContaining({ shiftUp: false, shiftDown: false }), shifts: { up: false, down: false } },
        { action: expect.objectContaining({ shiftUp: false, shiftDown: false }), shifts: { up: true, down: true } },
    ]);
});

it.each([
    { gears: [3, 0, 0, 4], targets: ['up', 'hold', 'hold'] },
    { gears: [2, 0, 1], targets: ['down', 'hold'] },
    { gears: [1, 0, 2], targets: ['up', 'hold'] },
    { gears: [1, 0, 1], targets: ['hold', 'hold'] },
    { gears: [3, 0, 5], targets: ['unknown', 'hold'] },
    { gears: [-1, 0, 1], targets: ['up', 'up'] },
    { gears: [1, 0, -1], targets: ['down', 'down'] },
    { gears: [0, 1, 0], targets: ['up', 'down'] },
    { gears: [1, 0, 2], targets: ['down'], times: [0, 0.1, 2] },
])('learns $targets from a recording with gears $gears', async ({ gears, targets, times = gears.map((_, i) => i / 10) }) => {
    const { fake, replay } = setup();
    const fixture = trainingFixture();
    const session = parseRecordedDrivingSession(gears.map((gear, index) => JSON.stringify({ ...fixture.live.currentTelemetry,
        Graphics_analyst_session_time: times[index], Physics_gear: gear + 1 })).join('\n'), 'acc');

    for (let index = 0; index < gears.length; index++) {
        const result = readRecordedDrivingObservation(session, times[index], fixture.observation.camera,
            fixture.referenceLap, 50, fake.configuration, fixture.circuitMap);
        expect(result.observation?.gear).toBe(gears[index]);
        await replay.consume(result);
    }

    expect(fake.learnDemonstrationAsync.mock.calls.map(([, action, shifts]) => ({ action, shifts }))).toEqual(
        targets.map((target) => ({ action: expect.objectContaining({ shiftUp: target === 'up', shiftDown: target === 'down' }),
            shifts: { up: target !== 'unknown', down: target !== 'unknown' } })));
    expect(replay.progress.samples).toBe(targets.length);
});

it.each(['gap', 'session', 'reference', 'position', 'duplicate clock', 'backwards telemetry'])(
    'drops history and pending future labels at a %s discontinuity', async (kind) => {
        const { fake, replay, frame } = setup();
        await replay.consume({ observation: frame(1000) });
        await replay.consume({ observation: frame(1100) });
        const at = kind === 'gap' ? 2701 : kind === 'duplicate clock' ? 1100 : 1200;
        const changed = frame(at);
        if (kind === 'session') changed.session = 'new-recorded-session';
        if (kind === 'reference') changed.reference = { ...changed.reference };
        if (kind === 'position') changed.normalizedPosition += 0.3;
        if (kind === 'backwards telemetry') changed.measuredAt = 1050;
        await replay.consume({ observation: changed });
        expect(fake.learnDemonstrationAsync).toHaveBeenCalledTimes(1);
        await replay.consume({ observation: { ...changed, capturedAt: at + 100, measuredAt: at + 100 } });
        expect(fake.learnDemonstrationAsync).toHaveBeenCalledTimes(2);
        expect(fake.learnDemonstrationAsync.mock.calls[1][0].cameras.slice(0, 3)).toEqual([null, null, null]);
        expect(fake.learnPlannedActionsAsync).not.toHaveBeenCalled();
    });

it('counts invalid inputs and incompatible samples as skipped and does not bridge them', async () => {
    const { fake, replay, frame } = setup();
    await replay.consume({ observation: frame(1000) });
    await replay.consume({ observation: frame(1100) });
    await replay.consume({ error: 'Telemetry gap.' });
    expect(replay.progress).toMatchObject({ samples: 1, skipped: 1, lastError: 'Telemetry gap.' });
    fake.matches.mockReturnValueOnce(false);
    await replay.consume({ observation: frame(1200) });
    expect(replay.progress.skipped).toBe(2);
    expect(replay.progress.lastError).toMatch(/do not match/);
    await replay.consume({ observation: frame(1300) });
    expect(fake.learnDemonstrationAsync).toHaveBeenCalledTimes(1);
    await replay.consume({ observation: frame(1400) });
    expect(replay.progress.samples).toBe(2);
    expect(fake.learnPlannedActionsAsync).not.toHaveBeenCalled();
});

it('evaluates fixed weights with running mean absolute action errors and releases each decision', async () => {
    const { fake, replay, frame } = setup(false);
    await replay.consume({ observation: frame(1000, { controls: { throttle: 0.3, brake: 0, steering: 0 } }) });
    await replay.consume({ observation: frame(1100, { controls: { throttle: 0, brake: 0.5, steering: -0.4 } }) });
    await replay.consume({ observation: frame(1200) });
    expect(replay.progress.samples).toBe(2);
    expect(replay.progress.pedalError).toBeCloseTo(0.6);
    expect(replay.progress.steeringError).toBeCloseTo(0.4);
    expect(fake.decideAsync.mock.calls.map(([, explore]) => explore)).toEqual([false, false]);
    expect(fake.releaseDecision).toHaveBeenCalledTimes(2);
    expect(fake.trainBatchAsync).not.toHaveBeenCalled();
    expect(fake.learnDemonstrationAsync).not.toHaveBeenCalled();
    expect(fake.learnPlannedActionsAsync).not.toHaveBeenCalled();
});

it('measures the full distance between opposite signed commands', async () => {
    const { fake, replay, frame } = setup(false);
    fake.decideAsync.mockResolvedValueOnce({ plan: [{ action: { pedal: -0.8, steering: 0 } }] } as PolicyDecision);
    await replay.consume({ observation: frame(1000, { controls: { throttle: 0.5, brake: 0, steering: 0 } }) });
    await replay.consume({ observation: frame(1100) });
    expect(replay.progress.pedalError).toBeCloseTo(1.3);
    expect(replay.progress.steeringError).toBe(0);
});

it('releases an inference that completes after cancellation without publishing metrics', async () => {
    const { fake, replay, frame } = setup(false);
    let finish!: (value: PolicyDecision) => void;
    fake.decideAsync.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    let active = true;
    await replay.consume({ observation: frame(1000) });
    const work = replay.consume({ observation: frame(1100) }, () => active);
    active = false;
    const decision = { plan: [{ action: { pedal: 1, steering: 1 } }] } as PolicyDecision;
    finish(decision);
    await work;
    expect(fake.releaseDecision).toHaveBeenCalledWith(decision);
    expect(replay.progress).toEqual({ samples: 0, skipped: 0, pedalError: 0, steeringError: 0 });
});

it('does not submit learning after cancellation while waiting for the GPU batch', async () => {
    const { fake, replay, frame } = setup();
    let active = true;
    await replay.consume({ observation: frame(1000) });
    fake.trainBatchAsync.mockImplementationOnce(async (run) => { active = false; await run(); });
    await replay.consume({ observation: frame(1100) }, () => active);
    expect(fake.learnDemonstrationAsync).not.toHaveBeenCalled();
    expect(fake.learnPlannedActionsAsync).not.toHaveBeenCalled();
    expect(replay.progress.samples).toBe(0);
});
