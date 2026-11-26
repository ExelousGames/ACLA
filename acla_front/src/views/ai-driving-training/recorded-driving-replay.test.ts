import { RecordedDrivingReplay } from './recorded-driving-replay';
import { RecordedDrivingBalance } from './recorded-driving-balance';
import { cloneConfig, emptyRecordedSamplingWeights } from './driving-model-config';
import { RECORDED_PLAN_WINDOW } from './driving-plan-training';
import type { DrivingPolicy, PlannedActionTarget, PolicyDecision, RecordedPlanTarget } from './driving-policy';
import type { DrivingObservation } from './driving-observation';
import type { DrivingPolicyInput } from './driving-memory';
import { assembleDrivingInput } from './driving-policy-input';
import { parseRecordedDrivingSession, readRecordedDrivingObservation } from './recorded-driving-session';
import { referenceLapFixture, trainingFixture } from './training-fixtures';

function setup(train = true) {
    const configuration = cloneConfig();
    configuration.memory.windowMs = 300;
    configuration.memory.intervalMs = 100;
    configuration.memory.features = ['speed', 'camera'];
    configuration.actions.planTimesSeconds = [0, 0.1, 0.2, 0.3];
    const learnRecordedPlanAsync = jest.fn(async (_input: DrivingPolicyInput, _targets: PlannedActionTarget[]) => undefined);
    const fake = {
        configuration, planTimes: configuration.actions.planTimesSeconds,
        matches: jest.fn(() => true),
        trainBatchAsync: jest.fn(async (run: () => Promise<void>) => run()),
        observeDemonstration: jest.fn((_input: DrivingPolicyInput) => undefined),
        learnRecordedPlanAsync,
        learnRecordedBatchAsync: jest.fn(async (batch: readonly RecordedPlanTarget[], active: () => boolean) => {
            if (!active()) return false;
            for (const { input, targets } of batch) await learnRecordedPlanAsync(input, targets);
            return true;
        }),
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
    const learnedByOrigin = () => learnRecordedPlanAsync.mock.calls.slice()
        .sort(([left], [right]) => left.recurrent!.capturedAt - right.recurrent!.capturedAt);
    return { fake, replay, frame, learnedByOrigin };
}

it('keeps zero-weight origins in recurrent history and uses their controls for other plans', async () => {
    const { fake, frame, learnedByOrigin } = setup();
    fake.configuration.recordedSamplingWeights = { ...emptyRecordedSamplingWeights(), Unlabeled: 100 };
    const frames = [frame(1000), frame(1100),
        frame(1200, { controls: { throttle: 0.4, brake: 0, steering: 0 } }), frame(1300)];
    const labels = [undefined, 'Hairpin', undefined, 'Hairpin'];
    const balance = new RecordedDrivingBalance(fake.configuration);
    frames.forEach((observation, index) => balance.count(observation, labels[index]));
    balance.prepare();
    const replay = new RecordedDrivingReplay(fake as unknown as DrivingPolicy, true, balance);

    for (let index = 0; index < frames.length; index++) {
        await replay.consume({ observation: frames[index] }, undefined, labels[index]);
    }
    expect(fake.observeDemonstration.mock.calls.map(([input]) => input.recurrent?.capturedAt)).toEqual([1000, 1100, 1200]);
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    await replay.finish();
    const learned = learnedByOrigin();
    expect(learned.map(([input]) => input.recurrent?.capturedAt)).toEqual([1000, 1200]);
    expect(learned[1][0].recurrent?.historyTimes).toEqual([1100, 1000, null]);
    const targets = learned[0][1];
    expect(targets.map(({ step }) => step)).toEqual([0, 1, 2]);
    expect(targets[1].action.pedal).toBe(0.4);
    expect(replay.progress.samples).toBe(3);
    expect(replay.trained).toEqual({ ...emptyRecordedSamplingWeights(), Unlabeled: 2 });
    expect(balance.complete).toBe(true);
});

it('trains every aligned origin once across uneven label ranges and drains the final batch', async () => {
    const { fake, frame, learnedByOrigin } = setup();
    fake.configuration.recordedSamplingWeights = { ...emptyRecordedSamplingWeights(), Unlabeled: 10, Hairpin: 90 };
    const unlabeled = Array.from({ length: 12 }, (_, index) => frame(1000 + index * 100));
    const hairpin = [frame(4000), frame(4100), frame(4200)];
    const balance = new RecordedDrivingBalance(fake.configuration);
    unlabeled.forEach((observation) => balance.count(observation));
    hairpin.forEach((observation, index) => balance.count(observation, 'Hairpin', index === 0));
    balance.prepare();
    const replay = new RecordedDrivingReplay(fake as unknown as DrivingPolicy, true, balance);

    for (const observation of unlabeled) await replay.consume({ observation });
    await replay.startRange();
    for (const observation of hairpin) await replay.consume({ observation }, undefined, 'Hairpin');
    await replay.finish();

    expect(learnedByOrigin().map(([input]) => input.recurrent?.capturedAt)).toEqual(
        [...unlabeled.slice(0, -1), ...hairpin.slice(0, -1)].map(({ capturedAt }) => capturedAt));
    expect(fake.learnRecordedBatchAsync.mock.calls.map(([batch]) => batch.length)).toEqual([8, 5]);
    expect(replay.trained).toEqual({ ...emptyRecordedSamplingWeights(), Unlabeled: 11, Hairpin: 2 });
    expect(balance.complete).toBe(true);
});

it('drains another window when closing the last range makes more than one window ready', async () => {
    const { fake, replay, frame, learnedByOrigin } = setup();
    fake.planTimes.push(1);
    const first = frame(1000);
    const origins = RECORDED_PLAN_WINDOW + 5;
    const times = Array.from({ length: origins + 1 }, (_, i) => first.capturedAt + i * 100);
    for (const at of times) {
        await replay.consume({ observation: { ...first, capturedAt: at, measuredAt: at } });
    }
    expect(fake.learnRecordedBatchAsync).not.toHaveBeenCalled();

    await replay.finish();

    expect(learnedByOrigin().map(([input]) => input.recurrent?.capturedAt)).toEqual(times.slice(0, -1));
    expect(fake.learnRecordedBatchAsync.mock.calls.at(-1)?.[0]).toHaveLength(5);
    expect(replay.trained.Unlabeled).toBe(origins);
});

it('learns next-sample controls from earlier observations and shifts future analog targets by one sample', async () => {
    const { fake, replay, frame, learnedByOrigin } = setup();
    const first = frame(1000, { gear: 3 });
    const second = frame(1100, { gear: 4, controls: { throttle: 0, brake: 0.6, steering: -0.4 } });
    const third = frame(1200, { gear: 3, controls: { throttle: 0.15, brake: 0, steering: 0.25 } });
    await replay.consume({ observation: first });
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    await replay.consume({ observation: second });
    await replay.consume({ observation: third });
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    await replay.finish();
    expect(fake.learnRecordedPlanAsync).toHaveBeenCalledTimes(2);
    const [[firstInput, firstTargets], [secondInput, secondTargets]] = learnedByOrigin();
    expect(firstInput.recurrent?.capturedAt).toBe(first.capturedAt);
    expect(secondInput.recurrent?.capturedAt).toBe(second.capturedAt);
    expect(firstInput.features.slice(0, 9)).toEqual(new Array(9).fill(0));
    expect(secondInput.features.slice(0, 3)).toEqual([1, 1 / 3, first.speed / fake.configuration.observation.speedScale]);
    expect(firstInput.cameras.slice(0, 3)).toEqual([null, null, null]);
    expect(secondInput.recordedHistory?.[0].camera).toEqual(first.camera);
    expect(secondInput.recordedHistory?.[0].camera).not.toBe(first.camera);
    expect(firstInput.recordedHistory).toEqual([]);
    expect(secondInput.recordedHistory).toEqual([{ capturedAt: first.capturedAt, features: first.features, camera: first.camera }]);
    const sequential = fake.observeDemonstration.mock.calls[1][0];
    expect(sequential.cameras[0]).toEqual(first.camera);
    expect(secondInput.cameras[0]).toBeNull();
    expect(assembleDrivingInput(secondInput, fake.configuration, -Infinity, secondInput.recordedHistory))
        .toEqual(assembleDrivingInput(sequential, fake.configuration, -Infinity, sequential.recordedHistory));
    const brake = { action: { pedal: -0.6, steering: -0.4, shiftUp: true, shiftDown: false },
        shifts: { up: true, down: true } };
    const accelerate = { action: { pedal: 0.15, steering: 0.25, shiftUp: false, shiftDown: true },
        shifts: { up: true, down: true } };
    expect(firstTargets).toEqual([{ step: 0, ...brake }, { step: 1, ...accelerate }]);
    expect(secondTargets).toEqual([{ step: 0, ...accelerate }]);
    expect(replay.progress.samples).toBe(2);
});

it('labels immediate and future recorded targets with brake throughout pedal overlap', async () => {
    const { fake, replay, frame, learnedByOrigin } = setup();
    const pedals = [[0.6, 0], [0.7, 0.2], [0.7, 0.2], [0.8, 0.1], [0.8, 0], [0, 0]];
    for (let i = 0; i < pedals.length; i++) {
        const [throttle, brake] = pedals[i];
        await replay.consume({ observation: frame(1000 + i * 100, { controls: { throttle, brake, steering: 0 } }) });
    }
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    await replay.consume({ error: 'Missing recorded frame.' });

    await replay.consume({ observation: frame(2000) });
    await replay.consume({ observation: frame(2100, { controls: { throttle: 0.6, brake: 0.1, steering: 0 } }) });
    await replay.finish();
    const groupedTargets = learnedByOrigin().map(([, targets]) => targets);
    expect(groupedTargets.map(([target]) => target.action.pedal)).toEqual([-0.2, -0.2, -0.1, 0.8, 0, -0.1]);
    expect(groupedTargets[0].map(({ action }) => action.pedal)).toEqual([-0.2, -0.2, -0.1, 0.8]);
    expect(groupedTargets[2].map(({ action }) => action.pedal)).toEqual([-0.1, 0.8, 0]);

    expect(groupedTargets[5]).toEqual([
        expect.objectContaining({ step: 0, action: expect.objectContaining({ pedal: -0.1 }) }),
    ]);
});

it('keeps steering unwind to zero in immediate and delayed targets', async () => {
    const { replay, frame, learnedByOrigin } = setup();
    const steering = [0, -0.06, 0, 0.001];
    for (let index = 0; index < steering.length; index++) {
        await replay.consume({ observation: frame(1000 + index * 100,
            { controls: { throttle: 0, brake: 0, steering: steering[index] } }) });
    }
    await replay.finish();
    const groupedTargets = learnedByOrigin().map(([, targets]) => targets);
    expect(groupedTargets.map(([target]) => target.action.steering)).toEqual([-0.06, 0, 0.001]);
    expect(groupedTargets[0]).toEqual([
        expect.objectContaining({ step: 0 }),
        expect.objectContaining({ step: 1, action: expect.objectContaining({ steering: 0 }) }),
        expect.objectContaining({ step: 2, action: expect.objectContaining({ steering: 0.001 }) }),
    ]);
});

it('labels each consecutive gear transition and masks ambiguous multi-gear jumps', async () => {
    const { replay, frame, learnedByOrigin } = setup();
    const gears = [3, 4, 3, 6, 6];
    for (let index = 0; index < gears.length; index++) await replay.consume({ observation: frame(1000 + index * 100, { gear: gears[index] }) });
    await replay.finish();
    expect(learnedByOrigin().map(([, [{ action, shifts }]]) => ({ action, shifts }))).toEqual([
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
    const { fake, replay, learnedByOrigin } = setup();
    const fixture = trainingFixture();
    const session = parseRecordedDrivingSession(gears.map((gear, index) => JSON.stringify({ ...fixture.live.currentTelemetry,
        Graphics_analyst_session_time: times[index], Physics_gear: gear + 1 })).join('\n'), 'acc');

    for (let index = 0; index < gears.length; index++) {
        const result = readRecordedDrivingObservation(session, times[index], fixture.observation.camera,
            fixture.referenceLap, 50, fake.configuration, fixture.circuitMap);
        expect(result.observation?.gear).toBe(gears[index]);
        await replay.consume(result);
    }

    await replay.finish();
    expect(learnedByOrigin().map(([, [{ action, shifts }]]) => ({ action, shifts }))).toEqual(
        targets.map((target) => ({ action: expect.objectContaining({ shiftUp: target === 'up', shiftDown: target === 'down' }),
            shifts: { up: target !== 'unknown', down: target !== 'unknown' } })));
    expect(replay.progress.samples).toBe(targets.length);
});

it.each(['gap', 'missing replay tick', 'session', 'reference', 'position', 'duplicate clock', 'telemetry still at the prior frame', 'range start'])(
    'keeps the preceding targets available for batching and resets history at %s', async (kind) => {
        const { fake, replay, frame, learnedByOrigin } = setup();
        await replay.consume({ observation: frame(1000) });
        const previous = frame(1100);
        if (kind === 'telemetry still at the prior frame') previous.measuredAt = 1050;
        await replay.consume({ observation: previous });
        const timestamps: Record<string, number> = { gap: 3000, 'missing replay tick': 1300, 'duplicate clock': 1100 };
        const at = timestamps[kind] ?? 1200;
        const changed = frame(at);
        if (kind === 'session') changed.session = 'new-recorded-session';
        if (kind === 'reference') changed.reference = { ...changed.reference };
        if (kind === 'position') changed.normalizedPosition = 0.55;
        if (kind === 'telemetry still at the prior frame') changed.measuredAt = previous.capturedAt;
        if (kind === 'range start') await replay.startRange();
        await replay.consume({ observation: changed });
        expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
        await replay.consume({ observation: { ...changed, capturedAt: at + 100, measuredAt: at + 100 } });
        expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
        await replay.finish();
        expect(fake.learnRecordedPlanAsync).toHaveBeenCalledTimes(2);
        expect(fake.learnRecordedBatchAsync).toHaveBeenCalledTimes(1);
        const learned = learnedByOrigin();
        expect(learned[1][0].cameras.slice(0, 3)).toEqual([null, null, null]);
        expect(learned.map(([, targets]) => targets.map(({ step }) => step))).toEqual([[0], [0]]);
    });

it('keeps history and future targets across missing lap position samples', async () => {
    const { replay, frame, learnedByOrigin } = setup();
    await replay.consume({ observation: frame(1000, { normalizedPosition: 0.25 }) });
    await replay.consume({ observation: frame(1100, { normalizedPosition: null }) });
    await replay.consume({ observation: frame(1200, { normalizedPosition: 0.75 }) });
    await replay.finish();
    expect(replay.progress).toMatchObject({ samples: 2, skipped: 0 });
    const calls = learnedByOrigin();
    expect(calls.map(([, targets]) => targets.map(({ step }) => step))).toEqual([[0, 1], [0]]);
    expect(calls[1][0].recurrent?.historyTimes).toEqual([1000, null, null]);
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
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    await replay.consume({ observation: frame(1400) });
    expect(replay.progress.samples).toBe(2);
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    await replay.finish();
    expect(fake.learnRecordedPlanAsync).toHaveBeenCalledTimes(2);
});

it('measures mean absolute error against next-sample controls and releases each decision', async () => {
    const { fake, replay, frame } = setup(false);
    await replay.consume({ observation: frame(1000, { gear: 3, controls: { throttle: 0.3, brake: 0, steering: 0 } }) });
    await replay.consume({ observation: frame(1100, { gear: 4, controls: { throttle: 0, brake: 0.5, steering: -0.4 } }) });
    await replay.consume({ observation: frame(1200, { gear: 3, controls: { throttle: 0.9, brake: 0, steering: 0.1 } }) });
    expect(replay.progress.samples).toBe(2);
    expect(replay.progress.pedalError).toBeCloseTo(0.7);
    expect(replay.progress.steeringError).toBeCloseTo(0.35);
    expect(fake.decideAsync.mock.calls.map(([input]) => input.recurrent?.capturedAt)).toEqual([1000, 1100]);
    expect(fake.decideAsync.mock.calls.map(([, explore]) => explore)).toEqual([false, false]);
    expect(fake.releaseDecision).toHaveBeenCalledTimes(2);
    expect(fake.trainBatchAsync).not.toHaveBeenCalled();
    await replay.finish();
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
});

it('measures the full distance between opposite signed commands', async () => {
    const { fake, replay, frame } = setup(false);
    fake.decideAsync.mockResolvedValueOnce({ plan: [{ action: { pedal: -0.8, steering: 0 } }] } as PolicyDecision);
    await replay.consume({ observation: frame(1000, { controls: { throttle: 0.1, brake: 0, steering: 0 } }) });
    await replay.consume({ observation: frame(1100, { controls: { throttle: 0.5, brake: 0, steering: 0 } }) });
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
    expect(fake.observeDemonstration).not.toHaveBeenCalled();
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    expect(replay.progress.samples).toBe(0);
});

it('flushes only targets already collected when stopping after a cancelled frame', async () => {
    const { fake, replay, frame } = setup();
    await replay.consume({ observation: frame(1000) });
    await replay.consume({ observation: frame(1100) });
    let active = true;
    fake.trainBatchAsync.mockImplementationOnce(async (run) => { active = false; await run(); });
    await replay.consume({ observation: frame(1200, { controls: { throttle: 0, brake: 0.5, steering: -0.4 } }) }, () => active);
    await replay.finish(() => active);
    expect(fake.learnRecordedPlanAsync).not.toHaveBeenCalled();
    await replay.finish();
    expect(fake.learnRecordedPlanAsync).toHaveBeenCalledTimes(1);
    expect(fake.learnRecordedPlanAsync.mock.calls[0][1].map(({ step }) => step)).toEqual([0]);
    expect(replay.progress.samples).toBe(1);
});
