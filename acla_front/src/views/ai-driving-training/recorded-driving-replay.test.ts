import { RecordedDrivingReplay } from './recorded-driving-replay';
import { RecordedDrivingBalance, recordedSampleGroup } from './recorded-driving-balance';
import { cloneConfig } from './driving-model-config';
import type { DrivingPolicy, PolicyDecision } from './driving-policy';
import type { DrivingObservation } from './driving-observation';
import type { DrivingPolicyInput } from './driving-memory';
import { parseRecordedDrivingSession, readRecordedDrivingObservation } from './recorded-driving-session';
import { referenceLapFixture, trainingFixture } from './training-fixtures';

function setup(train = true, selectSample?: (group: string) => boolean) {
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
        observeDemonstration: jest.fn(),
        learnPlannedActionsAsync: jest.fn(async (_input: DrivingPolicyInput, _targets: unknown) => undefined),
        decideAsync: jest.fn(async (_input: DrivingPolicyInput, _explore: boolean, _shifts: unknown) => ({
            plan: [{ action: { pedal: 0.5, steering: 0.2 } }],
        } as PolicyDecision)),
        releaseDecision: jest.fn(),
    };
    const replay = new RecordedDrivingReplay(fake as unknown as DrivingPolicy, train, selectSample);
    const lap = referenceLapFixture();
    const frame = (at: number, overrides: Partial<DrivingObservation> = {}): DrivingObservation => ({
        ...trainingFixture(at, 50, lap).observation, ...overrides,
    });
    return { fake, replay, frame };
}

it('balances distinct clip labels, games, tracks and saved ranges', () => {
    const { fake, frame } = setup();
    const { circuitMap } = trainingFixture();
    circuitMap.centerline_segments = [{ id: 'wrap', tags: ['corner'], start_position: 0.9, end_position: 0.1 },
        { id: 'longer-wrap', tags: ['consecutive corners'], start_position: 0.85, end_position: 0.15 },
        { id: 'separate', tags: ['corner'], start_position: 0.3, end_position: 0.4 }];
    const origin = frame(1000, { normalizedPosition: 0.95 });
    const group = recordedSampleGroup(origin, ' Racing ', circuitMap);
    expect(recordedSampleGroup({ ...origin, normalizedPosition: 0.05 }, 'racing', circuitMap)).toBe(group);
    const otherGroups = [
        recordedSampleGroup(origin, undefined, circuitMap),
        recordedSampleGroup({ ...origin, context: { ...origin.context, game: 'iracing' } }, 'racing', circuitMap),
        recordedSampleGroup({ ...origin, context: { ...origin.context, map: 'spa' } }, 'racing', circuitMap),
        recordedSampleGroup({ ...origin, normalizedPosition: 0.35 }, 'racing', circuitMap),
        recordedSampleGroup({ ...origin, normalizedPosition: 0.14 }, 'racing', circuitMap),
        recordedSampleGroup({ ...origin, normalizedPosition: 0.65 }, 'racing', circuitMap),
    ];
    expect(new Set([group, ...otherGroups]).size).toBe(7);

    const balance = new RecordedDrivingBalance(fake.configuration);
    const countPair = (key: string) => {
        balance.count(origin, key, true);
        balance.count(frame(1100, { normalizedPosition: 0.96 }), 'successor');
    };
    for (let i = 0; i < 6; i++) countPair(group);
    otherGroups.forEach(countPair);
    balance.count(undefined, undefined);
    balance.count(origin, group);
    balance.count(frame(1300, { normalizedPosition: 0.96 }), group);
    balance.prepare();
    expect(Array.from({ length: 6 }, () => balance.select(group))).toEqual([false, false, true, false, false, false]);
    for (const other of otherGroups) expect([balance.select(other), balance.select(other)]).toEqual([true, false]);
    expect(balance.select('successor')).toBe(false);
});

it('uses the shortest matching centerline range and one owner for a shared endpoint', () => {
    const { observation, circuitMap } = trainingFixture();
    const map = { ...circuitMap, centerline_segments: [
        { id: 'lap', tags: ['lap'], start_position: 0, end_position: 1 },
        { id: 'finish', tags: ['straight'], start_position: 0.9, end_position: 1 },
        { id: 'start', tags: ['straight'], start_position: 1, end_position: 0.1 },
        { id: 'corner', tags: ['corner'], start_position: 0.1, end_position: 0.5 },
        { id: 'apex', tags: ['apex'], start_position: 0.2, end_position: 0.3 },
        { id: 'invalid', tags: ['empty'], start_position: 1, end_position: 0 },
    ] };
    const at = (normalizedPosition: number) => recordedSampleGroup({ ...observation, normalizedPosition }, undefined, map);
    expect(at(0)).toBe(at(1));
    expect(at(1)).toBe(at(0.05));
    expect(at(0.95)).not.toBe(at(0));
    expect(at(0.1)).toBe(at(0.4));
    expect(at(0.2)).toBe(at(0.25));
    expect(at(0.25)).not.toBe(at(0.4));
    expect(at(0.3)).toBe(at(0.4));

    const fullLapMap = { ...map, centerline_segments: map.centerline_segments.slice(0, 1) };
    const fullLapAt = (normalizedPosition: number) => recordedSampleGroup({ ...observation, normalizedPosition }, undefined, fullLapMap);
    expect(fullLapAt(0)).toBe(fullLapAt(1));
    expect(fullLapAt(1)).toBe(at(0.75));
    expect(fullLapAt(1)).not.toBe(recordedSampleGroup(observation, undefined, null));
});

it('keeps omitted origins in recurrent history and uses their controls for retained plans', async () => {
    const { fake, replay, frame } = setup(true, (group) => group === 'selected');
    await replay.consume({ observation: frame(1000) }, undefined, 'selected');
    await replay.consume({ observation: frame(1100) }, undefined, 'omitted');
    await replay.consume({ observation: frame(1200, { controls: { throttle: 0.4, brake: 0, steering: 0 } }) }, undefined, 'selected');
    await replay.consume({ observation: frame(1300) }, undefined, 'omitted');
    expect(fake.learnDemonstrationAsync.mock.calls.map(([input]) => input.recurrent?.capturedAt)).toEqual([1000, 1200]);
    expect(fake.observeDemonstration).toHaveBeenCalledWith(expect.objectContaining({ recurrent: expect.objectContaining({ capturedAt: 1100 }) }));
    expect(fake.learnDemonstrationAsync.mock.calls[1][0].recurrent?.historyTimes).toEqual([1100, 1000, null]);
    expect(fake.learnPlannedActionsAsync.mock.calls.map(([input]) => input.recurrent?.capturedAt)).toEqual([1000, 1000]);
    expect(fake.learnPlannedActionsAsync.mock.calls[0][1]).toEqual([
        expect.objectContaining({ step: 1, action: expect.objectContaining({ pedal: 0.4 }) }),
    ]);
    expect(replay.progress).toMatchObject({ samples: 2, omittedForBalance: 1 });
});

it('learns next-sample controls from earlier observations and shifts future analog targets by one sample', async () => {
    const { fake, replay, frame } = setup();
    const first = frame(1000, { gear: 3 });
    const second = frame(1100, { gear: 4, controls: { throttle: 0, brake: 0.6, steering: -0.4 } });
    const third = frame(1200, { gear: 3, controls: { throttle: 0.15, brake: 0, steering: 0.25 } });
    await replay.consume({ observation: first });
    expect(fake.learnDemonstrationAsync).not.toHaveBeenCalled();
    await replay.consume({ observation: second });
    await replay.consume({ observation: third });
    expect(fake.learnDemonstrationAsync).toHaveBeenCalledTimes(2);
    const firstInput = fake.learnDemonstrationAsync.mock.calls[0][0];
    const secondInput = fake.learnDemonstrationAsync.mock.calls[1][0];
    expect(firstInput.recurrent?.capturedAt).toBe(first.capturedAt);
    expect(secondInput.recurrent?.capturedAt).toBe(second.capturedAt);
    expect(firstInput.features.slice(0, 9)).toEqual(new Array(9).fill(0));
    expect(secondInput.features.slice(0, 3)).toEqual([1, 1 / 3, first.speed / fake.configuration.observation.speedScale]);
    expect(firstInput.cameras.slice(0, 3)).toEqual([null, null, null]);
    expect(secondInput.cameras[0]).toEqual(first.camera);
    expect(secondInput.cameras[0]).not.toBe(first.camera);
    expect(fake.learnDemonstrationAsync.mock.calls[0].slice(1)).toEqual([
        { pedal: -0.6, steering: -0.4, shiftUp: true, shiftDown: false }, { up: true, down: true },
    ]);
    const delayedTargets = fake.learnPlannedActionsAsync.mock.calls.find(([input]) =>
        input.recurrent?.capturedAt === first.capturedAt)?.[1];
    expect(delayedTargets).toEqual([{ step: 1,
        action: { pedal: 0.15, steering: 0.25, shiftUp: false, shiftDown: true }, shifts: { up: true, down: true } }]);
    expect(replay.progress.samples).toBe(2);
});

it('labels immediate and future recorded targets with brake throughout pedal overlap', async () => {
    const { fake, replay, frame } = setup();
    const pedals = [[0.6, 0], [0.7, 0.2], [0.7, 0.2], [0.8, 0.1], [0.8, 0], [0, 0]];
    for (let i = 0; i < pedals.length; i++) {
        const [throttle, brake] = pedals[i];
        await replay.consume({ observation: frame(1000 + i * 100, { controls: { throttle, brake, steering: 0 } }) });
    }
    expect(fake.learnDemonstrationAsync.mock.calls.map(([, action]) => action)).toEqual(
        [-0.2, -0.2, -0.1, 0.8, 0].map((pedal) => expect.objectContaining({ pedal })));
    expect(fake.learnPlannedActionsAsync).toHaveBeenCalledWith(fake.learnDemonstrationAsync.mock.calls[0][0], [
        expect.objectContaining({ step: 1, action: expect.objectContaining({ pedal: -0.2 }) }),
    ]);
    expect(fake.learnPlannedActionsAsync).toHaveBeenCalledWith(fake.learnDemonstrationAsync.mock.calls[2][0], [
        expect.objectContaining({ step: 1, action: expect.objectContaining({ pedal: 0.8 }) }),
    ]);

    await replay.consume({ error: 'Missing recorded frame.' });
    await replay.consume({ observation: frame(2000) });
    await replay.consume({ observation: frame(2100, { controls: { throttle: 0.6, brake: 0.1, steering: 0 } }) });
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

it.each(['gap', 'missing replay tick', 'session', 'reference', 'position', 'duplicate clock', 'telemetry still at the prior frame', 'range start'])(
    'drops history and pending future labels after %s', async (kind) => {
        const { fake, replay, frame } = setup();
        await replay.consume({ observation: frame(1000) });
        const previous = frame(1100);
        if (kind === 'telemetry still at the prior frame') previous.measuredAt = 1050;
        await replay.consume({ observation: previous });
        const timestamps: Record<string, number> = { gap: 3000, 'missing replay tick': 1300, 'duplicate clock': 1100 };
        const at = timestamps[kind] ?? 1200;
        const changed = frame(at);
        if (kind === 'session') changed.session = 'new-recorded-session';
        if (kind === 'reference') changed.reference = { ...changed.reference };
        if (kind === 'position') changed.normalizedPosition += 0.3;
        if (kind === 'telemetry still at the prior frame') changed.measuredAt = previous.capturedAt;
        if (kind === 'range start') replay.startRange();
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

it('evaluates fixed weights against next-sample controls and releases each decision', async () => {
    const { fake, replay, frame } = setup(false);
    await replay.consume({ observation: frame(1000, { controls: { throttle: 0.3, brake: 0, steering: 0 } }) });
    await replay.consume({ observation: frame(1100, { controls: { throttle: 0, brake: 0.5, steering: -0.4 } }) });
    await replay.consume({ observation: frame(1200, { controls: { throttle: 0.9, brake: 0, steering: 0.1 } }) });
    expect(replay.progress.samples).toBe(2);
    expect(replay.progress.pedalError).toBeCloseTo(0.7);
    expect(replay.progress.steeringError).toBeCloseTo(0.35);
    expect(fake.decideAsync.mock.calls.map(([input]) => input.recurrent?.capturedAt)).toEqual([1000, 1100]);
    expect(fake.decideAsync.mock.calls.map(([, explore]) => explore)).toEqual([false, false]);
    expect(fake.releaseDecision).toHaveBeenCalledTimes(2);
    expect(fake.trainBatchAsync).not.toHaveBeenCalled();
    expect(fake.learnDemonstrationAsync).not.toHaveBeenCalled();
    expect(fake.learnPlannedActionsAsync).not.toHaveBeenCalled();
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
    expect(fake.learnDemonstrationAsync).not.toHaveBeenCalled();
    expect(fake.learnPlannedActionsAsync).not.toHaveBeenCalled();
    expect(replay.progress.samples).toBe(0);
});
