import { DrivingTrainingLoop } from './driving-training-loop';
import { DrivingPolicy } from './driving-policy';
import { NEUTRAL, ObservationResult, readDrivingObservation } from './driving-observation';
import { trainingFixture } from './training-fixtures';
import { normalizedAggression, referencePedalTargets } from './driving-reference';
import { ManualControls } from './driving-control';
import { DrivingControlOutput } from './driving-control-output';
import { DrivingMemory, DRIVING_MEMORY_INPUTS } from './driving-memory';

const flush = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };
beforeEach(() => { jest.useFakeTimers(); });
afterEach(() => { jest.useRealTimers(); });

it.each([true, false])('publishes one plan, applies only its first action and replans on fresh inputs (training=%s)', async (train) => {
    const run = setup();
    const decide = jest.spyOn(run.policy, 'decide');
    const planned = jest.spyOn(run.policy, 'learnPlannedActions');
    run.loop.start(run.policy, train); await flush();
    const plan = run.loop.state.plan!;
    expect(plan.map(({ timeSeconds }) => timeSeconds)).toEqual([0, 0.5, 1, 1.5, 2, 2.5, 3]);
    expect(run.controller.apply).toHaveBeenLastCalledWith(plan[0].action);
    run.input({ observation: run.fixture.observation }, 1100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.plan).toBe(plan);
    expect(decide).toHaveBeenCalledTimes(1);
    for (let capturedAt = 1200; capturedAt <= 4400; capturedAt += 200) {
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    expect(run.loop.state.plan).not.toBe(plan);
    expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.plan![0].action);
    if (train) expect(planned.mock.calls.some(([, targets]) => targets.some(({ step }) => step === 6))).toBe(true);
    else expect(planned).not.toHaveBeenCalled();
    await run.loop.stop();
    expect(run.loop.state.plan).toBeNull();
});

it.each(['stop', 'handoff', 'invalid', 'gap'] as const)('discards unfinished plan training after %s', async (boundary) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const planned = jest.spyOn(run.policy, 'learnPlannedActions');
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
    if (boundary !== 'gap') expect(run.loop.state.plan).toBeNull();
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

it.each([true, false])('uses later acknowledged human actions to teach the three-second plan only in training (training=%s)', async (train) => {
    const run = setup(true);
    const planned = jest.spyOn(run.policy, 'learnPlannedActions');
    const learn = jest.spyOn(run.policy, 'learn');
    const original = new DrivingMemory().encode(run.fixture.observation);
    run.manual.mockResolvedValue({ action: NEUTRAL, toggleControl: false });
    run.loop.startManual(); await flush();
    run.loop.start(run.policy, train); await flush();
    const later = { throttle: 0.8, brake: 0, steering: -0.6 };
    for (let capturedAt = 1200; capturedAt <= 4200; capturedAt += 200) {
        run.manual.mockResolvedValue({ action: capturedAt === 4000 ? later : NEUTRAL, toggleControl: false });
        run.input({ observation: { ...run.fixture.observation, capturedAt } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    if (train) expect(planned).toHaveBeenCalledWith(original,
        [{ step: 6, action: { ...later, shiftUp: false, shiftDown: false }, shifts: { up: true, down: true } }]);
    else expect(planned).not.toHaveBeenCalled();
    expect(learn).not.toHaveBeenCalled(); // Human driving never supplies an on-policy reward gradient.
    expect(run.loop.state.plan).toBeNull();
    await run.loop.stop();
});

function setup(withManual = false) {
    const fixture = trainingFixture();
    let input: ObservationResult = { observation: fixture.observation };
    let now = 1000;
    const controller = { connect: jest.fn(), disconnect: jest.fn(), apply: jest.fn().mockResolvedValue(undefined) };
    const publish = jest.fn();
    const policy = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length, () => 0.7);
    const manual = jest.fn<Promise<ManualControls | null>, []>().mockResolvedValue(null);
    const loop = new DrivingTrainingLoop(new DrivingControlOutput(controller), () => input, publish, () => now, withManual ? manual : undefined);
    return { fixture, controller, policy, loop, publish, manual,
        input: (next: ObservationResult, time: number, measuredAt = next.observation?.capturedAt) => {
            input = next.observation ? { observation: { ...next.observation, measuredAt: measuredAt! } } : next;
            now = time;
        } };
}

it.each([true, false])('passes three seconds of history to decisions and critic learning (training=%s)', async (train) => {
    const run = setup();
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const decide = jest.spyOn(run.policy, 'decide');
    const learn = jest.spyOn(run.policy, 'learn');
    run.loop.start(run.policy, train); await flush();
    const first = decide.mock.results[0].value;
    expect(first.features.slice(0, DRIVING_MEMORY_INPUTS)).toEqual(new Array(DRIVING_MEMORY_INPUTS).fill(0));
    for (let capturedAt = 1200; capturedAt <= 4000; capturedAt += 200) {
        const features = run.fixture.observation.features.slice();
        features[3] = 0.5;
        run.input({ observation: { ...run.fixture.observation, capturedAt, features } }, capturedAt);
        jest.advanceTimersByTime(200); await flush();
    }
    const current = decide.mock.calls[decide.mock.calls.length - 1][0];
    expect(current[14 * 22 + 1]).toBe(1); // Oldest input is exactly three seconds old.
    expect(current[14 * 22 + 5]).toBe(run.fixture.observation.features[3]);
    expect(learn).toHaveBeenCalledTimes(train ? 15 : 0);
    expect(learn.mock.calls[0]?.[0]).toBe(train ? first : undefined);
    expect(learn.mock.calls[learn.mock.calls.length - 1]?.[2]).toEqual(train ? current : undefined);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 4200 } }, 4200);
    jest.advanceTimersByTime(200); await flush();
    expect(decide.mock.calls[decide.mock.calls.length - 1][0][14 * 22 + 5]).toBe(0.5);
    await run.loop.stop();
});

it.each(['stop', 'handoff', 'invalid', 'gap'] as const)('discards history after %s', async (boundary) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const decide = jest.spyOn(run.policy, 'decide');
    run.manual.mockResolvedValue({ action: { throttle: 0.5, brake: 0, steering: 0 }, toggleControl: false });
    run.loop.start(run.policy, false); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(200); await flush();
    expect(decide.mock.calls[decide.mock.calls.length - 1][0][0]).toBe(1);
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
    expect(decide.mock.calls[decide.mock.calls.length - 1][0].slice(0, DRIVING_MEMORY_INPUTS).every((value) => value === 0)).toBe(true);
    await run.loop.stop();
});

it('learns human demonstrations with the history available before that action', async () => {
    const run = setup(true);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstration');
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
        expect.objectContaining(action), expect.anything());
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
    const run = setup(true);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstration');
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
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1300 } }, 1300);
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
    const learn = jest.spyOn(run.policy, 'learn');
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstration');
    const manual = { action: { throttle: 0.8, brake: 0.1, steering: -0.6 }, toggleControl: false };
    const before = run.policy.checkpoint();
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
    expect(demonstrate).toHaveBeenCalledTimes(train ? 1 : 0);
    if (train) expect(demonstrate).toHaveBeenLastCalledWith(new DrivingMemory().encode(run.fixture.observation),
        expect.objectContaining(manual.action), expect.anything());
    run.manual.mockResolvedValue({ ...manual, action: { ...manual.action, steering: 0.4 } });
    await tick(1300, 1350);
    expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining({ steering: 0.4 }));
    expect(demonstrate).toHaveBeenCalledTimes(train ? 1 : 0);
    run.loop.setControlSource('ai');
    await tick(1300, 1400);
    expect(run.loop.state).toMatchObject({ source: 'ai', action: NEUTRAL });
    await tick(1500);
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(learn).not.toHaveBeenCalled(); // No mixed-driver transition on return either.
    if (!train) expect(run.policy.checkpoint()).toEqual(before);
    await run.loop.stop();
});

it.each([true, false])('skips delayed transitions and resumes from fresh data across repeated handoffs (training=%s)', async (train) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    const learn = jest.spyOn(run.policy, 'learn');
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstration');
    run.manual.mockResolvedValue({ action: { throttle: 0.6, brake: 0, steering: 0.1 }, toggleControl: false });
    const frame = async (capturedAt: number, time = capturedAt) => {
        const paired = { ...run.fixture.live, receivedAt: capturedAt,
            currentTelemetry: { ...run.fixture.live.currentTelemetry, Physics_speed_kmh: 100 + capturedAt / 1000 } };
        const input = readDrivingObservation({ ...paired, receivedAt: time },
            { ...run.fixture.vision, capturedAt, telemetry: paired }, run.fixture.referenceLap, 50, time);
        expect(input.error).toBeUndefined();
        run.input(input, time);
        jest.advanceTimersByTime(50); await flush();
        return input.observation!;
    };
    const initial = run.policy.checkpoint();
    run.loop.start(run.policy, train); await flush();
    let time = 2000;
    for (const source of ['manual', 'ai', 'manual', 'ai'] as const) {
        run.input({ observation: { ...run.fixture.observation, capturedAt: time } }, time);
        run.loop.setControlSource(source);
        const steps = run.policy.steps;
        await frame(time + 100);
        expect(run.policy.steps).toBe(steps); // Never learn across the ownership change.
        await frame(time + 300, time + 1200);
        expect(run.policy.steps).toBe(steps + (train ? 1 : 0));
        const beforeGap = run.policy.checkpoint();
        const rewardBeforeGap = run.loop.state.episodeReward;
        // Fresh image, but captured before the previous command's acknowledgement.
        await frame(time + 1100, time + 2000);
        expect(run.policy.checkpoint()).toEqual(beforeGap);
        // The next usable image is fresh, but 1.6 seconds after the retained sample.
        const restart = await frame(time + 1900, time + 2800);
        expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source,
            reward: null, episodeReward: rewardBeforeGap });
        expect(run.policy.checkpoint()).toEqual(beforeGap);
        expect(run.controller.apply).toHaveBeenLastCalledWith(run.loop.state.action);
        expect(run.loop.state.action).not.toEqual(NEUTRAL);
        await frame(time + 2900);
        expect(run.policy.steps).toBe(steps + (train ? 2 : 0));
        if (train && source === 'manual') expect(demonstrate).toHaveBeenLastCalledWith(
            new DrivingMemory().encode(restart), expect.anything(), expect.anything());
        if (train && source === 'ai') expect(learn).toHaveBeenLastCalledWith(
            expect.objectContaining({ features: new DrivingMemory().encode(restart) }), expect.any(Number), expect.anything(), 1);
        time += 5000;
    }
    if (!train) {
        expect(run.policy.checkpoint()).toEqual(initial);
        expect(learn).not.toHaveBeenCalled();
        expect(demonstrate).not.toHaveBeenCalled();
    }
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
    ['session', 'reference', 'telemetry', 'capture', 'range', 'segmentation', 'context', 'features'].map((fault) => ({ source, fault }))))(
    'preserves $fault validation across a long gap under $source control', async ({ source, fault }) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    run.manual.mockResolvedValue({ action: { throttle: 0.5, brake: 0, steering: 0 }, toggleControl: false });
    run.loop.start(run.policy, true); await flush();
    if (source === 'manual') run.loop.setControlSource(source);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(50); await flush();
    const before = run.policy.checkpoint();
    const next = trainingFixture(2800, 50, fault === 'reference' ? undefined : run.fixture.referenceLap);
    if (fault === 'session') {
        next.live.sessionGeneration++;
        next.vision.telemetry!.sessionGeneration++;
    }
    if (fault === 'telemetry') next.live.receivedAt = 2299;
    if (fault === 'capture') next.vision.capturedAt = 1299;
    if (fault === 'range') next.vision.telemetry = { ...next.vision.telemetry!,
        currentTelemetry: { ...next.vision.telemetry!.currentTelemetry, Physics_gas: 2 } };
    const segment = next.vision.detections.segment;
    if (fault === 'segmentation' && segment?.task === 'segment') segment.instances[0].mask = new Uint8Array(1);
    const input = readDrivingObservation(next.live, next.vision, next.referenceLap, 50, 2800);
    if (fault === 'context') input.observation!.context = { ...input.observation!.context, car: 'different' };
    if (fault === 'features') input.observation!.features[0] = NaN;
    run.input(input, 2800);
    jest.advanceTimersByTime(50); await flush();
    expect(run.policy.checkpoint()).toEqual(before);
    if (source === 'ai') {
        expect(run.loop.state).toMatchObject({ mode: 'training', source: 'ai', action: NEUTRAL });
        expect(run.loop.state.status).toContain('retrying');
        expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    } else {
        expect(run.loop.state).toMatchObject({ mode: 'training', source: 'manual' });
        expect(run.loop.state.status).toContain('learning paused');
        expect(() => run.loop.setControlSource('ai')).toThrow();
    }
    expect(run.loop.state.status).toMatch(/session changed|reference changed|fresh, running live telemetry|fresh image|valid ranges|Invalid segmentation|matching model inputs|Invalid policy observation/);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 3000 } }, 3000);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ mode: 'training', source });
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(run.policy.checkpoint()).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 3100 } }, 3100);
    jest.advanceTimersByTime(50); await flush();
    expect(run.policy.steps).toBe(before.steps + 1);
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
    run.loop.start(run.policy, true);
    await run.loop.stop();
    resolve({ action: { throttle: 1, brake: 0, steering: 1 }, toggleControl: true });
    await flush();
    expect(run.controller.apply).toHaveBeenCalledTimes(1);
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.steps).toBe(0);
});

it('pulses each human paddle pull once and preserves cooldown across handovers', async () => {
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

it('learns a human shift acknowledged between images, after the pulse has released', async () => {
    const run = setup(true);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstration');
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
    expect(demonstrate).toHaveBeenCalledTimes(1);
    expect(demonstrate).toHaveBeenLastCalledWith(new DrivingMemory().encode(run.fixture.observation),
        expect.objectContaining({ ...input.action, shiftDown: true }), { up: true, down: true });
    expect(run.loop.state.action.shiftDown).toBe(false);
    await run.loop.stop();
});

it('does not learn from a failed manual write or a handover across a session boundary', async () => {
    const run = setup(true);
    const demonstrate = jest.spyOn(run.policy, 'learnDemonstration');
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
    const memory = new DrivingMemory();
    memory.encode(run.fixture.observation);
    expect(learn).toHaveBeenLastCalledWith(expect.anything(), reward.total,
        memory.encode({ ...run.fixture.observation, capturedAt: 1200 }), 0.2);
    expect(run.loop.state.episodeReward).toBe(reward.total);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(1);
    await run.loop.stop();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.loop.state.mode).toBe('idle');
});

it.each([true, false])('retries invalid inputs and resumes only from fresh data without losing progress (training=%s)', async (train) => {
    const run = setup();
    run.loop.start(run.policy, train); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    const before = run.policy.checkpoint();
    const episodeReward = run.loop.state.episodeReward;
    for (const time of [1300, 1400, 5000]) {
        run.input({ error: 'Telemetry disconnected.' }, time);
        jest.advanceTimersByTime(100); await flush();
        expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source: 'ai',
            action: NEUTRAL, reward: null, episodeReward, steps: before.steps });
        expect(run.loop.state.status).toContain('waiting for valid inputs; retrying: Telemetry disconnected.');
        expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
        expect(run.policy.checkpoint()).toEqual(before);
    }
    // Neither an old image nor telemetry measured before neutralization can resume AI.
    run.input({ observation: { ...run.fixture.observation, capturedAt: 4900 } }, 5100);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 5200 } }, 5200, 4900);
    jest.advanceTimersByTime(100); await flush();
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.loop.state.status).toContain('retrying');
    expect(run.policy.checkpoint()).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 5300 } }, 5300);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(run.loop.state.status).not.toContain('retrying');
    expect(run.loop.state.episodeReward).toBe(episodeReward);
    expect(run.policy.checkpoint()).toEqual(before); // No learning across the interrupted transition.
    run.input({ observation: { ...run.fixture.observation, capturedAt: 5400 } }, 5400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.steps).toBe(before.steps + (train ? 1 : 0));
    if (!train) expect(run.policy.checkpoint()).toEqual(before);
    await run.loop.stop();
});

it.each((['ai', 'manual'] as const).flatMap((source) => [true, false].map((train) => ({ source, train }))))(
    'discards a lap-position jump and retries under $source control (training=$train)', async ({ source, train }) => {
    const run = setup(true);
    const manual = { action: { throttle: 0.3, brake: 0, steering: 0.1 }, toggleControl: false };
    run.manual.mockResolvedValue(manual);
    if (source === 'manual') { run.loop.startManual(); await flush(); }
    run.loop.start(run.policy, train); await flush();
    const before = run.policy.checkpoint();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1050, normalizedPosition: 0.5 } }, 1050);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state).toMatchObject({ mode: train ? 'training' : 'evaluation', source,
        action: source === 'ai' ? NEUTRAL : manual.action, reward: null });
    expect(run.loop.state.status).toContain('Lap position jumped');
    expect(run.policy.checkpoint()).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1100 } }, 1100);
    jest.advanceTimersByTime(50); await flush();
    expect(run.loop.state.status).not.toContain('Lap position jumped');
    expect(run.loop.state.action).not.toEqual(NEUTRAL);
    expect(run.policy.checkpoint()).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(50); await flush();
    expect(run.policy.steps).toBe(train ? 1 : 0);
    if (!train) expect(run.policy.checkpoint()).toEqual(before);
    await run.loop.stop();
});

it('does not learn during evaluation, and waits on session changes', async () => {
    const run = setup();
    const before = run.policy.checkpoint();
    run.loop.start(run.policy, false); await flush();
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1200 } }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.policy.checkpoint()).toEqual(before);
    run.input({ observation: { ...run.fixture.observation, session: 'different', capturedAt: 1400 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('evaluation');
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    expect(run.policy.checkpoint()).toEqual(before);
    await run.loop.stop();
});

it.each(['stop', 'manual'] as const)('honors %s while a retry neutralization is awaiting acknowledgement', async (next) => {
    const run = setup(true);
    run.policy = new DrivingPolicy(run.fixture.observation.context, run.fixture.observation.features.length, () => 0.1);
    run.loop.start(run.policy, true); await flush();
    let acknowledge!: () => void;
    run.controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    run.input({ error: 'Invalid segmentation mask.' }, 1100);
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
    if (next === 'stop') {
        expect(run.loop.state).toMatchObject({ mode: 'idle', source: null, action: NEUTRAL });
        expect(run.controller.apply).toHaveBeenCalledTimes(count);
    } else {
        expect(run.loop.state).toMatchObject({ mode: 'training', source: 'manual', action: manual.action });
        expect(run.controller.apply).toHaveBeenLastCalledWith(expect.objectContaining(manual.action));
    }
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
    expect(learn.mock.calls[0][0].plan[0].action.shiftUp).toBe(true);
    expect(run.loop.state.action).toMatchObject({ shiftUp: false, shiftDown: false });
    run.input({ observation: { ...run.fixture.observation, capturedAt: 1400, gear: 4, normalizedPosition: 0.252 } }, 1400);
    jest.advanceTimersByTime(100); await flush();
    expect(learn.mock.calls[1][0].plan[0].shiftProbabilities).toEqual([1, 0, 0]);
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
    checkpoint.outputWeights.slice(3, 6).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === 2 ? 4 : 0; });
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
    if (train) expect(learn.mock.calls[0][0].plan[0].action.shiftDown).toBe(true);
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
    await flush();
    const sent = run.controller.apply.mock.calls[0][0];
    expect(sent).toEqual(decide.mock.results[0].value.plan[0].action);
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
    const memory = new DrivingMemory();
    memory.encode(withAggression(0, 1000, 0.25));
    expect(decide).toHaveBeenLastCalledWith(memory.encode(next), train, expect.anything());
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
    expect(() => run.loop.start(run.policy, true)).toThrow(/reference lap/);
    expect(run.controller.apply).not.toHaveBeenCalled();
    run.input({ observation: run.fixture.observation }, 1000);
    run.loop.start(run.policy, true); await flush();
    run.input({ error: 'Live Map reference changed. Waiting for matching Track Vision inputs.' }, 1200);
    jest.advanceTimersByTime(100); await flush();
    expect(run.loop.state.mode).toBe('training');
    expect(run.loop.state.status).toContain('retrying');
    expect(run.policy.steps).toBe(0);
    expect(run.controller.apply).toHaveBeenLastCalledWith(NEUTRAL);
    await run.loop.stop();
});
