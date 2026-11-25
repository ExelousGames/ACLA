import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingPolicy, type PlannedActionTarget } from './driving-policy';
import { DrivingMemory } from './driving-memory';
import { cloneConfig, emptyRecordedSamplingWeights, RECORDED_SAMPLING_CATEGORIES, type RecordedSamplingWeights } from './driving-model-config';
import { trainingFixture } from './training-fixtures';

function setup(weights?: RecordedSamplingWeights, random?: () => number) {
    const { observation } = trainingFixture();
    const config = cloneConfig();
    config.actions.planTimesSeconds = Array.from({ length: 31 }, (_, index) => index / 10);
    const policy = new DrivingPolicy(observation.context, observation.features.length, Math.random, config);
    const learn = jest.spyOn(policy, 'learnRecordedPlanAsync').mockResolvedValue(undefined);
    const batch = jest.spyOn(policy, 'learnRecordedBatchAsync').mockImplementation(async (plans) => {
        for (const { input, targets } of plans) await policy.learnRecordedPlanAsync(input, targets);
        return true;
    });
    return { policy, learn, batch, training: new DrivingPlanTraining(weights, random), features: new DrivingMemory().encode(observation) };
}
const shifts = { up: true, down: true };
const neutralTarget = { action: { pedal: 0, steering: 0 }, shifts };

it('keeps immediate and delayed targets together using the original input snapshot', async () => {
    const { policy, learn, training, features } = setup();
    const original = { ...features, features: features.features.slice(), cameras: features.cameras.slice() };
    await training.recordAsync(policy, features, neutralTarget, 1000);
    features.features[0] = 1;
    const targets: PlannedActionTarget[] = [{ step: 0, ...neutralTarget }];
    for (let step = 1; step <= 30; step++) {
        const action = { pedal: step / 30, steering: -step / 30, shiftUp: step === 3 };
        expect(learn).not.toHaveBeenCalled();
        await training.recordAsync(policy, undefined, { action, shifts }, 1000 + step * 100);
        targets.push({ step, action, shifts });
    }
    expect(learn).not.toHaveBeenCalled();
    await training.finishAsync(policy);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn).toHaveBeenCalledWith(original, targets);
    expect(learn.mock.calls[0][0].recurrent).toBe(original.recurrent);
    expect(training.hasPending).toBe(false);
});

it('waits for a time slot, tolerates small sampling offsets and skips missing targets', async () => {
    const { policy, learn, training, features } = setup();
    await training.recordAsync(policy, features, neutralTarget, 1000);
    await training.recordAsync(policy, undefined, neutralTarget, 1099);
    const action = { pedal: -0.1, steering: -0.5 };
    await training.recordAsync(policy, undefined, { action, shifts }, 1110);
    await training.recordAsync(policy, undefined, neutralTarget, 1361);
    await training.recordAsync(policy, undefined, neutralTarget, 1400);
    expect(learn).not.toHaveBeenCalled();
    await training.finishAsync(policy);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn).toHaveBeenCalledWith(features, [{ step: 0, ...neutralTarget }, { step: 1, action, shifts },
        { step: 4, ...neutralTarget }]);
});

it.each([
    { edge: 'the scheduled time', origin: 481.1, frame: 481.2, boundaryMs: 0, direction: -1 },
    { edge: 'the late limit', origin: 481.2, frame: 481.35, boundaryMs: 50, direction: 1 },
])('keeps brake and steering labels when frame time rounds outside $edge', async ({ origin, frame, boundaryMs, direction }) => {
    const { policy, learn, training, features } = setup();
    const recordedAt = (seconds: number) => (0.236614 + seconds) * 1000;
    const originMs = recordedAt(origin), frameMs = recordedAt(frame);
    expect(Math.sign(frameMs - (originMs + 100) - boundaryMs)).toBe(direction);

    await training.recordAsync(policy, features, neutralTarget, originMs);
    const action = { pedal: -0.55, steering: 0.4 };
    await training.recordAsync(policy, undefined, { action, shifts }, frameMs);
    await training.recordAsync(policy, undefined, neutralTarget, frameMs + 1);
    await training.finishAsync(policy);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn).toHaveBeenCalledWith(features, [{ step: 0, ...neutralTarget }, { step: 1, action, shifts }]);
});

it('flushes partial origins in order once without inventing missing targets', async () => {
    const { policy, learn, training, features } = setup();
    await training.recordAsync(policy, features, neutralTarget, 1000);
    const next = { ...features, features: [1, ...features.features.slice(1)] };
    const target = { action: { pedal: -0.5, steering: 0.4 }, shifts };
    await training.recordAsync(policy, next, target, 1100);
    await training.finishAsync(policy);
    await training.finishAsync(policy);
    expect(learn.mock.calls).toEqual([
        [features, [{ step: 0, ...neutralTarget }, { step: 1, ...target }]],
        [next, [{ step: 0, ...target }]],
    ]);
    expect(training.hasPending).toBe(false);
});

it('does not collect a cancelled frame or submit additional origins after cancellation', async () => {
    const { policy, learn, batch, training, features } = setup();
    let active = true;
    await training.recordAsync(policy, features, neutralTarget, 1000);
    await training.recordAsync(policy, features, neutralTarget, 1050);
    await training.recordAsync(policy, features, neutralTarget, 1100, () => false);
    batch.mockImplementationOnce(async () => { active = false; return false; });
    await training.finishAsync(policy, () => active);
    expect(learn).not.toHaveBeenCalled();
    expect(training.hasPending).toBe(true);

    await training.finishAsync(policy); // A graceful stop can flush targets observed before cancellation.
    expect(learn.mock.calls).toEqual([
        [features, [{ step: 0, ...neutralTarget }]], [features, [{ step: 0, ...neutralTarget }]],
    ]);
    expect(training.hasPending).toBe(false);
});

it('fills each optimizer batch from preset category percentages and reuses rare examples', async () => {
    const { policy, batch, training, features } = setup({ ...emptyRecordedSamplingWeights(), 'Race pace': 50, Cornering: 25, Shifting: 25 });
    for (let i = 0; i < 16; i++) {
        const category = i === 15 ? 'Shifting' : i === 14 ? 'Cornering' : 'Race pace';
        const action = { pedal: 0.5, steering: i === 14 ? 0.4 : 0, shiftUp: i === 15 };
        await training.recordAsync(policy, features, { action, shifts }, 1000 + i * 100, undefined, category);
    }
    await training.finishAsync(policy);
    expect(batch).toHaveBeenCalledTimes(2);
    for (const [plans] of batch.mock.calls) {
        const immediate = plans.map(({ targets }) => targets[0].action);
        expect(immediate.filter(({ steering, shiftUp }) => steering === 0 && !shiftUp)).toHaveLength(4);
        expect(immediate.filter(({ steering }) => steering === 0.4)).toHaveLength(2);
        expect(immediate.filter(({ shiftUp }) => shiftUp)).toHaveLength(2);
    }
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), 'Race pace': 8, Cornering: 4, Shifting: 4 });
});

it('waits for late categories and retains a cross-section of earlier origins for their fixed mix', async () => {
    const weights = { ...emptyRecordedSamplingWeights(), 'Race pace': 60, Cornering: 25, Shifting: 15 };
    const { policy, batch, training, features } = setup(weights, () => 0.99);
    for (let i = 0; i < 400; i++) {
        const input = { ...features, features: [i, ...features.features.slice(1)] };
        const category = i === 399 ? 'Shifting' : i === 398 ? 'Cornering' : 'Race pace';
        await training.recordAsync(policy, input, neutralTarget, 1000 + i * 100, undefined, category);
    }
    expect(batch).not.toHaveBeenCalled();
    for (let i = 400; i < 430; i++) {
        await training.recordAsync(policy, undefined, neutralTarget, 1000 + i * 100);
    }
    expect(batch).toHaveBeenCalledTimes(32);
    await training.finishAsync(policy);
    expect(batch).toHaveBeenCalledTimes(50);
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), 'Race pace': 240, Cornering: 100, Shifting: 60 });
    expect(batch.mock.calls.flatMap(([plans]) => plans).some(({ input }) => input.features[0] === 0)).toBe(true);
    for (const [plans] of batch.mock.calls) {
        expect(plans.filter(({ input }) => input.features[0] === 398)).toHaveLength(2);
        expect(plans.filter(({ input }) => input.features[0] === 399).length).toBeGreaterThanOrEqual(1);
    }
});

it('uses available categories to flush a stopped replay before every measured category arrives', async () => {
    const { policy, batch, training, features } = setup({ ...emptyRecordedSamplingWeights(), Traffic: 60, Cornering: 25, Shifting: 15 });
    for (let i = 0; i < 8; i++) {
        await training.recordAsync(policy, features, neutralTarget, 1000 + i * 100, undefined, i < 7 ? 'Traffic' : 'Cornering');
    }
    await training.finishAsync(policy, undefined, false);
    expect(batch).not.toHaveBeenCalled();
    await training.finishAsync(policy);
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), Traffic: 6, Cornering: 2 });
});

it('draws from all configured labels across batches smaller than the number of categories', async () => {
    const weights = emptyRecordedSamplingWeights();
    const expected = emptyRecordedSamplingWeights();
    for (const category of RECORDED_SAMPLING_CATEGORIES) {
        weights[category] = 100 / RECORDED_SAMPLING_CATEGORIES.length;
        expected[category] = 1;
    }
    const { policy, training, features } = setup(weights);
    for (let index = 0; index < RECORDED_SAMPLING_CATEGORIES.length; index++) {
        const category = RECORDED_SAMPLING_CATEGORIES[index];
        await training.recordAsync(policy, features, neutralTarget, 1000 + index * 100, undefined, category);
    }
    await training.finishAsync(policy);
    expect(training.sampled).toEqual(expected);
});
