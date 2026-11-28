import { DrivingPlanTraining, RECORDED_PLAN_WINDOW } from './driving-plan-training';
import { DrivingPolicy, type PlannedActionTarget } from './driving-policy';
import { DrivingMemory } from './driving-memory';
import { cloneConfig, emptyRecordedSamplingWeights, RECORDED_SAMPLING_CATEGORIES, type RecordedSamplingWeights } from './driving-model-config';
import { trainingFixture } from './training-fixtures';

function setup(weights?: RecordedSamplingWeights, batchSize = 1, batchCount = 1, random = () => 0.999) {
    const { observation } = trainingFixture();
    const config = cloneConfig();
    config.actions.planTimesSeconds = Array.from({ length: 31 }, (_, index) => index / 10);
    const policy = new DrivingPolicy(observation.context, observation.features.length, Math.random, config);
    const learn = jest.spyOn(policy, 'learnRecordedPlanAsync').mockResolvedValue(undefined);
    const batch = jest.spyOn(policy, 'learnRecordedBatchAsync').mockImplementation(async (plans) => {
        for (const { input, targets } of plans) await policy.learnRecordedPlanAsync(input, targets);
        return true;
    });
    return { policy, learn, batch, training: new DrivingPlanTraining(weights, batchSize, batchCount, random),
        features: new DrivingMemory().encode(observation) };
}
const steadyTarget = { action: { pedal: 0, steering: 0, gear: 3 } };

it('keeps immediate and delayed targets together using the original input snapshot', async () => {
    const { policy, learn, training, features } = setup();
    const original = { ...features, features: features.features.slice(), cameras: features.cameras.slice() };
    training.record(policy, features, steadyTarget, 1000);
    features.features[0] = 1;
    const targets: PlannedActionTarget[] = [{ step: 0, ...steadyTarget }];
    for (let step = 1; step <= 30; step++) {
        const action = { pedal: step / 30, steering: -step / 30, gear: step >= 3 ? 4 : 3 };
        expect(learn).not.toHaveBeenCalled();
        training.record(policy, undefined, { action }, 1000 + step * 100);
        targets.push({ step, action });
    }
    expect(learn).not.toHaveBeenCalled();
    await training.finishAsync(policy);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn).toHaveBeenCalledWith(original, targets);
    expect(learn.mock.calls[0][0].recurrent).toBe(original.recurrent);
    expect(training.completedBatches).toBe(1);
});

it('waits for a time slot, tolerates small sampling offsets and skips missing targets', async () => {
    const { policy, learn, training, features } = setup();
    training.record(policy, features, steadyTarget, 1000);
    training.record(policy, undefined, steadyTarget, 1099);
    const action = { pedal: -0.1, steering: -0.5, gear: 2 };
    training.record(policy, undefined, { action }, 1110);
    training.record(policy, undefined, steadyTarget, 1361);
    training.record(policy, undefined, steadyTarget, 1400);
    expect(learn).not.toHaveBeenCalled();
    await training.finishAsync(policy);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn).toHaveBeenCalledWith(features, [{ step: 0, ...steadyTarget }, { step: 1, action },
        { step: 4, ...steadyTarget }]);
});

it.each([
    { edge: 'the scheduled time', origin: 481.1, frame: 481.2, boundaryMs: 0, direction: -1 },
    { edge: 'the late limit', origin: 481.2, frame: 481.35, boundaryMs: 50, direction: 1 },
])('keeps brake and steering labels when frame time rounds outside $edge', async ({ origin, frame, boundaryMs, direction }) => {
    const { policy, learn, training, features } = setup();
    const recordedAt = (seconds: number) => (0.236614 + seconds) * 1000;
    const originMs = recordedAt(origin), frameMs = recordedAt(frame);
    expect(Math.sign(frameMs - (originMs + 100) - boundaryMs)).toBe(direction);

    training.record(policy, features, steadyTarget, originMs);
    const action = { pedal: -0.55, steering: 0.4, gear: 2 };
    training.record(policy, undefined, { action }, frameMs);
    training.record(policy, undefined, steadyTarget, frameMs + 1);
    await training.finishAsync(policy);
    expect(learn).toHaveBeenCalledTimes(1);
    expect(learn).toHaveBeenCalledWith(features, [{ step: 0, ...steadyTarget }, { step: 1, action }]);
});

it('flushes partial origins in order once without inventing missing targets', async () => {
    const { policy, learn, training, features } = setup(undefined, 2);
    training.record(policy, features, steadyTarget, 1000);
    const next = { ...features, features: [1, ...features.features.slice(1)] };
    const target = { action: { pedal: -0.5, steering: 0.4, gear: 2 } };
    training.record(policy, next, target, 1100);
    await training.finishAsync(policy);
    await training.finishAsync(policy);
    expect(learn.mock.calls).toEqual([
        [features, [{ step: 0, ...steadyTarget }, { step: 1, ...target }]],
        [next, [{ step: 0, ...target }]],
    ]);
    expect(training.completedBatches).toBe(1);
});

it('does not count a cancelled optimizer update', async () => {
    const { policy, learn, batch, training, features } = setup(undefined, 2);
    let active = true;
    training.record(policy, features, steadyTarget, 1000);
    training.record(policy, features, steadyTarget, 1050);
    batch.mockImplementationOnce(async () => { active = false; return false; });
    await training.finishAsync(policy, () => active);
    expect(learn).not.toHaveBeenCalled();
    expect(training.completedBatches).toBe(0);
    expect(training.sampled.Unlabeled).toBe(0);

    await training.finishAsync(policy);
    expect(learn.mock.calls).toEqual([
        [features, [{ step: 0, ...steadyTarget }]], [features, [{ step: 0, ...steadyTarget }]],
    ]);
    expect(training.completedBatches).toBe(1);
});

it('retries an interrupted clip batch with the same starting frames and saved targets', async () => {
    const { policy, batch, training, features } = setup(undefined, 2, 1, () => 0.37);
    for (let clip = 0; clip < 3; clip++) {
        const input = { ...features, features: [clip, ...features.features.slice(1)] };
        const target = { action: { ...steadyTarget.action, gear: clip + 1 } };
        training.record(policy, input, target, 1000 + clip * 100, 'Unlabeled', `clip-${clip}`);
        input.features[0] = -1;
        training.completeRange();
    }

    let active = true;
    batch.mockImplementationOnce(async () => { active = false; return false; });
    await training.finishAsync(policy, () => active);
    expect(training.completedBatches).toBe(0);
    expect(training.sampled.Unlabeled).toBe(0);

    await training.finishAsync(policy);

    expect(batch).toHaveBeenCalledTimes(2);
    const [[interrupted], [retried]] = batch.mock.calls;
    expect(retried).toEqual(interrupted);
    expect(new Set(retried.map(({ input }) => input.features[0])).size).toBe(2);
    for (const { input, targets } of retried) {
        expect(input.features[0]).toBeGreaterThanOrEqual(0);
        expect(targets).toEqual([{ step: 0, action: { ...steadyTarget.action, gear: input.features[0] + 1 } }]);
    }
    expect(training.completedBatches).toBe(1);
    expect(training.sampled.Unlabeled).toBe(2);
});

it('counts a re-observed timestamp once per clip without merging separate clips', async () => {
    const { policy, batch, training, features } = setup(undefined, 2, 2, () => 0.37);
    const record = (clip: string, marker: number) => {
        const input = { ...features, features: [marker, ...features.features.slice(1)] };
        training.record(policy, input, steadyTarget, 1000, 'Unlabeled', clip);
        training.completeRange();
    };
    record('first', 1);
    record('first', 99);
    record('second', 2);

    await training.finishAsync(policy);

    expect(batch).toHaveBeenCalledTimes(2);
    for (const [plans] of batch.mock.calls) {
        expect(new Set(plans.map(({ input }) => input.features[0]))).toEqual(new Set([1, 2]));
    }
});

it('fills each optimizer batch from preset category percentages and reuses rare examples', async () => {
    const { policy, batch, training, features } = setup({ ...emptyRecordedSamplingWeights(), Traffic: 50, Hairpin: 25, 'Shift up': 25 }, 8, 2);
    for (let i = 0; i < 16; i++) {
        const category = i === 15 ? 'Shift up' : i === 14 ? 'Hairpin' : 'Traffic';
        const action = { pedal: 0.5, steering: i === 14 ? 0.4 : 0, gear: i === 15 ? 4 : 3 };
        training.record(policy, features, { action }, 1000 + i * 100, category);
    }
    await training.finishAsync(policy);
    expect(batch).toHaveBeenCalledTimes(2);
    for (const [plans] of batch.mock.calls) {
        const immediate = plans.map(({ targets }) => targets[0].action);
        expect(immediate.filter(({ steering, gear }) => steering === 0 && gear === 3)).toHaveLength(4);
        expect(immediate.filter(({ steering }) => steering === 0.4)).toHaveLength(2);
        expect(immediate.filter(({ gear }) => gear === 4)).toHaveLength(2);
    }
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), Traffic: 8, Hairpin: 4, 'Shift up': 4 });
});

it('rotates clips and uses different starting frames on later visits while preserving category quotas', async () => {
    const weights = { ...emptyRecordedSamplingWeights(), Traffic: 66.7, Kink: 33.3 };
    const { policy, batch, training, features } = setup(weights, 3, 5, () => 0.37);
    const trafficClips = 5;
    for (let clip = 0; clip < trafficClips + 3; clip++) {
        const category = clip < trafficClips ? 'Traffic' : 'Kink';
        for (let frame = 0; frame < 2; frame++) {
            const input = { ...features, features: [clip, frame, ...features.features.slice(2)] };
            training.record(policy, input, steadyTarget, 1000 + (clip * 2 + frame) * 100, category, `clip-${clip}`);
        }
        training.completeRange();
    }

    await training.finishAsync(policy, undefined, 2);
    expect(training.completedBatches).toBe(2);
    await training.finishAsync(policy);

    const trafficDraws: { clip: number; frame: number }[] = [];
    const kinkDraws: number[] = [];
    for (const [plans] of batch.mock.calls) {
        const origins = plans.map(({ input }) => ({ clip: input.features[0], frame: input.features[1] }));
        const traffic = origins.filter(({ clip }) => clip < trafficClips);
        const kink = origins.filter(({ clip }) => clip >= trafficClips);
        expect(traffic).toHaveLength(2);
        expect(traffic[0].clip).not.toBe(traffic[1].clip);
        expect(kink).toHaveLength(1);
        trafficDraws.push(...traffic);
        kinkDraws.push(kink[0].clip);
    }

    expect(batch).toHaveBeenCalledTimes(5);
    expect(new Set(trafficDraws.slice(0, trafficClips).map(({ clip }) => clip)).size).toBe(trafficClips);
    expect(new Set(trafficDraws.map(({ clip, frame }) => `${clip}:${frame}`)).size).toBe(trafficDraws.length);
    expect(new Set(kinkDraws.slice(0, 3)).size).toBe(3);
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), Traffic: 10, Kink: 5 });
});

it('reaches every eligible clip even when the category contains more clips than the origin window', async () => {
    const clipCount = RECORDED_PLAN_WINDOW + 1;
    const batchSize = 17;
    const { policy, batch, training, features } = setup(undefined, batchSize, Math.ceil(clipCount / batchSize), () => 0.25);
    const clips = Array.from({ length: clipCount }, (_, index) => index);
    for (const clip of clips) {
        const input = { ...features, features: [clip, ...features.features.slice(1)] };
        training.record(policy, input, steadyTarget, 1000 + clip * 100, 'Unlabeled', `clip-${clip}`);
        training.completeRange();
    }

    await training.finishAsync(policy);

    const firstRotation = batch.mock.calls.flatMap(([plans]) => plans.map(({ input }) => input.features[0])).slice(0, clipCount);
    expect(new Set(firstRotation)).toEqual(new Set(clips));
    expect(firstRotation).not.toEqual(clips);
});

it('keeps late changes in target gear alongside steady targets and preserves category quotas', async () => {
    const weights = { ...emptyRecordedSamplingWeights(), Sweeper: 75, Traffic: 25 };
    const lateDraw = RECORDED_PLAN_WINDOW / 4;
    let draws = 0;
    const random = () => {
        // Keep the two late examples in the first two reservoir slots for this fixture.
        const draw = draws++ - lateDraw;
        return draw === 0 || draw === 1 ? (draw + 0.5) / (RECORDED_PLAN_WINDOW + draw + 1) : 0.999;
    };
    const { policy, batch, training, features } = setup(weights, 4, 3, random);
    policy.configuration.actions.planTimesSeconds = [0, 0.1];
    const steady = { ...features, features: [0, ...features.features.slice(1)] };
    const traffic = { ...features, features: [3, ...features.features.slice(1)] };
    for (let i = 0; i < RECORDED_PLAN_WINDOW; i++) {
        training.record(policy, steady, steadyTarget, i * 100, 'Sweeper');
    }
    training.completeRange();

    const followUp = { action: { pedal: 0.6, steering: -0.2, gear: 3 } };
    const memory = new DrivingMemory(policy.configuration, true);
    const { observation } = trainingFixture();
    memory.encode(observation);
    const origin = memory.encode({ ...observation, capturedAt: observation.capturedAt + 100, measuredAt: observation.measuredAt + 100 });
    expect(origin.recordedHistory).toHaveLength(1);
    const gearTargets = [{ pedal: 0.3, steering: 0.4, gear: 4 }, { pedal: -0.5, steering: -0.3, gear: 2 }];
    gearTargets.forEach((action, i) => {
        const at = (RECORDED_PLAN_WINDOW + i * 2) * 100;
        const input = { ...origin, features: [i + 1, ...origin.features.slice(1)] };
        training.record(policy, input, { action }, at, 'Sweeper');
        input.features[0] = -1;
        training.record(policy, undefined, followUp, at + 100);
        training.completeRange();
    });
    training.record(policy, traffic, steadyTarget, (RECORDED_PLAN_WINDOW + 4) * 100, 'Traffic');
    await training.finishAsync(policy);

    expect(batch).toHaveBeenCalledTimes(3);
    for (const [plans] of batch.mock.calls) {
        expect(plans).toHaveLength(4);
        expect(plans.filter(({ input }) => input.features[0] === 3)).toHaveLength(1);
    }
    const sampled = batch.mock.calls.flatMap(([plans]) => plans);
    expect(sampled.some(({ input, targets }) => input.features[0] === 0 && targets[0].action.gear === 3)).toBe(true);
    gearTargets.forEach((action, i) => {
        const sample = sampled.find(({ input }) => input.features[0] === i + 1);
        expect(sample?.targets).toEqual([{ step: 0, action }, { step: 1, ...followUp }]);
        expect(sample?.input.recordedHistory).toEqual(origin.recordedHistory);
    });
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), Sweeper: 9, Traffic: 3 });
});

it('waits for late categories and retains a cross-section of earlier origins for their fixed mix', async () => {
    const weights = { ...emptyRecordedSamplingWeights(), Traffic: 60, Hairpin: 25, 'Shift down': 15 };
    const { policy, batch, training, features } = setup(weights, 8, 50, () => 0.99);
    for (let i = 0; i < 400; i++) {
        const input = { ...features, features: [i, ...features.features.slice(1)] };
        const category = i === 399 ? 'Shift down' : i === 398 ? 'Hairpin' : 'Traffic';
        training.record(policy, input, steadyTarget, 1000 + i * 100, category);
    }
    expect(batch).not.toHaveBeenCalled();
    for (let i = 400; i < 430; i++) {
        training.record(policy, undefined, steadyTarget, 1000 + i * 100);
    }
    expect(batch).not.toHaveBeenCalled();
    await training.finishAsync(policy);
    expect(batch).toHaveBeenCalledTimes(50);
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), Traffic: 240, Hairpin: 100, 'Shift down': 60 });
    expect(batch.mock.calls.flatMap(([plans]) => plans).some(({ input }) => input.features[0] === 0)).toBe(true);
    for (const [plans] of batch.mock.calls) {
        expect(plans.filter(({ input }) => input.features[0] === 398)).toHaveLength(2);
        expect(plans.filter(({ input }) => input.features[0] === 399).length).toBeGreaterThanOrEqual(1);
    }
});

it('redistributes weights for categories without examples across categories with examples', async () => {
    const { policy, batch, training, features } = setup({ ...emptyRecordedSamplingWeights(), Traffic: 60, Hairpin: 25, 'Shift down': 15 }, 8);
    for (let i = 0; i < 8; i++) {
        training.record(policy, features, steadyTarget, 1000 + i * 100, i < 7 ? 'Traffic' : 'Hairpin');
    }
    expect(batch).not.toHaveBeenCalled();
    await training.finishAsync(policy);
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), Traffic: 6, Hairpin: 2 });
});

it('rejects a batch too small for its eligible labels before any optimizer update', async () => {
    const weights = emptyRecordedSamplingWeights();
    for (const category of RECORDED_SAMPLING_CATEGORIES) {
        weights[category] = 100 / RECORDED_SAMPLING_CATEGORIES.length;
    }
    const { policy, batch, training, features } = setup(weights, 1, RECORDED_SAMPLING_CATEGORIES.length);
    for (let index = 0; index < RECORDED_SAMPLING_CATEGORIES.length; index++) {
        const category = RECORDED_SAMPLING_CATEGORIES[index];
        training.record(policy, features, steadyTarget, 1000 + index * 100, category);
    }
    await expect(training.finishAsync(policy)).rejects.toThrow(/batch size/i);
    expect(batch).not.toHaveBeenCalled();
    expect(training.completedBatches).toBe(0);
});

it.each([2, 8])('reuses a tiny-weight label in every size-%i batch and excludes zero-weight and absent labels', async (size) => {
    const weights = { ...emptyRecordedSamplingWeights(), Traffic: 69.9, Hairpin: 0.1, 'Shift up': 30 };
    const { policy, batch, training, features } = setup(weights, size, 4);
    const examples = [
        { category: 'Traffic', marker: 1 },
        { category: 'Shift down', marker: 2 },
        { category: 'Hairpin', marker: 3 },
    ] as const;
    for (const { category, marker } of examples) {
        const input = { ...features, features: [marker, ...features.features.slice(1)] };
        training.record(policy, input, steadyTarget, 1000 + marker * 100, category);
    }
    await training.finishAsync(policy);
    expect(batch).toHaveBeenCalledTimes(4);
    for (const [plans] of batch.mock.calls) {
        expect(plans.filter(({ input }) => input.features[0] === 1)).toHaveLength(size - 1);
        expect(plans.filter(({ input }) => input.features[0] === 3)).toHaveLength(1);
    }
    expect(training.sampled).toEqual({ ...emptyRecordedSamplingWeights(), Traffic: (size - 1) * 4, Hairpin: 4 });
});

it('uses every label in every full batch and stops at the configured update count', async () => {
    const categories = RECORDED_SAMPLING_CATEGORIES;
    const weights = emptyRecordedSamplingWeights();
    for (const category of categories) weights[category] = 100 / categories.length;

    const batchSize = 31;
    const { policy, batch, training, features } = setup(weights, batchSize, 2);
    const origins = 783;
    for (let index = 0; index < origins; index++) {
        const labelIndex = index === origins - 1 ? categories.length - 1 : index % (categories.length - 1);
        const input = { ...features, features: [labelIndex, ...features.features.slice(1)] };
        training.record(policy, input, steadyTarget, 1000 + index * 100, categories[labelIndex]);
    }
    expect(batch).not.toHaveBeenCalled();

    await training.finishAsync(policy, undefined, 1);
    expect(training.completedBatches).toBe(1);
    await training.finishAsync(policy);
    await training.finishAsync(policy);
    expect(batch.mock.calls.map(([plans]) => plans.length)).toEqual([batchSize, batchSize]);
    for (const [plans] of batch.mock.calls) {
        expect(new Set(plans.map(({ input }) => input.features[0])).size).toBe(categories.length);
    }
    expect(training.completedBatches).toBe(2);
});
