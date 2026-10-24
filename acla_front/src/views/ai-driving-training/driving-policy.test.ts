import { DrivingPolicy } from './driving-policy';
import { DrivingMemory, DRIVING_MEMORY_INPUTS } from './driving-memory';
import { trainingFixture } from './training-fixtures';

it('learns human steering, pedals and shifts as targets with unchanged checkpoint dimensions', () => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    const policy = new DrivingPolicy(observation.context, observation.features.length, () => 0.6);
    const before = policy.decide(features, false);
    const checkpoint = policy.checkpoint();
    for (let i = 0; i < 30; i++) policy.learnDemonstration(features,
        { throttle: 1, brake: 0, steering: -1, shiftDown: true }, { up: true, down: true });
    const after = policy.decide(features, false);
    expect(after.action.throttle).toBeGreaterThan(before.action.throttle);
    expect(after.action.brake).toBeLessThan(before.action.brake);
    expect(after.action.steering).toBeLessThan(before.action.steering);
    expect(after.shiftProbabilities[2]).toBeGreaterThan(before.shiftProbabilities[2]);
    expect(policy.checkpoint()).toMatchObject({ version: 8, inputs: checkpoint.inputs, context: checkpoint.context, steps: 30 });
    expect(policy.checkpoint().outputWeights[3]).toEqual(checkpoint.outputWeights[3]);
    expect(DrivingPolicy.restore(policy.checkpoint()).decide(features, false)).toEqual(after);
});

it('does not train forced cooldown holds as human gear preferences', () => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    const policy = new DrivingPolicy(observation.context, observation.features.length);
    const before = policy.checkpoint();
    policy.learnDemonstration(features, { throttle: 0, brake: 1, steering: 1 }, { up: false, down: false });
    expect(policy.checkpoint().outputWeights.slice(4)).toEqual(before.outputWeights.slice(4));
});

it('learns from rewards, bounds actions, and exactly restores deterministic inference', () => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    let seed = 42;
    const random = () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 4294967296);
    const model = new DrivingPolicy(observation.context, observation.features.length, random);
    const before = model.checkpoint();
    for (let i = 0; i < 100; i++) {
        const decision = model.decide(features, true);
        expect(decision.action.throttle).toBeGreaterThanOrEqual(0);
        expect(decision.action.throttle).toBeLessThanOrEqual(1);
        expect(decision.action.brake).toBeGreaterThanOrEqual(0);
        expect(decision.action.brake).toBeLessThanOrEqual(1);
        expect(Math.abs(decision.action.steering)).toBeLessThanOrEqual(1);
        model.learn(decision, 1 - Math.abs(decision.action.steering), features, 0.2);
    }
    expect(model.steps).toBe(100);
    expect(model.checkpoint().hiddenWeights).not.toEqual(before.hiddenWeights);
    expect(model.checkpoint().outputWeights).not.toEqual(before.outputWeights);
    const restored = DrivingPolicy.restore(JSON.parse(JSON.stringify(model.checkpoint())));
    expect(restored.decide(features, false)).toEqual(model.decide(features, false));
    expect(restored.matches({ ...observation, context: { ...observation.context, map: 'spa' } })).toBe(false);
});

it('positive and negative advantages move the policy toward and away from the sampled action', () => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    const policy = new DrivingPolicy(observation.context, observation.features.length, () => 0.6);
    const decision = policy.decide(features, true);
    const initial = policy.decide(features, false).means[2];
    const negative = DrivingPolicy.restore(policy.checkpoint());
    policy.learn(decision, decision.value + 1, features, 0.2, true);
    negative.learn(decision, decision.value - 1, features, 0.2, true);
    const direction = Math.sign(decision.latent[2] - decision.means[2]);
    expect((policy.decide(features, false).means[2] - initial) * direction).toBeGreaterThan(0);
    expect((negative.decide(features, false).means[2] - initial) * direction).toBeLessThan(0);
});

it('rejects corrupt checkpoints and invalid observations', () => {
    const { observation } = trainingFixture();
    const model = new DrivingPolicy(observation.context, observation.features.length);
    const checkpoint = model.checkpoint();
    checkpoint.outputWeights[0][0] = NaN;
    expect(() => DrivingPolicy.restore(checkpoint)).toThrow(/checkpoint/);
    expect(() => model.decide([NaN], true)).toThrow(/observation/);
});

it.each([[0.1, 0], [0.7, 1], [0.95, 2]])('samples a mutually exclusive gear choice with draw %s', (draw, shift) => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    const model = new DrivingPolicy(observation.context, observation.features.length, () => draw);
    const decision = model.decide(features, true);
    expect(decision.shift).toBe(shift);
    expect(decision.action).toMatchObject({ shiftUp: shift === 1, shiftDown: shift === 2 });
    expect(decision.shiftProbabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
});

it.each([0.1, 0.7, 0.95])('reinforces or discourages the sampled gear choice according to reward (draw %s)', (draw) => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    const model = new DrivingPolicy(observation.context, observation.features.length, () => draw);
    const negative = DrivingPolicy.restore(model.checkpoint());
    const decision = model.decide(features, true);
    const before = decision.shiftProbabilities[decision.shift];
    model.learn(decision, decision.value + 1, features, 0.2, true);
    negative.learn(decision, decision.value - 1, features, 0.2, true);
    expect(model.decide(features, false).shiftProbabilities[decision.shift]).toBeGreaterThan(before);
    expect(negative.decide(features, false).shiftProbabilities[decision.shift]).toBeLessThan(before);
});

it('does not sample or train masked gear choices, including cooldown-forced holds', () => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    const model = new DrivingPolicy(observation.context, observation.features.length, () => 0.99);
    const before = model.checkpoint();
    const held = model.decide(features, true, { up: false, down: false });
    expect(held.shiftProbabilities).toEqual([1, 0, 0]);
    expect(held.action).toMatchObject({ shiftUp: false, shiftDown: false });
    model.learn(held, held.value + 1, features, 0.2, true);
    expect(model.checkpoint().outputWeights.slice(4)).toEqual(before.outputWeights.slice(4));
    const upOnly = model.decide(features, true, { up: true, down: false });
    expect(upOnly.shift).toBe(1);
    expect(upOnly.shiftProbabilities[2]).toBe(0);
    model.learn(upOnly, upOnly.value + 1, features, 0.2, true);
    expect(model.checkpoint().outputWeights[6]).toEqual(before.outputWeights[6]);
});

it.each([1, 2])('restores deterministic shifting choice %s and respects masks at evaluation', (shift) => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.outputWeights.slice(4).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === shift ? 4 : 0; });
    const model = DrivingPolicy.restore(JSON.parse(JSON.stringify(checkpoint)));
    const decision = model.decide(features, false);
    expect(decision.shift).toBe(shift);
    expect(decision.action).toMatchObject({ shiftUp: shift === 1, shiftDown: shift === 2 });
    expect(DrivingPolicy.restore(model.checkpoint()).decide(features, false)).toEqual(decision);
    expect(model.decide(features, false, { up: false, down: false }).shift).toBe(0);
});

it('rejects incorrect input dimensions and incomplete shifting heads', () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    expect(() => DrivingPolicy.restore({ ...checkpoint, inputs: checkpoint.inputs - 2 })).toThrow(/checkpoint/);
    const missingRotationRates = new DrivingPolicy(observation.context, observation.features.length - 3).checkpoint();
    expect(() => DrivingPolicy.restore(missingRotationRates)).toThrow(/checkpoint/);
    const withOrientation = new DrivingPolicy(observation.context, observation.features.length + 6).checkpoint();
    expect(() => DrivingPolicy.restore(withOrientation)).toThrow(/checkpoint/);
    expect(() => DrivingPolicy.restore({ ...checkpoint, outputWeights: checkpoint.outputWeights.slice(0, 4) })).toThrow(/checkpoint/);
});

it.each([1, 2, 3, 4, 5, 6, 7, 9])('rejects unsupported checkpoint version %s', (version) => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    expect(() => DrivingPolicy.restore({ ...checkpoint, version })).toThrow(/checkpoint/);
});

it('makes different decisions from the same current frame with different recent driving histories', () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.hiddenWeights.forEach((row) => row.fill(0));
    checkpoint.outputWeights.forEach((row) => row.fill(0));
    checkpoint.hiddenWeights[0][5] = 1; // Speed in the most recent history slot.
    checkpoint.outputWeights[2][0] = 1;
    const policy = DrivingPolicy.restore(checkpoint);
    const decideAfter = (speed: number) => {
        const memory = new DrivingMemory();
        const features = observation.features.slice();
        features[3] = speed;
        memory.encode({ ...observation, features });
        return policy.decide(memory.encode({ ...observation, capturedAt: 1200 }), false);
    };
    const slower = decideAfter(0.1), faster = decideAfter(0.8);
    expect(slower.features.slice(DRIVING_MEMORY_INPUTS)).toEqual(faster.features.slice(DRIVING_MEMORY_INPUTS));
    expect(faster.action.steering).toBeGreaterThan(slower.action.steering);
    expect(policy.matches(observation)).toBe(true);
    expect(() => policy.decide(observation.features, false)).toThrow(/observation/);
    expect(DrivingPolicy.restore(policy.checkpoint()).decide(faster.features, false)).toEqual(faster);
});
