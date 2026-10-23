import { DrivingPolicy } from './driving-policy';
import { trainingFixture } from './training-fixtures';

it('learns from rewards, bounds actions, and exactly restores deterministic inference', () => {
    const { observation } = trainingFixture();
    let seed = 42;
    const random = () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 4294967296);
    const model = new DrivingPolicy(observation.context, observation.features.length, random);
    const before = model.checkpoint();
    for (let i = 0; i < 100; i++) {
        const decision = model.decide(observation.features, true);
        expect(decision.action.throttle).toBeGreaterThanOrEqual(0);
        expect(decision.action.throttle).toBeLessThanOrEqual(1);
        expect(decision.action.brake).toBeGreaterThanOrEqual(0);
        expect(decision.action.brake).toBeLessThanOrEqual(1);
        expect(Math.abs(decision.action.steering)).toBeLessThanOrEqual(1);
        model.learn(decision, 1 - Math.abs(decision.action.steering), observation.features, 0.2);
    }
    expect(model.steps).toBe(100);
    expect(model.checkpoint().hiddenWeights).not.toEqual(before.hiddenWeights);
    expect(model.checkpoint().outputWeights).not.toEqual(before.outputWeights);
    const restored = DrivingPolicy.restore(JSON.parse(JSON.stringify(model.checkpoint())));
    expect(restored.decide(observation.features, false)).toEqual(model.decide(observation.features, false));
    expect(restored.matches({ ...observation, context: { ...observation.context, map: 'spa' } })).toBe(false);
});

it('positive and negative advantages move the policy toward and away from the sampled action', () => {
    const { observation } = trainingFixture();
    const policy = new DrivingPolicy(observation.context, observation.features.length, () => 0.6);
    const decision = policy.decide(observation.features, true);
    const initial = policy.decide(observation.features, false).means[2];
    const negative = DrivingPolicy.restore(policy.checkpoint());
    policy.learn(decision, decision.value + 1, observation.features, 0.2, true);
    negative.learn(decision, decision.value - 1, observation.features, 0.2, true);
    const direction = Math.sign(decision.latent[2] - decision.means[2]);
    expect((policy.decide(observation.features, false).means[2] - initial) * direction).toBeGreaterThan(0);
    expect((negative.decide(observation.features, false).means[2] - initial) * direction).toBeLessThan(0);
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
    const model = new DrivingPolicy(observation.context, observation.features.length, () => draw);
    const decision = model.decide(observation.features, true);
    expect(decision.shift).toBe(shift);
    expect(decision.action).toMatchObject({ shiftUp: shift === 1, shiftDown: shift === 2 });
    expect(decision.shiftProbabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
});

it.each([0.1, 0.7, 0.95])('reinforces or discourages the sampled gear choice according to reward (draw %s)', (draw) => {
    const { observation } = trainingFixture();
    const model = new DrivingPolicy(observation.context, observation.features.length, () => draw);
    const negative = DrivingPolicy.restore(model.checkpoint());
    const decision = model.decide(observation.features, true);
    const before = decision.shiftProbabilities[decision.shift];
    model.learn(decision, decision.value + 1, observation.features, 0.2, true);
    negative.learn(decision, decision.value - 1, observation.features, 0.2, true);
    expect(model.decide(observation.features, false).shiftProbabilities[decision.shift]).toBeGreaterThan(before);
    expect(negative.decide(observation.features, false).shiftProbabilities[decision.shift]).toBeLessThan(before);
});

it('does not sample or train masked gear choices, including cooldown-forced holds', () => {
    const { observation } = trainingFixture();
    const model = new DrivingPolicy(observation.context, observation.features.length, () => 0.99);
    const before = model.checkpoint();
    const held = model.decide(observation.features, true, { up: false, down: false });
    expect(held.shiftProbabilities).toEqual([1, 0, 0]);
    expect(held.action).toMatchObject({ shiftUp: false, shiftDown: false });
    model.learn(held, held.value + 1, observation.features, 0.2, true);
    expect(model.checkpoint().outputWeights.slice(4)).toEqual(before.outputWeights.slice(4));
    const upOnly = model.decide(observation.features, true, { up: true, down: false });
    expect(upOnly.shift).toBe(1);
    expect(upOnly.shiftProbabilities[2]).toBe(0);
    model.learn(upOnly, upOnly.value + 1, observation.features, 0.2, true);
    expect(model.checkpoint().outputWeights[6]).toEqual(before.outputWeights[6]);
});

it.each([1, 2])('restores deterministic shifting choice %s and respects masks at evaluation', (shift) => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.outputWeights.slice(4).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === shift ? 4 : 0; });
    const model = DrivingPolicy.restore(JSON.parse(JSON.stringify(checkpoint)));
    const decision = model.decide(observation.features, false);
    expect(decision.shift).toBe(shift);
    expect(decision.action).toMatchObject({ shiftUp: shift === 1, shiftDown: shift === 2 });
    expect(DrivingPolicy.restore(model.checkpoint()).decide(observation.features, false)).toEqual(decision);
    expect(model.decide(observation.features, false, { up: false, down: false }).shift).toBe(0);
});

it('rejects incorrect input dimensions and incomplete shifting heads', () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    expect(() => DrivingPolicy.restore({ ...checkpoint, inputs: checkpoint.inputs - 2 })).toThrow(/checkpoint/);
    expect(() => DrivingPolicy.restore({ ...checkpoint, outputWeights: checkpoint.outputWeights.slice(0, 4) })).toThrow(/checkpoint/);
});

it.each([1, 2, 4])('rejects unsupported checkpoint version %s', (version) => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    expect(() => DrivingPolicy.restore({ ...checkpoint, version })).toThrow(/checkpoint/);
});
