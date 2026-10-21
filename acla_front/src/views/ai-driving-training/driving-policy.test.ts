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
