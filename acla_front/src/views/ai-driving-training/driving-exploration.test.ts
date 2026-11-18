import { DrivingExploration } from './driving-exploration';
import { cloneConfig, drivingExplorationSigmas } from './driving-model-config';
import { drivingMemoryInputs } from './driving-memory';

function setup() {
    const config = cloneConfig();
    config.observation.features = ['controls', 'speed', 'yawRate'];
    config.memory.features = ['speed'];
    config.actions.planTimesSeconds = [0, 0.1, 0.2, 0.4, 0.5, 1];
    let seed = 42;
    const random = jest.fn(() => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; });
    const sampler = new DrivingExploration(config, random);
    const input = (values = [0, 0, 0, 0, 0]) => ({ features: [...new Array(drivingMemoryInputs(config)).fill(0), ...values], cameras: [] });
    return { config, sampler, input, random };
}

it('responds to state changes using the same persistent weights and recovers the same noise for the same state', () => {
    const { sampler, input, random } = setup();
    const first = sampler.plan(input(), 1000);
    random.mockClear();
    for (let feature = 0; feature < 5; feature++) {
        const values = [0, 0, 0, 0, 0]; values[feature] = 0.7;
        const changed = sampler.plan(input(values), 1000);
        changed[0].axes.forEach((value, axis) => {
            expect(value).not.toBe(first[0].axes[axis]);
            expect(changed[0].variances[axis]).toBeGreaterThan(first[0].variances[axis]);
        });
    }
    expect(sampler.plan(input(), 1000)).toEqual(first);
    expect(random).not.toHaveBeenCalled();
});

it('uses the configured current feature layout and ignores history and category bits', () => {
    const { config, input, random } = setup();
    config.observation.features = ['game', ...config.observation.features];
    const sampler = new DrivingExploration(config, random);
    const sample = input([...new Array(config.observation.categoryBits).fill(0), 0.1, 0.2, 0.3, 0.4, 0.5]);
    const first = sampler.plan(sample, 1000);
    sample.features.fill(1, 0, drivingMemoryInputs(config) + config.observation.categoryBits);
    expect(sampler.plan(sample, 1000)).toEqual(first);
    sample.features[sample.features.length - 1] += 0.01;
    expect(sampler.plan(sample, 1000)[0].axes).not.toEqual(first[0].axes);
});

it('changes continuously around refresh boundaries and is smoother than independent action noise', () => {
    const { sampler, input } = setup();
    const first = sampler.plan(input(), 1000);
    const before = sampler.plan(input(), 1499.99)[0];
    const boundary = sampler.plan(input(), 1500)[0];
    const after = sampler.plan(input(), 1500.01)[0];
    boundary.axes.forEach((value, axis) => {
        expect(value).toBeCloseTo(before.axes[axis], 7);
        expect(value).toBeCloseTo(after.axes[axis], 7);
        expect(value).toBe(first[4].axes[axis]);
    });
    const values: number[] = [];
    for (let i = 0; i < 1000; i++) values.push(sampler.plan(input(), 1600 + i * 20)[0].axes[0]);
    const energy = values.reduce((sum, value) => sum + value * value, 0) / values.length;
    const changes = values.slice(1).reduce((sum, value, i) => sum + (value - values[i]) ** 2, 0) / (values.length - 1);
    // Independent centered noise has expected squared change 2 * variance.
    expect(changes / energy).toBeLessThan(0.05);
});

it.each([0, 0.25, 0.5, 0.75])('retains calibrated marginal variance while blending (phase=%s)', (fraction) => {
    const { config, sampler, input } = setup();
    config.actions.planTimesSeconds = [0];
    const sums = [0, 0], squares = [0, 0], sigmas = drivingExplorationSigmas(config);
    const count = 2000;
    let expected = [0, 0];
    for (let i = 0; i < count; i++) {
        sampler.reset();
        sampler.plan(input(), 1000);
        const sample = sampler.plan(input([0.3, 0.2, -0.5, 0.8, 0.4]), 1000 + fraction * 500)[0];
        expected = sample.variances;
        sample.axes.forEach((value, axis) => {
            sums[axis] += value * sigmas[axis]; squares[axis] += (value * sigmas[axis]) ** 2;
        });
    }
    expected.forEach((variance, axis) => {
        expect(Math.abs(sums[axis] / count)).toBeLessThan(0.005);
        expect(Math.abs((squares[axis] / count - (sums[axis] / count) ** 2) / variance - 1)).toBeLessThan(0.1);
    });
});

it('keeps noise finite and nonzero at rest and bounds variance for extreme observations', () => {
    const { config, sampler, input } = setup();
    const sigmas = drivingExplorationSigmas(config);
    const rest = sampler.plan(input(), 1000)[0];
    expect(rest.variances).toEqual(sigmas.map((sigma) => sigma * sigma));
    rest.axes.forEach((value) => expect(Number.isFinite(value) && value !== 0).toBe(true));
    const extreme = sampler.plan(input([1e6, -1e6, 1e6, 1e6, -1e6]), 1000)[0];
    extreme.variances.forEach((variance, axis) => expect(variance).toBeCloseTo(2 * sigmas[axis] ** 2, 10));
    extreme.axes.forEach((value) => expect(Number.isFinite(value)).toBe(true));
});

it('keeps discrete gear draws until their own refresh interval', () => {
    const { config, sampler, input } = setup();
    config.learning.gearExplorationUpdateHz = 1;
    const first = sampler.plan(input(), 1000);
    expect(first.slice(0, -1).every((sample) => sample.shift === first[0].shift)).toBe(true);
    expect(first[5].shift).not.toBe(first[0].shift);
    expect(sampler.plan(input(), 2000)[0].shift).toBe(first[5].shift);
});

it.each(['explicit', 'rewind', 'gap'])('resets persistent weights on %s', (reason) => {
    const { sampler, input, random } = setup();
    const first = sampler.plan(input(), 1000);
    random.mockClear();
    if (reason === 'explicit') sampler.reset();
    const next = sampler.plan(input(), reason === 'rewind' ? 999 : reason === 'gap' ? 5000 : 1000);
    expect(next[0].axes).not.toEqual(first[0].axes);
    expect(random).toHaveBeenCalled();
});

it('retires expired weights instead of accumulating an episode of samples', () => {
    const { sampler, input } = setup();
    for (let i = 0; i < 200; i++) sampler.plan(input(), 1000 + i * 100);
    const caches = (sampler as any).samples as Map<number, number[]>[];
    expect(caches.slice(0, 2).every((cache) => cache.size <= 5)).toBe(true);
    expect(caches[2].size).toBeLessThanOrEqual(11);
});
