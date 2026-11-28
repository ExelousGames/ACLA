import { DrivingConsequenceBuffer } from './driving-consequence-buffer';
import type { DrivingPolicyInput } from './driving-memory';
import type { PolicyDecision } from './driving-policy';
import { drivingSequenceTargets } from './driving-rollout';
import { cloneConfig } from './driving-model-config';

const input: DrivingPolicyInput = { features: [1], cameras: [] };
const interval = (startedAt: number, endedAt: number, reward = 1) => ({
    startedAt, endedAt, reward,
    samples: [{ decision: { input: { features: [startedAt] } } as PolicyDecision, credits: [], rewardShare: 1 }],
});

it('uses later rewards within each rolling three-second window', () => {
    const buffer = new DrivingConsequenceBuffer(3);
    const samples = Array.from({ length: 4 }, (_, i) => interval(i * 1000, (i + 1) * 1000, i === 3 ? -7 : 1));
    samples.slice(0, 2).forEach((sample) => buffer.add(sample, input));
    expect(buffer.take()).toBeUndefined();
    buffer.add(samples[2], input);
    const first = buffer.take()!;
    expect(first.sequence.map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([
        [0, 1000], [1000, 2000], [2000, 3000],
    ]);
    expect(first.sequence.map(({ samples }) => samples.length)).toEqual([1, 0, 0]);
    buffer.add(samples[3], input);
    const second = buffer.take()!;
    expect(second.sequence.map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([
        [1000, 2000], [2000, 3000], [3000, 4000],
    ]);
    const learning = { ...cloneConfig().learning, discount: 0.5, discountIntervalSeconds: 1 };
    expect(drivingSequenceTargets(first.sequence, learning)).toEqual([
        expect.objectContaining({ decision: samples[0].samples[0].decision, elapsedSeconds: 3, reward: 1.75 }),
    ]);
    expect(drivingSequenceTargets(second.sequence, learning)).toEqual([
        expect.objectContaining({ decision: samples[1].samples[0].decision, elapsedSeconds: 3, reward: -0.25 }),
    ]);
    expect(first.sequence).toHaveLength(3);
    expect(buffer.hasReady).toBe(false);
});

it('uses the last observed endpoint within the horizon when telemetry crosses its limit', () => {
    const buffer = new DrivingConsequenceBuffer(0.4);
    buffer.add(interval(1000, 1150), input);
    const endpoint = { features: [2], cameras: [] };
    buffer.add(interval(1150, 1399), endpoint);
    expect(buffer.take()).toBeUndefined();
    buffer.add(interval(1399, 1450), input);
    const first = buffer.take()!;
    expect(first.sequence.map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([[1000, 1150], [1150, 1399]]);
    expect(first.input).toBe(endpoint);
    buffer.add(interval(1450, 1600), input);
    const second = buffer.take()!;
    expect(second.sequence.map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([[1150, 1399], [1399, 1450]]);
    expect(second.sequence.flatMap(({ samples }) => samples)).toHaveLength(1);
});

it('includes fifth-second rewards and bootstraps from the default horizon endpoint', () => {
    const learning = { ...cloneConfig().learning, discount: 0.5, discountIntervalSeconds: 1 };
    const buffer = new DrivingConsequenceBuffer(learning.sequenceSeconds);
    const start = { features: [0], cameras: [] }, successor = { features: [1], cameras: [] };
    const horizon = { features: [5], cameras: [] };
    const origin = { ...interval(0, 1000, 2), input: start, next: successor, nextGears: { max: 6 } };
    buffer.add(origin, successor);
    buffer.add(interval(1000, 2000, -5), input);
    buffer.add(interval(2000, 3000, 0), input);
    buffer.add(interval(3000, 4000, 0), input);
    expect(buffer.hasReady).toBe(false);
    buffer.add({ ...interval(4000, 5000, -8), next: horizon, nextGears: { max: 4 } }, horizon);
    const batch = buffer.take()!;
    expect(batch.input).toBe(horizon);
    const [target] = drivingSequenceTargets(batch.sequence, learning);
    expect(target.input).toBe(start);
    expect(target.next).toBe(horizon);
    expect(target.nextGears).toEqual({ max: 4 });
    expect(target).toMatchObject({ reward: -1, elapsedSeconds: 5, terminal: false });
});

it('keeps a late acknowledged command eligible for consequences three seconds after that command', () => {
    const buffer = new DrivingConsequenceBuffer(3), origin = interval(0, 1000, 10);
    const early = { ...origin.samples[0], startedAt: 0, rewardShare: 0.9 };
    const late = { ...origin.samples[0], startedAt: 900, rewardShare: 0.1 };
    buffer.add({ ...origin, samples: [early, late] }, input);
    buffer.add(interval(1000, 2000, 0), input);
    buffer.add(interval(2000, 3000, 0), input);
    const first = buffer.take()!;
    expect(first.sequence[0].samples).toEqual([early]);
    buffer.add(interval(3000, 3900, -8), input);
    const second = buffer.take()!;
    expect(second.sequence[0].samples).toEqual([late]);
    const learning = { ...cloneConfig().learning, discount: 0.5, discountIntervalSeconds: 1 };
    const target = drivingSequenceTargets(second.sequence, learning);
    expect(target).toHaveLength(1);
    expect(target[0].elapsedSeconds).toBe(3.9);
    expect(target[0].reward).toBe(9);
    expect(first.sequence[0].samples).toEqual([early]);
});

it('groups separate commands from one observed interval when they share a real bootstrap endpoint', () => {
    const buffer = new DrivingConsequenceBuffer(3), origin = interval(0, 1000);
    const actions = [0, 100, 200].map((startedAt) => ({ ...origin.samples[0], startedAt, rewardShare: 1 / 3 }));
    buffer.add({ ...origin, samples: actions }, input);
    buffer.add(interval(1000, 2000), input);
    buffer.add(interval(2000, 3000), input);
    buffer.add(interval(3000, 3300), input);
    const batch = buffer.take()!;
    expect(batch.sequence[0].samples).toEqual(actions);
    expect(batch.sequence[batch.sequence.length - 1].endedAt).toBe(3000);
    expect(drivingSequenceTargets(batch.sequence, cloneConfig().learning).map(({ elapsedSeconds }) => elapsedSeconds)).toEqual([3, 3, 3]);
    expect(buffer.take()).toBeUndefined();
});

it('uses one observed interval as the minimum return when observations are slower than the horizon', () => {
    const buffer = new DrivingConsequenceBuffer(0.001), sample = interval(1000, 1200);
    buffer.add(sample, input);
    expect(buffer.take()).toEqual({ sequence: [sample], input });
    expect(buffer.finish()).toEqual([]);
});

it('combines matured actions that share a bootstrap endpoint without crediting later actions early', () => {
    const buffer = new DrivingConsequenceBuffer(1);
    for (const [from, to] of [[0, 100], [100, 200], [200, 300], [300, 900], [900, 1200]]) {
        buffer.add(interval(from, to), input);
    }
    const first = buffer.take()!;
    expect(first.sequence.map(({ samples }) => samples.length)).toEqual([1, 1, 0, 0]);
    expect(first.sequence[first.sequence.length - 1].endedAt).toBe(900);
    const second = buffer.take()!;
    expect(second.sequence.map(({ samples }) => samples.length)).toEqual([1, 0, 0]);
    expect(second.sequence[0].startedAt).toBe(200);
    expect(second.sequence[second.sequence.length - 1].endedAt).toBe(1200);
});

it('bounds stalled training by dropping old matured windows without shortening retained returns', () => {
    const buffer = new DrivingConsequenceBuffer(1);
    for (let at = 0; at < 5000; at += 500) buffer.add(interval(at, at + 500), input);
    const starts: number[] = [];
    while (buffer.hasReady) {
        const { sequence } = buffer.take()!;
        starts.push(sequence[0].startedAt);
        expect(sequence[sequence.length - 1].endedAt - sequence[0].startedAt).toBe(1000);
        expect(sequence.flatMap(({ samples }) => samples)).toHaveLength(1);
    }
    expect(starts).toEqual([3000, 3500, 4000]);
});

it('also bounds a very high observation rate by batch count', () => {
    const buffer = new DrivingConsequenceBuffer(0.3);
    for (let at = 0; at < 600; at++) buffer.add(interval(at, at + 1), input);
    let count = 0;
    while (buffer.hasReady) {
        const { sequence } = buffer.take()!;
        expect(sequence[sequence.length - 1].endedAt - sequence[0].startedAt).toBe(300);
        count++;
    }
    expect(count).toBe(256);
});

it('flushes every remaining action once on handoff and discards both queues on reset', () => {
    const buffer = new DrivingConsequenceBuffer(3);
    for (let at = 0; at < 6000; at += 1000) buffer.add(interval(at, at + 1000), input);
    const first = buffer.take()!;
    const remaining = buffer.finish();
    const actions = [first, ...remaining].flatMap(({ sequence }) => sequence.flatMap(({ samples }) =>
        samples.map(({ decision }) => decision.input.features[0])));
    expect(actions).toEqual([0, 1000, 2000, 3000, 4000, 5000]);
    expect(remaining[remaining.length - 1].sequence.map(({ startedAt }) => startedAt)).toEqual([4000, 5000]);
    expect(buffer.finish()).toEqual([]);
    buffer.add(interval(6000, 7000), input);
    buffer.add(interval(7000, 9000), input);
    buffer.reset();
    expect(buffer.finish()).toEqual([]);
});
