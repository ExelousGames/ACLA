import { DrivingConsequenceBuffer } from './driving-consequence-buffer';
import type { DrivingPolicyInput } from './driving-memory';

const input: DrivingPolicyInput = { features: [1], cameras: [] };
const interval = (startedAt: number, endedAt: number) => ({ startedAt, endedAt, reward: 1, samples: [] });

it('requires the configured duration and leaves the next window intact when a batch is taken', () => {
    const buffer = new DrivingConsequenceBuffer(0.4);
    buffer.add(interval(1000, 1150), input);
    buffer.add(interval(1150, 1399), input);
    expect(buffer.take()).toBeUndefined();
    const endpoint = { features: [2], cameras: [] };
    buffer.add(interval(1399, 1450), endpoint);
    buffer.add(interval(1450, 1600), input);
    const first = buffer.take()!;
    expect(first.sequence.map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([
        [1000, 1150], [1150, 1399], [1399, 1450],
    ]);
    expect(first.input).toBe(endpoint);
    buffer.add(interval(1600, 1850), input);
    expect(buffer.take()!.sequence.map(({ startedAt, endedAt }) => [startedAt, endedAt])).toEqual([[1450, 1600], [1600, 1850]]);
    expect(first.sequence).toHaveLength(3);
    expect(buffer.hasReady).toBe(false);
});

it('bounds a stalled consumer by dropping whole old batches without shortening their duration', () => {
    const buffer = new DrivingConsequenceBuffer(1);
    for (let at = 0; at < 5000; at += 500) buffer.add(interval(at, at + 500), input);
    expect(buffer.take()!.sequence).toEqual([interval(3000, 3500), interval(3500, 4000)]);
    expect(buffer.take()!.sequence).toEqual([interval(4000, 4500), interval(4500, 5000)]);
    expect(buffer.take()).toBeUndefined();
});

it('flushes complete and partial windows once on handoff and discards both on reset', () => {
    const buffer = new DrivingConsequenceBuffer(1);
    buffer.add(interval(0, 1000), input);
    buffer.add(interval(1000, 1200), input);
    expect(buffer.finish().map(({ sequence }) => sequence)).toEqual([[interval(0, 1000)], [interval(1000, 1200)]]);
    expect(buffer.finish()).toEqual([]);
    buffer.add(interval(1200, 2200), input);
    buffer.add(interval(2200, 2400), input);
    buffer.reset();
    expect(buffer.finish()).toEqual([]);
});
