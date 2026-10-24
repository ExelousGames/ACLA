import { DrivingMemory, DRIVING_MEMORY_INPUTS } from './driving-memory';
import { trainingFixture } from './training-fixtures';

it('starts immediately with absent history and preserves all current inputs', () => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation);
    expect(features.slice(0, DRIVING_MEMORY_INPUTS)).toEqual(new Array(DRIVING_MEMORY_INPUTS).fill(0));
    expect(features.slice(DRIVING_MEMORY_INPUTS)).toEqual(observation.features);
});

it('uses elapsed time, retains a full three seconds and expires older samples', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    memory.encode(observation);
    for (const capturedAt of [1300, 2100, 3000]) memory.encode({ ...observation, capturedAt });
    const full = memory.encode({ ...observation, capturedAt: 4000 });
    expect(full.slice(14 * 22, 14 * 22 + 6)).toEqual([1, 1, ...observation.features.slice(0, 4)]);
    const expired = memory.encode({ ...observation, capturedAt: 4001 });
    expect(expired.slice(14 * 22, 15 * 22)).toEqual(new Array(22).fill(0));
});

it('uses the newest sample in each time bin without treating repeated images as history', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    const initial = memory.encode(observation);
    expect(memory.encode(observation)).toEqual(initial);
    const recent = { ...observation, capturedAt: 1050, features: observation.features.map(() => 0.4) };
    memory.encode(recent);
    const features = memory.encode({ ...observation, capturedAt: 1200 });
    expect(features.slice(0, 6)).toEqual([1, 150 / 3000, 0.4, 0.4, 0.4, 0.4]);
    expect(features.slice(22, DRIVING_MEMORY_INPUTS)).toEqual(new Array(DRIVING_MEMORY_INPUTS - 22).fill(0));
});

it('copies historical controls and motion so later mutations cannot alter a learning input', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    memory.encode(observation);
    const original = observation.features.slice();
    observation.features.fill(0);
    const encoded = memory.encode({ ...observation, capturedAt: 1200 });
    expect(encoded.slice(2, 22)).toEqual([...original.slice(0, 4), ...original.slice(52, 65), ...original.slice(-3)]);
    memory.encode({ ...observation, capturedAt: 1400 });
    memory.reset();
    expect(encoded.slice(2, 6)).toEqual(original.slice(0, 4));
    expect(memory.encode({ ...observation, capturedAt: 1600 }).slice(0, DRIVING_MEMORY_INPUTS).every((v) => v === 0)).toBe(true);
});

it.each(['session', 'reference', 'context', 'gap', 'clock'] as const)('clears history at a %s boundary', (boundary) => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    memory.encode(observation);
    memory.encode({ ...observation, capturedAt: 1200 });
    const next = { ...observation, capturedAt: 1400 };
    if (boundary === 'session') next.session = 'new-session';
    if (boundary === 'reference') next.reference = { ...observation.reference };
    if (boundary === 'context') next.context = { ...observation.context, classes: ['road', 'wall'] };
    if (boundary === 'gap') next.capturedAt = 2701;
    if (boundary === 'clock') next.capturedAt = 1100;
    expect(memory.encode(next).slice(0, DRIVING_MEMORY_INPUTS).every((value) => value === 0)).toBe(true);
});
