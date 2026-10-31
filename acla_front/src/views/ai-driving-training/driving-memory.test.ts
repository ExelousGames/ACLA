import { DrivingMemory, drivingMemoryInputs, drivingMemorySlotInputs } from './driving-memory';
import { trainingFixture } from './training-fixtures';

it('starts immediately with absent history and preserves all current inputs', () => {
    const { observation } = trainingFixture();
    const features = new DrivingMemory().encode(observation).features;
    expect(features.slice(0, drivingMemoryInputs())).toEqual(new Array(drivingMemoryInputs()).fill(0));
    expect(features.slice(drivingMemoryInputs())).toEqual(observation.features);
});

it('retains every observation input in 17 history slots with 300 ms sampling and a partial oldest bin', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    const samples = Array.from({ length: 18 }, (_, i) => ({ ...observation, capturedAt: i === 17 ? 6000 : 1000 + i * 300,
        features: observation.features.map((_, j) => (i + j / observation.features.length) / 18) }));
    let encoded: number[] = [];
    samples.forEach((sample) => { encoded = memory.encode(sample).features; });
    const slotInputs = observation.features.length + 2;
    expect(encoded.length).toBe(17 * slotInputs + observation.features.length);
    for (let slot = 0; slot < 17; slot++) {
        expect(encoded.slice(slot * slotInputs, (slot + 1) * slotInputs))
            .toEqual([1, (200 + slot * 300) / 5000, ...samples[16 - slot].features]);
    }
    expect(encoded.slice(17 * slotInputs)).toEqual(samples[17].features);
});

it('uses elapsed time, retains exactly five seconds and expires older samples from the partial bin', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    memory.encode(observation);
    for (let capturedAt = 1300; capturedAt <= 5800; capturedAt += 300) memory.encode({ ...observation, capturedAt });
    const full = memory.encode({ ...observation, capturedAt: 6000 }).features;
    expect(full).toHaveLength(17 * drivingMemorySlotInputs() + observation.features.length);
    expect(full.slice(16 * drivingMemorySlotInputs(), 16 * drivingMemorySlotInputs() + 6)).toEqual([1, 1, ...observation.features.slice(0, 4)]);
    const expired = memory.encode({ ...observation, capturedAt: 6001 }).features;
    expect(expired.slice(16 * drivingMemorySlotInputs(), 17 * drivingMemorySlotInputs())).toEqual(new Array(drivingMemorySlotInputs()).fill(0));
});

it('uses the newest sample in each time bin without treating repeated images as history', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    const initial = memory.encode(observation).features;
    expect(memory.encode(observation).features).toEqual(initial);
    const recent = { ...observation, capturedAt: 1050, features: observation.features.map(() => 0.4) };
    memory.encode(recent);
    const features = memory.encode({ ...observation, capturedAt: 1300 }).features;
    expect(features.slice(0, 6)).toEqual([1, 250 / 5000, 0.4, 0.4, 0.4, 0.4]);
    expect(features.slice(drivingMemorySlotInputs(), drivingMemoryInputs())).toEqual(new Array(drivingMemoryInputs() - drivingMemorySlotInputs()).fill(0));
});

it('copies the entire historical observation so later mutations cannot alter a learning input', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    memory.encode(observation);
    const original = observation.features.slice();
    observation.features.fill(0);
    const encoded = memory.encode({ ...observation, capturedAt: 1200 }).features;
    expect(encoded.slice(2, drivingMemorySlotInputs())).toEqual(original);
    memory.encode({ ...observation, capturedAt: 1400 });
    memory.reset();
    expect(encoded.slice(2, drivingMemorySlotInputs())).toEqual(original);
    expect(memory.encode({ ...observation, capturedAt: 1600 }).features.slice(0, drivingMemoryInputs()).every((v) => v === 0)).toBe(true);
});

it('retains full-resolution camera frames aligned to history and owns their pixels', () => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    observation.camera.rgb.fill(255);
    const first = memory.encode(observation);
    observation.camera.rgb.fill(0);
    const second = memory.encode({ ...observation, capturedAt: 1200,
        camera: { ...observation.camera, rgb: observation.camera.rgb.slice() } });
    expect(first.cameras).toHaveLength(18);
    expect(first.cameras[17]!.width).toBe(768);
    expect(first.cameras[17]!.height).toBe(768);
    expect(first.cameras[17]!.rgb.length).toBe(768 * 768 * 3);
    expect(first.cameras[17]!.rgb.every((value) => value === 255)).toBe(true);
    expect(second.cameras[0]).toBe(first.cameras[17]);
    expect(second.cameras[17]!.rgb.every((value) => value === 0)).toBe(true);
    expect(second.cameras.slice(1, 17)).toEqual(new Array(16).fill(null));
    memory.reset();
    expect(memory.encode({ ...observation, capturedAt: 1400 }).cameras.slice(0, 17))
        .toEqual(new Array(17).fill(null));
});

it.each(['session', 'reference', 'context', 'gap', 'clock'] as const)('clears history at a %s boundary', (boundary) => {
    const { observation } = trainingFixture();
    const memory = new DrivingMemory();
    memory.encode(observation);
    memory.encode({ ...observation, capturedAt: 1200 });
    const next = { ...observation, capturedAt: 1400 };
    if (boundary === 'session') next.session = 'new-session';
    if (boundary === 'reference') next.reference = { ...observation.reference };
    if (boundary === 'context') next.context = { ...observation.context, car: 'different' };
    if (boundary === 'gap') next.capturedAt = 2701;
    if (boundary === 'clock') next.capturedAt = 1100;
    expect(memory.encode(next).features.slice(0, drivingMemoryInputs()).every((value) => value === 0)).toBe(true);
});
