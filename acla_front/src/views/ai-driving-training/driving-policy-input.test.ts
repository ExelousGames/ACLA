import { cloneConfig, drivingPolicyInputs } from './driving-model-config';
import { DrivingMemory } from './driving-memory';
import { assembleDrivingInput } from './driving-policy-input';
import { DrivingPolicy, emptyTrainingDocument, readTrainingDocument } from './driving-policy';
import { trainingFixture } from './training-fixtures';

it('processes sparse history chronologically, aligning custom feature layouts and cameras', () => {
    const config = cloneConfig();
    config.memory.features = ['controls', 'speed'];
    const { observation } = trainingFixture();
    const memory = new DrivingMemory(config);
    const frame = (capturedAt: number, speed: number) => {
        const features = observation.features.slice(); features[3] = speed;
        return { ...observation, capturedAt, features };
    };
    memory.encode(frame(1000, 0.1));
    memory.encode(frame(1400, 0.2));
    const input = assembleDrivingInput(memory.encode(frame(1600, 0.3)), config);
    const width = drivingPolicyInputs(config);
    expect(input.steps).toBe(3);
    expect([input.features[5], input.features[width + 5], input.features[2 * width + 5]]).toEqual([0.1, 0.2, 0.3]);
    expect([input.features[1], input.features[width + 1], input.features[2 * width + 1]]).toEqual([600 / 5000, 200 / 5000, 0]);
    expect(input.features.slice(2, 5)).toEqual([0, 0, 0]);
    expect(input.cameras).toHaveLength(1);
    expect(input.cameras[0].offset).toBeGreaterThan(2 * width);
    memory.reset();
    expect(assembleDrivingInput(memory.encode(frame(1800, 0.4)), config).steps).toBe(1);
});

it('assembles all 17 history bins and the current observation into 18 chronological LSTM steps', () => {
    const config = cloneConfig(), { observation } = trainingFixture();
    const memory = new DrivingMemory(config);
    for (let capturedAt = 1000; capturedAt <= 5800; capturedAt += 300) memory.encode({ ...observation, capturedAt });
    const input = assembleDrivingInput(memory.encode({ ...observation, capturedAt: 6000 }), config);
    const width = drivingPolicyInputs(config);
    expect(input.steps).toBe(18);
    expect(input.cameras).toHaveLength(18);
    expect(Array.from({ length: input.steps }, (_, i) => input.features[i * width + 1]))
        .toEqual([...Array.from({ length: 17 }, (_, i) => (5000 - i * 300) / 5000), 0]);
});

it('rejects pooled-image configurations and missing or malformed CNN/LSTM weights without migration', () => {
    const { observation } = trainingFixture(), config = cloneConfig();
    config.network.hiddenUnits = 2;
    const saved = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    for (const lstmWeights of [undefined, [], [[NaN]], saved.model.lstmWeights.slice(1), saved.model.lstmWeights.map((row) => row.slice(1))]) {
        expect(() => DrivingPolicy.restore({ ...saved, model: { ...saved.model, lstmWeights } })).toThrow(/checkpoint/);
    }
    expect(() => DrivingPolicy.restore({ ...saved, model: { ...saved.model, cameraWeights: [] } })).toThrow(/checkpoint/);
    for (const cameraWeights of [undefined, [[[NaN]]], saved.model.cameraWeights.slice(1),
        saved.model.cameraWeights.map((rows) => rows.map((row) => row.slice(1)))]) {
        expect(() => DrivingPolicy.restore({ ...saved, model: { ...saved.model, cameraWeights } })).toThrow(/checkpoint/);
    }
    const old: any = emptyTrainingDocument();
    old.configuration.network.type = 'lstm';
    old.configuration.observation.cameraEncoding = 'rgb-bytes-average-pool';
    expect(() => readTrainingDocument(old)).toThrow(/configuration/);
});

it.each(['dimensions', 'bytes', 'nonfinite'])('rejects malformed GPU input: %s', (fault) => {
    const { observation } = trainingFixture(), config = cloneConfig();
    const input = new DrivingMemory(config).encode(observation);
    if (fault === 'nonfinite') input.features[0] = NaN;
    else input.cameras[input.cameras.length - 1] = { ...observation.camera,
        ...(fault === 'dimensions' ? { width: 1 } : { rgb: new Uint8Array(3) }) };
    expect(() => assembleDrivingInput(input, config)).toThrow(/observation/);
});
