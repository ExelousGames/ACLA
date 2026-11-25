import { cloneConfig, drivingCameraInputs, drivingFeatureSizes, drivingPolicyInputs } from './driving-model-config';
import { DrivingMemory } from './driving-memory';
import { assembleDrivingInput } from './driving-policy-input';
import { DrivingPolicy, emptyTrainingDocument, readTrainingDocument } from './driving-policy';
import { trainingFixture } from './training-fixtures';

it('processes sparse history chronologically, aligning custom feature layouts and cameras', () => {
    const config = cloneConfig();
    config.memory.features = ['controls', 'gForce', 'speed'];
    const { observation } = trainingFixture();
    const memory = new DrivingMemory(config);
    const frame = (capturedAt: number, speed: number) => {
        const features = observation.features.slice(); features[3] = speed;
        features.splice(60, 3, speed, -speed, speed * 2);
        return { ...observation, capturedAt, features };
    };
    memory.encode(frame(1000, 0.1));
    memory.encode(frame(1400, 0.2));
    const input = assembleDrivingInput(memory.encode(frame(1600, 0.3)), config);
    const width = drivingPolicyInputs(config);
    expect(input.steps).toBe(3);
    expect([input.features[5], input.features[width + 5], input.features[2 * width + 5]]).toEqual([0.1, 0.2, 0.3]);
    expect(Array.from({ length: input.steps }, (_, i) => input.features.slice(i * width + 62, i * width + 65)))
        .toEqual([[0.1, -0.1, 0.2], [0.2, -0.2, 0.4], [0.3, -0.3, 0.6]]);
    expect([input.features[1], input.features[width + 1], input.features[2 * width + 1]]).toEqual([600 / 5000, 200 / 5000, 0]);
    expect(input.features.slice(2, 5)).toEqual([0, 0, 0]);
    expect(input.cameras).toHaveLength(1);
    expect(input.cameras[0].offset).toBeGreaterThan(2 * width);
    memory.reset();
    expect(assembleDrivingInput(memory.encode(frame(1800, 0.4)), config).steps).toBe(1);
});

it('assembles all 17 history bins and the current observation into 18 chronological GRU steps', () => {
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

it('passes centerline through custom current, memory and recorded history layouts while zeroing reserved slots', () => {
    const config = cloneConfig(), { observation } = trainingFixture();
    config.observation.features = ['position', 'speed', 'camera', 'centerline', 'targetSpeed', 'gear', 'referenceSpeed'];
    config.memory.features = ['gear', 'referenceSpeed', 'position', 'targetSpeed', 'centerline', 'speed'];
    const memory = new DrivingMemory(config, true);
    const cameraInputs = drivingCameraInputs(config), centerlineWidth = drivingFeatureSizes(config).centerline;
    const makeFrame = (capturedAt: number, speed: number, gear: number, route: number) => ({
        ...observation, capturedAt, features: [route, -route, speed, ...new Array(centerlineWidth).fill(route), route, gear, -route],
    });
    const expected = (age: number, speed: number, gear: number, route: number) => [1, age, 0, 0, speed,
        ...new Array(cameraInputs).fill(0), ...new Array(centerlineWidth).fill(route), 0, gear, 0];
    memory.encode(makeFrame(1000, 0.3, 0.1, 0.8));
    const input = memory.encode(makeFrame(1200, 0.5, 0.2, -0.9));
    const assembled = assembleDrivingInput(input, config);
    expect(assembled.steps).toBe(2);
    const current = expected(0, 0.5, 0.2, -0.9);
    expect(assembled.features).toEqual([...expected(200 / config.memory.windowMs, 0.3, 0.1, 0.8), ...current]);
    expect(assembled.cameras.map(({ offset }) => offset)).toEqual([drivingPolicyInputs(config) + 5]);

    const recurrent = assembleDrivingInput(input, config, 1000);
    expect(recurrent.steps).toBe(1);
    expect(recurrent.features).toEqual(current);
    expect(recurrent.cameras.map(({ offset }) => offset)).toEqual([5]);

    const recorded = assembleDrivingInput(input, config, -Infinity, input.recordedHistory);
    expect(recorded.steps).toBe(2);
    expect(recorded.features).toEqual([...expected(0, 0.3, 0.1, 0.8), ...current]);
    expect(recorded.cameras.map(({ offset }) => offset)).toEqual([5, drivingPolicyInputs(config) + 5]);
});

it('rejects old configurations and missing or malformed CNN/projection/GRU weights without migration', () => {
    const { observation } = trainingFixture(), config = cloneConfig();
    config.network.hiddenUnits = 2;
    const saved = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    for (const gruWeights of [undefined, [], [[NaN]], saved.model.gruWeights.slice(1), saved.model.gruWeights.map((row) => row.slice(1))]) {
        expect(() => DrivingPolicy.restore({ ...saved, model: { ...saved.model, gruWeights } })).toThrow(/checkpoint/);
    }
    expect(() => DrivingPolicy.restore({ ...saved, model: { ...saved.model, cameraWeights: [] } })).toThrow(/checkpoint/);
    for (const cameraWeights of [undefined, [[[NaN]]], saved.model.cameraWeights.slice(1),
        saved.model.cameraWeights.map((rows) => rows.map((row) => row.slice(1)))]) {
        expect(() => DrivingPolicy.restore({ ...saved, model: { ...saved.model, cameraWeights } })).toThrow(/checkpoint/);
    }
    for (const cameraProjectionWeights of [undefined, [], [[NaN]], saved.model.cameraProjectionWeights.slice(1),
        saved.model.cameraProjectionWeights.map((row) => row.slice(1))]) {
        expect(() => DrivingPolicy.restore({ ...saved, model: { ...saved.model, cameraProjectionWeights } })).toThrow(/checkpoint/);
    }
    const square: any = emptyTrainingDocument();
    square.configuration.observation.cameraWidth = square.configuration.observation.cameraHeight = 768;
    square.configuration.network.cameraLayers = [
        { kernelSize: 8, stride: 8, channels: 8 }, { kernelSize: 4, stride: 4, channels: 16 },
        { kernelSize: 3, stride: 3, channels: 16 },
    ];
    expect(() => readTrainingDocument(square)).toThrow(/configuration/);
    const old: any = emptyTrainingDocument();
    old.configuration.network.type = 'cnn-lstm';
    expect(() => readTrainingDocument(old)).toThrow(/configuration/);
    expect(() => DrivingPolicy.restore({ ...saved, configuration: old.configuration })).toThrow(/configuration/);
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
