import { mockDrivingGpu } from './driving-gpu-test-model';
import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import AiDrivingTraining from './AiDrivingTraining';
import type { ObservationResult } from './driving-observation';
import { circuitMapFixture, referenceLapFixture, trainingFixture } from './training-fixtures';
import { DrivingPolicy, emptyTrainingDocument } from './driving-policy';
import { DrivingGpuModel } from './driving-gpu-model';
import { DrivingTrainingLoop } from './driving-training-loop';
import { cloneConfig } from './driving-model-config';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import { SessionScreenFrameReader } from 'views/live-session/live-session-recording/session-screen-frame';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';

let mockInput: ObservationResult;
const mockRead = jest.fn();
const mockFindComponentRef = jest.fn();
jest.mock('contexts/OperationComponentRefContext', () => ({
    OPERATION_COMPONENT_NAMES: { LIVE_SESSION: 'live-session' },
    useOptionalOperationComponentRefDirectory: () => ({ findComponentRef: mockFindComponentRef }),
}));
jest.mock('./driving-observation', () => ({
    ...jest.requireActual('./driving-observation'),
    readDrivingObservation: (...args: unknown[]) => args[5] === undefined
        ? jest.requireActual('./driving-observation').readDrivingObservation(...args) : mockRead(...args),
}));
// Match definition-list values to their terms through semantic queries.
const getDefinition = (label: string) => screen.getAllByRole('definition')[
    screen.getAllByRole('term').findIndex((term) => term.textContent === label)];
const flush = async () => { await act(async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); }); };
const screenFrame = ({ vision }: ReturnType<typeof trainingFixture>) => ({ capturedAt: vision.capturedAt,
    camera: vision.camera!, telemetry: vision.telemetry!, processingMs: 5, publishedAt: vision.capturedAt + 5 });
const MODEL_PATH = 'C:/models/driving.json';
const files = new Map<string, string>();
let selectedFile: string | null;
const renderTraining = async (onOpenLiveSession = () => undefined) => {
    const view = render(<AiDrivingTraining onOpenLiveSession={onOpenLiveSession} />);
    await flush();
    return view;
};

beforeEach(() => {
    jest.useFakeTimers();
    // Keep interpolated UI controls deterministic; policy tests cover exploration.
    jest.spyOn(Math, 'random').mockReturnValue(0.5);
    mockDrivingGpu();
    jest.spyOn(DrivingPolicy.prototype, 'initializeGpu');
    localStorage.clear();
    files.clear();
    files.set(MODEL_PATH, JSON.stringify(emptyTrainingDocument()));
    selectedFile = MODEL_PATH;
    window.drivingModelFiles = {
        current: jest.fn(async () => selectedFile),
        choose: jest.fn(async () => MODEL_PATH),
        select: jest.fn(async (filePath) => { selectedFile = filePath; }),
        read: jest.fn(async (filePath) => {
            const json = files.get(filePath);
            if (json === undefined) throw new Error('Model file not found.');
            return json;
        }),
        write: jest.fn(async (filePath, json) => { files.set(filePath, json); }),
    };
    mockInput = { error: 'Waiting for live telemetry.' };
    mockRead.mockReset().mockImplementation(() => mockInput);
    mockFindComponentRef.mockReset();
    window.virtualController = { connect: jest.fn().mockResolvedValue(undefined), apply: jest.fn().mockResolvedValue(undefined),
        disconnect: jest.fn().mockResolvedValue(undefined),
        installAccPreset: jest.fn().mockImplementation(async (deviceId) => ({
            name: `ACLA AI Training - vJoy ${deviceId}`, path: '/preset.json', deviceId,
        })) };
});
afterEach(async () => { await flush(); delete window.virtualController; delete window.drivingModelFiles; jest.restoreAllMocks(); jest.useRealTimers(); });

it('forwards every fresh telemetry event between controller ticks without a camera', async () => {
    const collect = jest.spyOn(DrivingTrainingLoop.prototype, 'recordTelemetry');
    const { live } = trainingFixture(Date.now());
    await renderTraining();
    act(() => { liveTelemetryStore.resetSession(); });
    collect.mockClear();
    act(() => {
        liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence: 1, committedSequence: 1, committedCount: 1,
            sample: { ...live.currentTelemetry, Physics_gas: 0.2 } });
        liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence: 2, committedSequence: 2, committedCount: 2,
            sample: { ...live.currentTelemetry, Physics_gas: 0.8 } });
    });
    expect(collect.mock.calls.map(([sample]) => sample.controls.throttle)).toEqual([0.2, 0.8]);
    expect(collect.mock.calls.map(([sample]) => sample.sampleIndex)).toEqual([0, 1]);
    expect(collect.mock.calls[1][0].measuredAt).toBe(collect.mock.calls[0][0].measuredAt);
    expect(mockInput.observation).toBeUndefined();
});

it('passes the Live Map centerline and boundaries to observations and displays track rewards', async () => {
    const circuitMap: CircuitMapDto = { id: 'monza', game: 'acc', source_track_key: 'monza', circuit_name: 'Monza', resolution: 1000,
        samples: { ...circuitMapFixture().samples, ...Object.fromEntries(([['left_boundary', -5], ['right_boundary', 5]] as const).map(([side, x]) => [side,
            [-10, 0, 10].map((y, i) => ({ x, y, z: 0, normalized_position: 0.24 + i * 0.01, bin: 240 + i * 10,
                sample_count: 1, updated_at: '' }))])) } };
    const referenceLap = referenceLapFixture();
    let fixture = trainingFixture(Date.now(), 50, referenceLap);
    const map = { getReferenceLap: () => referenceLap, getCircuitMap: () => circuitMap };
    jest.spyOn(SessionScreenFrameReader.prototype, 'read').mockImplementation(() => screenFrame(fixture));
    mockFindComponentRef.mockImplementation((name) => name === 'visualization:live-trajectory-map' ? { current: map } : null);
    jest.spyOn(liveTelemetryStore, 'getSnapshot').mockImplementation(() => fixture.live);
    mockRead.mockImplementation((...args) => jest.requireActual('./driving-observation').readDrivingObservation(...args));
    await renderTraining();
    expect(mockRead).toHaveBeenLastCalledWith(fixture.live, screenFrame(fixture), referenceLap, 50, expect.any(Number), expect.anything(), circuitMap);
    expect(getDefinition('Centerline')).toHaveTextContent('50 steps behind · 50 steps ahead');
    expect(screen.getByLabelText('Off-track detection')).toHaveTextContent('Waiting for a scored transition.');
    expect(getDefinition('On-track reward').textContent).toBe('0');
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    act(() => { jest.advanceTimersByTime(200); }); await flush();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    ['front_left', 'front_right', 'rear_left', 'rear_right'].forEach((wheel, i) => {
        Object.assign(fixture.live.currentTelemetry, { [`Physics_tyre_contact_point_${wheel}_x`]: 7 + i % 2,
            [`Physics_tyre_contact_point_${wheel}_y`]: i < 2 ? 2 : -2 });
    });
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(getDefinition('Off-track penalty').textContent).toBe('-1');
    expect(getDefinition('On-track reward').textContent).toBe('0');
    expect(screen.getByLabelText('Off-track detection')).toHaveTextContent('Off track');
    act(() => { jest.advanceTimersByTime(200); }); await flush();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    Object.assign(fixture.live.currentTelemetry, { Graphics_normalized_car_position: 0.252 });
    ['front_left', 'front_right', 'rear_left', 'rear_right'].forEach((wheel, i) => {
        Object.assign(fixture.live.currentTelemetry, { [`Physics_tyre_contact_point_${wheel}_x`]: i % 2 ? 1 : -1,
            [`Physics_tyre_contact_point_${wheel}_y`]: i < 2 ? 4 : 0 });
    });
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(getDefinition('On-track reward').textContent).toBe('0.3');
    expect(getDefinition('Off-track penalty').textContent).toBe('0');
    expect(screen.getByLabelText('Off-track detection')).toHaveTextContent('On track');
    act(() => { jest.advanceTimersByTime(200); }); await flush();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(getDefinition('Off-track penalty').textContent).toBe('0');
    expect(screen.getByLabelText('Off-track detection')).toHaveTextContent('Off-track detection unavailable');
    expect(getDefinition('On-track reward').textContent).toBe('0');
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
});

it('shows a GPU startup failure without starting AI controls', async () => {
    mockInput = { observation: trainingFixture(Date.now()).observation };
    jest.spyOn(DrivingPolicy.prototype, 'initializeGpu').mockRejectedValue(new Error('A hardware WebGPU device is required.'));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('A hardware WebGPU device is required.');
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
});

it('shows each control-change penalty and sends the combined reward to learning', async () => {
    const { observation } = trainingFixture(Date.now()), config = cloneConfig();
    config.learning.sequenceSeconds = 0.001;
    config.network.hiddenUnits = 2;
    config.network.cameraProjectionUnits = 2;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 1; });
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, () => 0.5, config).checkpoint();
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    const infer = DrivingPolicy.prototype.decideAsync;
    jest.spyOn(DrivingPolicy.prototype, 'decideAsync').mockImplementation(async function (this: DrivingPolicy, ...args) {
        const decision = await infer.apply(this, args);
        decision.plan.forEach((point, step) => {
            point.action.steering = step === 1 ? -0.4 : 0.4;
            point.action.pedal = step === 1 ? -0.6 : 0.8;
            point.latent[0] = Math.atanh(point.action.pedal);
            point.latent[1] = Math.atanh(point.action.steering);
        });
        return decision;
    });
    const learn = jest.spyOn(DrivingPolicy.prototype, 'learnAsync');
    await renderTraining();
    expect(getDefinition('Steering-change penalty').textContent).toBe('0');
    expect(getDefinition('Throttle-change penalty').textContent).toBe('0');
    expect(getDefinition('Brake-change penalty').textContent).toBe('0');
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    for (let i = 0; i < 3; i++) {
        act(() => { jest.advanceTimersByTime(100); }); await flush();
    }
    mockInput = { observation: { ...observation, capturedAt: Date.now(), measuredAt: Date.now() } };
    act(() => { jest.advanceTimersByTime(20); }); await flush();
    expect(getDefinition('Normalized position progress').textContent).toBe('0');
    expect(Number(getDefinition('Acceleration / deceleration').textContent)).toBeCloseTo(-0.12);
    expect(Number(getDefinition('Steering-change penalty').textContent)).toBeCloseTo(-0.064);
    expect(Number(getDefinition('Throttle-change penalty').textContent)).toBeCloseTo(-0.064);
    expect(Number(getDefinition('Brake-change penalty').textContent)).toBeCloseTo(-0.036);
    expect(learn).toHaveBeenCalledTimes(3);
    // Every replay proxy shares the same observed endpoints and full Bellman reward.
    learn.mock.calls.forEach(([, reward]) => expect(reward).toBeCloseTo(-0.284));
    const total = learn.mock.calls[0][1];
    expect(screen.getByText('Episode reward').previousElementSibling?.textContent).toBe(String(total));
    expect(screen.getByText(/Holding a steering angle has no penalty/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
});

it.each([[30, 0, 0, 0], [30, 23, 0, 0], [10 / 3.6, 0, 0, 0], [30, 24, -6, 0.4], [30, 36, 6, 0.4]])(
    'shows impact confidence, severity, penalty and unavailable timing after a jolt (from=%s, forward=%s, right=%s, steering=%s)', async (initial, forward, right, steering) => {
    const observation = { ...trainingFixture(Date.now()).observation, speed: initial * 3.6,
        localVelocity: [0, initial, 0] as [number, number, number] };
    observation.controls = { ...observation.controls, steering };
    mockInput = { observation };
    await renderTraining();
    expect(screen.getByLabelText('Impact estimate')).toHaveTextContent('Waiting for two fresh motion samples.');
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    act(() => { jest.advanceTimersByTime(200); }); await flush();
    mockInput = { observation: { ...observation, capturedAt: Date.now(), measuredAt: Date.now(),
        localVelocity: [right, forward, 0], speed: Math.hypot(right, forward) * 3.6 } };
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    if (initial === 30 && forward === 0) {
        expect(getDefinition('Impact penalty').textContent).toBe('-5');
        expect(screen.getByLabelText('Impact estimate')).toHaveTextContent('Possible impact · Confidence 100% · Estimated severity 100/100');
    } else {
        expect(Number(getDefinition('Impact penalty').textContent)).toBeLessThan(-0.4);
        expect(screen.getByLabelText('Impact estimate')).toHaveTextContent('Possible impact');
    }
    expect(screen.getByLabelText('Impact estimate')).toHaveTextContent(/Excess Δv .* m\/s/);
    act(() => { jest.setSystemTime(Date.now() + 500); });
    mockInput = { observation: { ...observation, capturedAt: Date.now(), measuredAt: Date.now() } };
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(getDefinition('Impact penalty').textContent).toBe('0');
    expect(screen.getByLabelText('Impact estimate')).toHaveTextContent('Impact estimate unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
});

it.each([20, 80])('shows a brief %s km/h impact from live telemetry between slow AI frames', async (kph) => {
    const fixture = trainingFixture(Date.now());
    act(() => liveTelemetryStore.resetSession());
    let sequence = 0;
    const speed = kph / 3.6;
    const publish = (forward = speed, right = 0) => {
        sequence++;
        liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence,
            committedSequence: sequence, committedCount: sequence, sample: { ...fixture.live.currentTelemetry,
                Physics_local_velocity_x: right, Physics_local_velocity_y: forward, Physics_speed_kmh: Math.hypot(right, forward) * 3.6 } });
    };
    act(() => publish());
    const live = liveTelemetryStore.getSnapshot();
    const initial = { ...fixture.observation, speed: kph, localVelocity: [0, speed, 0] as [number, number, number],
        session: JSON.stringify([live.sessionGeneration, live.streamGeneration, 'acc', 'GT3', 'monza']) };
    mockInput = { observation: initial };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    act(() => { jest.advanceTimersByTime(100); publish(speed * 0.6, speed * 0.3); }); await flush();
    for (let i = 0; i < 7; i++) {
        act(() => { jest.advanceTimersByTime(100); publish(); }); await flush();
    }
    mockInput = { observation: { ...initial, capturedAt: Date.now(), measuredAt: Date.now() } };
    act(() => { jest.advanceTimersByTime(100); publish(); }); await flush();
    expect(Number(getDefinition('Impact penalty').textContent)).toBeLessThan(0);
    expect(screen.getByLabelText('Impact estimate')).toHaveTextContent('Possible impact');
    // The next image must follow the acknowledgement of the newly applied action.
    act(() => { jest.advanceTimersByTime(100); publish(); }); await flush();
    mockInput = { observation: { ...initial, capturedAt: Date.now(), measuredAt: Date.now() } };
    act(() => { jest.advanceTimersByTime(100); publish(); }); await flush();
    expect(Number(getDefinition('Impact penalty').textContent)).toBe(0);
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
});

it('shows sustained blocked motion in the reward without an impact', async () => {
    const fixture = trainingFixture(Date.now());
    act(() => liveTelemetryStore.resetSession());
    let sequence = 0;
    const publish = () => {
        sequence++;
        liveTelemetryStore.publishFrame({ type: 'frame', game: 'acc', sequence,
            committedSequence: sequence, committedCount: sequence, sample: { ...fixture.live.currentTelemetry,
                Physics_local_velocity_x: 0, Physics_local_velocity_y: 0, Physics_speed_kmh: 0, Physics_gas: 0.8 } });
    };
    act(() => publish());
    const live = liveTelemetryStore.getSnapshot();
    const initial = { ...fixture.observation, speed: 0, localVelocity: [0, 0, 0] as [number, number, number],
        controls: { ...fixture.observation.controls, throttle: 0.8 },
        session: JSON.stringify([live.sessionGeneration, live.streamGeneration, 'acc', 'GT3', 'monza']) };
    mockInput = { observation: initial };
    await renderTraining();
    expect(getDefinition('Blocked-motion penalty').textContent).toBe('0');
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    for (let i = 0; i <= 10; i++) {
        act(() => { jest.advanceTimersByTime(100); publish(); }); await flush();
    }
    mockInput = { observation: { ...initial, capturedAt: Date.now(), measuredAt: Date.now() } };
    act(() => { jest.advanceTimersByTime(20); }); await flush();
    expect(Number(getDefinition('Blocked-motion penalty').textContent)).toBeCloseTo(-0.85);
    expect(getDefinition('Impact penalty').textContent).toBe('0');
    const rewardBreakdown = within(screen.getByRole('region', { name: 'Reward signal' })).getAllByRole('definition')
        .reduce((total, entry) => total + Number(entry.textContent), 0);
    expect(screen.getByText((text) => Math.abs(Number(text) - rewardBreakdown) < 1e-9, { selector: 'strong' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
});

it('honors Stop while preparing the GPU', async () => {
    mockInput = { observation: trainingFixture(Date.now()).observation };
    let ready!: () => void;
    jest.spyOn(DrivingPolicy.prototype, 'initializeGpu').mockImplementation(() => new Promise<void>((resolve) => { ready = resolve; }));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    expect(screen.getByText('Preparing GPU…')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    ready(); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
});

it('enables a restored model with unchanged inputs even when its context keys were reordered', async () => {
    const { observation } = trainingFixture(Date.now());
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 10;
    const { game, car, map } = checkpoint.model.context;
    checkpoint.model.context = { map, car, game };
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.queryByText(/does not match|belongs to different inputs/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Run trained model' })).toBeEnabled();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
});

it.each(['Start training', 'Run trained model'])('waits for fresh telemetry after loading a model for %s', async (button) => {
    let fixture = trainingFixture(Date.now());
    const config = cloneConfig();
    config.network.hiddenUnits = 3;
    config.network.cameraProjectionUnits = 8;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    const checkpoint = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    const map = { getReferenceLap: () => fixture.referenceLap, getCircuitMap: () => fixture.circuitMap };
    jest.spyOn(SessionScreenFrameReader.prototype, 'read').mockImplementation(() => screenFrame(fixture));
    mockFindComponentRef.mockImplementation((name) => name === 'visualization:live-trajectory-map' ? { current: map } : null);
    jest.spyOn(liveTelemetryStore, 'getSnapshot').mockImplementation(() => fixture.live);
    mockRead.mockImplementation((...args) => jest.requireActual('./driving-observation').readDrivingObservation(...args));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.getByRole('button', { name: button })).toBeEnabled();

    const restore = DrivingPolicy.restore;
    jest.spyOn(DrivingPolicy, 'restore').mockImplementation((document) => {
        const restored = restore(document);
        // Large JSON validation/copying blocks delivery of the next telemetry event.
        jest.setSystemTime(Date.now() + config.runtime.telemetryMaxAgeMs + 1);
        return restored;
    });
    fireEvent.click(screen.getByRole('button', { name: button })); await flush();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
    fixture = trainingFixture(Date.now());
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('AI');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
});

it.each(['Start training', 'Run trained model'])('allows %s with a saved model from another game, car and map', async (button) => {
    const { observation } = trainingFixture(Date.now());
    const config = cloneConfig();
    config.network.hiddenUnits = 3;
    config.network.cameraProjectionUnits = 8;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation: { ...observation, context: { game: 'iracing', car: 'MX5', map: 'spa' }, session: 'another-session' } };
    const view = await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: button })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: button })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('AI');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
    view.unmount(); await flush();
    mockInput = { observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: button })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('AI');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(window.drivingModelFiles!.choose).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
});

it('shows input layout mismatches and recovers on another combination without replacing the saved model', async () => {
    const { observation } = trainingFixture(Date.now());
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation: { ...observation, context: { game: 'iracing', car: 'MX5', map: 'spa' },
        features: observation.features.slice(16) } };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    const warning = screen.getByRole('alert');
    expect(warning).not.toHaveTextContent('Game:');
    expect(warning).not.toHaveTextContent('Car:');
    expect(warning).not.toHaveTextContent('Track:');
    expect(warning).not.toHaveTextContent('Segmentation labels');
    expect(warning).toHaveTextContent(`Input count: saved ${observation.features.length}; live ${observation.features.length - 16}.`);
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Run trained model' })).toBeDisabled();
    mockInput = { observation: { ...mockInput.observation!, features: observation.features } };
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Run trained model' })).toBeEnabled();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
});

it('displays the complete action plan in one table and clears it on Stop', async () => {
    const { observation } = trainingFixture(Date.now());
    const config = cloneConfig();
    config.network.hiddenUnits = 3;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    config.network.cameraProjectionUnits = 8;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    await renderTraining();
    const table = screen.getByRole('table', { name: 'Planned driving actions' });
    expect(within(table).getByText('Waiting for AI control and fresh inputs.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    fireEvent.click(screen.getByRole('button', { name: 'Run trained model' })); await flush();
    expect(screen.getByText(/One action plan predicts 51 control steps from 0 to 5 seconds/)).toBeInTheDocument();
    expect(within(table).getAllByRole('row')).toHaveLength(52);
    expect(within(table).getAllByRole('rowheader').map((cell) => cell.textContent))
        .toEqual(Array.from({ length: 51 }, (_, step) => `${(step / 10).toFixed(1)} s`));
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(within(table).getByText('Waiting for AI control and fresh inputs.')).toBeInTheDocument();
});

it('uses pixels from the recording stream with Track Vision closed and clears them when sharing stops', async () => {
    const fixture = trainingFixture(Date.now());
    const track = { readyState: 'live', stop: jest.fn() };
    const stream = { getVideoTracks: () => [track] };
    const getScreenCaptureStream = jest.fn((): typeof stream | null => stream);
    const map = { getReferenceLap: () => fixture.referenceLap, getCircuitMap: () => fixture.circuitMap };
    mockFindComponentRef.mockImplementation((name) => name === 'live-session' ? { current: { getScreenCaptureStream } }
        : name === 'visualization:live-trajectory-map' ? { current: map } : null);
    const telemetry = jest.spyOn(liveTelemetryStore, 'getSnapshot').mockReturnValue(fixture.live);
    mockRead.mockImplementation((...args) => jest.requireActual('./driving-observation').readDrivingObservation(...args));
    const video = document.createElement('video');
    const emitFrame = jest.fn();
    const cancelFrame = jest.fn();
    Object.defineProperties(video, {
        videoWidth: { value: 1440 }, videoHeight: { value: 1080 }, readyState: { value: 2 },
        play: { value: jest.fn().mockResolvedValue(undefined) }, pause: { value: jest.fn() },
        requestVideoFrameCallback: { value: emitFrame.mockReturnValue(1) }, cancelVideoFrameCallback: { value: cancelFrame },
    });
    const dom: { createElement(tag: string, options?: ElementCreationOptions): HTMLElement } = document;
    const createElement = dom.createElement.bind(dom);
    jest.spyOn(dom, 'createElement').mockImplementation((tag, options) => tag === 'video' ? video : createElement(tag, options));
    const rgba = new Uint8ClampedArray(384 * 216 * 4);
    rgba.set([12, 34, 56, 255]);
    const drawImage = jest.fn();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        drawImage, fillRect: jest.fn(), getImageData: () => ({ data: rgba }),
    } as any);
    await renderTraining();
    expect(video.srcObject).toBe(stream);
    expect(screen.getByText(/A fresh image is required/)).toBeInTheDocument();
    act(() => { jest.advanceTimersByTime(250); });
    emitFrame.mock.calls[0][0]();
    const latest = { ...fixture.live, receivedAt: Date.now(), currentTelemetry: { ...fixture.live.currentTelemetry, Physics_speed_kmh: 150 } };
    telemetry.mockReturnValue(latest);
    act(() => { jest.advanceTimersByTime(250); });
    const frame = mockRead.mock.calls[mockRead.mock.calls.length - 1][1];
    expect(drawImage).toHaveBeenCalledWith(video, 48, 0, 288, 216);
    rgba.fill(0);
    expect(Array.from(frame.camera.rgb.slice(0, 3))).toEqual([12, 34, 56]);
    expect(frame.telemetry).toBe(fixture.live);
    expect(getDefinition('Speed')).toHaveTextContent('108.0 km/h');
    expect(mockFindComponentRef).not.toHaveBeenCalledWith('visualization:track-vision');
    getScreenCaptureStream.mockReturnValue(null);
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.getByText(/A fresh image is required/)).toBeInTheDocument();
    expect(cancelFrame).toHaveBeenCalledWith(1);
    expect(video.srcObject).toBeNull();
    expect(track.stop).not.toHaveBeenCalled();
});

it('shows direct observation inputs without summary metadata', async () => {
    mockInput = { observation: trainingFixture(Date.now()).observation };
    await renderTraining();
    act(() => { jest.advanceTimersByTime(250); });
    ['Current observation', 'Observation history', 'Total policy inputs'].forEach((label) => {
        expect(screen.queryByText(label)).not.toBeInTheDocument();
    });
    expect(getDefinition('Camera image')).toHaveTextContent('384 × 216 RGB');
    expect(getDefinition('Camera image')).not.toHaveTextContent(/history/);
    expect(screen.getByText(/5-second observation history covers all AI observation inputs/)).toHaveTextContent('sampled in 300 ms bins');
    for (const label of ['Reconstructed scene', 'Depth model', 'Segmentation labels']) {
        expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
    expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();
    mockInput = { error: 'Screen capture camera pixels must be 384 × 216 RGB.' };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Camera image')).toHaveTextContent('384 × 216 RGB · Waiting');
    expect(screen.getByText('Screen capture camera pixels must be 384 × 216 RGB.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
});

it.each([true, false])('displays the live consumer sample without independently polling while control is pending (training=%s)', async (train) => {
    const { observation } = trainingFixture(Date.now()), config = cloneConfig();
    config.network.hiddenUnits = 3;
    config.network.cameraProjectionUnits = 8;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    checkpoint.model.outputWeights.slice(2, 5).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === 0 ? 4 : 0; });
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation: { ...observation, speed: 108 } };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: train ? 'Start training' : 'Run trained model' })); await flush();

    let release!: () => void;
    (window.virtualController!.apply as jest.Mock).mockImplementationOnce(() => new Promise<void>((resolve) => { release = resolve; }));
    mockInput = { observation: { ...observation, speed: 128, capturedAt: Date.now() + 20, measuredAt: Date.now() + 20 } };
    act(() => { jest.advanceTimersByTime(20); }); await flush();
    expect(release).toBeDefined();
    mockRead.mockClear();
    mockInput = { observation: { ...observation, speed: 208, capturedAt: Date.now() + 20, measuredAt: Date.now() + 20 } };
    act(() => { jest.advanceTimersByTime(750); }); await flush();
    expect(mockRead).not.toHaveBeenCalled();
    expect(getDefinition('Speed')).toHaveTextContent('128.0 km/h');

    // Configuration changes also wait for the consumer's next sample.
    fireEvent.change(screen.getByLabelText('Driving aggression'), { target: { value: '60' } }); await flush();
    expect(mockRead).not.toHaveBeenCalled();
    expect(getDefinition('Speed')).toHaveTextContent('128.0 km/h');
    release(); await flush();

    (window.virtualController!.apply as jest.Mock).mockImplementationOnce(() => new Promise<void>((resolve) => { release = resolve; }));
    mockInput = { error: 'Waiting for a fresh camera observation.' };
    act(() => { jest.advanceTimersByTime(20); }); await flush();
    mockRead.mockClear();
    mockInput = { observation: { ...observation, speed: 208 } };
    act(() => { jest.advanceTimersByTime(750); }); await flush();
    expect(mockRead).not.toHaveBeenCalled();
    expect(within(screen.getByRole('region', { name: 'AI observation inputs' })).getByText('Waiting for a fresh camera observation.')).toBeInTheDocument();
    expect(getDefinition('Speed')).toHaveTextContent('—');
    release(); await flush();

    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(mockRead).toHaveBeenCalled();
    expect(getDefinition('Speed')).toHaveTextContent('208.0 km/h');
});

it('keeps all input delays visible across stale frames and recovery using the saved freshness limits', async () => {
    const document = emptyTrainingDocument();
    document.configuration.runtime.visionMaxAgeMs = 1000;
    document.configuration.runtime.telemetryMaxAgeMs = 300;
    files.set(MODEL_PATH, JSON.stringify(document));
    let fixture = trainingFixture(Date.now() - 900);
    jest.spyOn(liveTelemetryStore, 'getSnapshot').mockImplementation(() => ({ ...fixture.live, receivedAt: Date.now() - 40 }));
    const reader = jest.spyOn(SessionScreenFrameReader.prototype, 'read').mockReturnValue({ ...screenFrame(fixture), captureIntervalMs: 67 });
    mockInput = { observation: fixture.observation };
    await renderTraining();
    expect(getDefinition('Live telemetry age')).toHaveTextContent('40 ms / 300 ms limit · Fresh');
    expect(getDefinition('Image age for AI')).toHaveTextContent('900 ms / 1000 ms limit · Fresh');
    expect(getDefinition('Telemetry age at image capture')).toHaveTextContent('0 ms / 300 ms limit · Fresh');
    expect(screen.queryByText(/segmentation|depth|mask readback|position model/i)).not.toBeInTheDocument();
    expect(getDefinition('Screen frame copy and RGB conversion')).toHaveTextContent('5 ms');
    expect(getDefinition('Capture interval')).toHaveTextContent('67 ms');
    expect(getDefinition('Time since frame published')).toHaveTextContent('895 ms');

    mockInput = { error: 'Share the game screen from the Live Session recording bar. A fresh image is required.' };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Image age for AI')).toHaveTextContent('1150 ms / 1000 ms limit · Stale');
    expect(getDefinition('Image age for AI')).toHaveAttribute('data-stale', 'true');
    expect(getDefinition('Paired telemetry age now')).toHaveTextContent('1150 ms');
    expect(getDefinition('Time since frame published')).toHaveTextContent('1145 ms');
    expect(getDefinition('Observation validation / assembly')).toHaveTextContent('Inputs invalid');
    expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();

    fixture = trainingFixture(Date.now());
    reader.mockReturnValue(screenFrame(fixture));
    mockInput = { observation: fixture.observation };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Image age for AI')).toHaveTextContent('250 ms / 1000 ms limit · Fresh');
    expect(getDefinition('Image age for AI')).toHaveAttribute('data-stale', 'false');
    reader.mockReturnValue(null);
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Image age for AI')).toHaveTextContent('Unavailable');
    expect(getDefinition('Time since frame published')).toHaveTextContent('Unavailable');
    expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();
});

it('shows completed driving model timings without running timers and retains the last cycle when inputs become invalid or driving stops', async () => {
    const { observation } = trainingFixture(Date.now());
    const config = cloneConfig();
    config.network.hiddenUnits = 3;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    config.network.cameraProjectionUnits = 8;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    let finish!: () => void;
    const decide = DrivingPolicy.prototype.decideAsync;
    jest.spyOn(DrivingPolicy.prototype, 'decideAsync').mockImplementation(function (this: DrivingPolicy, input, explore, shifts) {
        const decision = decide.call(this, input, explore, shifts);
        return new Promise((resolve) => { finish = () => resolve(decision); });
    });
    await renderTraining();
    const table = screen.getByRole('table', { name: 'Driving model processing' });
    expect(within(table).getByRole('row', { name: 'Driving policy inference Unavailable' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Run trained model' })); await flush();
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(within(table).getByRole('row', { name: 'Driving policy inference Running' })).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: 'Total driving model cycle Running' })).toBeInTheDocument();
    finish(); await flush();
    mockInput = { error: 'Waiting for a fresh image.' };
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(within(table).getByRole('row', { name: 'Driving policy inference 250 ms' })).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: 'Training CPU preparation / submission Not run' })).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: 'Training GPU completion wait Not run' })).toBeInTheDocument();
    expect(getDefinition('Replay count')).toHaveTextContent('0');
    expect(getDefinition('History length')).toHaveTextContent('0');
    expect(within(table).getByRole('row', { name: 'Total driving model cycle 250 ms · Complete' })).toBeInTheDocument();
    expect(screen.getByText(/Latest model cycle: Evaluation · AI control/)).toHaveTextContent('Finished 250 ms ago.');
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(within(table).getByRole('row', { name: 'Total driving model cycle 250 ms · Complete' })).toBeInTheDocument();
    expect(screen.getByText(/Latest model cycle: Evaluation · AI control/)).toHaveTextContent('Finished 500 ms ago.');
});

it('reports the saved history scope when a model selects only some observation inputs', async () => {
    const document = emptyTrainingDocument();
    document.configuration.memory.windowMs = 600;
    document.configuration.memory.intervalMs = 300;
    document.configuration.memory.features = ['speed', 'camera'];
    files.set(MODEL_PATH, JSON.stringify(document));
    await renderTraining();
    expect(screen.getByText(/observation history covers the observation inputs selected in this saved model/)).toBeInTheDocument();
});

it('shows motion inputs with axes and units, and clears them when observations are unavailable', async () => {
    await renderTraining();
    const labels = ['Local velocity', 'G-force', 'Yaw rate'];
    labels.forEach((label) => expect(getDefinition(label)).toHaveTextContent('—'));
    mockInput = { observation: trainingFixture(Date.now()).observation };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Local velocity')).toHaveTextContent('X 1.00 · Y 30.00 · Z 0.00 m/s');
    expect(getDefinition('G-force')).toHaveTextContent('X 0.50 · Y -1.00 · Z 1.00 g');
    expect(screen.queryByText('Reference error')).not.toBeInTheDocument();
    expect(screen.queryByText('World-space velocity')).not.toBeInTheDocument();
    expect(screen.queryByText('Orientation')).not.toBeInTheDocument();
    expect(getDefinition('Yaw rate')).toHaveTextContent('-0.200 rad/s');
    const observation = trainingFixture(Date.now()).observation;
    mockInput = { observation: { ...observation, context: { ...observation.context, game: 'iracing' },
        localVelocity: [1, 0, 30], gForce: [-0.5, 1, 2], yawRate: -0.2 } };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Local velocity')).toHaveTextContent('X 1.00 · Y 0.00 · Z 30.00 m/s');
    expect(getDefinition('G-force')).toHaveTextContent('X -0.50 · Y 1.00 · Z 2.00 g');
    expect(screen.queryByText('Reference error')).not.toBeInTheDocument();
    expect(getDefinition('Yaw rate')).toHaveTextContent('-0.200 rad/s');
    mockInput = { error: 'Waiting for fresh, running live telemetry.' };
    act(() => { jest.advanceTimersByTime(250); });
    labels.forEach((label) => expect(getDefinition(label)).toHaveTextContent('—'));
});

it('installs an ACC preset for the selected device and explains how to activate it', async () => {
    await renderTraining();
    fireEvent.change(screen.getByRole('combobox', { name: 'vJoy device' }), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Install ACC preset' })); await flush();
    expect(window.virtualController!.installAccPreset).toHaveBeenCalledWith(3);
    expect(screen.getByText(/Installed “ACLA AI Training - vJoy 3”/)).toHaveTextContent('load this preset');
    expect(window.virtualController!.connect).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    fireEvent.change(screen.getByRole('combobox', { name: 'vJoy device' }), { target: { value: '2' } });
    expect(screen.queryByText(/Installed “/)).not.toBeInTheDocument();
});

it('locks the device choice during installation and reports failures without claiming success', async () => {
    let rejectInstall!: (reason: Error) => void;
    (window.virtualController!.installAccPreset as jest.Mock).mockImplementation(() => new Promise((_, reject) => { rejectInstall = reject; }));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Install ACC preset' }));
    expect(screen.getByRole('button', { name: 'Installing ACC preset…' })).toBeDisabled();
    expect(screen.getByRole('combobox', { name: 'vJoy device' })).toBeDisabled();
    await act(async () => { rejectInstall(new Error('ACC controls were not found.')); });
    expect(screen.getByRole('alert')).toHaveTextContent('ACC controls were not found');
    expect(screen.queryByText(/Installed “/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Install ACC preset' })).toBeEnabled();
});

it('keeps training disabled until the controller and all live inputs are ready', async () => {
    const openLive = jest.fn();
    await renderTraining(openLive);
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByLabelText('AI start status')).toHaveTextContent('Connect the virtual wheel.');
    fireEvent.click(screen.getByRole('button', { name: 'Open Live Session' }));
    expect(openLive).toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByLabelText('AI start status')).toHaveTextContent('Waiting for live telemetry.');
    mockInput = { observation: trainingFixture(Date.now()).observation };
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
    expect(screen.getByLabelText('AI start status')).toHaveTextContent('Ready to train. Run trained model becomes available after the model has learned at least one step.');
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    expect(screen.getByRole('button', { name: 'Install ACC preset' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Hold to test shift up' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Test shift down in 5 seconds' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    expect(JSON.parse(files.get(MODEL_PATH)!)).toMatchObject({ algorithm: 'soft-actor-critic', format: 'kestrel-self-driving-model' });
});

it.each([true, false])('keeps the session active through invalid inputs and resumes automatically (training=%s)', async (train) => {
    // Keep shift pulses out of this input-recovery test's short observation intervals.
    jest.spyOn(Math, 'random').mockReturnValue(0.1);
    const observation = trainingFixture(Date.now()).observation;
    // UI recovery uses the GPU stub; keep repeated checkpoint serialization small.
    const config = cloneConfig();
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    config.network.cameraProjectionUnits = 8;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    fireEvent.click(screen.getByRole('button', { name: train ? 'Start training' : 'Run trained model' })); await flush();
    for (const error of ['Waiting for fresh, running live telemetry.', 'Invalid camera pixels.']) {
        mockInput = { error };
        act(() => { jest.advanceTimersByTime(250); }); await flush();
        expect(screen.getByText(`AI control • waiting for valid inputs; retrying: ${error}`)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Run trained model' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'New model' })).toBeDisabled();
        expect(screen.getByLabelText('Current control owner')).toHaveTextContent('AI');
        expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    }
    const recover = async () => {
        act(() => {
            jest.advanceTimersByTime(100);
            mockInput = { observation: { ...observation, capturedAt: Date.now(), measuredAt: Date.now() } };
        });
        await flush();
    };
    await recover();
    expect(screen.getByText(train ? 'AI control • soft actor–critic' : 'AI control • fixed policy')).toBeInTheDocument();
    expect(within(screen.getByRole('region', { name: 'AI driver' })).getByText('10', { selector: 'strong' })).toBeInTheDocument();
    expect(window.virtualController!.apply).not.toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    await recover();
    expect(within(screen.getByRole('region', { name: 'AI driver' })).getByText('10', { selector: 'strong' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
    const saved = JSON.parse(files.get(MODEL_PATH)!);
    expect(saved.model.steps).toBe(10); // Recovery has not collected a complete training sequence yet.
    expect(saved).toEqual(checkpoint);
});

it('removes retired saves and starts a new JSON model', async () => {
    const { observation } = trainingFixture(Date.now());
    const keys = ['acla.self-driving.policy.v1', 'acla.self-driving.policy.v13', 'acla.self-driving.policy.v987'];
    keys.forEach((key) => localStorage.setItem(key, '{"version":13}'));
    localStorage.setItem('unrelated', 'keep');
    mockInput = { observation };
    await renderTraining();
    keys.forEach((key) => expect(localStorage.getItem(key)).toBeNull());
    expect(screen.queryByRole('button', { name: 'Export previous model' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    const document = JSON.parse(files.get(MODEL_PATH)!);
    expect(document.format).toBe('kestrel-self-driving-model');
    expect(document).not.toHaveProperty('version');
    expect(document.model.steps).toBe(0);
    expect(localStorage.getItem('unrelated')).toBe('keep');
});

it.each([1, 2])('shows and releases a saved model’s learned shift choice %s', async (shift) => {
    const { observation } = trainingFixture(Date.now());
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 10;
    checkpoint.model.outputWeights.slice(2, 5).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === shift ? 4 : 0; });
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.getByText('3 / 6000')).toBeInTheDocument();
    expect(screen.getByText(/Disable automatic gears/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Run trained model' })); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: shift === 1, shiftDown: shift === 2 }));
    const label = shift === 1 ? 'Shift up output' : 'Shift down output';
    expect(screen.getByLabelText(label)).toHaveTextContent('Pressed');
    // Let each 20 ms control tick settle throughout the 100 ms paddle pulse.
    for (let elapsed = 0; elapsed < 100; elapsed += 20) {
        act(() => { jest.advanceTimersByTime(20); }); await flush();
    }
    expect(screen.getByLabelText(label)).toHaveTextContent('Released');
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: false, shiftDown: false }));
    fireEvent.keyDown(window, { key: 'Escape' }); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
});

it('maps physical controls and switches vJoy ownership with one UI button or one controller button', async () => {
    const device = { id: 'wheel', name: 'Test wheel', axes: { axis0: 0.75, axis1: 0.8, axis2: 0.1 }, buttons: [false, false] };
    window.virtualController!.readPhysicalControls = jest.fn().mockResolvedValue([device]);
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toBeDisabled();
    for (const [label, axis] of [['Steering', 'axis0'], ['Throttle', 'axis1'], ['Brake', 'axis2']]) {
        fireEvent.change(screen.getByRole('combobox', { name: label }), { target: { value: JSON.stringify(['wheel', axis]) } });
    }
    expect(screen.getByLabelText('Manual throttle preview')).toHaveTextContent('0.800');
    fireEvent.click(screen.getByRole('checkbox', { name: 'Invert throttle' }));
    expect(screen.getByLabelText('Manual throttle preview')).toHaveTextContent('0.200');
    expect(JSON.parse(localStorage.getItem('acla.self-driving.manual-mapping.v1')!).throttle.inverted).toBe(true);
    fireEvent.change(screen.getByRole('combobox', { name: 'Take over / Return to AI button' }), { target: { value: JSON.stringify(['wheel', 0]) } });
    fireEvent.click(screen.getByRole('button', { name: 'Enable controller takeover' }));
    const handoverButton = screen.getByRole('button', { name: 'Start human driving' });
    expect(screen.queryByRole('button', { name: 'Return to AI' })).not.toBeInTheDocument();
    mockInput = { observation: trainingFixture(Date.now()).observation };
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    expect(screen.getByRole('combobox', { name: 'Steering' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Take over' })).toBe(handoverButton);
    fireEvent.click(handoverButton);
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    expect(screen.getByText('You are driving')).toBeInTheDocument();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({
        throttle: expect.closeTo(0.2), brake: 0.1, steering: expect.closeTo(0.5),
    }));
    expect(screen.getByRole('button', { name: 'Return to AI' })).toBe(handoverButton);
    expect(screen.queryByRole('button', { name: 'Take over' })).not.toBeInTheDocument();
    fireEvent.click(handoverButton);
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    expect(screen.getByText('AI is driving')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Take over' })).toBe(handoverButton);
    device.buttons[0] = true;
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    expect(screen.getByRole('button', { name: 'Return to AI' })).toBe(handoverButton);
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    expect(screen.getByText('You are driving')).toBeInTheDocument();
    device.buttons[0] = false;
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    device.buttons[0] = true;
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    expect(screen.getByRole('button', { name: 'Take over' })).toBe(handoverButton);
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    expect(screen.getByText('AI is driving')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(screen.getByRole('button', { name: 'Start human driving' })).toBe(handoverButton);
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
});

it('starts human driving without AI inputs, arms training without stealing control, and explicitly hands off', async () => {
    const device = { id: 'wheel', name: 'Test wheel', axes: { axis0: 0.5, axis1: 0.6, axis2: 0.1 }, buttons: [] };
    window.virtualController!.readPhysicalControls = jest.fn().mockResolvedValue([device]);
    localStorage.setItem('acla.self-driving.manual-mapping.v1', JSON.stringify({
        steering: { deviceId: 'wheel', axis: 'axis0', inverted: false },
        throttle: { deviceId: 'wheel', axis: 'axis1', inverted: false },
        brake: { deviceId: 'wheel', axis: 'axis2', inverted: false },
    }));
    await renderTraining();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
    expect(screen.getByRole('heading', { name: 'Human takeover' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Enable controller takeover' }));
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start human driving' })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('Human');
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({ throttle: 0.6, brake: 0.1, steering: 0 }));
    expect(screen.getByRole('button', { name: 'Return to AI' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Hold to test throttle' })).toBeDisabled();
    expect(localStorage.getItem('acla.self-driving.model')).toBeNull();
    const observation = trainingFixture(Date.now()).observation;
    mockInput = { observation };
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('Human');
    expect(screen.getByRole('button', { name: 'Return to AI' })).toBeEnabled();
    mockInput = { error: 'Capture disconnected.' };
    act(() => { jest.advanceTimersByTime(20); }); await flush();
    act(() => { jest.advanceTimersByTime(230); }); await flush();
    expect(screen.getByRole('button', { name: 'Return to AI' })).toBeDisabled();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('Human');
    expect(screen.getByText(/Human control • learning paused/)).toBeInTheDocument();
    mockInput = { observation: { ...observation, capturedAt: Date.now(), measuredAt: Date.now() } };
    act(() => { jest.advanceTimersByTime(20); }); await flush();
    act(() => { jest.advanceTimersByTime(230); }); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Return to AI' })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('AI');
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
});

it('locks human driving during a calibration test and returns to no driver on cancellation', async () => {
    const device = { id: 'wheel', name: 'Test wheel', axes: { axis0: 0.5, axis1: 0, axis2: 0 }, buttons: [] };
    window.virtualController!.readPhysicalControls = jest.fn().mockResolvedValue([device]);
    localStorage.setItem('acla.self-driving.manual-mapping.v1', JSON.stringify({
        steering: { deviceId: 'wheel', axis: 'axis0', inverted: false },
        throttle: { deviceId: 'wheel', axis: 'axis1', inverted: false },
        brake: { deviceId: 'wheel', axis: 'axis2', inverted: false },
    }));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Enable controller takeover' }));
    expect(screen.getByRole('button', { name: 'Start human driving' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Test throttle in 5 seconds' })); await flush();
    expect(screen.getByRole('button', { name: 'Start human driving' })).toBeDisabled();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('Calibration test');
    fireEvent.keyDown(window, { key: 'Escape' }); await flush();
    expect(screen.getByRole('button', { name: 'Start human driving' })).toBeEnabled();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
});

it('shows driver setup errors without claiming the controller is connected', async () => {
    (window.virtualController!.connect as jest.Mock).mockRejectedValue(new Error('Install vJoy and enable X, Y and Z axes.'));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Install vJoy');
    expect(screen.getByRole('button', { name: 'Disconnect' })).toBeDisabled();
});

it('offers only wheel output and reconnects to the selected vJoy device', async () => {
    await renderTraining();
    expect(screen.getByRole('heading', { name: 'Steering wheel (vJoy)' })).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Controller output' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Xbox|XInput|gamepad/i })).not.toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'vJoy device' })).toHaveValue('1');
    fireEvent.change(screen.getByRole('combobox', { name: 'vJoy device' }), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(window.virtualController!.connect).toHaveBeenLastCalledWith({ deviceId: 3 });
    expect(screen.getByRole('combobox', { name: 'vJoy device' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Disconnect' })); await flush();
    expect(screen.getByRole('combobox', { name: 'vJoy device' })).toBeEnabled();
    fireEvent.change(screen.getByRole('combobox', { name: 'vJoy device' }), { target: { value: '4' } });
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(window.virtualController!.connect).toHaveBeenLastCalledWith({ deviceId: 4 });
});

it('keeps a timed binding test alive in the simulator and releases automatically', async () => {
    const api = window.virtualController!;
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Test throttle in 5 seconds' })); await flush();
    expect(screen.getByRole('button', { name: 'Install ACC preset' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Hold to test steering' })).toBeDisabled();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    fireEvent.blur(window); await flush();
    act(() => { jest.advanceTimersByTime(4900); }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 1, brake: 0, steering: 0 });
    act(() => { jest.advanceTimersByTime(10000); }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();
    const count = (api.apply as jest.Mock).mock.calls.length;
    act(() => { jest.advanceTimersByTime(1000); }); await flush();
    expect(api.apply).toHaveBeenCalledTimes(count);
});

it.each(['Stop', 'Escape', 'Disconnect'])('cancels a pending timed test with %s', async (stop) => {
    const api = window.virtualController!;
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Test brake in 5 seconds' })); await flush();
    if (stop === 'Escape') fireEvent.keyDown(window, { key: 'Escape' });
    else fireEvent.click(screen.getByRole('button', { name: stop }));
    await flush();
    act(() => { jest.advanceTimersByTime(16000); }); await flush();
    expect((api.apply as jest.Mock).mock.calls.every(([action]) => action.brake === 0)).toBe(true);
});

it.each([-1, -0.2, 0, 0.2, 1])('supports steering calibration position %s', async (position) => {
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.change(screen.getByRole('combobox', { name: 'Timed steering position' }), { target: { value: String(position) } });
    fireEvent.click(screen.getByRole('button', { name: 'Test steering in 5 seconds' })); await flush();
    act(() => { jest.advanceTimersByTime(4900); }); await flush();
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: position });
});

it('cancels a timed test on unmount before any non-neutral input is sent', async () => {
    const api = window.virtualController!;
    const { unmount } = await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Test throttle in 5 seconds' })); await flush();
    unmount();
    act(() => { jest.advanceTimersByTime(16000); }); await flush();
    expect((api.apply as jest.Mock).mock.calls.every(([action]) => action.throttle === 0)).toBe(true);
    expect(api.disconnect).toHaveBeenCalled();
});

it('stops timed tests and reports a rejected device write', async () => {
    const api = window.virtualController!;
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    (api.apply as jest.Mock).mockRejectedValueOnce(new Error('vJoy device disconnected.'));
    fireEvent.click(screen.getByRole('button', { name: 'Test brake in 5 seconds' })); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('vJoy device disconnected');
    expect(screen.getByRole('button', { name: 'Test brake in 5 seconds' })).toBeDisabled();
    const count = (api.apply as jest.Mock).mock.calls.length;
    act(() => { jest.advanceTimersByTime(16000); }); await flush();
    expect(api.apply).toHaveBeenCalledTimes(count);
});

it('releases a held test axis on Escape and disconnects on unmount', async () => {
    const api = window.virtualController!;
    const { unmount } = await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.keyDown(screen.getByRole('button', { name: 'Hold to test throttle' }), { key: ' ' }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0.25, brake: 0, steering: 0 });
    fireEvent.keyDown(window, { key: 'Escape' }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    const count = (api.apply as jest.Mock).mock.calls.length;
    act(() => { jest.advanceTimersByTime(500); }); await flush();
    expect(api.apply).toHaveBeenCalledTimes(count);
    fireEvent.keyDown(screen.getByRole('button', { name: 'Hold to test steering' }), { key: ' ' }); await flush();
    fireEvent.blur(window); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    unmount();
    expect(api.disconnect).toHaveBeenCalled();
});

it.each([
    ['shiftUp', 'shift up', 'Shift up output'], ['shiftDown', 'shift down', 'Shift down output'],
])('presses and releases %s from pointer and keyboard controls', async (key, label, outputLabel) => {
    const api = window.virtualController!;
    await renderTraining();
    const button = screen.getByRole('button', { name: `Hold to test ${label}` });
    expect(button).toBeDisabled();
    expect(screen.getByRole('button', { name: `Test ${label} in 5 seconds` })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    button.setPointerCapture = jest.fn();
    const neutral = { throttle: 0, brake: 0, steering: 0 };
    fireEvent.pointerDown(button, { pointerId: 1 }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ ...neutral, [key]: true });
    expect(button).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByLabelText(outputLabel)).toHaveTextContent('Pressed');
    fireEvent.pointerUp(button); await flush();
    expect(api.apply).toHaveBeenLastCalledWith(neutral);
    expect(button).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByLabelText(outputLabel)).toHaveTextContent('Released');
    fireEvent.keyDown(button, { key: 'Enter' }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ ...neutral, [key]: true });
    fireEvent.keyUp(button, { key: 'Enter' }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith(neutral);
});

it.each([['shiftUp', 'shift up'], ['shiftDown', 'shift down']])('pulses %s once after the countdown while the simulator has focus', async (key, label) => {
    const api = window.virtualController!;
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: `Test ${label} in 5 seconds` })); await flush();
    const neutral = { throttle: 0, brake: 0, steering: 0 };
    fireEvent.blur(window); await flush();
    expect(screen.getByRole('button', { name: 'Hold to test shift up' })).toBeDisabled();
    act(() => { jest.advanceTimersByTime(4900); }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith(neutral);
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ ...neutral, [key]: true });
    act(() => { jest.advanceTimersByTime(300); }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith(neutral);
    expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();
    const count = (api.apply as jest.Mock).mock.calls.length;
    act(() => { jest.advanceTimersByTime(10000); }); await flush();
    expect(api.apply).toHaveBeenCalledTimes(count);
});

it.each(['Stop', 'Escape', 'Disconnect', 'blur', 'pointerCancel', 'lostPointerCapture', 'unmount'])('releases a held shift button on %s', async (stop) => {
    const api = window.virtualController!;
    const { unmount } = await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    const button = screen.getByRole('button', { name: 'Hold to test shift down' });
    fireEvent.keyDown(button, { key: ' ' }); await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0, shiftDown: true });
    if (stop === 'unmount') unmount();
    else if (stop === 'Escape') fireEvent.keyDown(window, { key: 'Escape' });
    else if (stop === 'blur') fireEvent.blur(window);
    else if (stop === 'pointerCancel') fireEvent.pointerCancel(button);
    else if (stop === 'lostPointerCapture') fireEvent.lostPointerCapture(button);
    else fireEvent.click(screen.getByRole('button', { name: stop }));
    await flush();
    expect(api.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    const count = (api.apply as jest.Mock).mock.calls.length;
    act(() => { jest.advanceTimersByTime(1000); }); await flush();
    expect(api.apply).toHaveBeenCalledTimes(count);
});

it('saves exploration with an untrained model, restores it on remount and resets its defaults', async () => {
    const view = await renderTraining();
    const labels = ['Throttle/brake exploration', 'Steering exploration'];
    const sliders = labels.map((name) => screen.getByRole('slider', { name }));
    const gears = screen.getByRole('slider', { name: 'Gear exploration' });
    const speedLimit = () => screen.getByRole('slider', { name: 'Throttle exploration speed limit' });
    const maxGear = () => screen.getByRole('combobox', { name: 'Maximum gear to explore' });
    const frequencyLabels = ['Throttle/brake', 'Steering', 'Gear'].map((control) => `${control} exploration update frequency`);
    const frequencies = frequencyLabels.map((name) => screen.getByRole('slider', { name }));
    const defaultFrequencies = ['2', '2', '10'];
    frequencies.forEach((slider, i) => expect(slider).toHaveValue(defaultFrequencies[i]));
    sliders.forEach((slider) => expect(slider).toHaveValue('0.05'));
    expect(gears).toHaveValue('0.25');
    expect(speedLimit()).toHaveValue('400');
    expect(maxGear()).toHaveDisplayValue('Unlimited');
    fireEvent.change(maxGear(), { target: { value: '4' } });
    fireEvent.change(speedLimit(), { target: { value: '120' } });
    expect(speedLimit()).toHaveAttribute('aria-valuetext', '120 km/h');
    fireEvent.change(sliders[0], { target: { value: '0.15' } });
    expect(sliders[1]).toHaveValue('0.05');
    fireEvent.change(sliders[1], { target: { value: '0.35' } });
    expect(sliders[0]).toHaveValue('0.15');
    const frequencyValues = ['2.5', '0.5', '5'];
    frequencies.forEach((slider, i) => {
        fireEvent.change(slider, { target: { value: frequencyValues[i] } });
        frequencies.forEach((other, j) => expect(other).toHaveValue(j <= i ? frequencyValues[j] : defaultFrequencies[j]));
    });
    expect(screen.getByText('2.5 Hz · every 0.4 s')).toBeInTheDocument();
    fireEvent.change(gears, { target: { value: '0.6' } }); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toMatchObject({ model: null,
        configuration: { learning: { pedalExplorationSigma: 0.15, throttleExplorationSpeedLimitKmh: 120,
            steeringExplorationSigma: 0.35, shiftTemperature: 0.6, maxExplorationGear: 4, pedalExplorationUpdateHz: 2.5,
            steeringExplorationUpdateHz: 0.5, gearExplorationUpdateHz: 5 } } });
    view.unmount();
    await renderTraining();
    labels.forEach((name, i) => expect(screen.getByRole('slider', { name })).toHaveValue(['0.15', '0.35'][i]));
    expect(screen.getByRole('slider', { name: 'Gear exploration' })).toHaveValue('0.6');
    expect(speedLimit()).toHaveValue('120');
    expect(maxGear()).toHaveValue('4');
    frequencyLabels.forEach((name, i) => expect(screen.getByRole('slider', { name })).toHaveValue(frequencyValues[i]));
    fireEvent.click(screen.getByRole('button', { name: 'Reset exploration' })); await flush();
    labels.forEach((name) => expect(screen.getByRole('slider', { name })).toHaveValue('0.05'));
    expect(screen.getByRole('slider', { name: 'Gear exploration' })).toHaveValue('0.25');
    expect(speedLimit()).toHaveValue('400');
    expect(maxGear()).toHaveDisplayValue('Unlimited');
    frequencyLabels.forEach((name, i) => expect(screen.getByRole('slider', { name })).toHaveValue(defaultFrequencies[i]));
    expect(JSON.parse(files.get(MODEL_PATH)!).configuration.learning).toMatchObject({ pedalExplorationSigma: 0.05, throttleExplorationSpeedLimitKmh: 400,
        steeringExplorationSigma: 0.05, shiftTemperature: 0.25, maxExplorationGear: 0,
        pedalExplorationUpdateHz: 2, steeringExplorationUpdateHz: 2, gearExplorationUpdateHz: 10 });
});

it.each([true, false])('loads exploration into the GPU and locks it for the run without changing weights (training=%s)', async (train) => {
    const { observation } = trainingFixture(Date.now()), config = cloneConfig();
    config.network.hiddenUnits = 2;
    config.network.cameraProjectionUnits = 2;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 1; });
    config.learning.pedalExplorationSigma = 0.12;
    config.learning.steeringExplorationSigma = 0.14;
    config.learning.shiftTemperature = 0.4;
    config.learning.pedalExplorationUpdateHz = 1;
    config.learning.steeringExplorationUpdateHz = 2;
    config.learning.gearExplorationUpdateHz = 3;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 7;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    const decide = jest.spyOn(DrivingPolicy.prototype, 'decideAsync');
    await renderTraining();
    const sliders = ['Throttle/brake exploration', 'Steering exploration']
        .map((name) => screen.getByRole('slider', { name }));
    const gears = screen.getByRole('slider', { name: 'Gear exploration' });
    const maxGear = screen.getByRole('combobox', { name: 'Maximum gear to explore' });
    const frequencies = ['Throttle/brake', 'Steering', 'Gear']
        .map((control) => screen.getByRole('slider', { name: `${control} exploration update frequency` }));
    frequencies.forEach((slider, i) => expect(slider).toHaveValue(String(i + 1)));
    sliders.forEach((slider, i) => expect(slider).toHaveValue(['0.12', '0.14'][i]));
    expect(gears).toHaveValue('0.4');
    fireEvent.change(maxGear, { target: { value: String(observation.gear) } });
    sliders.forEach((slider, i) => fireEvent.change(slider, { target: { value: ['0.2', '0.4'][i] } }));
    frequencies.forEach((slider, i) => fireEvent.change(slider, { target: { value: String(4 - i) } }));
    fireEvent.change(gears, { target: { value: '0.7' } }); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!).model).toEqual(checkpoint.model);
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    const startLabel = train ? 'Start training' : 'Run trained model';
    fireEvent.click(screen.getByRole('button', { name: startLabel }));
    sliders.forEach((slider) => expect(slider).toBeDisabled());
    expect(gears).toBeDisabled();
    expect(maxGear).toBeDisabled();
    frequencies.forEach((slider) => expect(slider).toBeDisabled());
    await flush();
    expect(DrivingGpuModel.create).toHaveBeenLastCalledWith(expect.objectContaining({
        learning: expect.objectContaining({ pedalExplorationSigma: 0.2,
            steeringExplorationSigma: 0.4, shiftTemperature: 0.7, pedalExplorationUpdateHz: 4,
            steeringExplorationUpdateHz: 3, gearExplorationUpdateHz: 2 }),
    }), checkpoint.model);
    expect(decide).toHaveBeenCalledWith(expect.anything(), train, { up: !train, down: true });
    sliders.forEach((slider) => expect(slider).toBeDisabled());
    expect(gears).toBeDisabled();
    frequencies.forEach((slider) => expect(slider).toBeDisabled());
    expect(screen.getByRole('button', { name: 'Reset exploration' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    sliders.forEach((slider) => expect(slider).toBeEnabled());
    expect(gears).toBeEnabled();
    expect(maxGear).toBeEnabled();
    frequencies.forEach((slider) => expect(slider).toBeEnabled());
    fireEvent.change(frequencies[2], { target: { value: '0.5' } });
    fireEvent.change(sliders[0], { target: { value: '0.08' } }); await flush();
    fireEvent.click(screen.getByRole('button', { name: startLabel })); await flush();
    expect(DrivingGpuModel.create).toHaveBeenCalledTimes(2);
    expect(DrivingGpuModel.create).toHaveBeenLastCalledWith(expect.objectContaining({
        learning: expect.objectContaining({ pedalExplorationSigma: 0.08,
            steeringExplorationSigma: 0.4, shiftTemperature: 0.7, pedalExplorationUpdateHz: 4,
            steeringExplorationUpdateHz: 3, gearExplorationUpdateHz: 0.5 }),
    }), checkpoint.model);
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
});

it('keeps exploration disabled until a model file is selected', async () => {
    selectedFile = null;
    await renderTraining();
    ['Throttle/brake exploration', 'Steering exploration'].forEach((name) =>
        expect(screen.getByRole('slider', { name })).toBeDisabled());
    expect(screen.getByRole('slider', { name: 'Gear exploration' })).toBeDisabled();
    ['Throttle/brake', 'Steering', 'Gear'].forEach((control) =>
        expect(screen.getByRole('slider', { name: `${control} exploration update frequency` })).toBeDisabled());
    expect(screen.getByRole('button', { name: 'Reset exploration' })).toBeDisabled();
    expect(window.drivingModelFiles!.write).not.toHaveBeenCalled();
});

it('persists the aggression setting, restores it on remount and resets to reference', async () => {
    const view = await renderTraining();
    const slider = screen.getByRole('slider', { name: 'Driving aggression' });
    expect(slider).toHaveValue('50');
    expect(slider).toHaveAttribute('min', '0');
    expect(slider).toHaveAttribute('max', '100');
    fireEvent.change(slider, { target: { value: '0' } }); await flush();
    expect(screen.getByText(/Uses the same speed targets as Live Phrases/)).toHaveTextContent('100% braking adjustment');
    expect(JSON.parse(files.get(MODEL_PATH)!).configuration.aggression).toBe(0);
    view.unmount();
    await renderTraining();
    expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue('0');
    fireEvent.click(screen.getByRole('button', { name: 'Reset to reference' })); await flush();
    expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue('50');
    expect(JSON.parse(files.get(MODEL_PATH)!).configuration.aggression).toBe(50);
});

it.each(['null', '"100"', '-1', '101', 'invalid'])('uses reference aggression when the saved setting is invalid: %s', async (value) => {
    localStorage.setItem('acla.self-driving.aggression', value);
    await renderTraining();
    expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue('50');
});

it.each([true, false])('uses Live Map’s speed targets and applies slider changes during training=%s', async (train) => {
    const referenceLap = referenceLapFixture();
    let fixture = trainingFixture(Date.now(), 50, referenceLap);
    const map = { getReferenceLap: () => referenceLap };
    jest.spyOn(SessionScreenFrameReader.prototype, 'read').mockImplementation(() => screenFrame(fixture));
    mockFindComponentRef.mockImplementation((name) => name === 'visualization:live-trajectory-map' ? { current: map } : null);
    mockRead.mockImplementation((_live, detected, lap, aggression) =>
        jest.requireActual('./driving-observation').readDrivingObservation(fixture.live, detected, lap, aggression, Date.now(), undefined, fixture.circuitMap));
    const config = cloneConfig();
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    config.network.cameraProjectionUnits = 8;
    const checkpoint = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    const slider = screen.getByRole('slider', { name: 'Driving aggression' });
    fireEvent.change(slider, { target: { value: '0' } });
    fireEvent.click(screen.getByRole('button', { name: train ? 'Start training' : 'Run trained model' })); await flush();
    expect(slider).toBeEnabled();
    const sampledAtStart = mockRead.mock.calls.length;
    fireEvent.change(slider, { target: { value: '100' } });
    expect(mockRead).toHaveBeenCalledTimes(sampledAtStart);
    // The loop samples the changed setting; the panel then displays that sample.
    act(() => { jest.advanceTimersByTime(20); }); await flush();
    act(() => { jest.advanceTimersByTime(230); }); await flush();
    expect(mockRead).toHaveBeenLastCalledWith(expect.anything(), screenFrame(fixture), referenceLap, 100, expect.any(Number), expect.objectContaining({ aggression: 100 }), null);
    expect(screen.queryByRole('table', { name: 'Live pedals · current aggression' })).not.toBeInTheDocument();
    expect(getDefinition('Reference speed')).toHaveTextContent('100.0 km/h');
    expect(getDefinition('Aggression target speed')).toHaveTextContent('150.0 km/h');
    expect(screen.getByText(/Raises reference speed by 50%/)).toBeInTheDocument();
    expect(screen.queryByText('Reference error')).not.toBeInTheDocument();
    expect(screen.queryByText('Reference pedal reward')).not.toBeInTheDocument();
    expect(screen.queryByText('Drive fast')).not.toBeInTheDocument();
    expect(screen.queryByText('Penalize instability')).not.toBeInTheDocument();
    expect(screen.getByText('Acceleration / deceleration')).toBeInTheDocument();
    expect(getDefinition('Normalized position progress').textContent).toBe('0');
    expect(screen.queryByText('Drive toward the reference line')).not.toBeInTheDocument();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    fixture.live.currentTelemetry.Graphics_normalized_car_position = 0.252;
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(screen.getByText(/Last scored at aggression 0/)).toBeInTheDocument();
    expect(Number(getDefinition('Acceleration / deceleration').textContent)).toBeLessThan(0);
    const progress = Number(getDefinition('Normalized position progress').textContent);
    expect(progress).toBeGreaterThan(0);
    expect(screen.queryByText('Follow the reference line')).not.toBeInTheDocument();
    expect(screen.queryByText('Drive toward the reference line')).not.toBeInTheDocument();
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    fixture.live.currentTelemetry.Graphics_normalized_car_position = 0.252;
    fixture.live.currentTelemetry.Physics_speed_kmh = 0.5;
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(Number(getDefinition('Normalized position progress').textContent)).toBeLessThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Reset to reference' }));
    expect(slider).toHaveValue('50');
    expect(screen.getByText(/Reference uses the recorded speed targets/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    const saved = JSON.parse(files.get(MODEL_PATH)!);
    // Rewards update immediately; this short session has not completed a sequence.
    expect(saved.model.steps).toBe(10);
    expect(saved.model).toEqual(checkpoint.model);
});

it('restores a saved model after remount', async () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 321;
    const oldKeys = [1, 2, 3, 4, 5, 6, 7].map((version) => `acla.self-driving.policy.v${version}`);
    oldKeys.forEach((key) => localStorage.setItem(key, 'old-model'));
    localStorage.setItem('acla.self-driving.aggression', '75');
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    const view = await renderTraining();
    expect(screen.getByText('321')).toBeInTheDocument();
    expect(oldKeys.map((key) => localStorage.getItem(key))).toEqual(oldKeys.map(() => null));
    expect(localStorage.getItem('acla.self-driving.aggression')).toBeNull();
    view.unmount();
    await renderTraining();
    expect(screen.getByText('321')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save model' })).toBeEnabled();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
});

it('opens a full model file and saves subsequent changes back to that file', async () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 44;
    const otherPath = 'C:/models/other.json';
    files.set(otherPath, JSON.stringify(checkpoint));
    await renderTraining();
    (window.drivingModelFiles!.choose as jest.Mock).mockResolvedValue(otherPath);
    fireEvent.click(screen.getByRole('button', { name: 'Open model' })); await flush();
    expect(screen.getByText(otherPath)).toBeInTheDocument();
    expect(screen.getByText('44')).toBeInTheDocument();
    expect(selectedFile).toBe(otherPath);
    fireEvent.change(screen.getByRole('slider', { name: 'Driving aggression' }), { target: { value: '80' } }); await flush();
    expect(JSON.parse(files.get(otherPath)!).configuration.aggression).toBe(80);
    expect(JSON.parse(files.get(otherPath)!).model).toEqual(checkpoint.model);
    expect(JSON.parse(files.get(MODEL_PATH)!).model).toBeNull();
    expect(localStorage.getItem('acla.self-driving.model')).toBeNull();
});

it('rejects an unsupported model file without replacing the current model', async () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 321;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    await renderTraining();
    files.set('unsupported.json', JSON.stringify({ version: 13, algorithm: 'online-actor-critic', ...checkpoint.model }));
    (window.drivingModelFiles!.choose as jest.Mock).mockResolvedValue('unsupported.json');
    fireEvent.click(screen.getByRole('button', { name: 'Open model' })); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Invalid or incompatible self-driving model checkpoint.');
    expect(screen.getByText('321')).toBeInTheDocument();
    expect(selectedFile).toBe(MODEL_PATH);
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
});

it('removes the saved model on New model and stays empty after remount', async () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 321;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    const view = await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'New model' })); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!).model).toBeNull();
    view.unmount();
    await renderTraining();
    expect(screen.queryByText('321')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save model' })).toBeEnabled();
    expect(JSON.parse(files.get(MODEL_PATH)!).model).toBeNull();
});

it('reloads JSON weights, aggression and plan timing every time evaluation starts', async () => {
    const { observation } = trainingFixture(Date.now());
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    for (const pedalBias of [1, -1]) {
        // Simulate an updated JSON between runs without remounting the page.
        checkpoint.configuration.aggression = 50 + pedalBias * 30;
        checkpoint.configuration.actions.planTimesSeconds = [0, 0.4];
        checkpoint.model.outputWeights = checkpoint.model.outputWeights.slice(0,
            checkpoint.configuration.actions.planTimesSeconds.length * checkpoint.configuration.actions.outputs.length + 1);
        checkpoint.model.sac.logStds = checkpoint.model.sac.logStds.slice(0, checkpoint.configuration.actions.planTimesSeconds.length * 2);
        checkpoint.model.outputWeights.forEach((row) => row.fill(0));
        checkpoint.model.outputWeights[0][checkpoint.configuration.network.hiddenUnits] = pedalBias;
        files.set(MODEL_PATH, JSON.stringify(checkpoint));
        fireEvent.click(screen.getByRole('button', { name: 'Run trained model' })); await flush();
        expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({
            throttle: Math.max(0, Math.tanh(pedalBias)), brake: Math.max(0, -Math.tanh(pedalBias)),
        }));
        expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue(String(50 + pedalBias * 30));
        const plan = within(screen.getByRole('table', { name: 'Planned driving actions' }));
        expect(plan.getAllByRole('row')).toHaveLength(3);
        expect(plan.getByRole('columnheader', { name: 'Throttle/brake' })).toBeVisible();
        expect(within(plan.getAllByRole('row')[1]).getAllByRole('cell').map((cell) => cell.textContent))
            .toEqual([Math.tanh(pedalBias).toFixed(3), '0.000', 'Hold']);
        fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
        expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
    }
}, 15000);

it('opens configuration before training and saves the same complete JSON', async () => {
    const document = emptyTrainingDocument();
    document.configuration.aggression = 80;
    document.configuration.network.hiddenUnits = 12;
    selectedFile = null;
    files.set(MODEL_PATH, JSON.stringify(document));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Open model' })); await flush();
    expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue('80');
    expect(screen.getByText(/12 recurrent units/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Save model' })); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(document);
    expect(window.drivingModelFiles!.write).toHaveBeenCalledWith(MODEL_PATH, JSON.stringify(document));
});

it.each(['mount', 'start'])('does not overwrite corrupt saved JSON discovered at %s', async (when) => {
    const { observation } = trainingFixture(Date.now());
    mockInput = { observation };
    if (when === 'mount') files.set(MODEL_PATH, '{broken');
    const view = await renderTraining();
    if (when === 'start') {
        fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
        files.set(MODEL_PATH, '{broken');
        fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    }
    expect(screen.getByRole('alert')).toBeInTheDocument();
    view.unmount();
    expect(files.get(MODEL_PATH)).toBe('{broken');
});

it('discards browser models and requires a chosen file before training', async () => {
    selectedFile = null;
    files.clear();
    const { observation } = trainingFixture(Date.now());
    const legacyConfig = cloneConfig();
    legacyConfig.network.hiddenUnits = 32;
    legacyConfig.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    legacyConfig.network.cameraProjectionUnits = 8;
    const oldModel = new DrivingPolicy(observation.context, observation.features.length, Math.random, legacyConfig).checkpoint();
    oldModel.model.steps = 321;
    localStorage.setItem('acla.self-driving.model', JSON.stringify(oldModel));
    localStorage.setItem('acla.self-driving.policy.v13', 'old');
    localStorage.setItem('acla.self-driving.aggression', '80');
    localStorage.setItem('unrelated', 'keep');
    mockInput = { observation };
    await renderTraining();
    expect(localStorage.getItem('acla.self-driving.model')).toBeNull();
    expect(localStorage.getItem('acla.self-driving.policy.v13')).toBeNull();
    expect(localStorage.getItem('acla.self-driving.aggression')).toBeNull();
    expect(localStorage.getItem('unrelated')).toBe('keep');
    expect(screen.queryByText('321')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(window.drivingModelFiles!.write).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'New model' })); await flush();
    expect(window.drivingModelFiles!.choose).toHaveBeenCalledWith('new');
    expect(screen.getByText(MODEL_PATH)).toBeInTheDocument();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(emptyTrainingDocument());
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!).model.steps).toBe(0);
    expect(localStorage.getItem('acla.self-driving.model')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Export model' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Import model checkpoint')).not.toBeInTheDocument();
});

it.each(['New model', 'Open model'])('cancels %s without changing the active file', async (button) => {
    const before = files.get(MODEL_PATH);
    await renderTraining();
    (window.drivingModelFiles!.choose as jest.Mock).mockResolvedValue(null);
    fireEvent.click(screen.getByRole('button', { name: button })); await flush();
    expect(files.get(MODEL_PATH)).toBe(before);
    expect(selectedFile).toBe(MODEL_PATH);
    expect(window.drivingModelFiles!.write).not.toHaveBeenCalled();
    expect(screen.getByText(MODEL_PATH)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: button })).toBeEnabled();
});

it('keeps the current file selected when creation fails', async () => {
    await renderTraining();
    (window.drivingModelFiles!.choose as jest.Mock).mockResolvedValue('C:/models/new.json');
    (window.drivingModelFiles!.write as jest.Mock).mockRejectedValueOnce(new Error('Disk full'));
    fireEvent.click(screen.getByRole('button', { name: 'New model' })); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Disk full');
    expect(selectedFile).toBe(MODEL_PATH);
    expect(screen.getByText(MODEL_PATH)).toBeInTheDocument();
    expect(files.has('C:/models/new.json')).toBe(false);
});

it('reports a failed save, blocks stale reloads and permits retrying the in-memory model', async () => {
    mockInput = { observation: trainingFixture(Date.now()).observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    (window.drivingModelFiles!.write as jest.Mock).mockRejectedValueOnce(new Error('Disk full'));
    fireEvent.change(screen.getByRole('slider', { name: 'Driving aggression' }), { target: { value: '80' } }); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent(`Could not save model to ${MODEL_PATH}. Disk full`);
    expect(screen.getByText(/Model not saved/)).toBeInTheDocument();
    expect(JSON.parse(files.get(MODEL_PATH)!).configuration.aggression).toBe(50);
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Save the model successfully before starting again.');
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Save model' })); await flush();
    expect(JSON.parse(files.get(MODEL_PATH)!).configuration.aggression).toBe(80);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByText(/Model saved to file/)).toBeInTheDocument();
});

it('honors Stop while reading the model before starting', async () => {
    mockInput = { observation: trainingFixture(Date.now()).observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    let finish!: (json: string) => void;
    (window.drivingModelFiles!.read as jest.Mock).mockImplementationOnce(() => new Promise<string>((resolve) => { finish = resolve; }));
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    finish(files.get(MODEL_PATH)!); await flush();
    expect(DrivingPolicy.prototype.initializeGpu).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
});

it.each([0, 49])('automatically saves when a completed consequence window reaches the save interval (initial steps=%s)', async (initialSteps) => {
    jest.spyOn(Math, 'random').mockReturnValue(0.1);
    const observation = trainingFixture(Date.now()).observation;
    const config = cloneConfig();
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    config.network.cameraProjectionUnits = 8;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = initialSteps;
    checkpoint.configuration.runtime.saveEverySteps = initialSteps ? 50 : 1;
    checkpoint.configuration.learning.sequenceSeconds = 0.3;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    (window.drivingModelFiles!.write as jest.Mock).mockClear();
    const advance = async () => {
        act(() => {
            jest.advanceTimersByTime(100);
            mockInput = { observation: { ...observation, capturedAt: Date.now(), measuredAt: Date.now() } };
        });
        await flush();
    };
    await advance(); await advance(); await advance(); await advance();
    const saved = JSON.parse(files.get(MODEL_PATH)!);
    expect(saved.model.steps).toBeGreaterThan(initialSteps);
    expect(saved.model.steps).toBeGreaterThanOrEqual(checkpoint.configuration.runtime.saveEverySteps);
    expect(saved.model.lstmWeights).not.toEqual(checkpoint.model.lstmWeights);
    expect(window.drivingModelFiles!.write).toHaveBeenCalledWith(MODEL_PATH, expect.any(String));
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    expect(localStorage.getItem('acla.self-driving.model')).toBeNull();
});

it('waits for a departing page’s GPU checkpoint before reopening the file', async () => {
    const { observation } = trainingFixture();
    const config = cloneConfig();
    config.network.hiddenUnits = 3;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 2; });
    config.network.cameraProjectionUnits = 8;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length, Math.random, config).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    const view = await renderTraining();
    const updated = { ...checkpoint, model: { ...checkpoint.model, steps: 25 } };
    let finish!: (document: typeof checkpoint) => void;
    jest.spyOn(DrivingPolicy.prototype, 'checkpointAsync').mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    view.unmount(); await flush();
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />); await flush();
    expect(within(screen.getByRole('region', { name: 'AI driver' })).queryByText('10', { selector: 'strong' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open model' })).toBeDisabled();
    finish(updated); await flush();
    expect(screen.getByText('25')).toBeInTheDocument();
    expect(JSON.parse(files.get(MODEL_PATH)!).model.steps).toBe(25);
});
