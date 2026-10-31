import { mockDrivingGpu } from './driving-gpu-test-model';
import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import AiDrivingTraining from './AiDrivingTraining';
import type { ObservationResult } from './driving-observation';
import { referenceLapFixture, trainingFixture } from './training-fixtures';
import { DrivingPolicy, emptyTrainingDocument } from './driving-policy';
import { cloneConfig } from './driving-model-config';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import type { VisionTimingSnapshot } from 'views/live-session/track-vision/vision-timing';

let mockInput: ObservationResult;
const mockRead = jest.fn();
const mockFindComponentRef = jest.fn();
jest.mock('contexts/OperationComponentRefContext', () => ({
    useOptionalOperationComponentRefDirectory: () => ({ findComponentRef: mockFindComponentRef }),
}));
jest.mock('./driving-observation', () => ({
    ...jest.requireActual('./driving-observation'),
    readDrivingObservation: (...args: unknown[]) => args.length === 5
        ? jest.requireActual('./driving-observation').readDrivingObservation(...args) : mockRead(...args),
}));
// Match definition-list values to their terms through semantic queries.
const getDefinition = (label: string) => screen.getAllByRole('definition')[
    screen.getAllByRole('term').findIndex((term) => term.textContent === label)];
const flush = async () => { await act(async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); }); };
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

it('shows saved and live mismatch values and recovers without replacing the saved model', async () => {
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
    expect(warning).toHaveTextContent('Game: saved “acc”; live “iracing”.');
    expect(warning).toHaveTextContent('Car: saved “GT3”; live “MX5”.');
    expect(warning).toHaveTextContent('Track: saved “monza”; live “spa”.');
    expect(warning).not.toHaveTextContent('Segmentation labels');
    expect(warning).toHaveTextContent(`Input count: saved ${observation.features.length}; live ${observation.features.length - 16}.`);
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Run trained model' })).toBeDisabled();
    mockInput = { observation };
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Run trained model' })).toBeEnabled();
    expect(JSON.parse(files.get(MODEL_PATH)!)).toEqual(checkpoint);
});

it('displays the complete action plan in one table and clears it on Stop', async () => {
    const { observation } = trainingFixture(Date.now());
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    mockInput = { observation };
    await renderTraining();
    const table = screen.getByRole('table', { name: 'Planned driving actions' });
    expect(within(table).getByText('Waiting for AI control and fresh inputs.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    fireEvent.click(screen.getByRole('button', { name: 'Run trained model' })); await flush();
    expect(within(table).getAllByRole('row')).toHaveLength(8);
    expect(within(table).getAllByRole('rowheader').map((cell) => cell.textContent))
        .toEqual(['0.0 s', '0.5 s', '1.0 s', '1.5 s', '2.0 s', '2.5 s', '3.0 s']);
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(within(table).getByText('Waiting for AI control and fresh inputs.')).toBeInTheDocument();
});

it('uses the independent camera/position output without reading segmentation or requesting depth readback', async () => {
    const fixture = trainingFixture(Date.now());
    const driving = { ...fixture.vision, detections: {}, reconstructedScene: null };
    const requestDepthValues = jest.fn().mockResolvedValue(undefined);
    const getLatestDetection = jest.fn(() => fixture.vision);
    mockFindComponentRef.mockImplementation((name) => name === 'visualization:track-vision' ? { current: {
        getLatestDetection, getLatestDrivingDetection: () => driving, requestDepthValues,
    } } : null);
    await renderTraining();
    act(() => { jest.advanceTimersByTime(250); });
    expect(mockRead.mock.calls[mockRead.mock.calls.length - 1][1]).toBe(driving);
    expect(getLatestDetection).not.toHaveBeenCalled();
    expect(requestDepthValues).not.toHaveBeenCalled();
});

it('shows direct observation inputs without summary metadata', async () => {
    mockInput = { observation: trainingFixture(Date.now()).observation };
    await renderTraining();
    act(() => { jest.advanceTimersByTime(250); });
    ['Current observation', 'Observation history', 'Total policy inputs'].forEach((label) => {
        expect(screen.queryByText(label)).not.toBeInTheDocument();
    });
    expect(getDefinition('Camera image')).toHaveTextContent('768 × 768 RGB');
    expect(getDefinition('Camera image')).not.toHaveTextContent(/history/);
    expect(screen.getByText(/5-second observation history covers all AI observation inputs/)).toHaveTextContent('sampled in 300 ms bins');
    for (const label of ['Reconstructed scene', 'Depth model', 'Segmentation labels']) {
        expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
    expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();
    mockInput = { error: 'Track Vision camera pixels must be 768 × 768 RGB.' };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Camera image')).toHaveTextContent('768 × 768 RGB · Waiting');
    expect(screen.getByText('Track Vision camera pixels must be 768 × 768 RGB.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
});

it('keeps all input delays visible across stale frames and recovery using the saved freshness limits', async () => {
    const document = emptyTrainingDocument();
    document.configuration.runtime.visionMaxAgeMs = 1000;
    document.configuration.runtime.telemetryMaxAgeMs = 300;
    files.set(MODEL_PATH, JSON.stringify(document));
    let fixture = trainingFixture(Date.now() - 900);
    jest.spyOn(liveTelemetryStore, 'getSnapshot').mockImplementation(() => ({ ...fixture.live, receivedAt: Date.now() - 40 }));
    const timing: VisionTimingSnapshot = { capturedAt: fixture.vision.capturedAt, stages: {
        capture: 2, camera: 3, references: 4, segment: 145, depth: 64, depthReadback: 30, segmentReadback: 13,
        position: 600, preview: 2, track: 10, scene: 12, birdsEye: 8,
    }, currentStage: null, totalMs: 893, finishedAt: Date.now() - 20, state: 'complete' };
    fixture.vision.timing = timing;
    let latestTiming = timing;
    let detection: typeof fixture.vision | null = fixture.vision;
    const vision = { getLatestDrivingDetection: () => detection, getTimingSnapshot: () => latestTiming };
    mockFindComponentRef.mockImplementation((name) => name.endsWith('track-vision') ? { current: vision } : null);
    mockInput = { observation: fixture.observation };
    await renderTraining();
    expect(getDefinition('Live telemetry age')).toHaveTextContent('40 ms / 300 ms limit · Fresh');
    expect(getDefinition('Image age for AI')).toHaveTextContent('900 ms / 1000 ms limit · Fresh');
    expect(getDefinition('Telemetry age at image capture')).toHaveTextContent('0 ms / 300 ms limit · Fresh');
    const table = screen.getByRole('table', { name: 'Latest capture processing' });
    expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();
    expect(within(table).getByRole('row', { name: 'Position model 600 ms' })).toBeInTheDocument();
    expect(getDefinition('Published frame processing')).toHaveTextContent('893 ms');
    expect(getDefinition('Time since frame published')).toHaveTextContent('20 ms');
    expect(screen.queryByRole('table', { name: 'Published frame timing breakdown' })).not.toBeInTheDocument();

    latestTiming = { ...timing, stages: { segment: 145, depth: 64, segmentReadback: 99 }, state: 'processing',
        currentStage: 'position', stageElapsedMs: 700, totalMs: 1008, finishedAt: undefined };
    mockInput = { error: 'Start Track Vision capture in Live Session. A fresh image is required.' };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Image age for AI')).toHaveTextContent('1150 ms / 1000 ms limit · Stale');
    expect(getDefinition('Image age for AI')).toHaveAttribute('data-stale', 'true');
    expect(getDefinition('Paired telemetry age now')).toHaveTextContent('1150 ms');
    expect(getDefinition('Time since frame published')).toHaveTextContent('270 ms');
    expect(within(table).getByRole('row', { name: 'Position model 700 ms · Running' })).toBeInTheDocument();
    expect(getDefinition('Observation validation / assembly')).toHaveTextContent('Inputs invalid');
    expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();

    for (const currentStage of ['segment', 'segmentReadback', 'depth', 'depthReadback'] as const) {
        latestTiming = { ...latestTiming, currentStage, stages: {} };
        act(() => { jest.advanceTimersByTime(250); });
        expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/Currently processing:/)).not.toBeInTheDocument();
        expect(within(table).getByRole('row', { name: 'Position model Pending' })).toBeInTheDocument();
    }

    fixture = trainingFixture(Date.now());
    detection = fixture.vision;
    mockInput = { observation: fixture.observation };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Image age for AI')).toHaveTextContent('250 ms / 1000 ms limit · Fresh');
    expect(getDefinition('Image age for AI')).toHaveAttribute('data-stale', 'false');
    detection = null;
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Image age for AI')).toHaveTextContent('Unavailable');
    expect(getDefinition('Time since frame published')).toHaveTextContent('Unavailable');
    expect(screen.queryByText(/segmentation|depth|mask readback/i)).not.toBeInTheDocument();
});

it('shows live driving model timings and retains the last cycle when inputs become invalid or driving stops', async () => {
    const { observation } = trainingFixture(Date.now());
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
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
    expect(within(table).getByRole('row', { name: 'Driving policy inference 250 ms · Running' })).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: 'Total driving model cycle 250 ms · Running' })).toBeInTheDocument();
    finish(); await flush();
    mockInput = { error: 'Waiting for a fresh image.' };
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(within(table).getByRole('row', { name: 'Driving policy inference 250 ms' })).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: 'Training update / GPU submission Not run' })).toBeInTheDocument();
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

it('shows motion and reference inputs with axes and units, and clears them when observations are unavailable', async () => {
    await renderTraining();
    const labels = ['Local velocity', 'Yaw rate', 'Reference error'];
    labels.forEach((label) => expect(getDefinition(label)).toHaveTextContent('—'));
    mockInput = { observation: { ...trainingFixture(Date.now()).observation,
        referenceError: [-23.25, 4.5] } };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Local velocity')).toHaveTextContent('X 1.00 · Y 30.00 · Z 0.00 m/s');
    expect(getDefinition('Reference error')).toHaveTextContent('X (right) -23.25 · Y (forward) 4.50 coordinate units');
    expect(screen.queryByText('World-space velocity')).not.toBeInTheDocument();
    expect(screen.queryByText('Orientation')).not.toBeInTheDocument();
    expect(getDefinition('Yaw rate')).toHaveTextContent('-0.200 rad/s');
    const observation = trainingFixture(Date.now()).observation;
    mockInput = { observation: { ...observation, context: { ...observation.context, game: 'iracing' },
        localVelocity: [1, 0, 30], yawRate: -0.2, referenceError: [12.34, -26.75] } };
    act(() => { jest.advanceTimersByTime(250); });
    expect(getDefinition('Local velocity')).toHaveTextContent('X 1.00 · Y 0.00 · Z 30.00 m/s');
    expect(getDefinition('Reference error')).toHaveTextContent('X (right) 12.34 · Y (forward) -26.75 coordinate units');
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
    fireEvent.click(screen.getByRole('button', { name: 'Open Live Session' }));
    expect(openLive).toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    mockInput = { observation: trainingFixture(Date.now()).observation };
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    expect(screen.getByRole('button', { name: 'Install ACC preset' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Hold to test shift up' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Test shift down in 5 seconds' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    expect(JSON.parse(files.get(MODEL_PATH)!)).toMatchObject({ algorithm: 'online-actor-critic', format: 'kestrel-self-driving-model' });
});

it.each([true, false])('keeps the session active through invalid inputs and resumes automatically (training=%s)', async (train) => {
    // Keep shift pulses out of this input-recovery test's short observation intervals.
    jest.spyOn(Math, 'random').mockReturnValue(0.1);
    const observation = trainingFixture(Date.now()).observation;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
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
    expect(screen.getByText(train ? 'AI control • online actor–critic' : 'AI control • fixed policy')).toBeInTheDocument();
    expect(within(screen.getByRole('region', { name: 'AI driver' })).getByText('10', { selector: 'strong' })).toBeInTheDocument();
    expect(window.virtualController!.apply).not.toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    await recover();
    expect(within(screen.getByRole('region', { name: 'AI driver' })).getByText(train ? '11' : '10', { selector: 'strong' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('No driver');
    const saved = JSON.parse(files.get(MODEL_PATH)!);
    expect(saved.model.steps).toBe(train ? 11 : 10);
    expect(saved).toEqual(train ? expect.not.objectContaining(checkpoint) : checkpoint);
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
    checkpoint.model.outputWeights.slice(3, 6).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === shift ? 4 : 0; });
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
    // Allow each asynchronous 50 ms control tick to settle before the next tick.
    act(() => { jest.advanceTimersByTime(50); }); await flush();
    act(() => { jest.advanceTimersByTime(50); }); await flush();
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
        throttle: expect.closeTo(0.2), brake: 0.1, steering: expect.closeTo(0.48 / 0.98),
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
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(screen.getByRole('button', { name: 'Return to AI' })).toBeDisabled();
    expect(screen.getByLabelText('Current control owner')).toHaveTextContent('Human');
    expect(screen.getByText(/Human control • learning paused/)).toBeInTheDocument();
    mockInput = { observation: { ...observation, capturedAt: Date.now(), measuredAt: Date.now() } };
    act(() => { jest.advanceTimersByTime(250); }); await flush();
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

it('persists the aggression setting, restores it on remount and resets to reference', async () => {
    const view = await renderTraining();
    const slider = screen.getByRole('slider', { name: 'Driving aggression' });
    expect(slider).toHaveValue('50');
    expect(slider).toHaveAttribute('min', '0');
    expect(slider).toHaveAttribute('max', '100');
    fireEvent.change(slider, { target: { value: '0' } }); await flush();
    expect(screen.getByText(/Cautious spreads the slower-speed area/)).toHaveTextContent('20.0 coordinate units');
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

it.each([true, false])('uses Live Map’s selected lap and applies slider changes during training=%s', async (train) => {
    const referenceLap = referenceLapFixture();
    let fixture = trainingFixture(Date.now(), 50, referenceLap);
    const map = { getReferenceLap: () => referenceLap };
    const vision = { getLatestDetection: () => fixture.vision };
    mockFindComponentRef.mockImplementation((name) => ({ current: name.endsWith('track-vision') ? vision : map }));
    mockRead.mockImplementation((_live, detected, lap, aggression) =>
        jest.requireActual('./driving-observation').readDrivingObservation(fixture.live, detected, lap, aggression, Date.now()));
    const checkpoint = new DrivingPolicy(fixture.observation.context, fixture.observation.features.length).checkpoint();
    checkpoint.model.steps = 10;
    files.set(MODEL_PATH, JSON.stringify(checkpoint));
    await renderTraining();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    const slider = screen.getByRole('slider', { name: 'Driving aggression' });
    fireEvent.change(slider, { target: { value: '0' } });
    fireEvent.click(screen.getByRole('button', { name: train ? 'Start training' : 'Run trained model' })); await flush();
    expect(slider).toBeEnabled();
    fireEvent.change(slider, { target: { value: '100' } });
    expect(mockRead).toHaveBeenLastCalledWith(expect.anything(), fixture.vision, referenceLap, 100, expect.any(Number), expect.objectContaining({ aggression: 100 }));
    expect(screen.queryByRole('table', { name: 'Live pedals · current aggression' })).not.toBeInTheDocument();
    expect(getDefinition('Reference speed')).toHaveTextContent('100.0 km/h');
    expect(getDefinition('Aggression target speed')).toHaveTextContent('120.0 km/h');
    expect(screen.getByText(/Aggressive condenses the slower-speed area/)).toBeInTheDocument();
    expect(screen.getByText('X (right) 2.00 · Y (forward) 0.00 coordinate units')).toBeInTheDocument();
    expect(screen.queryByText('Reference pedal reward')).not.toBeInTheDocument();
    expect(screen.queryByText('Drive fast')).not.toBeInTheDocument();
    expect(screen.getByText('Acceleration / deceleration')).toBeInTheDocument();
    expect(screen.getByText(/holding the same speed earns no bonus/)).toBeInTheDocument();
    expect(getDefinition('Normalized position progress')).toHaveTextContent('0.000');
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    fixture.live.currentTelemetry.Graphics_normalized_car_position = 0.252;
    fixture.vision.position!.normalizedPosition = 0.252;
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(screen.getByText(/Last scored at aggression 0/)).toBeInTheDocument();
    const progress = Number(getDefinition('Normalized position progress').textContent);
    expect(progress).toBeGreaterThan(0);
    expect(progress).toBeLessThan(Number(getDefinition('Follow the reference line').textContent) * 0.25);
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    fixture.live.currentTelemetry.Graphics_normalized_car_position = 0.252;
    fixture.vision.position!.normalizedPosition = 0.252;
    fixture.live.currentTelemetry.Physics_speed_kmh = 0;
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(Number(getDefinition('Normalized position progress').textContent)).toBeLessThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Reset to reference' }));
    expect(slider).toHaveValue('50');
    expect(screen.getByText(/Reference uses the recorded speed targets/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    const saved = JSON.parse(files.get(MODEL_PATH)!);
    expect(saved.model.steps).toBe(train ? 12 : 10);
    expect(saved).toEqual(train ? expect.not.objectContaining(checkpoint) : checkpoint);
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
    for (const throttleBias of [1, 2]) {
        // Simulate an updated JSON between runs without remounting the page.
        checkpoint.configuration.aggression = throttleBias * 30;
        checkpoint.configuration.actions.planTimesSeconds = [0, 0.4];
        checkpoint.model.outputWeights = checkpoint.model.outputWeights.slice(0, 13);
        checkpoint.model.outputWeights.forEach((row) => row.fill(0));
        checkpoint.model.outputWeights[0][checkpoint.configuration.network.hiddenUnits] = throttleBias;
        files.set(MODEL_PATH, JSON.stringify(checkpoint));
        fireEvent.click(screen.getByRole('button', { name: 'Run trained model' })); await flush();
        expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({
            throttle: 1 / (1 + Math.exp(-throttleBias)),
        }));
        expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue(String(throttleBias * 30));
        expect(within(screen.getByRole('table', { name: 'Planned driving actions' })).getAllByRole('row')).toHaveLength(3);
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

it('automatically saves trained weights to the selected file at the configured interval', async () => {
    jest.spyOn(Math, 'random').mockReturnValue(0.1);
    const observation = trainingFixture(Date.now()).observation;
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.configuration.runtime.saveEverySteps = 1;
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
    await advance(); await advance();
    const saved = JSON.parse(files.get(MODEL_PATH)!);
    expect(saved.model.steps).toBeGreaterThan(0);
    expect(saved.model.lstmWeights).not.toEqual(checkpoint.model.lstmWeights);
    expect(window.drivingModelFiles!.write).toHaveBeenCalledWith(MODEL_PATH, expect.any(String));
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    expect(localStorage.getItem('acla.self-driving.model')).toBeNull();
});

it('waits for a departing page’s GPU checkpoint before reopening the file', async () => {
    const { observation } = trainingFixture();
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
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
