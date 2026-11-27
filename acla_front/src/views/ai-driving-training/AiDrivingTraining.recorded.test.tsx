import trainingLabels from 'data/recorded-training-labels.json';
import { mockDrivingGpu } from './driving-gpu-test-model';
import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import AiDrivingTraining from './AiDrivingTraining';
import { DrivingPolicy, emptyTrainingDocument } from './driving-policy';
import { cloneConfig, DEFAULT_DRIVING_CONFIG, emptyRecordedSamplingWeights } from './driving-model-config';
import { circuitMapFixture, trainingFixture } from './training-fixtures';
import { parseRecordedDrivingSession } from './recorded-driving-session';
import type { RecordedDrivingFrame, RecordedFrameViewerHandle } from './RecordedFrameViewer';
import { loadRecordedDrivingReferences } from './recorded-driving-reference';

let mockFrames: RecordedDrivingFrame[] = [];
let mockWaitForCancel = false;
let mockReplayTimeOffset = 0;
let mockDecodedReplays = 0;
let mockFrameMeasurements = 0;
let mockSelectedTrainingClips = 1;
jest.mock('contexts/OperationComponentRefContext', () => ({ useOptionalOperationComponentRefDirectory: () => null }));
jest.mock('./recorded-driving-reference', () => ({
    ...jest.requireActual('./recorded-driving-reference'), loadRecordedDrivingReferences: jest.fn(),
}));
jest.mock('./RecordedFrameViewer', () => {
    const React = require('react');
    return { __esModule: true, default: React.forwardRef((props: any, ref: any) => {
        const replay: RecordedFrameViewerHandle['replay'] = async (consume, signal, onProgress) => {
            mockDecodedReplays++;
            const totalFrames = mockFrames.length;
            let processedFrames = 0;
            onProgress?.({ recordingNumber: 1, recordingCount: 1, processedFrames, totalFrames });
            for (const frame of mockFrames) {
                if (signal.aborted) return;
                const session = frame.recordingId ? { ...frame.session, id: `${frame.session.id}-replay` } : frame.session;
                await consume({ ...frame, session, telemetrySeconds: frame.telemetrySeconds + mockReplayTimeOffset });
                processedFrames++;
                onProgress?.({ recordingNumber: 1, recordingCount: 1, processedFrames, totalFrames });
            }
            if (mockWaitForCancel && !signal.aborted) await new Promise<void>((resolve) => signal.addEventListener('abort', () => resolve(), { once: true }));
        };
        const measure: RecordedFrameViewerHandle['measure'] = async (consume, signal) => {
            mockFrameMeasurements++;
            for (const { camera: _camera, ...frame } of mockFrames) {
                if (signal.aborted) return;
                await consume(frame);
            }
        };
        React.useImperativeHandle(ref, () => ({ replay, measure, countTrainingClips: () => mockSelectedTrainingClips }));
        return <button disabled={props.disabled} onClick={() => {
            props.onFrame(mockFrames[0]); props.onReplayReadyChange(mockSelectedTrainingClips > 0);
            props.onReplaySelectionChange?.();
        }}>Load test recording</button>;
    }) };
});
const flush = async () => { await act(async () => { for (let i = 0; i < 80; i++) await Promise.resolve(); }); };
let savedJson: string;

beforeEach(() => {
    jest.useFakeTimers(); mockDrivingGpu(); mockWaitForCancel = false; mockReplayTimeOffset = 0;
    mockDecodedReplays = 0; mockFrameMeasurements = 0; mockSelectedTrainingClips = 1;
    const fixture = trainingFixture(1000);
    const session = parseRecordedDrivingSession([0, 0.1, 0.2].map((time) => JSON.stringify({ ...fixture.live.currentTelemetry,
        Graphics_analyst_session_time: time, Graphics_status: 2 })).join('\n'), 'acc');
    mockFrames = session.rows.map((row) => ({ session, videoSeconds: row.timeSeconds,
        telemetrySeconds: session.startTimeSeconds + row.timeSeconds, camera: fixture.observation.camera }));
    (loadRecordedDrivingReferences as jest.Mock).mockResolvedValue({ laps: [fixture.referenceLap], maps: new Map([['monza', fixture.circuitMap]]) });
    const config = cloneConfig(); config.network.hiddenUnits = 2; config.network.cameraProjectionUnits = 2;
    config.recordedBatchCount = 1;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 1; });
    savedJson = JSON.stringify(emptyTrainingDocument(config));
    window.drivingModelFiles = { current: jest.fn(async () => 'C:/models/test.json'), choose: jest.fn(), select: jest.fn(),
        read: jest.fn(async () => savedJson), write: jest.fn(async (_path, json) => { savedJson = json; }) };
    window.virtualController = { connect: jest.fn(), apply: jest.fn(), disconnect: jest.fn(async () => undefined), installAccPreset: jest.fn() };
});
afterEach(async () => { await flush(); delete window.drivingModelFiles; delete window.virtualController; jest.restoreAllMocks(); jest.useRealTimers(); });

async function loadRecording() {
    const view = render(<AiDrivingTraining onOpenLiveSession={() => undefined} />); await flush();
    expect(screen.getByLabelText('AI observation input source')).toHaveValue('live');
    fireEvent.change(screen.getByLabelText('AI observation input source'), { target: { value: 'recorded' } });
    fireEvent.click(screen.getByRole('button', { name: 'Load test recording' })); await flush();
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    return view;
}

it('counts selected training clips without a model or frame replay', async () => {
    window.drivingModelFiles!.current = jest.fn(async () => null);
    mockSelectedTrainingClips = 27;
    const initializeGpu = jest.spyOn(DrivingPolicy.prototype, 'initializeGpu');
    await loadRecording();
    expect(screen.getByRole('button', { name: 'Train from annotations' })).toBeDisabled();
    const count = screen.getByRole('button', { name: 'Count training clips' });
    const status = () => screen.getByRole('status', { name: 'Training clip count' });
    expect(count).toBeEnabled();
    fireEvent.click(count);
    expect(status()).toHaveTextContent('27 training clips selected.');

    mockSelectedTrainingClips = 1;
    fireEvent.click(screen.getByRole('button', { name: 'Load test recording' }));
    expect(screen.queryByText('27 training clips selected.')).not.toBeInTheDocument();
    fireEvent.click(count);
    expect(status()).toHaveTextContent('1 training clip selected.');

    mockSelectedTrainingClips = 0;
    fireEvent.click(screen.getByRole('button', { name: 'Load test recording' }));
    expect(count).toBeEnabled();
    fireEvent.click(count);
    expect(status()).toHaveTextContent('0 training clips selected.');
    expect(mockFrameMeasurements).toBe(0);
    expect(mockDecodedReplays).toBe(0);
    expect(initializeGpu).not.toHaveBeenCalled();
    expect(window.drivingModelFiles!.read).not.toHaveBeenCalled();
    expect(window.drivingModelFiles!.write).not.toHaveBeenCalled();
});

it('trains and evaluates local recording without connecting or writing virtual controls', async () => {
    const view = await loadRecording();
    const aiDriver = within(screen.getByRole('region', { name: 'AI driver' }));
    expect(screen.queryByRole('button', { name: 'Connect wheel' })).not.toBeInTheDocument();
    expect(aiDriver.getByRole('button', { name: 'Train from annotations' })).toBeEnabled();
    fireEvent.click(aiDriver.getByRole('button', { name: 'Train from annotations' })); await flush();
    expect(JSON.parse(savedJson).model.steps).toBe(8);
    expect(screen.getByText(/Completed\. Learning from recording/)).toBeInTheDocument();
    expect(aiDriver.queryByRole('progressbar', { name: 'Annotation training progress' })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Driving aggression'), { target: { value: '73' } }); await flush();
    expect(JSON.parse(savedJson).configuration.aggression).toBe(73);
    const trained = savedJson;
    fireEvent.click(aiDriver.getByRole('button', { name: 'Evaluate annotations' })); await flush();
    expect(screen.getByText(/Completed\. Evaluating recording.*pedal MAE.*steering MAE/)).toBeInTheDocument();
    expect(savedJson).toBe(trained);
    fireEvent.keyDown(window, { key: 'Escape' }); await flush();
    expect(window.virtualController!.connect).not.toHaveBeenCalled();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
    view.unmount(); await flush();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
});

it('locks source and model changes during replay and cancels before fitting', async () => {
    mockWaitForCancel = true;
    const learnBatch = jest.spyOn(DrivingPolicy.prototype, 'learnRecordedBatchAsync');
    const originalJson = savedJson;
    await loadRecording();
    const aiDriver = within(screen.getByRole('region', { name: 'AI driver' }));
    fireEvent.click(aiDriver.getByRole('button', { name: 'Train from annotations' }));
    const progress = aiDriver.getByRole('progressbar', { name: 'Annotation training progress' });
    expect(progress).not.toHaveAttribute('value');
    await flush();
    expect(progress).toHaveAttribute('value', '1');
    expect(progress).toHaveAttribute('aria-valuetext', 'Recording 1 of 1 · 3 / 3 frames processed');
    expect(screen.getByLabelText('AI observation input source')).toBeDisabled();
    expect(aiDriver.getByRole('spinbutton', { name: 'Shift down (%)' })).toBeDisabled();
    expect(aiDriver.getByRole('spinbutton', { name: 'Recorded batch size' })).toBeDisabled();
    expect(aiDriver.getByRole('spinbutton', { name: 'Recorded batch count' })).toBeDisabled();
    expect(aiDriver.getByText('Model file')).toBeInTheDocument();
    for (const name of ['New model', 'Open model', 'Save model']) {
        expect(aiDriver.getByRole('button', { name })).toBeDisabled();
    }
    fireEvent.click(aiDriver.getByRole('button', { name: 'Stop replay' })); await flush();
    expect(screen.getByText(/^Stopped\. Collecting samples from recording/)).toBeInTheDocument();
    expect(progress).not.toBeInTheDocument();
    expect(learnBatch).not.toHaveBeenCalled();
    expect(savedJson).toBe(originalJson);
    expect(window.drivingModelFiles!.write).not.toHaveBeenCalled();
    expect(screen.getByLabelText('AI observation input source')).toBeEnabled();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
});

it('shows completed batches and saves only committed draws when stopped during fitting', async () => {
    const checkpoint = JSON.parse(savedJson);
    checkpoint.configuration.recordedBatchCount = 3;
    checkpoint.configuration.runtime.saveEverySteps = 8;
    savedJson = JSON.stringify(checkpoint);
    const learnRecordedBatch = DrivingPolicy.prototype.learnRecordedBatchAsync;
    let releaseSecondBatch!: () => void;
    const secondBatch = new Promise<void>((resolve) => { releaseSecondBatch = resolve; });
    let attemptedBatches = 0;
    const learnBatch = jest.spyOn(DrivingPolicy.prototype, 'learnRecordedBatchAsync').mockImplementation(async function (
        this: DrivingPolicy, batch, active,
    ) {
        attemptedBatches++;
        if (attemptedBatches === 2) await secondBatch;
        return learnRecordedBatch.call(this, batch, active);
    });

    await loadRecording();
    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();
    expect(learnBatch).toHaveBeenCalledTimes(2);
    expect(screen.getByRole('progressbar', { name: 'Annotation training progress' }))
        .toHaveAttribute('aria-valuetext', '1 / 3 batches completed');
    fireEvent.click(screen.getByRole('button', { name: 'Stop replay' }));
    releaseSecondBatch(); await flush();

    expect(screen.getByText(/^Stopped\..*1 \/ 3 batches completed/)).toBeInTheDocument();
    expect(learnBatch).toHaveBeenCalledTimes(2);
    expect(JSON.parse(savedJson).model.steps).toBe(8);
});

it.each(['Train from annotations', 'Evaluate annotations'])('keeps the consumed frame visible during %s between replay updates', async (action) => {
    mockFrames[0].session.rows.forEach((row, index) => { row.sample = { ...row.sample, Physics_speed_kmh: 108 + index * 10 }; });
    await loadRecording();
    if (action === 'Evaluate annotations') {
        fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();
    }
    mockWaitForCancel = true;
    fireEvent.click(screen.getByRole('button', { name: action })); await flush();
    const inputs = within(screen.getByLabelText('AI observation inputs'));
    const speed = () => inputs.getByText('Speed', { selector: 'dt' }).nextElementSibling;
    expect(screen.getByRole('button', { name: 'Stop replay' })).toBeEnabled();
    expect(speed()).toHaveTextContent('128.0 km/h');
    for (let tick = 0; tick < 3; tick++) {
        act(() => { jest.advanceTimersByTime(250); }); await flush();
        expect(speed()).toHaveTextContent('128.0 km/h');
    }
    fireEvent.click(screen.getByRole('button', { name: 'Stop replay' })); await flush();
});

it.each(['Train from annotations', 'Evaluate annotations'])('keeps skipped sample errors visible during %s between replay updates', async (action) => {
    await loadRecording();
    if (action === 'Evaluate annotations') {
        fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();
    }
    const lastRow = mockFrames[0].session.rows[2];
    lastRow.sample = { ...lastRow.sample, Graphics_status: 1 };
    mockWaitForCancel = true;
    fireEvent.click(screen.getByRole('button', { name: action })); await flush();
    const inputs = within(screen.getByLabelText('AI observation inputs'));
    const error = 'This recorded telemetry sample is not marked as running (Graphics_status = 2).';
    expect(inputs.getByText(error)).toBeInTheDocument();
    expect(inputs.getByText('Speed', { selector: 'dt' }).nextElementSibling).toHaveTextContent('—');
    act(() => { jest.advanceTimersByTime(750); }); await flush();
    expect(inputs.getByText(error)).toBeInTheDocument();
    expect(inputs.queryByText('All model inputs ready')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop replay' })); await flush();
});

it('reports unavailable recording references and supports retry', async () => {
    (loadRecordedDrivingReferences as jest.Mock).mockRejectedValueOnce(new Error('Reference download failed.'));
    await loadRecording();
    expect(screen.getByText(/Could not load recorded training references/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reload recorded references' })); await flush();
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(screen.getByText('All model inputs ready')).toBeInTheDocument();
});

it('trains one model across recordings and clears imitation history between close ranges', async () => {
    const first = mockFrames.map((frame, index) => ({ ...frame, recordingName: 'first.mp4', startsRange: index === 0 || index === 2 }));
    const secondSession = { ...first[0].session, id: 'second' };
    const second = mockFrames.map((frame, index) => ({ ...frame, session: secondSession, recordingName: 'second.mp4', startsRange: index === 0 }));
    mockFrames = [...first, ...second];
    await loadRecording();
    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();
    expect(JSON.parse(savedJson).model.steps).toBe(8);
    expect(loadRecordedDrivingReferences).toHaveBeenCalledWith(secondSession);
    expect(screen.getByText(/Completed\. Learning from recording · second\.mp4.*3 samples/)).toBeInTheDocument();
});

it('saves a fixed sampling mix, trains on tagged origins, and evaluates every aligned pair', async () => {
    const { session: source, camera } = mockFrames[0];
    const session = parseRecordedDrivingSession(Array.from({ length: 12 }, (_, index) => JSON.stringify({
        ...source.rows[0].sample, Graphics_analyst_session_time: index / 10,
        Physics_gear: index < 4 ? 4 : 5, Physics_steer_angle: index >= 2 && index < 6 ? 0.3 : 0,
    })).join('\n'), 'acc');
    mockFrames = session.rows.map((row, index) => ({ session, videoSeconds: row.timeSeconds, telemetrySeconds: row.timeSeconds,
        camera, startsRange: [0, 5, 8, 10].includes(index),
        clipLabel: index < 5 ? 'Shift down' : index < 8 ? 'Recover off track' : index < 10 ? 'Hairpin' : 'Shift up' }));
    await loadRecording();
    const weights = within(screen.getByRole('group', { name: 'Fixed sampling weights' }));
    const categories = [...trainingLabels, 'Unlabeled'];
    expect(weights.getAllByRole('spinbutton', { name: /\(%\)$/ })).toHaveLength(categories.length);
    categories.forEach((label) => expect(weights.getByRole('spinbutton', { name: `${label} (%)` })).toBeInTheDocument());
    expect(weights.queryByRole('spinbutton', { name: 'Race pace (%)' })).not.toBeInTheDocument();
    expect(weights.queryByRole('spinbutton', { name: 'Shifting (%)' })).not.toBeInTheDocument();
    expect(weights.queryByRole('spinbutton', { name: 'Cornering (%)' })).not.toBeInTheDocument();
    const shiftDown = weights.getByRole('spinbutton', { name: 'Shift down (%)' });
    const shiftUp = weights.getByRole('spinbutton', { name: 'Shift up (%)' });
    const hairpin = weights.getByRole('spinbutton', { name: 'Hairpin (%)' });
    const unlabeled = weights.getByRole('spinbutton', { name: 'Unlabeled (%)' });
    const saveWeights = weights.getByRole('button', { name: 'Save sampling weights' });
    expect(shiftDown).toHaveValue(7.5);
    expect(shiftUp).toHaveValue(7.5);
    expect(hairpin).toHaveValue(0);
    expect(unlabeled).toHaveValue(45);
    fireEvent.change(hairpin, { target: { value: '25' } });
    fireEvent.change(shiftDown, { target: { value: '12' } });
    fireEvent.change(shiftUp, { target: { value: '15' } });
    expect(saveWeights).toBeDisabled();
    fireEvent.change(unlabeled, { target: { value: '8' } });
    expect(saveWeights).toBeEnabled();
    jest.mocked(window.drivingModelFiles!.write).mockRejectedValueOnce(new Error('Test save failure'));
    fireEvent.click(saveWeights); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Test save failure');
    expect(saveWeights).toBeEnabled();
    fireEvent.click(saveWeights); await flush();
    expect(JSON.parse(savedJson).configuration.recordedSamplingWeights).toEqual({
        ...DEFAULT_DRIVING_CONFIG.recordedSamplingWeights, 'Shift down': 12, 'Shift up': 15, Hairpin: 25, Unlabeled: 8,
    });

    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();
    expect(screen.getByText(/Completed\. Learning from recording.*8 samples.*Shift down 2 \(4\).*Shift up 2 \(1\).*Recover off track 1 \(2\).*Hairpin 3 \(1\)/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(8);
    fireEvent.change(shiftDown, { target: { value: '13' } });
    fireEvent.click(screen.getByRole('button', { name: 'Evaluate annotations' })); await flush();
    expect(screen.getByText(/Completed\. Evaluating recording.*8 samples/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(8);
    expect(shiftDown).toHaveValue(13);
    expect(JSON.parse(savedJson).configuration.recordedSamplingWeights['Shift down']).toBe(12);
});

it('requires a valid whole recorded batch size and keeps the last saved value on save failure', async () => {
    await loadRecording();
    const batchSize = screen.getByRole('spinbutton', { name: 'Recorded batch size' });
    const saveBatchSize = screen.getByRole('button', { name: 'Save batch size' });
    const train = screen.getByRole('button', { name: 'Train from annotations' });
    const write = jest.mocked(window.drivingModelFiles!.write);
    expect(batchSize).toHaveValue(8);
    expect(saveBatchSize).toBeDisabled();

    for (const value of ['', '0', '8.5', '257']) {
        fireEvent.change(batchSize, { target: { value } });
        expect(saveBatchSize).toBeDisabled();
        expect(train).toBeDisabled();
    }
    expect(write).not.toHaveBeenCalled();

    fireEvent.change(batchSize, { target: { value: '256' } });
    expect(saveBatchSize).toBeEnabled();
    expect(train).toBeDisabled();
    write.mockRejectedValueOnce(new Error('Test batch-size save failure'));
    fireEvent.click(saveBatchSize); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Test batch-size save failure');
    expect(JSON.parse(savedJson).configuration.recordedBatchSize).toBe(8);
    expect(batchSize).toHaveValue(256);
    expect(saveBatchSize).toBeEnabled();
    expect(train).toBeDisabled();

    fireEvent.click(saveBatchSize); await flush();
    expect(JSON.parse(savedJson).configuration.recordedBatchSize).toBe(256);
    expect(train).toBeEnabled();
});

it('defaults the recorded batch count for older models and only saves a valid whole count', async () => {
    const oldModel = JSON.parse(savedJson);
    delete oldModel.configuration.recordedBatchCount;
    savedJson = JSON.stringify(oldModel);
    await loadRecording();
    const batchCount = screen.getByRole('spinbutton', { name: 'Recorded batch count' });
    const saveBatchCount = screen.getByRole('button', { name: 'Save batch count' });
    const train = screen.getByRole('button', { name: 'Train from annotations' });
    const write = jest.mocked(window.drivingModelFiles!.write);
    expect(batchCount).toHaveValue(100);
    expect(saveBatchCount).toBeDisabled();

    for (const value of ['', '0', '-1', '2.5', '9007199254740992']) {
        fireEvent.change(batchCount, { target: { value } });
        expect(saveBatchCount).toBeDisabled();
        expect(train).toBeDisabled();
    }
    expect(write).not.toHaveBeenCalled();

    fireEvent.change(batchCount, { target: { value: '3' } });
    write.mockRejectedValueOnce(new Error('Test batch-count save failure'));
    fireEvent.click(saveBatchCount); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Test batch-count save failure');
    expect(JSON.parse(savedJson).configuration).not.toHaveProperty('recordedBatchCount');
    expect(batchCount).toHaveValue(3);
    expect(saveBatchCount).toBeEnabled();
    expect(train).toBeDisabled();

    fireEvent.click(saveBatchCount); await flush();
    expect(JSON.parse(savedJson).configuration.recordedBatchCount).toBe(3);
    expect(train).toBeEnabled();
});

it('runs the saved number and size of batches and restores both settings when reopening the model', async () => {
    const { session: source, camera } = mockFrames[0];
    const session = parseRecordedDrivingSession(Array.from({ length: 12 }, (_, index) => JSON.stringify({
        ...source.rows[0].sample, Graphics_analyst_session_time: index / 10,
    })).join('\n'), 'acc');
    mockFrames = session.rows.map(({ timeSeconds }) => ({
        session, videoSeconds: timeSeconds, telemetrySeconds: timeSeconds, camera,
    }));
    const learnBatch = jest.spyOn(DrivingPolicy.prototype, 'learnRecordedBatchAsync');
    const view = await loadRecording();
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Recorded batch size' }), { target: { value: '10' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save batch size' })); await flush();
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Recorded batch count' }), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save batch count' })); await flush();
    expect(JSON.parse(savedJson).configuration).toMatchObject({ recordedBatchSize: 10, recordedBatchCount: 3 });

    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();
    expect(screen.getByText(/Completed\. Learning from recording.*11 samples.*3 \/ 3 batches completed/)).toBeInTheDocument();
    expect(learnBatch.mock.calls.map(([batch]) => batch.length)).toEqual([10, 10, 10]);
    expect(JSON.parse(savedJson).model.steps).toBe(30);
    expect(JSON.parse(savedJson).configuration).toMatchObject({ recordedBatchSize: 10, recordedBatchCount: 3 });

    view.unmount(); await flush();
    await loadRecording();
    expect(screen.getByRole('spinbutton', { name: 'Recorded batch size' })).toHaveValue(10);
    expect(screen.getByRole('spinbutton', { name: 'Recorded batch count' })).toHaveValue(3);
});

it('requires room for every eligible weighted label and includes a low-weight label in each batch', async () => {
    const document = JSON.parse(savedJson);
    document.configuration.recordedBatchSize = 1;
    document.configuration.recordedBatchCount = 3;
    document.configuration.recordedSamplingWeights = { ...emptyRecordedSamplingWeights(), Warmup: 99, Hairpin: 1 };
    savedJson = JSON.stringify(document);
    mockFrames = mockFrames.map((frame, index) => {
        const row = frame.session.rows[index];
        row.sample = { ...row.sample, Physics_steer_angle: index === 2 ? 0.5 : 0 };
        return { ...frame, clipLabel: index === 0 ? 'Warmup' : 'Hairpin' };
    });
    const learnBatch = jest.spyOn(DrivingPolicy.prototype, 'learnRecordedBatchAsync');
    await loadRecording();

    const train = screen.getByRole('button', { name: 'Train from annotations' });
    fireEvent.click(train); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent(
        'Recorded batch size must be at least 2 to include all 2 eligible labels with positive sampling weights.',
    );
    expect(mockDecodedReplays).toBe(0);
    expect(learnBatch).not.toHaveBeenCalled();
    expect(window.drivingModelFiles!.write).not.toHaveBeenCalled();

    fireEvent.change(screen.getByRole('spinbutton', { name: 'Recorded batch size' }), { target: { value: '4' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save batch size' })); await flush();
    fireEvent.click(train); await flush();
    expect(mockDecodedReplays).toBe(1);
    expect(learnBatch).toHaveBeenCalledTimes(3);
    for (const [batch] of learnBatch.mock.calls) {
        const steering = batch.map(({ targets }) => targets[0].action.steering);
        expect(steering).toHaveLength(4);
        expect(steering).toContain(0);
        expect(steering).toContain(0.5);
    }
    expect(screen.getByText(/Completed\. Learning from recording.*3 \/ 3 batches completed/)).toBeInTheDocument();
});

it('uses the same telemetry times when counting samples and replaying decoded frames', async () => {
    const fixture = trainingFixture();
    const map = { ...fixture.circuitMap, centerline_segments: [
        { id: 'entry', start_position: 0.2, end_position: 0.27, tags: ['corner'] },
        { id: 'exit', start_position: 0.27, end_position: 0.35, tags: ['corner'] },
    ] };
    jest.mocked(loadRecordedDrivingReferences).mockResolvedValue({ laps: [fixture.referenceLap], maps: new Map([['monza', map]]) });
    const positions = [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.28, 0.3, 0.31, 0.32];
    const session = parseRecordedDrivingSession(positions.map((position, index) => JSON.stringify({
        ...fixture.live.currentTelemetry, Graphics_analyst_session_time: index / 10, Graphics_normalized_car_position: position,
    })).join('\n'), 'acc');
    mockFrames = session.rows.map(({ timeSeconds }) => ({ session, videoSeconds: timeSeconds, telemetrySeconds: timeSeconds,
        camera: fixture.observation.camera, clipLabel: 'Warmup' }));
    mockReplayTimeOffset = -0.015;

    await loadRecording();
    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();

    expect(screen.getByText(/Completed\. Learning from recording.*9 samples.*Warmup 8 \(9\)/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(8);
});

it('samples across simulators and tracks and reuses the measured recording references', async () => {
    const fixture = trainingFixture();
    const references = jest.mocked(loadRecordedDrivingReferences);
    references.mockClear().mockImplementation(async ({ game, rows }) => {
        const track = rows[0].sample.Static_track!;
        const lap = { ...fixture.referenceLap, game, track };
        return { laps: [lap], maps: new Map([[track, circuitMapFixture(lap)]]) };
    });
    const recordings = [
        { game: 'acc', track: 'monza', frames: 7 },
        { game: 'acc', track: 'spa', frames: 3 },
        { game: 'iracing', track: 'monza', frames: 5 },
    ] as const;
    mockFrames = recordings.flatMap(({ game, track, frames }) => {
        const session = parseRecordedDrivingSession(Array.from({ length: frames }, (_, index) => JSON.stringify({
            ...fixture.live.currentTelemetry, Static_track: track, Graphics_analyst_session_time: index / 10,
        })).join('\n'), game);
        return session.rows.map(({ timeSeconds }, index) => ({ session, recordingId: session.id, startsRange: index === 0,
            videoSeconds: timeSeconds, telemetrySeconds: timeSeconds, camera: fixture.observation.camera, clipLabel: 'Warmup' }));
    });

    await loadRecording();
    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();

    expect(screen.getByText(/Completed\. Learning from recording.*12 samples.*Warmup 8 \(12\)/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(8);
    expect(references).toHaveBeenCalledTimes(recordings.length);
});
