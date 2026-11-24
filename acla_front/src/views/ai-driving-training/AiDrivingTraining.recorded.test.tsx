import { mockDrivingGpu } from './driving-gpu-test-model';
import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import AiDrivingTraining from './AiDrivingTraining';
import { emptyTrainingDocument } from './driving-policy';
import { cloneConfig } from './driving-model-config';
import { circuitMapFixture, trainingFixture } from './training-fixtures';
import { parseRecordedDrivingSession } from './recorded-driving-session';
import type { RecordedDrivingFrame, RecordedFrameViewerHandle } from './RecordedFrameViewer';
import { loadRecordedDrivingReferences } from './recorded-driving-reference';

let mockFrames: RecordedDrivingFrame[] = [];
let mockWaitForCancel = false;
let mockReplayTimeOffset = 0;
jest.mock('contexts/OperationComponentRefContext', () => ({ useOptionalOperationComponentRefDirectory: () => null }));
jest.mock('./recorded-driving-reference', () => ({
    ...jest.requireActual('./recorded-driving-reference'), loadRecordedDrivingReferences: jest.fn(),
}));
jest.mock('./RecordedFrameViewer', () => {
    const React = require('react');
    return { __esModule: true, default: React.forwardRef((props: any, ref: any) => {
        const replay: RecordedFrameViewerHandle['replay'] = async (consume, signal, onProgress) => {
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
            for (const { camera: _camera, ...frame } of mockFrames) {
                if (signal.aborted) return;
                await consume(frame);
            }
        };
        React.useImperativeHandle(ref, () => ({ replay, measure }));
        return <button disabled={props.disabled} onClick={() => {
            props.onFrame(mockFrames[0]); props.onReplayReadyChange(true);
        }}>Load test recording</button>;
    }) };
});
const flush = async () => { await act(async () => { for (let i = 0; i < 80; i++) await Promise.resolve(); }); };
let savedJson: string;

beforeEach(() => {
    jest.useFakeTimers(); mockDrivingGpu(); mockWaitForCancel = false; mockReplayTimeOffset = 0;
    const fixture = trainingFixture(1000);
    const session = parseRecordedDrivingSession([0, 0.1, 0.2].map((time) => JSON.stringify({ ...fixture.live.currentTelemetry,
        Graphics_analyst_session_time: time, Graphics_status: 2 })).join('\n'), 'acc');
    mockFrames = session.rows.map((row) => ({ session, videoSeconds: row.timeSeconds,
        telemetrySeconds: session.startTimeSeconds + row.timeSeconds, camera: fixture.observation.camera }));
    (loadRecordedDrivingReferences as jest.Mock).mockResolvedValue({ laps: [fixture.referenceLap], maps: new Map([['monza', fixture.circuitMap]]) });
    const config = cloneConfig(); config.network.hiddenUnits = 2; config.network.cameraProjectionUnits = 2;
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

it('trains and evaluates local recording without connecting or writing virtual controls', async () => {
    const view = await loadRecording();
    const aiDriver = within(screen.getByRole('region', { name: 'AI driver' }));
    expect(screen.queryByRole('button', { name: 'Connect wheel' })).not.toBeInTheDocument();
    expect(aiDriver.getByRole('button', { name: 'Train from annotations' })).toBeEnabled();
    fireEvent.click(aiDriver.getByRole('button', { name: 'Train from annotations' })); await flush();
    expect(JSON.parse(savedJson).model.steps).toBe(2);
    expect(screen.getByText(/Completed\. Learning from recording/)).toBeInTheDocument();
    expect(aiDriver.queryByRole('progressbar', { name: 'Annotation training progress' })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Driving aggression'), { target: { value: '73' } }); await flush();
    expect(JSON.parse(savedJson).configuration.aggression).toBe(73);
    const trained = savedJson;
    fireEvent.click(aiDriver.getByRole('button', { name: 'Evaluate annotations' })); await flush();
    expect(screen.getByText(/Completed\. Evaluating recording.*pedal MAE/)).toBeInTheDocument();
    expect(savedJson).toBe(trained);
    fireEvent.keyDown(window, { key: 'Escape' }); await flush();
    expect(window.virtualController!.connect).not.toHaveBeenCalled();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
    view.unmount(); await flush();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
});

it('locks source and model changes during replay, stops and saves on cancellation', async () => {
    mockWaitForCancel = true;
    await loadRecording();
    const aiDriver = within(screen.getByRole('region', { name: 'AI driver' }));
    fireEvent.click(aiDriver.getByRole('button', { name: 'Train from annotations' }));
    const progress = aiDriver.getByRole('progressbar', { name: 'Annotation training progress' });
    expect(progress).not.toHaveAttribute('value');
    await flush();
    expect(progress).toHaveAttribute('value', '1');
    expect(progress).toHaveAttribute('aria-valuetext', 'Recording 1 of 1 · 3 / 3 frames processed');
    expect(screen.getByLabelText('AI observation input source')).toBeDisabled();
    expect(aiDriver.getByText('Model file')).toBeInTheDocument();
    for (const name of ['New model', 'Open model', 'Save model']) {
        expect(aiDriver.getByRole('button', { name })).toBeDisabled();
    }
    fireEvent.click(aiDriver.getByRole('button', { name: 'Stop replay' })); await flush();
    expect(screen.getByText(/Stopped\. Learning from recording/)).toBeInTheDocument();
    expect(progress).not.toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(2);
    expect(screen.getByLabelText('AI observation input source')).toBeEnabled();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
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
    expect(JSON.parse(savedJson).model.steps).toBe(3);
    expect(loadRecordedDrivingReferences).toHaveBeenCalledWith(secondSession);
    expect(screen.getByText(/Completed\. Learning from recording · second\.mp4.*3 samples/)).toBeInTheDocument();
});

it('uses balanced labeled clip samples for training and evaluates every aligned pair', async () => {
    const { session: source, camera } = mockFrames[0];
    const session = parseRecordedDrivingSession(Array.from({ length: 10 }, (_, index) => JSON.stringify({
        ...source.rows[0].sample, Graphics_analyst_session_time: index / 10,
    })).join('\n'), 'acc');
    mockFrames = session.rows.map((row, index) => ({ session, videoSeconds: row.timeSeconds, telemetrySeconds: row.timeSeconds,
        camera, startsRange: index === 0 || index === 7, clipLabel: index < 7 ? 'Race pace' : 'Recovery' }));
    await loadRecording();
    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();
    expect(screen.getByText(/Completed\. Learning from recording.*4 samples.*4 set aside for balance/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(4);
    fireEvent.click(screen.getByRole('button', { name: 'Evaluate annotations' })); await flush();
    expect(screen.getByText(/Completed\. Evaluating recording.*8 samples/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(4);
});

it('balances saved centerline ranges using the same telemetry times in both passes', async () => {
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
        camera: fixture.observation.camera, clipLabel: 'Race pace' }));
    mockReplayTimeOffset = -0.015;

    await loadRecording();
    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();

    expect(screen.getByText(/Completed\. Learning from recording.*6 samples.*3 set aside for balance/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(6);
});

it('balances across simulators and tracks and reuses the measured recording references', async () => {
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
            videoSeconds: timeSeconds, telemetrySeconds: timeSeconds, camera: fixture.observation.camera, clipLabel: 'Race pace' }));
    });

    await loadRecording();
    fireEvent.click(screen.getByRole('button', { name: 'Train from annotations' })); await flush();

    expect(screen.getByText(/Completed\. Learning from recording.*6 samples.*6 set aside for balance/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(6);
    expect(references).toHaveBeenCalledTimes(recordings.length);
});
