import { mockDrivingGpu } from './driving-gpu-test-model';
import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import AiDrivingTraining from './AiDrivingTraining';
import { emptyTrainingDocument } from './driving-policy';
import { cloneConfig } from './driving-model-config';
import { trainingFixture } from './training-fixtures';
import { parseRecordedDrivingSession } from './recorded-driving-session';
import type { RecordedDrivingFrame } from './RecordedFrameViewer';
import { loadRecordedDrivingReferences } from './recorded-driving-reference';

let mockFrames: RecordedDrivingFrame[] = [];
let mockWaitForCancel = false;
jest.mock('contexts/OperationComponentRefContext', () => ({ useOptionalOperationComponentRefDirectory: () => null }));
jest.mock('./recorded-driving-reference', () => ({
    ...jest.requireActual('./recorded-driving-reference'), loadRecordedDrivingReferences: jest.fn(),
}));
jest.mock('./RecordedFrameViewer', () => {
    const React = require('react');
    return { __esModule: true, default: React.forwardRef((props: any, ref: any) => {
        React.useImperativeHandle(ref, () => ({ replay: async (consume: any, signal: AbortSignal) => {
            for (const frame of mockFrames) { if (signal.aborted) return; await consume(frame); }
            if (mockWaitForCancel && !signal.aborted) await new Promise<void>((resolve) => signal.addEventListener('abort', () => resolve(), { once: true }));
        } }));
        return <button disabled={props.disabled} onClick={() => props.onFrame(mockFrames[0])}>Load test recording</button>;
    }) };
});
const flush = async () => { await act(async () => { for (let i = 0; i < 80; i++) await Promise.resolve(); }); };
let savedJson: string;

beforeEach(() => {
    jest.useFakeTimers(); mockDrivingGpu(); mockWaitForCancel = false;
    const fixture = trainingFixture(1000);
    const session = parseRecordedDrivingSession([0, 0.1, 0.2].map((time) => JSON.stringify({ ...fixture.live.currentTelemetry,
        Graphics_analyst_session_time: time, Graphics_status: 2 })).join('\n'), 'acc');
    mockFrames = session.rows.map((row) => ({ session, videoSeconds: row.timeSeconds, offsetSeconds: 0, camera: fixture.observation.camera }));
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
    expect(screen.queryByRole('button', { name: 'Connect wheel' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Train from recording' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Train from recording' })); await flush();
    expect(JSON.parse(savedJson).model.steps).toBe(2);
    expect(screen.getByText(/Completed\. Learning from recording/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Driving aggression'), { target: { value: '73' } }); await flush();
    expect(JSON.parse(savedJson).configuration.aggression).toBe(73);
    const trained = savedJson;
    fireEvent.click(screen.getByRole('button', { name: 'Evaluate recording' })); await flush();
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
    fireEvent.click(screen.getByRole('button', { name: 'Train from recording' })); await flush();
    expect(screen.getByLabelText('AI observation input source')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Open model' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop replay' })); await flush();
    expect(screen.getByText(/Stopped\. Learning from recording/)).toBeInTheDocument();
    expect(JSON.parse(savedJson).model.steps).toBe(2);
    expect(screen.getByLabelText('AI observation input source')).toBeEnabled();
    expect(window.virtualController!.apply).not.toHaveBeenCalled();
});

it.each(['Train from recording', 'Evaluate recording'])('keeps the consumed frame visible during %s between replay updates', async (action) => {
    mockFrames[0].session.rows.forEach((row, index) => { row.sample = { ...row.sample, Physics_speed_kmh: 108 + index * 10 }; });
    await loadRecording();
    if (action === 'Evaluate recording') {
        fireEvent.click(screen.getByRole('button', { name: 'Train from recording' })); await flush();
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

it.each(['Train from recording', 'Evaluate recording'])('keeps skipped sample errors visible during %s between replay updates', async (action) => {
    await loadRecording();
    if (action === 'Evaluate recording') {
        fireEvent.click(screen.getByRole('button', { name: 'Train from recording' })); await flush();
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
    expect(screen.getByRole('button', { name: 'Train from recording' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Reload recorded references' })); await flush();
    act(() => { jest.advanceTimersByTime(250); }); await flush();
    expect(screen.getByRole('button', { name: 'Train from recording' })).toBeEnabled();
});
