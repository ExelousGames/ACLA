import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import AiDrivingTraining from './AiDrivingTraining';
import type { ObservationResult } from './driving-observation';
import { referenceLapFixture, trainingFixture } from './training-fixtures';
import { DrivingPolicy } from './driving-policy';

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
const flush = async () => { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); };

beforeEach(() => {
    jest.useFakeTimers();
    localStorage.clear();
    mockInput = { error: 'Waiting for live telemetry.' };
    mockRead.mockReset().mockImplementation(() => mockInput);
    mockFindComponentRef.mockReset();
    window.virtualController = { connect: jest.fn().mockResolvedValue(undefined), apply: jest.fn().mockResolvedValue(undefined),
        disconnect: jest.fn().mockResolvedValue(undefined),
        installAccPreset: jest.fn().mockImplementation(async (deviceId) => ({
            name: `ACLA AI Training - vJoy ${deviceId}`, path: '/preset.json', deviceId,
        })) };
});
afterEach(() => { delete window.virtualController; jest.useRealTimers(); });

it('installs an ACC preset for the selected device and explains how to activate it', async () => {
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    render(<AiDrivingTraining onOpenLiveSession={openLive} />);
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
    expect(JSON.parse(localStorage.getItem('acla.self-driving.policy.v3')!)).toMatchObject({ algorithm: 'online-actor-critic', version: 3 });
});

it.each([1, 2])('ignores version-%s local models and starts training a new model', async (version) => {
    const { observation } = trainingFixture();
    const checkpoint = { ...new DrivingPolicy(observation.context, observation.features.length - 3).checkpoint(), version, steps: 123 };
    localStorage.setItem(`acla.self-driving.policy.v${version}`, JSON.stringify(checkpoint));
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Export model' })).toBeDisabled();
    expect(localStorage.getItem('acla.self-driving.policy.v3')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    mockInput = { observation: trainingFixture(Date.now()).observation };
    act(() => { jest.advanceTimersByTime(250); });
    fireEvent.click(screen.getByRole('button', { name: 'Start training' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(JSON.parse(localStorage.getItem('acla.self-driving.policy.v3')!).version).toBe(3);
});

it.each([1, 2])('shows and releases a saved model’s learned shift choice %s', async (shift) => {
    const { observation } = trainingFixture(Date.now());
    const checkpoint = new DrivingPolicy(observation.context, observation.features.length).checkpoint();
    checkpoint.steps = 10;
    checkpoint.outputWeights.slice(4).forEach((row, i) => { row.fill(0); row[row.length - 1] = i === shift ? 4 : 0; });
    localStorage.setItem('acla.self-driving.policy.v3', JSON.stringify(checkpoint));
    mockInput = { observation };
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    expect(screen.getByText('3 / 6000')).toBeInTheDocument();
    expect(screen.getByText(/Disable automatic gears/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Run trained model' })); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: shift === 1, shiftDown: shift === 2 }));
    const label = shift === 1 ? 'Shift up output' : 'Shift down output';
    expect(screen.getByLabelText(label)).toHaveTextContent('Pressed');
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(screen.getByLabelText(label)).toHaveTextContent('Released');
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith(expect.objectContaining({ shiftUp: false, shiftDown: false }));
    fireEvent.keyDown(window, { key: 'Escape' }); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    expect(JSON.parse(localStorage.getItem('acla.self-driving.policy.v3')!)).toEqual(checkpoint);
});

it('shows driver setup errors without claiming the controller is connected', async () => {
    (window.virtualController!.connect as jest.Mock).mockRejectedValue(new Error('Install vJoy and enable X, Y and Z axes.'));
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Install vJoy');
    expect(screen.getByRole('button', { name: 'Disconnect' })).toBeDisabled();
});

it('offers only wheel output and reconnects to the selected vJoy device', async () => {
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    expect(screen.getByRole('heading', { name: 'Steering wheel (vJoy)' })).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Controller output' })).not.toBeInTheDocument();
    expect(screen.queryByText(/Xbox|XInput|gamepad/i)).not.toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'vJoy device' })).toHaveValue('1');
    fireEvent.change(screen.getByRole('combobox', { name: 'vJoy device' }), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(window.virtualController!.connect).toHaveBeenLastCalledWith({ backend: 'vjoy', deviceId: 3 });
    expect(screen.getByRole('combobox', { name: 'vJoy device' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Disconnect' })); await flush();
    expect(screen.getByRole('combobox', { name: 'vJoy device' })).toBeEnabled();
    fireEvent.change(screen.getByRole('combobox', { name: 'vJoy device' }), { target: { value: '4' } });
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    expect(window.virtualController!.connect).toHaveBeenLastCalledWith({ backend: 'vjoy', deviceId: 4 });
});

it('keeps a timed binding test alive in the simulator and releases automatically', async () => {
    const api = window.virtualController!;
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Test brake in 5 seconds' })); await flush();
    if (stop === 'Escape') fireEvent.keyDown(window, { key: 'Escape' });
    else fireEvent.click(screen.getByRole('button', { name: stop }));
    await flush();
    act(() => { jest.advanceTimersByTime(16000); }); await flush();
    expect((api.apply as jest.Mock).mock.calls.every(([action]) => action.brake === 0)).toBe(true);
});

it.each([-1, -0.2, 0, 0.2, 1])('supports steering calibration position %s', async (position) => {
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.change(screen.getByRole('combobox', { name: 'Timed steering position' }), { target: { value: String(position) } });
    fireEvent.click(screen.getByRole('button', { name: 'Test steering in 5 seconds' })); await flush();
    act(() => { jest.advanceTimersByTime(4900); }); await flush();
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: position });
});

it('cancels a timed test on unmount before any non-neutral input is sent', async () => {
    const api = window.virtualController!;
    const { unmount } = render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Test throttle in 5 seconds' })); await flush();
    unmount();
    act(() => { jest.advanceTimersByTime(16000); }); await flush();
    expect((api.apply as jest.Mock).mock.calls.every(([action]) => action.throttle === 0)).toBe(true);
    expect(api.disconnect).toHaveBeenCalled();
});

it('stops timed tests and reports a rejected device write', async () => {
    const api = window.virtualController!;
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    const { unmount } = render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    const { unmount } = render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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

it('persists the aggression setting, restores it on remount and resets to reference', () => {
    const first = render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    const slider = screen.getByRole('slider', { name: 'Driving aggression' });
    expect(slider).toHaveValue('50');
    expect(slider).toHaveAttribute('min', '0');
    expect(slider).toHaveAttribute('max', '100');
    fireEvent.change(slider, { target: { value: '0' } });
    expect(screen.getByText(/Reward earlier braking and later throttle/)).toHaveTextContent('20.0 coordinate units');
    expect(localStorage.getItem('acla.self-driving.aggression')).toBe('0');
    first.unmount();
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue('0');
    fireEvent.click(screen.getByRole('button', { name: 'Reset to reference' }));
    expect(screen.getByRole('slider', { name: 'Driving aggression' })).toHaveValue('50');
    expect(localStorage.getItem('acla.self-driving.aggression')).toBe('50');
});

it.each(['null', '"100"', '-1', '101', 'invalid'])('uses reference aggression when the saved setting is invalid: %s', (value) => {
    localStorage.setItem('acla.self-driving.aggression', value);
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
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
    checkpoint.steps = 10;
    localStorage.setItem('acla.self-driving.policy.v3', JSON.stringify(checkpoint));
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'Connect wheel' })); await flush();
    act(() => { jest.advanceTimersByTime(250); });
    const slider = screen.getByRole('slider', { name: 'Driving aggression' });
    fireEvent.change(slider, { target: { value: '0' } });
    fireEvent.click(screen.getByRole('button', { name: train ? 'Start training' : 'Run trained model' })); await flush();
    expect(slider).toBeEnabled();
    fireEvent.change(slider, { target: { value: '100' } });
    expect(mockRead).toHaveBeenLastCalledWith(expect.anything(), fixture.vision, referenceLap, 100);
    const table = screen.getByRole('table', { name: 'Live pedals · current aggression' });
    expect(within(table).getAllByRole('row')[1]).toHaveTextContent('Throttle50.0%100.0%30.0%');
    expect(within(table).getAllByRole('row')[2]).toHaveTextContent('Brake50.0%0.0%0.0%');
    expect(screen.getByText('2.00 coordinate units')).toBeInTheDocument();
    expect(screen.getByText('Reference pedal reward')).toBeInTheDocument();
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    fixture = trainingFixture(Date.now(), 50, referenceLap);
    fixture.live.currentTelemetry.Graphics_normalized_car_position = 0.252;
    fixture.vision.position!.normalizedPosition = 0.252;
    act(() => { jest.advanceTimersByTime(100); }); await flush();
    expect(screen.getByText(/Last scored at aggression 0/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reset to reference' }));
    expect(slider).toHaveValue('50');
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    const saved = JSON.parse(localStorage.getItem('acla.self-driving.policy.v3')!);
    expect(saved.steps).toBe(train ? 11 : 10);
    if (!train) expect(saved).toEqual(checkpoint);
});

it('restores a saved model after remount', () => {
    const { observation } = trainingFixture();
    const checkpoint = { ...new DrivingPolicy(observation.context, observation.features.length).checkpoint(), steps: 321 };
    localStorage.setItem('acla.self-driving.policy.v3', JSON.stringify(checkpoint));
    const first = render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    expect(screen.getByText('321')).toBeInTheDocument();
    first.unmount();
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    expect(screen.getByText('321')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Export model' })).toBeEnabled();
    expect(JSON.parse(localStorage.getItem('acla.self-driving.policy.v3')!)).toEqual(checkpoint);
});

it('imports and saves a version-3 checkpoint', async () => {
    const { observation } = trainingFixture();
    const checkpoint = { ...new DrivingPolicy(observation.context, observation.features.length).checkpoint(), steps: 44 };
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    const text = JSON.stringify(checkpoint);
    const file = new File([text], 'model.json', { type: 'application/json' });
    Object.defineProperty(file, 'text', { value: async () => text });
    fireEvent.change(screen.getByLabelText('Import model checkpoint'), { target: { files: [file] } }); await flush();
    expect(JSON.parse(localStorage.getItem('acla.self-driving.policy.v3')!)).toEqual(checkpoint);
    expect(screen.getByText('44')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it.each([1, 2])('rejects a version-%s import without replacing the current model', async (version) => {
    const { observation } = trainingFixture();
    const checkpoint = { ...new DrivingPolicy(observation.context, observation.features.length).checkpoint(), steps: 321 };
    localStorage.setItem('acla.self-driving.policy.v3', JSON.stringify(checkpoint));
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    const text = JSON.stringify({ ...new DrivingPolicy(observation.context, observation.features.length - 3).checkpoint(), version, steps: 44 });
    const file = new File([text], 'unsupported.json', { type: 'application/json' });
    Object.defineProperty(file, 'text', { value: async () => text });
    fireEvent.change(screen.getByLabelText('Import model checkpoint'), { target: { files: [file] } }); await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Invalid or incompatible self-driving model checkpoint.');
    expect(screen.getByText('321')).toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem('acla.self-driving.policy.v3')!)).toEqual(checkpoint);
});

it('removes the saved model on New model and stays empty after remount', () => {
    const { observation } = trainingFixture();
    const checkpoint = { ...new DrivingPolicy(observation.context, observation.features.length).checkpoint(), steps: 321 };
    localStorage.setItem('acla.self-driving.policy.v3', JSON.stringify(checkpoint));
    const first = render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    fireEvent.click(screen.getByRole('button', { name: 'New model' }));
    expect(localStorage.getItem('acla.self-driving.policy.v3')).toBeNull();
    first.unmount();
    render(<AiDrivingTraining onOpenLiveSession={() => undefined} />);
    expect(screen.queryByText('321')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Export model' })).toBeDisabled();
    expect(localStorage.getItem('acla.self-driving.policy.v3')).toBeNull();
});
