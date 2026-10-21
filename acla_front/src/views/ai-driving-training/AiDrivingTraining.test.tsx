import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import AiDrivingTraining from './AiDrivingTraining';
import type { ObservationResult } from './driving-observation';
import { trainingFixture } from './training-fixtures';

let mockInput: ObservationResult;
jest.mock('./driving-observation', () => ({
    ...jest.requireActual('./driving-observation'),
    readDrivingObservation: (...args: unknown[]) => args.length === 3
        ? jest.requireActual('./driving-observation').readDrivingObservation(...args) : mockInput,
}));
const flush = async () => { await act(async () => { await Promise.resolve(); await Promise.resolve(); }); };

beforeEach(() => {
    jest.useFakeTimers();
    localStorage.clear();
    mockInput = { error: 'Waiting for live telemetry.' };
    window.virtualController = { connect: jest.fn().mockResolvedValue(undefined), apply: jest.fn().mockResolvedValue(undefined),
        disconnect: jest.fn().mockResolvedValue(undefined) };
});
afterEach(() => { delete window.virtualController; jest.useRealTimers(); });

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
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' })); await flush();
    expect(window.virtualController!.apply).toHaveBeenLastCalledWith({ throttle: 0, brake: 0, steering: 0 });
    expect(JSON.parse(localStorage.getItem('acla.self-driving.policy.v1')!).algorithm).toBe('online-actor-critic');
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
