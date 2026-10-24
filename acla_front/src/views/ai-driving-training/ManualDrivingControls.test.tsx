import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import ManualDrivingControls from './ManualDrivingControls';
import type { PhysicalController } from './driving-control';

const MAPPING_KEY = 'acla.self-driving.manual-mapping.v1';
const props = { connected: true, running: false, source: 'ai' as const, onSourceChange: jest.fn() };
let devices: PhysicalController[];
let read: jest.Mock;
const flush = async () => { await act(async () => { await Promise.resolve(); }); };
const poll = async (milliseconds = 50) => {
    await act(async () => { jest.advanceTimersByTime(milliseconds); });
};
const detect = async (label: string) => {
    fireEvent.click(screen.getByRole('button', { name: `Detect ${label.toLowerCase()}` }));
    await flush();
};

beforeEach(() => {
    jest.useFakeTimers();
    localStorage.clear();
    devices = [
        { id: 'wheel', name: 'Wheel', axes: { axis0: 0.5 }, buttons: [false, false, false, false] },
        { id: 'pedals', name: 'Pedals', axes: { axis1: 1, axis2: 0 }, buttons: [] },
    ];
    read = jest.fn().mockImplementation(async () => JSON.parse(JSON.stringify(devices)));
    window.virtualController = { connect: jest.fn(), disconnect: jest.fn(), apply: jest.fn(),
        installAccPreset: jest.fn(), readPhysicalControls: read };
});

afterEach(() => { delete window.virtualController; jest.useRealTimers(); });

it('detects and saves axes, pedal inversion and every button across separate devices', async () => {
    const view = render(<ManualDrivingControls {...props} />);
    await flush();
    await detect('steering');
    expect(screen.getByRole('status', { name: 'Controller detection' })).toHaveTextContent('Turn the wheel to the right');
    devices[0].axes.axis0 = 0.51;
    await poll();
    expect(screen.getByRole('combobox', { name: 'Steering' })).toHaveValue('');
    devices[0].axes.axis0 = 0.9;
    await poll();
    expect(screen.getByRole('status', { name: 'Controller detection' })).toHaveTextContent('Steering mapped to Wheel · axis AXIS0');
    devices[0].axes.axis0 = 0.5;
    await detect('throttle');
    devices[1].axes.axis1 = 0.1;
    await poll();
    expect(screen.getByRole('checkbox', { name: 'Invert throttle' })).toBeChecked();
    devices[1].axes.axis1 = 1;
    await detect('brake');
    devices[1].axes.axis2 = 0.8;
    await poll();
    expect(screen.getByLabelText('Manual steering preview')).toHaveTextContent('0.000');
    expect(screen.getByLabelText('Manual throttle preview')).toHaveTextContent('0.000');
    expect(screen.getByLabelText('Manual brake preview')).toHaveTextContent('0.800');
    const labels = ['shift up', 'shift down', 'take over / return to AI'];
    for (let index = 0; index < labels.length; index++) {
        devices[0].buttons.fill(false);
        await detect(labels[index]);
        devices[0].buttons[index] = true;
        await poll();
    }
    const expected = {
        steering: { deviceId: 'wheel', axis: 'axis0', inverted: false },
        throttle: { deviceId: 'pedals', axis: 'axis1', inverted: true },
        brake: { deviceId: 'pedals', axis: 'axis2', inverted: false },
        shiftUp: { deviceId: 'wheel', button: 0 }, shiftDown: { deviceId: 'wheel', button: 1 },
        toggleControl: { deviceId: 'wheel', button: 2 },
    };
    expect(JSON.parse(localStorage.getItem(MAPPING_KEY)!)).toEqual(expected);
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toBeEnabled();
    view.unmount();
    render(<ManualDrivingControls {...props} />);
    await flush();
    expect(screen.getByRole('combobox', { name: 'Throttle' })).toHaveValue(JSON.stringify(['pedals', 'axis1']));
    expect(screen.getByRole('checkbox', { name: 'Invert throttle' })).toBeChecked();
    expect(screen.getByRole('combobox', { name: 'Take over / Return to AI button' })).toHaveValue(JSON.stringify(['wheel', 2]));
    expect(screen.queryByRole('combobox', { name: 'Take over button' })).not.toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Return to AI button' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toHaveAttribute('aria-pressed', 'false');
});

it.each([
    [{ takeover: { deviceId: 'wheel', button: 2 }, resume: { deviceId: 'wheel', button: 3 } }, 2],
    [{ resume: { deviceId: 'wheel', button: 3 } }, 3],
    [{ toggleControl: { deviceId: 'wheel', button: 1 }, takeover: { deviceId: 'wheel', button: 2 } }, 1],
])('restores one toggle from saved bindings %j', async (bindings, button) => {
    const axes = {
        steering: { deviceId: 'wheel', axis: 'axis0', inverted: false },
        throttle: { deviceId: 'pedals', axis: 'axis1', inverted: true },
        brake: { deviceId: 'pedals', axis: 'axis2', inverted: false },
    };
    localStorage.setItem(MAPPING_KEY, JSON.stringify({ ...axes, ...bindings }));
    render(<ManualDrivingControls {...props} />);
    await flush();
    expect(screen.getByRole('combobox', { name: 'Take over / Return to AI button' })).toHaveValue(JSON.stringify(['wheel', button]));
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toBeEnabled();
    expect(JSON.parse(localStorage.getItem(MAPPING_KEY)!)).toEqual({ ...axes, toggleControl: { deviceId: 'wheel', button } });
});

it.each(['x', 'y', 'z', 'r', 'u', 'v'])('requires remapping saved legacy axis %s', async (axis) => {
    localStorage.setItem(MAPPING_KEY, JSON.stringify({
        steering: { deviceId: 'wheel', axis, inverted: false },
        throttle: { deviceId: 'pedals', axis: 'axis1', inverted: true },
        brake: { deviceId: 'pedals', axis: 'axis2', inverted: false },
    }));
    render(<ManualDrivingControls {...props} />);
    await flush();
    expect(screen.getByRole('combobox', { name: 'Steering' })).toHaveValue('');
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toBeDisabled();
    expect(JSON.parse(localStorage.getItem(MAPPING_KEY)!)).toEqual({});
});

it('detects a newly connected gamepad and restores extended axis and button bindings', async () => {
    devices = [];
    const view = render(<ManualDrivingControls {...props} />);
    await flush();
    expect(screen.getByText(/No physical controllers detected/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Detect steering' })).toBeDisabled();
    devices = [{ id: 'sdl:pad:usb1', name: 'Physical controller',
        axes: { axis0: 0.5, axis4: 0, axis9: 0 }, buttons: Array(68).fill(false) }];
    await poll(250);
    for (const [label, axis] of [['steering', 'axis0'], ['throttle', 'axis4'], ['brake', 'axis9']]) {
        await detect(label);
        devices[0].axes[axis] = 1;
        await poll();
        devices[0].axes[axis] = axis === 'axis0' ? 0.5 : 0;
    }
    await detect('take over / return to AI');
    devices[0].buttons[67] = true;
    await poll();
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toBeEnabled();
    view.unmount();
    render(<ManualDrivingControls {...props} />);
    await flush();
    expect(screen.getByRole('combobox', { name: 'Brake' })).toHaveValue(JSON.stringify(['sdl:pad:usb1', 'axis9']));
    expect(screen.getByRole('combobox', { name: 'Take over / Return to AI button' })).toHaveValue(JSON.stringify(['sdl:pad:usb1', 67]));
    expect(screen.getByLabelText('Manual throttle preview')).toHaveTextContent('0.000');
    expect(screen.getByLabelText('Manual brake preview')).toHaveTextContent('0.000');
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toBeEnabled();
});

it('requires releasing and pressing a button that was held before detection', async () => {
    devices[0].buttons[2] = true;
    render(<ManualDrivingControls {...props} />);
    await flush();
    await detect('take over / return to AI');
    await poll();
    expect(screen.getByRole('combobox', { name: 'Take over / Return to AI button' })).toHaveValue('');
    devices[0].buttons[2] = false;
    await poll();
    devices[0].buttons[2] = true;
    await poll();
    expect(screen.getByRole('combobox', { name: 'Take over / Return to AI button' })).toHaveValue(JSON.stringify(['wheel', 2]));
});

it('disables takeover and other mapping changes while detecting', async () => {
    localStorage.setItem(MAPPING_KEY, JSON.stringify({
        steering: { deviceId: 'wheel', axis: 'axis0', inverted: false },
        throttle: { deviceId: 'pedals', axis: 'axis1', inverted: true },
        brake: { deviceId: 'pedals', axis: 'axis2', inverted: false },
    }));
    render(<ManualDrivingControls {...props} />);
    await flush();
    fireEvent.click(screen.getByRole('button', { name: 'Enable controller takeover' }));
    await detect('steering');
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', { name: 'Enable controller takeover' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Detect throttle' })).toBeDisabled();
    expect(screen.getByRole('combobox', { name: 'Brake' })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: 'Invert throttle' })).toBeDisabled();
});

it.each(['cancel', 'timeout', 'running', 'disconnect', 'unmount'])('ignores an in-flight reading after %s', async (stop) => {
    const saved = { steering: { deviceId: 'wheel', axis: 'axis0', inverted: true } };
    localStorage.setItem(MAPPING_KEY, JSON.stringify(saved));
    const view = render(<ManualDrivingControls {...props} />);
    await flush();
    await detect('steering');
    let resolve!: (result: PhysicalController[]) => void;
    read.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    await poll();
    if (stop === 'cancel') fireEvent.click(screen.getByRole('button', { name: 'Cancel detection' }));
    if (stop === 'timeout') await poll(15000);
    if (stop === 'running') view.rerender(<ManualDrivingControls {...props} running />);
    if (stop === 'disconnect') view.rerender(<ManualDrivingControls {...props} connected={false} />);
    if (stop === 'unmount') view.unmount();
    expect(Boolean(screen.queryByText(/No clear input detected/))).toBe(stop === 'timeout');
    await act(async () => { resolve([{ ...devices[0], axes: { axis0: 1 } }, devices[1]]); });
    expect(JSON.parse(localStorage.getItem(MAPPING_KEY)!)).toEqual(saved);
    const calls = read.mock.calls.length;
    if (['running', 'disconnect', 'unmount'].includes(stop)) await poll(500);
    expect(read).toHaveBeenCalledTimes(calls);
});

it.each(['unplug', 'error'])('stops detection on a controller %s without changing saved mappings', async (failure) => {
    render(<ManualDrivingControls {...props} />);
    await flush();
    await detect('throttle');
    if (failure === 'unplug') devices = [devices[0]];
    else read.mockRejectedValue(new Error('Cannot read physical controllers.'));
    await poll();
    expect(screen.queryByRole('button', { name: 'Cancel detection' })).not.toBeInTheDocument();
    expect(screen.getByRole('status', { name: 'Controller detection' })).toHaveTextContent(failure === 'unplug' ? 'Controller disconnected' : 'Detection stopped');
    expect(JSON.parse(localStorage.getItem(MAPPING_KEY)!)).toEqual({});
});
