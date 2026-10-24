import { detectAxisBinding, detectButtonBinding, DrivingControlSelector, ManualMapping, mapManualControls, PhysicalController } from './driving-control';

const devices: PhysicalController[] = [
    { id: 'wheel', name: 'Wheel', axes: { axis0: 0.75 }, buttons: [false, false, false, false] },
    { id: 'pedals', name: 'Pedals', axes: { axis1: 0.2, axis2: 0.7 }, buttons: [] },
];
const mapping: ManualMapping = {
    steering: { deviceId: 'wheel', axis: 'axis0', inverted: false },
    throttle: { deviceId: 'pedals', axis: 'axis1', inverted: true },
    brake: { deviceId: 'pedals', axis: 'axis2', inverted: true },
    shiftUp: { deviceId: 'wheel', button: 0 }, shiftDown: { deviceId: 'wheel', button: 1 },
    toggleControl: { deviceId: 'wheel', button: 2 },
};

it.each<[number, number, boolean]>([[0, 0.8, false], [1, 0.2, true], [0.5, 0.9, false], [0.5, 0.1, true]])(
    'detects axis movement from %s to %s and infers inversion %s', (start, end, inverted) => {
        const before: PhysicalController[] = [{ ...devices[1], axes: { axis1: start } }, devices[0]];
        const after: PhysicalController[] = [{ ...devices[0], axes: { axis0: 0.76 } }, { ...devices[1], axes: { axis1: end } }];
        expect(detectAxisBinding(before, after)).toEqual({ deviceId: 'pedals', axis: 'axis1', inverted });
    });

it('ignores jitter, ambiguous movement, invalid readings and devices without a baseline', () => {
    expect(detectAxisBinding(devices, [{ ...devices[0], axes: { axis0: 0.76 } }])).toBeNull();
    expect(detectAxisBinding(devices, [{ ...devices[1], axes: { axis1: 0.7, axis2: 0.2 } }])).toBeNull();
    expect(detectAxisBinding(devices, [{ ...devices[0], axes: { axis0: NaN, axis1: 1 } }])).toBeNull();
    expect(detectAxisBinding(devices, [{ ...devices[0], axes: { axis0: 2 } }])).toBeNull();
    expect(detectAxisBinding([], devices)).toBeNull();
});

it('detects one new button press by device identity, ignoring held, released and simultaneous buttons', () => {
    const pressed = [{ ...devices[0], buttons: [true, false, false, false] }];
    expect(detectButtonBinding([devices[1], devices[0]], pressed)).toEqual({ deviceId: 'wheel', button: 0 });
    expect(detectButtonBinding(pressed, pressed)).toBeNull();
    expect(detectButtonBinding(pressed, devices)).toBeNull();
    expect(detectButtonBinding(devices, [{ ...devices[0], buttons: [true, true, false, false] }])).toBeNull();
    expect(detectButtonBinding([], pressed)).toBeNull();
});

it('maps separate pedals, axis inversion and calibrated steering without changing the model observation', () => {
    const result = mapManualControls(devices, mapping);
    expect(result.action.throttle).toBeCloseTo(0.8);
    expect(result.action.brake).toBeCloseTo(0.3);
    expect(result.action.steering).toBeCloseTo(0.48 / 0.98);
    expect(result).toMatchObject({ toggleControl: false, action: { shiftUp: false, shiftDown: false } });
    expect(mapManualControls([{ ...devices[0], axes: { axis0: 0.501 } }, devices[1]], mapping).action.steering).toBe(0);
});

it('rejects unplugged devices, unavailable axes/buttons and conflicting handover buttons', () => {
    expect(() => mapManualControls([devices[0]], mapping)).toThrow(/disconnected/);
    expect(() => mapManualControls(devices, { ...mapping, throttle: { ...mapping.throttle!, axis: 'axis5' } })).toThrow(/unavailable/);
    expect(() => mapManualControls(devices, { ...mapping, toggleControl: { deviceId: 'wheel', button: 6 } })).toThrow(/unavailable/);
    expect(() => mapManualControls(devices, { ...mapping, toggleControl: mapping.shiftUp })).toThrow(/different button/);
});

it('latches takeover until explicit return, detects paddle edges and suppresses opposing shifts', () => {
    const selector = new DrivingControlSelector();
    const input = mapManualControls(devices, mapping);
    expect(selector.select(input).source).toBe('ai');
    expect(selector.select({ ...input, toggleControl: true }).source).toBe('manual');
    expect(selector.select(input).source).toBe('manual');
    const shifted = { ...input, action: { ...input.action, shiftUp: true } };
    expect(selector.select(shifted).action!.shiftUp).toBe(true);
    expect(selector.select(shifted).action!.shiftUp).toBe(false);
    expect(selector.select({ ...shifted, action: { ...shifted.action, shiftDown: true } }).action)
        .toMatchObject({ shiftUp: false, shiftDown: false });
    expect(() => selector.select(null)).toThrow(/unavailable/);
    expect(selector.select({ ...input, toggleControl: true }).source).toBe('ai');
    expect(selector.select(input).source).toBe('ai');
});

it('switches once per physical press in both directions, including while the button is held', () => {
    const selector = new DrivingControlSelector();
    const input = mapManualControls(devices, mapping);
    const pressed = mapManualControls([{ ...devices[0], buttons: [false, false, true, false] }, devices[1]], mapping);
    expect(pressed.toggleControl).toBe(true);
    expect(selector.select(pressed).source).toBe('manual');
    expect(selector.select(pressed).source).toBe('manual');
    expect(selector.select(input).source).toBe('manual');
    expect(selector.select(pressed).source).toBe('ai');
    expect(selector.select(pressed).source).toBe('ai');
    expect(selector.select(input).source).toBe('ai');
    expect(selector.select(pressed).source).toBe('manual');
});

it('allows takeover without AI observations and consumes blocked return presses until released', () => {
    const selector = new DrivingControlSelector();
    const input = mapManualControls(devices, mapping);
    const pressed = { ...input, toggleControl: true };
    expect(selector.select(pressed, false).source).toBe('manual');
    selector.select(input, false);
    expect(selector.select(pressed, false).source).toBe('manual');
    expect(selector.select(pressed, true).source).toBe('manual');
    selector.select(input);
    expect(selector.select(pressed).source).toBe('ai');
});
