import { clamp, DrivingAction } from './driving-observation';

export interface PhysicalController {
    id: string;
    name: string;
    /** Device axes normalized to 0–1, with centered sticks at 0.5. */
    axes: Record<string, number>;
    buttons: boolean[];
}
export interface AxisBinding { deviceId: string; axis: string; inverted: boolean }
export interface ButtonBinding { deviceId: string; button: number }
export interface ManualMapping {
    steering?: AxisBinding;
    throttle?: AxisBinding;
    brake?: AxisBinding;
    shiftUp?: ButtonBinding;
    shiftDown?: ButtonBinding;
    toggleControl?: ButtonBinding;
}
export interface ManualControls { action: DrivingAction; toggleControl: boolean }
export type ControlSource = 'ai' | 'manual';

/** Detect deliberate movement from rest, ignoring axis jitter and ambiguous inputs. */
export function detectAxisBinding(baseline: PhysicalController[], devices: PhysicalController[]): AxisBinding | null {
    const candidates: { binding: AxisBinding; movement: number }[] = [];
    for (const device of devices) {
        const before = baseline.find((candidate) => candidate.id === device.id);
        if (!before) continue;
        for (const [axis, value] of Object.entries(device.axes)) {
            const start = before.axes[axis];
            if (![start, value].every((sample) => Number.isFinite(sample) && sample >= 0 && sample <= 1)) continue;
            const delta = value - start;
            if (Math.abs(delta) >= 0.2) candidates.push({
                binding: { deviceId: device.id, axis, inverted: delta < 0 }, movement: Math.abs(delta),
            });
        }
    }
    candidates.sort((a, b) => b.movement - a.movement);
    if (!candidates.length || (candidates[1] && candidates[0].movement - candidates[1].movement < 0.1)) return null;
    return candidates[0].binding;
}

/** Only a fresh press can bind a button, including one held when detection started. */
export function detectButtonBinding(previous: PhysicalController[], devices: PhysicalController[]): ButtonBinding | null {
    const candidates: ButtonBinding[] = [];
    for (const device of devices) {
        const before = previous.find((candidate) => candidate.id === device.id);
        if (!before) continue;
        device.buttons.forEach((pressed, button) => {
            if (pressed && before.buttons[button] === false) candidates.push({ deviceId: device.id, button });
        });
    }
    return candidates.length === 1 ? candidates[0] : null;
}

export function mapManualControls(devices: PhysicalController[], mapping: ManualMapping): ManualControls {
    const device = (id: string) => {
        const found = devices.find((candidate) => candidate.id === id);
        if (!found) throw new Error('Mapped controller disconnected. Controls released; reconnect and restart.');
        return found;
    };
    const axis = (key: 'steering' | 'throttle' | 'brake') => {
        const binding = mapping[key];
        if (!binding) throw new Error('Map steering, throttle and brake before enabling takeover.');
        const raw = device(binding.deviceId).axes[binding.axis];
        if (!Number.isFinite(raw) || raw < 0 || raw > 1) throw new Error(`Mapped ${key} axis is unavailable.`);
        const value = binding.inverted ? 1 - raw : raw;
        if (key !== 'steering') return value;
        const centered = value * 2 - 1;
        return Math.sign(centered) * clamp((Math.abs(centered) - 0.02) / 0.98, 0, 1);
    };
    const button = (key: 'shiftUp' | 'shiftDown' | 'toggleControl') => {
        const binding = mapping[key];
        if (!binding) return false;
        const pressed = device(binding.deviceId).buttons[binding.button];
        if (typeof pressed !== 'boolean') throw new Error(`Mapped ${key} button is unavailable.`);
        return pressed;
    };
    const bindings = [mapping.shiftUp, mapping.shiftDown, mapping.toggleControl].filter(Boolean);
    if (new Set(bindings.map((binding) => JSON.stringify([binding!.deviceId, binding!.button]))).size !== bindings.length) {
        throw new Error('Use a different button for each shift and the control toggle.');
    }
    return { action: { steering: axis('steering'), throttle: axis('throttle'), brake: axis('brake'),
        shiftUp: button('shiftUp'), shiftDown: button('shiftDown') },
        toggleControl: button('toggleControl') };
}

/** One latched owner of the output. Physical input is never an observation feature. */
export class DrivingControlSelector {
    source: ControlSource = 'ai';
    private held = { toggleControl: false, shiftUp: false, shiftDown: false };

    select(input: ManualControls | null, allowAI = true): { source: ControlSource; action?: DrivingAction } {
        if (!input) {
            if (this.source === 'manual') throw new Error('Manual controller is unavailable. Controls released.');
            return { source: 'ai' };
        }
        // Switch once per press; a blocked return must be released before retrying.
        if (input.toggleControl && !this.held.toggleControl) {
            if (this.source === 'ai') this.source = 'manual';
            else if (allowAI) this.source = 'ai';
        }
        const shiftUp = Boolean(input.action.shiftUp), shiftDown = Boolean(input.action.shiftDown);
        const action = { ...input.action,
            shiftUp: shiftUp && !shiftDown && !this.held.shiftUp,
            shiftDown: shiftDown && !shiftUp && !this.held.shiftDown };
        this.held = { toggleControl: input.toggleControl, shiftUp, shiftDown };
        return { source: this.source, action: this.source === 'manual' ? action : undefined };
    }
}
