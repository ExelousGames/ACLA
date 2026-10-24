import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { ControlSource, detectAxisBinding, detectButtonBinding, ManualControls, ManualMapping, mapManualControls, PhysicalController } from './driving-control';

const MAPPING_KEY = 'acla.self-driving.manual-mapping.v1';
const AXES = ['steering', 'throttle', 'brake'] as const;
const BUTTONS = ['shiftUp', 'shiftDown', 'toggleControl'] as const;
const LABELS = { steering: 'Steering', throttle: 'Throttle', brake: 'Brake', shiftUp: 'Shift up',
    shiftDown: 'Shift down', toggleControl: 'Take over / Return to AI' };
const DETECTION_PROMPTS = { steering: 'Turn the wheel to the right or move your steering stick right.', throttle: 'Press the throttle pedal or trigger fully.',
    brake: 'Press the brake pedal or trigger fully.', shiftUp: 'Press your shift up button or paddle.',
    shiftDown: 'Press your shift down button or paddle.', toggleControl: 'Press the button to switch between human and AI control.' };

function savedMapping(): ManualMapping {
    try {
        const data = JSON.parse(localStorage.getItem(MAPPING_KEY) || '{}');
        if (!data || typeof data !== 'object' || Array.isArray(data)) return {};
        const toggleControl = data.toggleControl ?? data.takeover ?? data.resume;
        delete data.takeover;
        delete data.resume;
        if (toggleControl !== undefined) data.toggleControl = toggleControl;
        for (const [key, value] of Object.entries(data) as [string, any][]) {
            if (!value || typeof value.deviceId !== 'string') return {};
            if (AXES.includes(key as any)) {
                if (typeof value.axis !== 'string' || !/^axis\d+$/.test(value.axis) || typeof value.inverted !== 'boolean') return {};
            } else if (!BUTTONS.includes(key as any) || !Number.isSafeInteger(value.button) || value.button < 0) return {};
        }
        return data;
    } catch { return {}; }
}

export interface ManualDrivingControlsHandle { read(): Promise<ManualControls | null> }

const ManualDrivingControls = forwardRef<ManualDrivingControlsHandle, {
    connected: boolean; running: boolean; source: ControlSource | null; onSourceChange(source: ControlSource): void;
    testing?: boolean; aiActive?: boolean; aiReady?: boolean;
}>(function ManualDrivingControls({ connected, running, source, onSourceChange, testing = false, aiActive = running, aiReady = true }, ref) {
    const [mapping, setMapping] = useState(savedMapping);
    const [enabled, setEnabled] = useState(false);
    const [devices, setDevices] = useState<PhysicalController[]>([]);
    const [error, setError] = useState('');
    const [detecting, setDetecting] = useState<keyof ManualMapping | null>(null);
    const [detectionMessage, setDetectionMessage] = useState('');
    const mounted = useRef(false);
    useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);

    const readDevices = async () => {
        const api = window.virtualController;
        if (!api?.readPhysicalControls) throw new Error('Controller takeover requires the updated Windows desktop app.');
        const result = await api.readPhysicalControls();
        if (mounted.current) { setDevices(result); setError(''); }
        return result;
    };
    useImperativeHandle(ref, () => ({ read: async () => {
        if (!enabled) return null;
        return mapManualControls(await readDevices(), mapping);
    } }));

    const changeMapping = useCallback((next: ManualMapping | ((current: ManualMapping) => ManualMapping)) => {
        setMapping(next); setEnabled(false); setDetectionMessage('');
    }, []);
    useEffect(() => {
        try { localStorage.setItem(MAPPING_KEY, JSON.stringify(mapping)); }
        catch { setError('Controller mapping could not be saved on this device.'); }
    }, [mapping]);

    useEffect(() => {
        if (!connected || running) {
            if (!connected) setDevices([]);
            setDetecting(null); setDetectionMessage('');
            return;
        }
        if (!window.virtualController?.readPhysicalControls) return;
        let cancelled = false, busy = false;
        let baseline: PhysicalController[] | undefined, previous: PhysicalController[] | undefined;
        const finishDetection = (message: string) => {
            cancelled = true;
            setDetecting(null); setDetectionMessage(message);
        };
        const poll = async () => {
            if (busy || cancelled) return;
            busy = true;
            try {
                const result = await window.virtualController!.readPhysicalControls!();
                if (cancelled) return;
                setDevices(result); setError('');
                if (detecting) {
                    if (!result.length || baseline?.some((device) => !result.some((current) => current.id === device.id))) {
                        finishDetection('Controller disconnected. Reconnect and try Detect again.');
                        return;
                    }
                    if (baseline && previous) {
                        const binding = AXES.includes(detecting as typeof AXES[number])
                            ? detectAxisBinding(baseline, result) : detectButtonBinding(previous, result);
                        if (binding) {
                            changeMapping((current) => ({ ...current, [detecting]: binding }));
                            const device = result.find((candidate) => candidate.id === binding.deviceId)!;
                            const input = 'axis' in binding ? `axis ${binding.axis.toUpperCase()}` : `button ${binding.button + 1}`;
                            finishDetection(`${LABELS[detecting]} mapped to ${device.name} · ${input}.`);
                            return;
                        }
                    }
                    baseline ??= result;
                    previous = result;
                }
            } catch (reason) {
                if (!cancelled) {
                    setDevices([]); setError(reason instanceof Error ? reason.message : 'Cannot read the controller.');
                    if (detecting) finishDetection('Detection stopped. Check the controller connection and try again.');
                }
            } finally { busy = false; }
        };
        void poll();
        const timer = setInterval(() => { void poll(); }, detecting ? 50 : 250);
        const timeout = detecting ? setTimeout(() => {
            finishDetection('No clear input detected. Center the wheel or stick, release pedals, triggers and buttons, then try Detect again.');
        }, 15000) : undefined;
        return () => { cancelled = true; clearInterval(timer); clearTimeout(timeout); };
    }, [connected, running, detecting, changeMapping]);

    const detectButton = (key: keyof ManualMapping) => <button type="button" aria-label={`Detect ${LABELS[key].toLowerCase()}`}
        disabled={!devices.length || detecting !== null} onClick={() => {
            setEnabled(false); setDetectionMessage(''); setDetecting(key);
        }}>{detecting === key ? 'Detecting…' : 'Detect'}</button>;
    let preview: ManualControls | undefined, mappingError = '';
    try { preview = mapManualControls(devices, mapping); }
    catch (reason) { mappingError = reason instanceof Error ? reason.message : 'Check the controller mapping.'; }
    const humanDriving = source === 'manual';

    return <section className="driving-card driving-manual" aria-label="Human takeover">
        <div className="driving-card-title"><h2>Human takeover</h2><span>{running ? source === 'manual' ? 'You are driving' : 'AI is driving' : enabled && preview ? 'Ready' : 'Set up your controls'}</span></div>
        <p>Drive through the same vJoy wheel as the AI. Human driving works without a model, telemetry or screen capture.</p>
        <div className="driving-buttons">
            <button type="button" className="driving-primary" aria-pressed={enabled}
                disabled={running || testing || !connected || detecting !== null || (!enabled && !preview)}
                onClick={() => setEnabled((current) => !current)}>
                {enabled && <span aria-hidden="true">✓&nbsp;</span>}Enable controller takeover</button>
            <button className="driving-primary" disabled={!connected || testing || (humanDriving
                ? !running || !aiActive || !aiReady : !enabled || !preview)}
                onClick={() => onSourceChange(humanDriving ? 'ai' : 'manual')}>
                {humanDriving ? 'Return to AI' : running ? 'Take over' : 'Start human driving'}</button>
        </div>
        <p>Connect your physical gamepad, wheel, joystick or pedals by USB or Bluetooth. Center steering and release pedals and buttons, then click Detect and follow the prompt. Check the live preview and enable controller takeover.</p>
        <fieldset disabled={running || testing || !connected}>
            <legend>Physical controller mapping</legend>
            {(detecting || detectionMessage) && <p role="status" aria-label="Controller detection">{detecting ? `${LABELS[detecting]}: ${DETECTION_PROMPTS[detecting]} Move only this control; keep a button pressed until detected.` : detectionMessage}</p>}
            {detecting && <button type="button" onClick={() => { setDetecting(null); setDetectionMessage('Detection cancelled.'); }}>Cancel detection</button>}
            {!devices.length && <p>{connected ? 'No physical controllers detected. Connect or pair your controller and check that Windows recognizes it. Detection refreshes automatically; vJoy outputs are excluded.' : 'Connect the vJoy output above to start physical controller detection.'}</p>}
            {AXES.map((key) => <div className="driving-manual-axis" key={key}>
                <label htmlFor={`manual-${key}`}>{LABELS[key]}</label>
                <div className="driving-manual-binding"><select id={`manual-${key}`} disabled={detecting !== null}
                    value={mapping[key] ? JSON.stringify([mapping[key]!.deviceId, mapping[key]!.axis]) : ''}
                    onChange={(event) => {
                        const [deviceId, axis] = event.target.value ? JSON.parse(event.target.value) : [];
                        changeMapping({ ...mapping, [key]: deviceId ? { deviceId, axis, inverted: false } : undefined });
                    }}>
                    <option value="">Choose axis</option>
                    {mapping[key] && !devices.some((device) => device.id === mapping[key]!.deviceId)
                        && <option value={JSON.stringify([mapping[key]!.deviceId, mapping[key]!.axis])}>Mapped device disconnected</option>}
                    {devices.flatMap((device) => Object.keys(device.axes).map((axis) =>
                        <option key={`${device.id}:${axis}`} value={JSON.stringify([device.id, axis])}>{device.name} · {axis.toUpperCase()} · {(device.axes[axis] * 100).toFixed(0)}%</option>))}
                </select>{detectButton(key)}</div>
                <label className="driving-manual-invert"><input type="checkbox" checked={mapping[key]?.inverted ?? false} disabled={!mapping[key] || detecting !== null}
                    onChange={(event) => changeMapping({ ...mapping, [key]: { ...mapping[key]!, inverted: event.target.checked } })} />Invert {LABELS[key].toLowerCase()}</label>
                <output aria-label={`Manual ${key} preview`}>{preview ? preview.action[key].toFixed(3) : '—'}</output>
            </div>)}
            <div className="driving-output-options">{BUTTONS.map((key) => <React.Fragment key={key}>
                <label htmlFor={`manual-${key}`}>{LABELS[key]} button</label>
                <div className="driving-manual-binding"><select id={`manual-${key}`} disabled={detecting !== null}
                    value={mapping[key] ? JSON.stringify([mapping[key]!.deviceId, mapping[key]!.button]) : ''}
                    onChange={(event) => {
                        const [deviceId, button] = event.target.value ? JSON.parse(event.target.value) : [];
                        changeMapping({ ...mapping, [key]: deviceId ? { deviceId, button } : undefined });
                    }}>
                    <option value="">Not mapped</option>
                    {mapping[key] && !devices.some((device) => device.id === mapping[key]!.deviceId)
                        && <option value={JSON.stringify([mapping[key]!.deviceId, mapping[key]!.button])}>Mapped device disconnected</option>}
                    {devices.flatMap((device) => device.buttons.map((pressed, button) =>
                        <option key={`${device.id}:${button}`} value={JSON.stringify([device.id, button])}>{device.name} · Button {button + 1}{pressed ? ' · Pressed' : ''}</option>))}
                </select>{detectButton(key)}</div>
            </React.Fragment>)}</div>
        </fieldset>
        {(error || (connected && !detecting && mappingError)) && <p className="driving-error">{error || mappingError}</p>}
        <p className="driving-note">Takeover stays active until you return control to AI. Map the Take over / Return to AI button to switch while the simulator has focus. Each press switches control once; release before pressing again. Returning to AI requires an active AI session with ready inputs. Steering, pedals and mapped shift buttons all pass through the same vJoy device; keep the simulator bound to vJoy. Invert pedals if their preview is 1 at rest. Calibrate axis ranges in Windows controller settings.</p>
        <p className="driving-note">During training, your applied controls teach the model. Missing AI observations pause learning while you keep driving. Evaluation and standalone human driving never update weights. A mapped controller disconnect releases controls. Manual shifts use one press per paddle pull and the same cooldown as AI.</p>
    </section>;
});

export default ManualDrivingControls;
