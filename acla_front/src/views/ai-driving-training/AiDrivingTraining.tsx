import React, { useEffect, useRef, useState } from 'react';
import { useOptionalOperationComponentRefDirectory } from 'contexts/OperationComponentRefContext';
import { getVisualizationComponentName } from 'views/session-shared/visualization/visualization-component-names';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import type { TrackVisionHandle } from 'views/live-session/track-vision/LiveTrackVision';
import { NEUTRAL, ObservationResult, readDrivingObservation } from './driving-observation';
import { DrivingPolicy } from './driving-policy';
import { DrivingTrainingLoop, initialTrainingState } from './driving-training-loop';
import type { AccControlPreset, ControllerAction, VirtualController } from './virtual-controller';
import './AiDrivingTraining.css';

const STORAGE_KEY = 'acla.self-driving.policy.v2';
const TEST_CONTROLS = [
    { key: 'throttle', label: 'throttle', value: 0.25 },
    { key: 'brake', label: 'brake', value: 0.25 },
    { key: 'steering', label: 'steering', value: 0.25 },
    { key: 'shiftDown', label: 'shift down', value: true },
    { key: 'shiftUp', label: 'shift up', value: true },
] as const;
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Self-driving training failed.';

export default function AiDrivingTraining({ onOpenLiveSession }: { onOpenLiveSession: () => void }) {
    const directory = useOptionalOperationComponentRefDirectory();
    const directoryRef = useRef(directory);
    directoryRef.current = directory;
    const policy = useRef<DrivingPolicy | null>(null);
    const loop = useRef<DrivingTrainingLoop | null>(null);
    const controller = useRef<VirtualController | null>(null);
    const mounted = useRef(false);
    const connectingGeneration = useRef(0);
    const testTimer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
    const testMode = useRef<'hold' | 'timed' | null>(null);
    const [deviceId, setDeviceId] = useState(1);
    const [steeringTest, setSteeringTest] = useState(1);
    const [testStatus, setTestStatus] = useState('');
    const [connected, setConnected] = useState(false);
    const [connecting, setConnecting] = useState(false);
    const [installingPreset, setInstallingPreset] = useState(false);
    const [accPreset, setAccPreset] = useState<AccControlPreset | null>(null);
    const [state, setState] = useState(initialTrainingState);
    const [inputs, setInputs] = useState<ObservationResult>({ error: 'Waiting for Live Session inputs.' });
    const [modelSteps, setModelSteps] = useState<number | null>(null);
    const [error, setError] = useState('');
    const [saved, setSaved] = useState(false);
    const [testAction, setTestAction] = useState<ControllerAction>(NEUTRAL);
    const running = state.mode !== 'idle';

    const save = () => {
        if (!policy.current) return;
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(policy.current.checkpoint()));
            if (mounted.current) setSaved(true);
        } catch { if (mounted.current) setError('The model could not be saved locally. Export a checkpoint to keep it.'); }
    };
    const read = (): ObservationResult => {
        const vision = directoryRef.current?.findComponentRef<TrackVisionHandle>(getVisualizationComponentName('track-vision'))?.current;
        return readDrivingObservation(liveTelemetryStore.getSnapshot(), vision?.getLatestDetection() ?? null);
    };

    useEffect(() => {
        mounted.current = true;
        try {
            const savedModel = localStorage.getItem(STORAGE_KEY);
            if (savedModel) {
                policy.current = DrivingPolicy.restore(JSON.parse(savedModel));
                setModelSteps(policy.current.steps);
                setSaved(true);
            } else if (localStorage.getItem('acla.self-driving.policy.v1')) {
                setError('The saved model predates learned gear shifting. Create a new model to train with gear changes. The old checkpoint is preserved.');
            }
        } catch { setError('The saved model is incompatible. Create a new model or import a checkpoint.'); }
        const api = window.virtualController;
        if (api) {
            controller.current = {
                connect: (options) => api.connect(options), disconnect: () => api.disconnect(),
                apply: async (action) => {
                    try { await api.apply(action); }
                    catch (reason) { if (mounted.current) setConnected(false); throw reason; }
                },
            };
            loop.current = new DrivingTrainingLoop(controller.current, read, (next) => {
                if (!mounted.current) return;
                setState(next);
                setModelSteps(policy.current?.steps ?? null);
                if (next.mode === 'idle' || (next.steps > 0 && next.steps % 50 === 0)) save();
            });
        }
        const timer = setInterval(() => setInputs(read()), 250);
        const releaseTest = () => {
            clearInterval(testTimer.current);
            testTimer.current = undefined;
            testMode.current = null;
            if (mounted.current) { setTestAction(NEUTRAL); setTestStatus(''); }
        };
        const emergencyStop = () => {
            releaseTest();
            void loop.current?.stop('Emergency stop. Controller released.');
        };
        const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') emergencyStop(); };
        const onBlur = () => {
            if (testMode.current !== 'hold') return;
            releaseTest();
            void api?.apply(NEUTRAL).catch((reason) => { if (mounted.current) setError(errorMessage(reason)); });
        };
        const onUnload = () => { emergencyStop(); void api?.disconnect(); };
        window.addEventListener('keydown', onKey);
        window.addEventListener('blur', onBlur);
        window.addEventListener('beforeunload', onUnload);
        return () => {
            mounted.current = false;
            // This ref is a cancellation counter, not a DOM node.
            // eslint-disable-next-line react-hooks/exhaustive-deps
            connectingGeneration.current++;
            releaseTest();
            clearInterval(timer);
            window.removeEventListener('keydown', onKey);
            window.removeEventListener('blur', onBlur);
            window.removeEventListener('beforeunload', onUnload);
            void loop.current?.stop();
            void api?.disconnect().catch(() => undefined);
            save();
        };
        // The live component directory is read through a ref, without restarting a running episode.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const connect = async () => {
        const generation = ++connectingGeneration.current;
        setConnecting(true); setError('');
        try {
            if (!controller.current) throw new Error('Virtual control requires the Windows desktop app.');
            await controller.current.connect({ backend: 'vjoy', deviceId });
            if (mounted.current && generation === connectingGeneration.current) setConnected(true);
        } catch (reason) { if (mounted.current) setError(errorMessage(reason)); }
        finally { if (mounted.current && generation === connectingGeneration.current) setConnecting(false); }
    };
    const installAccPreset = async () => {
        setInstallingPreset(true); setAccPreset(null); setError('');
        try {
            if (!window.virtualController?.installAccPreset) throw new Error('ACC preset installation requires the updated Windows desktop app. Restart the app after updating.');
            const preset = await window.virtualController.installAccPreset(deviceId);
            if (mounted.current) setAccPreset(preset);
        } catch (reason) { if (mounted.current) setError(errorMessage(reason)); }
        finally { if (mounted.current) setInstallingPreset(false); }
    };
    const stopTest = () => {
        clearInterval(testTimer.current); testTimer.current = undefined; setTestAction(NEUTRAL);
        testMode.current = null; setTestStatus('');
        void controller.current?.apply(NEUTRAL).catch((reason) => { if (mounted.current) setError(errorMessage(reason)); });
    };
    const disconnect = async () => {
        stopTest();
        await loop.current?.stop();
        await controller.current?.disconnect();
        setConnected(false);
    };
    const start = (train: boolean) => {
        setError('');
        try {
            const input = read();
            if (!input.observation) throw new Error(input.error);
            if (!policy.current) policy.current = new DrivingPolicy(input.observation.context, input.observation.features.length);
            setSaved(false);
            loop.current?.start(policy.current, train);
        } catch (reason) { setError(errorMessage(reason)); }
    };
    const newModel = () => {
        policy.current = null; setModelSteps(null); setSaved(false); setError('');
        try { localStorage.removeItem(STORAGE_KEY); } catch { setError('Unable to remove the locally saved model.'); }
    };
    const exportModel = () => {
        if (!policy.current) return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(policy.current.checkpoint())], { type: 'application/json' }));
        const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'acla-driving-policy.json'; anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    const importModel = async (file?: File) => {
        if (!file) return;
        try {
            if (file.size > 2000000) throw new Error('The checkpoint is too large.');
            const imported = DrivingPolicy.restore(JSON.parse(await file.text()));
            if (!mounted.current || loop.current?.state.mode !== 'idle') return;
            policy.current = imported; setModelSteps(imported.steps); setError(''); save();
        } catch (reason) { setError(errorMessage(reason)); }
    };
    const test = (action: ControllerAction) => {
        if (running || !connected || testTimer.current) return;
        testMode.current = 'hold';
        setTestAction(action);
        const apply = () => { void controller.current?.apply(action).catch((reason) => {
            clearInterval(testTimer.current); testTimer.current = undefined;
            testMode.current = null;
            if (mounted.current) { setTestAction(NEUTRAL); setTestStatus(''); setError(errorMessage(reason)); }
        }); };
        apply(); testTimer.current = setInterval(apply, 100);
    };
    const timedTest = ({ key, label, value }: typeof TEST_CONTROLS[number]) => {
        if (running || !connected || testTimer.current) return;
        testMode.current = 'timed';
        setError('');
        const startsAt = Date.now() + 5000;
        const action = { ...NEUTRAL, [key]: typeof value === 'boolean' ? true : key === 'steering' ? steeringTest : 1 };
        const duration = typeof value === 'boolean' ? 300 : 10000;
        let busy = false;
        const tick = () => {
            const elapsed = Date.now() - startsAt;
            if (elapsed >= duration) { stopTest(); return; }
            setTestStatus(elapsed < 0 ? `${label} test starts in ${Math.ceil(-elapsed / 1000)}s — switch to simulator controls.`
                : `${label} test releases in ${Math.ceil((duration - elapsed) / 1000)}s.`);
            if (busy) return;
            const next = elapsed < 0 ? NEUTRAL : action;
            setTestAction(next);
            busy = true;
            void controller.current?.apply(next).catch((reason) => {
                clearInterval(testTimer.current); testTimer.current = undefined; testMode.current = null;
                if (mounted.current) { setTestAction(NEUTRAL); setTestStatus(''); setError(errorMessage(reason)); }
            }).finally(() => { busy = false; });
        };
        tick(); testTimer.current = setInterval(tick, 100);
    };
    const output: ControllerAction = running ? state.action : testAction;
    const observation = inputs.observation;
    const mismatched = observation && policy.current && !policy.current.matches(observation);

    return <section className="driving-training" aria-label="AI Self-Driving Training">
        <header className="driving-training-header">
            <div><span className="driving-eyebrow">SIMULATOR LAB</span><h1>AI Self-Driving Training</h1>
                <p>Learn to follow the reference line with smooth, stable control.</p></div>
            <button onClick={onOpenLiveSession}>Open Live Session</button>
        </header>
        <div className="driving-status" role="status"><span data-active={running} />{state.status}</div>
        {error && <p className="driving-error" role="alert">{error}</p>}
        <div className="driving-grid">
            <section className="driving-card">
                <h2>Live inputs</h2>
                <p>Start live telemetry, add Live Map and Track Vision, then start screen capture in Live Session.</p>
                <div className="driving-input-status" data-ready={Boolean(observation)}>{observation ? 'All model inputs ready' : inputs.error}</div>
                <dl className="driving-details">
                    <dt>Game / car</dt><dd>{observation ? `${observation.context.game} / ${observation.context.car}` : 'Waiting'}</dd>
                    <dt>Map</dt><dd>{observation?.context.map ?? 'Waiting'}</dd>
                    <dt>Speed</dt><dd>{observation ? `${observation.speed.toFixed(1)} km/h` : '—'}</dd>
                    <dt>Gear / RPM</dt><dd>{observation ? `${observation.gear < 0 ? 'R' : observation.gear === 0 ? 'N' : observation.gear} / ${observation.rpm}` : '—'}</dd>
                    <dt>Lap position</dt><dd>{observation ? `${(observation.normalizedPosition * 100).toFixed(1)}%` : '—'}</dd>
                    <dt>Reference error</dt><dd>{observation ? `${observation.referenceDistance.toFixed(2)} m` : '—'}</dd>
                    <dt>Position + segmentation</dt><dd>{observation ? `${observation.context.classes.length} classes · ${observation.features.length} features` : 'Waiting'}</dd>
                </dl>
                <p className="driving-note">The policy uses velocity XYZ, speed, current gear, engine RPM, game, car, map, normalized position, rotation, current controls, position-model deltas and segmentation masks.</p>
            </section>
            <section className="driving-card">
                <div className="driving-card-title"><h2>Steering wheel (vJoy)</h2><span>{connected ? 'Connected' : 'Disconnected'}</span></div>
                <div className="driving-output-options">
                    <label htmlFor="driving-device">vJoy device</label>
                    <select id="driving-device" value={deviceId} disabled={connected || connecting || installingPreset}
                        onChange={(event) => { setDeviceId(Number(event.target.value)); setAccPreset(null); }}>
                        {Array.from({ length: 16 }, (_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}
                    </select>
                </div>
                <p>Works independently of your physical wheel brand. Bind vJoy X → steering, Y → throttle, Z → brake, button 1 → shift up, and button 2 → shift down in the simulator.</p>
                <p className="driving-note">vJoy is included with the Windows installer. Restart Windows if setup requests it, then enable X, Y, Z and at least 2 buttons on the selected device in Configure vJoy. In iRacing, use Configure Controls or Settings → Driving → Input Calibration.</p>
                <div className="driving-buttons"><button onClick={() => { void installAccPreset(); }}
                    disabled={installingPreset || connecting || running || Boolean(testTimer.current)}>
                    {installingPreset ? 'Installing ACC preset…' : 'Install ACC preset'}</button></div>
                <p className="driving-note">For ACC, save your current controls, install the training preset, then load it in Options → Controls → Wheel. This automatically maps steering, throttle, brake, shift up (button 1), and shift down (button 2) on the selected vJoy device. If the preset is missing, restart ACC. Load your normal preset again for manual driving.</p>
                {accPreset && <p role="status">Installed “{accPreset.name}”. In ACC, open Options → Controls → Wheel and load this preset, then check all three input meters and both shift buttons with the timed tests.</p>}
                <div className="driving-controls">
                    {(['throttle', 'brake', 'steering'] as const).map((key) => <div key={key}>
                        <div><label htmlFor={`driving-${key}`}>{key}</label><output>{output[key].toFixed(3)}</output></div>
                        <meter id={`driving-${key}`} min={key === 'steering' ? -1 : 0} max={1} value={output[key]} />
                    </div>)}
                    {(['shiftDown', 'shiftUp'] as const).map((key) => <div key={key}>
                        <div><span>{key === 'shiftUp' ? 'Shift up · button 1' : 'Shift down · button 2'}</span>
                            <output aria-label={key === 'shiftUp' ? 'Shift up output' : 'Shift down output'}>{output[key] ? 'Pressed' : 'Released'}</output></div>
                    </div>)}
                </div>
                <div className="driving-buttons"><button onClick={() => { void connect(); }} disabled={connected || connecting}>{connecting ? 'Connecting…' : 'Connect wheel'}</button>
                    <button onClick={() => { void disconnect().catch((reason) => setError(errorMessage(reason))); }} disabled={!connected}>Disconnect</button></div>
                <div className="driving-buttons" aria-label="Test controller controls">
                    {TEST_CONTROLS.map(({ key, label, value }) => <button key={key} disabled={!connected || running || testMode.current === 'timed'}
                        aria-pressed={typeof value === 'boolean' ? Boolean(output[key]) : undefined}
                        onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); test({ ...NEUTRAL, [key]: value }); }}
                        onPointerUp={() => { if (testMode.current === 'hold') stopTest(); }}
                        onPointerCancel={() => { if (testMode.current === 'hold') stopTest(); }}
                        onLostPointerCapture={() => { if (testMode.current === 'hold') stopTest(); }}
                        onBlur={() => { if (testMode.current === 'hold') stopTest(); }}
                        onKeyDown={(event) => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); test({ ...NEUTRAL, [key]: value }); } }}
                        onKeyUp={(event) => { if (testMode.current === 'hold' && (event.key === ' ' || event.key === 'Enter')) stopTest(); }}>
                        Hold to test {label}</button>)}
                </div>
                <div className="driving-output-options">
                    <label htmlFor="driving-steering-test">Timed steering position</label>
                    <select id="driving-steering-test" value={steeringTest} disabled={running || Boolean(testTimer.current)}
                        onChange={(event) => setSteeringTest(Number(event.target.value))}>
                        <option value={-1}>Full left</option><option value={1}>Full right</option>
                        <option value={0}>Center</option><option value={-0.2}>20% left (90° for a 900° range)</option>
                        <option value={0.2}>20% right (90° for a 900° range)</option>
                    </select>
                </div>
                <div className="driving-buttons" aria-label="Timed controller tests">
                    {TEST_CONTROLS.map((control) => <button key={control.key}
                        disabled={!connected || running || Boolean(testTimer.current)} onClick={() => timedTest(control)}>Test {control.label} in 5 seconds</button>)}
                </div>
                <p className="driving-note">Use timed tests in the simulator’s controls setup: switch windows during the 5-second countdown. Axes stay active for 10 seconds; shift buttons press once for 0.3 seconds, then release. Pedal tests use the full range. For iRacing steering calibration, test full left, full right, then center; use 20% left for its 90° step with a 900° range.</p>
                {testStatus && <p role="status">{testStatus}</p>}
                <p className="driving-note">Connected means the virtual device is ready; verify the simulator’s input meters before training. Stop or Escape in this app cancels tests. Training controls release on stale input or capture loss.</p>
            </section>
            <section className="driving-card">
                <h2>Reinforcement learning model</h2>
                <p>Online actor–critic · 32 hidden units · continuous throttle, brake and steering, plus learned hold gear, shift up and shift down.</p>
                <p className="driving-note">Disable automatic gears in the simulator and bind both shift buttons before training. The model learns gear choices from driving rewards. Shifts use 0.1-second presses with at least 0.5 seconds between shifts. Downshifts are allowed in every gear, including neutral and reverse.</p>
                <div className="driving-metrics"><div><strong>{modelSteps ?? 0}</strong><span>Learning steps</span></div>
                    <div><strong>{state.episodeReward.toFixed(2)}</strong><span>Episode reward</span></div></div>
                {mismatched && <p className="driving-error">This model belongs to different inputs. Create a new model or import a matching checkpoint.</p>}
                <div className="driving-buttons">
                    <button className="driving-primary" disabled={!connected || !observation || running || Boolean(mismatched) || Boolean(testTimer.current)} onClick={() => start(true)}>Start training</button>
                    <button disabled={!connected || !observation || running || !modelSteps || Boolean(mismatched) || Boolean(testTimer.current)} onClick={() => start(false)}>Run trained model</button>
                    <button className="driving-stop" disabled={!running && !testTimer.current} onClick={() => { stopTest(); void loop.current?.stop(); }}>Stop</button>
                </div>
                <div className="driving-buttons">
                    <button disabled={running} onClick={newModel}>New model</button>
                    <button disabled={modelSteps === null || running} onClick={exportModel}>Export model</button>
                    <label className="driving-import">Import model<input aria-label="Import model checkpoint" type="file" accept=".json,application/json" disabled={running}
                        onChange={(event) => { void importModel(event.target.files?.[0]); event.target.value = ''; }} /></label>
                </div>
                <p className="driving-note">{saved ? 'Checkpoint saved on this device.' : 'Checkpoints save every 50 learning steps and when stopped.'} Training explores controls; use a practice session and reposition the car between episodes. A new policy starts untrained.</p>
            </section>
            <section className="driving-card">
                <h2>Reward signal</h2>
                <dl className="driving-rewards">
                    <div><dt>Drive fast</dt><dd>{state.reward?.speed.toFixed(3) ?? '0.000'}</dd><p>Covering the same section faster earns a larger speed bonus. It requires forward lap progress and decreases farther from the reference line.</p></div>
                    <div><dt>Follow the reference line</dt><dd>{state.reward?.line.toFixed(3) ?? '0.000'}</dd><p>Forward lap progress earns more reward closer to the reference line. Standing still earns no progress reward.</p></div>
                    <div><dt>Converge smoothly</dt><dd>{state.reward?.convergence.toFixed(3) ?? '0.000'}</dd><p>Reducing position error earns reward, weighted by how smoothly the car approaches the line.</p></div>
                    <div><dt>Penalize instability</dt><dd>{state.reward?.instability.toFixed(3) ?? '0.000'}</dd><p>Excessive yaw, pitch and roll changes, and overlapping pedals reduce reward.</p></div>
                </dl>
                <p className="driving-note">Rewards use the position model’s reference delta. Stop or input loss ends the episode; learning resumes only after Start training.</p>
            </section>
        </div>
    </section>;
}
