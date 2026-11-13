import React, { useEffect, useRef, useState } from 'react';
import { useOptionalOperationComponentRefDirectory } from 'contexts/OperationComponentRefContext';
import { getVisualizationComponentName } from 'views/session-shared/visualization/visualization-component-names';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import type { TrackVisionHandle } from 'views/live-session/track-vision/LiveTrackVision';
import type { LiveTrajectoryMapHandle } from 'views/live-session/LiveTrajectoryMap';
import { NEUTRAL, ObservationResult, readDrivingObservation } from './driving-observation';
import { DrivingPolicy, DrivingTrainingDocument, emptyTrainingDocument } from './driving-policy';
import { cloneConfig } from './driving-model-config';
import { discardBrowserModels, drivingModelFiles, loadDrivingModel, saveDrivingModel, trackDrivingModelSave } from './driving-model-storage';
import { DrivingTrainingLoop, initialTrainingState } from './driving-training-loop';
import { DrivingControlOutput } from './driving-control-output';
import type { ControlSource } from './driving-control';
import { aggressionRadius } from './driving-reference';
import { IMPACT_CONFIDENCE_THRESHOLD } from './driving-impact';
import { DrivingImpactMonitor } from './driving-impact-monitor';
import type { AccControlPreset, ControllerAction, VirtualController } from './virtual-controller';
import ManualDrivingControls, { ManualDrivingControlsHandle } from './ManualDrivingControls';
import ObservationInputDelays, { ObservationDelaySnapshot } from './ObservationInputDelays';
import './AiDrivingTraining.css';

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
    const configuration = useRef(cloneConfig());
    const documentReady = useRef(false);
    const loop = useRef<DrivingTrainingLoop | null>(null);
    const impactMonitor = useRef(new DrivingImpactMonitor());
    const controller = useRef<VirtualController | null>(null);
    const controlOutput = useRef<DrivingControlOutput | null>(null);
    const manualControls = useRef<ManualDrivingControlsHandle>(null);
    const mounted = useRef(false);
    const connectingGeneration = useRef(0);
    const startGeneration = useRef(0);
    const testTimer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
    const testMode = useRef<'hold' | 'timed' | null>(null);
    const [deviceId, setDeviceId] = useState(1);
    const [steeringTest, setSteeringTest] = useState(1);
    const [testStatus, setTestStatus] = useState('');
    const [connected, setConnected] = useState(false);
    const [connecting, setConnecting] = useState(false);
    const [disconnecting, setDisconnecting] = useState(false);
    const [installingPreset, setInstallingPreset] = useState(false);
    const [accPreset, setAccPreset] = useState<AccControlPreset | null>(null);
    const [state, setState] = useState(initialTrainingState);
    const [inputs, setInputs] = useState<ObservationResult & { delays?: ObservationDelaySnapshot }>({ error: 'Waiting for Live Session inputs.' });
    const [modelSteps, setModelSteps] = useState<number | null>(null);
    const [error, setError] = useState('');
    const [saved, setSaved] = useState(false);
    const modelPath = useRef<string | null>(null);
    const [filePath, setFilePath] = useState<string | null>(null);
    const [fileBusy, setFileBusy] = useState(true);
    const [saving, setSaving] = useState(false);
    const [initializingGpu, setInitializingGpu] = useState(false);
    const saveTask = useRef<Promise<boolean> | null>(null);
    const saveGeneration = useRef(0);
    const lastSaveStep = useRef(-1);
    const [aggression, setAggression] = useState(configuration.current.aggression);
    const aggressionRef = useRef(aggression);
    const [testAction, setTestAction] = useState<ControllerAction>(NEUTRAL);
    const running = state.mode !== 'idle';
    const aiActive = state.mode === 'training' || state.mode === 'evaluation';

    const save = (): Promise<boolean> => {
        const destination = modelPath.current;
        if (!documentReady.current || !destination) return Promise.resolve(true);
        const current = policy.current;
        const generation = ++saveGeneration.current;
        const previous = saveTask.current;
        if (mounted.current) { setSaved(false); setSaving(true); }
        const task = (async () => {
            try {
                await previous;
                if (policy.current !== current || generation !== saveGeneration.current) return true;
                const document = current ? await current.checkpointAsync() : emptyTrainingDocument(configuration.current);
                if (policy.current !== current || generation !== saveGeneration.current) return true;
                await saveDrivingModel(destination, document);
                if (mounted.current && generation === saveGeneration.current) setSaved(current?.steps === document.model?.steps);
                return true;
            } catch (reason) {
                if (mounted.current && policy.current === current && generation === saveGeneration.current) {
                    setSaved(false);
                    setError(`Could not save model to ${destination}. ${errorMessage(reason)}`);
                }
                return false;
            } finally { if (mounted.current && generation === saveGeneration.current) setSaving(false); }
        })();
        saveTask.current = task;
        trackDrivingModelSave(destination, task);
        return task;
    };
    const read = (): ObservationResult & { delays: ObservationDelaySnapshot } => {
        const vision = directoryRef.current?.findComponentRef<TrackVisionHandle>(getVisualizationComponentName('track-vision'))?.current;
        const map = directoryRef.current?.findComponentRef<LiveTrajectoryMapHandle>(getVisualizationComponentName('live-trajectory-map'))?.current;
        const live = liveTelemetryStore.getSnapshot();
        const detection = vision?.getLatestDrivingDetection
            ? vision.getLatestDrivingDetection() : vision?.getLatestDetection() ?? null;
        const checkedAt = Date.now(), started = performance.now();
        const result = readDrivingObservation(live, detection,
            map?.getReferenceLap() ?? null, aggressionRef.current, checkedAt, configuration.current);
        return { ...result, delays: { checkedAt, telemetryReceivedAt: live.receivedAt,
            capturedAt: detection?.capturedAt, pairedTelemetryReceivedAt: detection?.telemetry?.receivedAt,
            publishedTiming: detection?.timing, latestTiming: vision?.getTimingSnapshot?.(),
            observationMs: performance.now() - started, observationReady: Boolean(result.observation),
            modelTiming: loop.current?.getModelTimingSnapshot() } };
    };
    const changeAggression = (value: number) => {
        configuration.current.aggression = value;
        aggressionRef.current = value;
        setAggression(value);
        setInputs(read());
        save();
    };

    const activateDocument = (document: DrivingTrainingDocument) => {
        const restored = document.model ? DrivingPolicy.restore(document) : null;
        saveGeneration.current++;
        policy.current?.dispose();
        policy.current = restored;
        lastSaveStep.current = -1;
        documentReady.current = true;
        configuration.current = restored?.configuration ?? cloneConfig(document.configuration);
        aggressionRef.current = configuration.current.aggression;
        setAggression(aggressionRef.current);
        setModelSteps(restored?.steps ?? null);
        setInputs(read());
    };

    useEffect(() => {
        mounted.current = true;
        let cancelled = false;
        const monitor = impactMonitor.current;
        const observeImpact = () => monitor.observe(liveTelemetryStore.getSnapshot());
        const unsubscribeImpact = liveTelemetryStore.subscribeSnapshot(observeImpact);
        observeImpact();
        discardBrowserModels();
        void (async () => {
            try {
                const destination = await window.drivingModelFiles?.current();
                if (cancelled || !destination) return;
                const document = await loadDrivingModel(destination);
                if (cancelled) return;
                modelPath.current = destination; setFilePath(destination);
                activateDocument(document); setSaved(true);
            } catch (reason) {
                if (!cancelled) setError(`Could not open the model file. ${errorMessage(reason)}`);
            } finally { if (!cancelled) setFileBusy(false); }
        })();
        const api = window.virtualController;
        if (api) {
            controller.current = {
                connect: (options) => api.connect(options), disconnect: () => api.disconnect(),
                apply: async (action) => {
                    try { await api.apply(action); }
                    catch (reason) { if (mounted.current) setConnected(false); throw reason; }
                },
            };
            controlOutput.current = new DrivingControlOutput(controller.current);
            loop.current = new DrivingTrainingLoop(controlOutput.current, read, (next) => {
                if (!mounted.current) return;
                setState(next);
                setModelSteps(policy.current?.steps ?? null);
                if (next.mode === 'training') setSaved(false);
                if (next.mode === 'idle' || (next.steps > 0 && next.steps !== lastSaveStep.current
                    && next.steps % configuration.current.runtime.saveEverySteps === 0)) {
                    lastSaveStep.current = next.steps;
                    void save();
                }
            }, Date.now, () => manualControls.current?.read() ?? Promise.resolve(null),
            (previous, current) => monitor.between(previous, current));
        }
        const timer = setInterval(() => setInputs(read()), 250);
        const releaseTest = () => {
            clearInterval(testTimer.current);
            testTimer.current = undefined;
            testMode.current = null;
            if (mounted.current) { setTestAction(NEUTRAL); setTestStatus(''); }
        };
        const emergencyStop = () => {
            startGeneration.current++;
            releaseTest();
            void loop.current?.stop('Emergency stop. Controller released.');
        };
        const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') emergencyStop(); };
        const onBlur = () => {
            if (testMode.current !== 'hold') return;
            releaseTest();
            void controlOutput.current?.release().catch((reason) => { if (mounted.current) setError(errorMessage(reason)); });
        };
        const onUnload = () => { emergencyStop(); void api?.disconnect(); };
        window.addEventListener('keydown', onKey);
        window.addEventListener('blur', onBlur);
        window.addEventListener('beforeunload', onUnload);
        return () => {
            cancelled = true;
            unsubscribeImpact();
            mounted.current = false;
            // This ref is a cancellation counter, not a DOM node.
            // eslint-disable-next-line react-hooks/exhaustive-deps
            connectingGeneration.current++;
            // This ref is also a cancellation counter, not a DOM node.
            // eslint-disable-next-line react-hooks/exhaustive-deps
            startGeneration.current++;
            releaseTest();
            clearInterval(timer);
            window.removeEventListener('keydown', onKey);
            window.removeEventListener('blur', onBlur);
            window.removeEventListener('beforeunload', onUnload);
            void loop.current?.stop();
            void api?.disconnect().catch(() => undefined);
            const current = policy.current;
            void save().finally(() => current?.dispose());
        };
        // The live component directory is read through a ref, without restarting a running episode.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const connect = async () => {
        const generation = ++connectingGeneration.current;
        setConnecting(true); setError('');
        try {
            if (!controller.current) throw new Error('Virtual control requires the Windows desktop app.');
            await controller.current.connect({ deviceId });
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
        if (!testMode.current) return;
        clearInterval(testTimer.current); testTimer.current = undefined; setTestAction(NEUTRAL);
        testMode.current = null; setTestStatus('');
        void controlOutput.current?.release().catch((reason) => { if (mounted.current) setError(errorMessage(reason)); });
    };
    const disconnect = async () => {
        startGeneration.current++;
        setConnected(false); setDisconnecting(true);
        stopTest();
        try {
            await loop.current?.stop();
            await controller.current?.disconnect();
        } finally { if (mounted.current) setDisconnecting(false); }
    };
    const start = async (train: boolean) => {
        setError('');
        try {
            if (!connected || testTimer.current || initializingGpu || fileBusy || !modelPath.current) return;
            const generation = ++startGeneration.current;
            setInitializingGpu(true);
            if (await saveTask.current === false) throw new Error('Save the model successfully before starting again.');
            if (!mounted.current || generation !== startGeneration.current) return;
            // Reconstruct from the selected file for every training/evaluation run.
            documentReady.current = false;
            const document = await loadDrivingModel(modelPath.current);
            if (!mounted.current || generation !== startGeneration.current) return;
            activateDocument(document);
            const input = read();
            if (!input.observation) throw new Error(input.error);
            if (!train && (!policy.current || !policy.current.steps)) throw new Error('Train a model before evaluation.');
            if (!policy.current) {
                policy.current = new DrivingPolicy(input.observation.context, input.observation.features.length, Math.random, configuration.current);
                configuration.current = policy.current.configuration;
            }
            await saveDrivingModel(modelPath.current, policy.current.checkpoint());
            if (!mounted.current || generation !== startGeneration.current) return;
            setSaved(true);
            await loop.current?.start(policy.current, train);
        } catch (reason) { if (mounted.current) setError(errorMessage(reason)); }
        finally { if (mounted.current) setInitializingGpu(false); }
    };
    const changeSource = (source: ControlSource) => {
        setError('');
        if (!connected || testTimer.current) return;
        try {
            if (source === 'manual' && loop.current?.state.mode === 'idle') loop.current.startManual();
            else loop.current?.setControlSource(source);
        } catch (reason) { setError(errorMessage(reason)); }
    };
    const chooseModel = async (mode: 'new' | 'open') => {
        if (running || initializingGpu || fileBusy) return;
        setFileBusy(true); setError('');
        try {
            await saveTask.current;
            if (!mounted.current) return;
            const destination = await drivingModelFiles().choose(mode);
            if (!mounted.current || !destination) return;
            const document = mode === 'new' ? emptyTrainingDocument() : await loadDrivingModel(destination);
            if (!mounted.current) return;
            if (mode === 'new') await saveDrivingModel(destination, document);
            await drivingModelFiles().select(destination);
            if (!mounted.current) return;
            modelPath.current = destination; setFilePath(destination);
            activateDocument(document); setSaved(true);
            saveTask.current = null;
        } catch (reason) { if (mounted.current) setError(errorMessage(reason)); }
        finally { if (mounted.current) setFileBusy(false); }
    };
    const test = (action: ControllerAction) => {
        if (loop.current?.state.mode !== 'idle' || !connected || testTimer.current) return;
        const lease = controlOutput.current!.acquire('test');
        testMode.current = 'hold';
        let busy = false;
        const apply = () => {
            if (busy) return;
            busy = true;
            void lease.apply(action).then((applied) => { if (applied && mounted.current) setTestAction(action); }).catch((reason) => {
                clearInterval(testTimer.current); testTimer.current = undefined;
                testMode.current = null;
                if (mounted.current) { setTestAction(NEUTRAL); setTestStatus(''); setError(errorMessage(reason)); }
            }).finally(() => { busy = false; });
        };
        apply(); testTimer.current = setInterval(apply, 100);
    };
    const timedTest = ({ key, label, value }: typeof TEST_CONTROLS[number]) => {
        if (loop.current?.state.mode !== 'idle' || !connected || testTimer.current) return;
        const lease = controlOutput.current!.acquire('test');
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
            busy = true;
            void lease.apply(next).then((applied) => { if (applied && mounted.current) setTestAction(next); }).catch((reason) => {
                clearInterval(testTimer.current); testTimer.current = undefined; testMode.current = null;
                if (mounted.current) { setTestAction(NEUTRAL); setTestStatus(''); setError(errorMessage(reason)); }
            }).finally(() => { busy = false; });
        };
        tick(); testTimer.current = setInterval(tick, 100);
    };
    const config = configuration.current;
    const fullObservationHistory = config.observation.features.every((name) => config.memory.features.includes(name));
    const planEnd = config.actions.planTimesSeconds[config.actions.planTimesSeconds.length - 1];
    const output: ControllerAction = running ? state.action : testAction;
    const observation = inputs.observation;
    const inputMismatches = observation ? policy.current?.inputMismatches(observation) ?? [] : [];
    const mismatched = inputMismatches.length > 0;

    return <section className="driving-training" aria-label="AI Self-Driving Training">
        <header className="driving-training-header">
            <div><span className="driving-eyebrow">SIMULATOR LAB</span><h1>AI Self-Driving Training</h1>
                <p>Two drivers. One virtual wheel. Choose who controls the car.</p></div>
            <button onClick={onOpenLiveSession}>Open Live Session</button>
        </header>
        <div className="driving-status" role="status"><span data-active={running} />{state.status}</div>
        {error && <p className="driving-error" role="alert">{error}</p>}
        <section className="driving-routing" aria-label="Shared control system">
            <div className="driving-route-source" data-active={state.source === 'manual'}>
                <span className="driving-eyebrow">HUMAN INPUT</span><strong>Human takeover</strong>
                <span>Wheel, pedals or gamepad</span>
            </div>
            <span className="driving-route-arrow" aria-hidden="true">→</span>
            <div className="driving-route-owner">
                <span className="driving-eyebrow">CONTROL OWNER</span>
                <strong aria-label="Current control owner">{testMode.current ? 'Calibration test' : state.source === 'manual' ? 'Human' : state.source === 'ai' ? 'AI' : 'No driver'}</strong>
                <span>{connected ? `vJoy ${deviceId} → Simulator` : 'Connect vJoy to drive'}</span>
                <button className="driving-stop" disabled={!running && !testTimer.current && !initializingGpu}
                    onClick={() => { startGeneration.current++; stopTest(); void loop.current?.stop(); }}>Stop</button>
            </div>
            <span className="driving-route-arrow" aria-hidden="true">←</span>
            <div className="driving-route-source" data-active={state.source === 'ai'}>
                <span className="driving-eyebrow">AI INPUT</span><strong>Driving policy</strong>
                <span>{aiActive ? state.mode === 'training' ? 'Training enabled' : 'Evaluation · learning off' : 'Not started'}</span>
            </div>
            <p>One driver owns steering, pedals and shifts at a time. Human takeover stays active until you explicitly return control to AI. Stop releases every input.</p>
        </section>
        <div className="driving-grid">
            <ManualDrivingControls ref={manualControls} connected={connected} running={running} source={state.source}
                testing={Boolean(testMode.current)} aiActive={aiActive} aiReady={Boolean(observation && !mismatched)} onSourceChange={changeSource} />
            <section className="driving-card driving-ai-input" aria-label="AI driver">
                <div className="driving-card-title"><h2>AI driver</h2><span>{aiActive ? state.source === 'ai' ? 'Driving' : 'Human has control' : 'Not started'}</span></div>
                <p>Start AI training or evaluation when live inputs are ready. If you are already driving, you keep control until Return to AI.</p>
                <p>CNN–LSTM actor–critic · {config.network.hiddenUnits} recurrent units · continuous throttle/brake and steering, plus learned hold gear, shift up and shift down. Throttle/brake ranges from −1 (full brake) through 0 (both released) to 1 (full throttle).</p>
                <p>{config.memory.windowMs / 1000}-second observation history covers {fullObservationHistory ? 'all AI observation inputs' : 'the observation inputs selected in this saved model'} during training and evaluation, sampled in {config.memory.intervalMs} ms bins. Memory starts fresh after stopping, changing drivers or interrupted inputs.</p>
                <p>One action plan predicts controls at {config.actions.planTimesSeconds.join(', ')} seconds. The AI follows these steps while computing, replacing the whole plan as soon as a new output is ready. Reward trains each executed step; human driving teaches future steps from demonstrated actions.</p>
                <p className="driving-note">Each model JSON contains its inputs, memory, network, training settings and weights. Training and evaluation load that JSON each time they start.</p>
                <table className="driving-pedals" aria-label="Planned driving actions">
                    <caption>Action plan · 0–{planEnd} seconds</caption>
                    <thead><tr><th scope="col">Time</th><th scope="col">Throttle / brake</th><th scope="col">Steering</th><th scope="col">Shift</th></tr></thead>
                    <tbody>{state.plan ? state.plan.map(({ timeSeconds, action }) => <tr key={timeSeconds}>
                        <th scope="row">{timeSeconds.toFixed(1)} s</th>
                        <td>{(action.throttle - action.brake).toFixed(3)}</td>
                        <td>{action.steering.toFixed(3)}</td><td>{action.shiftUp ? 'Up' : action.shiftDown ? 'Down' : 'Hold'}</td>
                    </tr>) : <tr><td colSpan={4}>Waiting for AI control and fresh inputs.</td></tr>}</tbody>
                </table>
                <p className="driving-note">Disable automatic gears in the simulator and bind both shift buttons before training. The model learns gear choices from driving rewards. Shifts use {config.actions.shiftPulseMs / 1000}-second presses with at least {config.actions.shiftCooldownMs / 1000} seconds between shifts. Downshifts are allowed in every gear, including neutral and reverse.</p>
                <div className="driving-metrics"><div><strong>{modelSteps ?? 0}</strong><span>Learning steps</span></div>
                    <div><strong>{state.episodeReward.toFixed(2)}</strong><span>Episode reward</span></div></div>
                {mismatched && <div className="driving-error" role="alert">
                    <p>The saved model does not match the current live session.</p>
                    <ul>{inputMismatches.map((difference) => <li key={difference}>{difference}</li>)}</ul>
                    <p>Use matching live inputs, open a matching model file, or create a new model.</p>
                </div>}
                <div className="driving-buttons">
                    <button className="driving-primary" disabled={!filePath || fileBusy || !connected || !observation || aiActive || initializingGpu || Boolean(mismatched) || Boolean(testTimer.current)} onClick={() => { void start(true); }}>Start training</button>
                    <button disabled={!filePath || fileBusy || !connected || !observation || aiActive || initializingGpu || !modelSteps || Boolean(mismatched) || Boolean(testTimer.current)} onClick={() => { void start(false); }}>Run trained model</button>
                </div>
                <p className="driving-model-path"><strong>Model file</strong><br />{filePath || 'Choose New model to select a save location, or Open model to continue training a saved file.'}</p>
                <div className="driving-buttons">
                    <button disabled={running || initializingGpu || fileBusy} onClick={() => { void chooseModel('new'); }}>New model</button>
                    <button disabled={running || initializingGpu || fileBusy} onClick={() => { void chooseModel('open'); }}>Open model</button>
                    <button disabled={!filePath || fileBusy || initializingGpu || saving || !documentReady.current} onClick={() => { setError(''); void save(); }}>Save model</button>
                </div>
                <p className="driving-note" role="status">{saving ? 'Saving model…' : saved ? 'Model saved to file.' : 'Model not saved.'} The complete model saves to the selected file every {config.runtime.saveEverySteps} learning steps and when stopped.</p>
                <p className="driving-note">Training explores controls; use a practice session and reposition the car between episodes. A new policy starts untrained.</p>
            </section>
            <section className="driving-card driving-shared-output" aria-label="Shared vJoy output">
                <div className="driving-card-title"><h2>Steering wheel (vJoy)</h2><span>{connected ? 'Connected' : 'Disconnected'}</span></div>
                <p>Shared output for human and AI driving. Steering, pedals and shifts use one set of simulator bindings.</p>
                <div className="driving-output-options">
                    <label htmlFor="driving-device">vJoy device</label>
                    <select id="driving-device" value={deviceId} disabled={connected || connecting || disconnecting || installingPreset}
                        onChange={(event) => { setDeviceId(Number(event.target.value)); setAccPreset(null); }}>
                        {Array.from({ length: 16 }, (_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}
                    </select>
                </div>
                <p>Works independently of your physical wheel brand. Bind vJoy X → steering, Y → throttle, Z → brake, button 1 → shift up, and button 2 → shift down in the simulator.</p>
                <p className="driving-note">vJoy is included with the Windows installer. Restart Windows if setup requests it, then enable X, Y, Z and at least 2 buttons on the selected device in Configure vJoy. In iRacing, use Configure Controls or Settings → Driving → Input Calibration.</p>
                <div className="driving-buttons"><button onClick={() => { void installAccPreset(); }}
                    disabled={installingPreset || connecting || disconnecting || running || Boolean(testTimer.current)}>
                    {installingPreset ? 'Installing ACC preset…' : 'Install ACC preset'}</button></div>
                <p className="driving-note">For ACC, save your current controls, install the training preset, then load it in Options → Controls → Wheel. This automatically maps steering, throttle, brake, shift up (button 1), and shift down (button 2) on the selected vJoy device. If the preset is missing, restart ACC. Keep this preset loaded for controller takeover during training.</p>
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
                <div className="driving-buttons"><button onClick={() => { void connect(); }} disabled={connected || connecting || disconnecting}>{connecting ? 'Connecting…' : 'Connect wheel'}</button>
                    <button onClick={() => { void disconnect().catch((reason) => setError(errorMessage(reason))); }} disabled={!connected}>{disconnecting ? 'Disconnecting…' : 'Disconnect'}</button></div>
                <p className="driving-note">Hold tests release when this app loses focus. To test shifting in ACC, use “Test shift up in 5 seconds” or “Test shift down in 5 seconds” below, then switch to ACC during the countdown. Check that both shift bindings use the same vJoy device as steering and pedals.</p>
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
                <p className="driving-note">Connected means the virtual device is ready; verify the simulator’s input meters before driving. Stop or Escape in this app cancels tests. AI driving neutralizes controls and retries delayed or invalid inputs, then resumes automatically; human driving continues while learning pauses.</p>
            </section>
            <section className="driving-card driving-live-inputs" aria-label="AI observation inputs">
                <h2>AI observation inputs</h2>
                <p>Start live telemetry, add Live Map and Track Vision, then start screen capture in Live Session. The camera image is the only visual input.</p>
                <div className="driving-input-status" data-ready={Boolean(observation)}>{observation ? 'All model inputs ready' : inputs.error}</div>
                <div className="driving-aggression">
                    <div><label htmlFor="driving-aggression">Driving aggression</label><output htmlFor="driving-aggression">{aggression}</output></div>
                    <input id="driving-aggression" type="range" min="0" max="100" step="1" value={aggression} disabled={fileBusy || initializingGpu}
                        aria-describedby="driving-aggression-meaning" onChange={(event) => changeAggression(Number(event.target.value))} />
                    <div className="driving-aggression-scale"><span>0 · Cautious</span><span>50 · Reference</span><span>100 · Aggressive</span></div>
                    <p id="driving-aggression-meaning">{aggression < 50 ? 'Cautious spreads the slower-speed area to outer points and lowers speed targets through braking and throttle sections.'
                        : aggression > 50 ? 'Aggressive condenses the slower-speed area and raises speed targets through braking and throttle sections.' : 'Reference uses the recorded speed targets through braking and throttle sections.'}
                        {' '}Area adjustment: up to {aggressionRadius(aggression, config.reward.aggressionMaxRadius).toFixed(1)} coordinate units along the reference line (maximum {config.reward.aggressionMaxRadius}; limited between nearby pedal events).</p>
                    <button disabled={fileBusy || initializingGpu} onClick={() => changeAggression(50)}>Reset to reference</button>
                </div>
                <dl className="driving-details">
                    <dt>Game / car</dt><dd>{observation ? `${observation.context.game} / ${observation.context.car}` : 'Waiting'}</dd>
                    <dt>Map</dt><dd>{observation?.context.map ?? 'Waiting'}</dd>
                    <dt>Speed</dt><dd>{observation ? `${observation.speed.toFixed(1)} km/h` : '—'}</dd>
                    <dt>Reference speed</dt><dd>{observation ? `${observation.speedTargets.reference.toFixed(1)} km/h` : '—'}</dd>
                    <dt>Aggression target speed</dt><dd>{observation ? `${observation.speedTargets.target.toFixed(1)} km/h` : '—'}</dd>
                    <dt>Local velocity</dt><dd>{observation ? `${observation.localVelocity.map((value, axis) => `${['X', 'Y', 'Z'][axis]} ${value.toFixed(2)}`).join(' · ')} m/s` : '—'}</dd>
                    <dt>G-force</dt><dd>{observation ? `${observation.gForce.map((value, axis) => `${['X', 'Y', 'Z'][axis]} ${value.toFixed(2)}`).join(' · ')} g` : '—'}</dd>
                    <dt>Yaw rate</dt><dd>{observation ? `${observation.yawRate.toFixed(3)} rad/s` : '—'}</dd>
                    <dt>Gear / RPM</dt><dd>{observation ? `${observation.gear < 0 ? 'R' : observation.gear === 0 ? 'N' : observation.gear} / ${observation.rpm}` : '—'}</dd>
                    <dt>Lap position</dt><dd>{observation ? `${(observation.normalizedPosition * 100).toFixed(1)}%` : '—'}</dd>
                    <dt>Camera image</dt><dd>{config.observation.cameraWidth} × {config.observation.cameraHeight} RGB{observation ? '' : ' · Waiting'}</dd>
                    <dt>Model compute</dt><dd>{initializingGpu ? 'Preparing GPU…' : 'WebGPU · CNN, LSTM and training'}</dd>
                </dl>
                <p className="driving-note">The policy uses car-local velocity XYZ, g-force XYZ, yaw rate, speed, current gear, engine RPM, game, car, map, normalized position, current controls, camera pixels, aggression, reference speed and target speed. Acceleration and deceleration timing uses the reference lap’s throttle and brake events together with its speed profile. It generates the pedal commands. Aggression applies during training and evaluation; further training teaches its response. Evaluation does not update weights. Models saved with a different observation input set need to be trained again.</p>
            </section>
            <section className="driving-card">
                <h2>Reward signal</h2>
                <dl className="driving-rewards">
                    <div><dt>Acceleration / deceleration</dt><dd>{state.reward?.timing.toFixed(3) ?? '0.000'}</dd>
                        <p>Measured acceleration or deceleration earns reward when it moves speed toward the aggression-adjusted goal, even while far from that goal or holding the pedals steady. Moving away from the goal is penalized; holding the same speed earns no bonus. Both speed readings are compared with the same target, so a changing target alone earns no reward. Cautious driving spreads the slower-speed area outward and lowers its targets; aggressive driving condenses it and raises its targets. Overshooting reduces credit. Reward is weighted by forward progress. Stationary or reverse driving earns no acceleration/deceleration bonus.</p>
                        {state.reward && <p>Last scored at aggression {state.reward.speedComparison.aggression}: speed {state.reward.speedComparison.measured.toFixed(1)} / reference {state.reward.speedComparison.reference.toFixed(1)} / target {state.reward.speedComparison.target.toFixed(1)} km/h; speed gap closed {state.reward.speedComparison.speedGoalProgress.toFixed(2)} km/h (negative means moving away); measured acceleration {state.reward.speedComparison.measuredAcceleration.toFixed(2)} m/s² (negative means deceleration).</p>}
                    </div>
                    <div><dt>Normalized position progress</dt><dd>{state.reward?.progress.toFixed(3) ?? '0.000'}</dd><p>Faster increases in normalized lap position per second earn more reward; backward progress is penalized. Start/finish crossings preserve the direction of progress. Zero speed or unchanged normalized position subtracts {(0.25 * config.reward.progressWeight * 0.01).toFixed(3)} reward per second.</p></div>
                    <div><dt>Impact penalty</dt><dd>{state.reward?.impact.toFixed(3) ?? '0.000'}</dd>
                        <p>Live telemetry checks for impacts between AI frames, including brief bounces followed by sliding. With the brake released, unexpected slowing uses a smaller coasting allowance at every speed. With steering also centered, sideways deflection adds evidence of a diagonal impact. The strongest detected impact within each training interval applies a penalty. Confidence of at least {IMPACT_CONFIDENCE_THRESHOLD * 100}% applies a penalty scaled by confidence and severity. Severity reaches 100 at 10 m/s of excess velocity change. These are heuristic estimates; spins and extreme braking can resemble impacts. Car damage telemetry is not used.</p>
                        <p aria-label="Impact estimate">{!state.reward ? 'Waiting for two fresh motion samples.'
                            : !state.reward.impactEstimate.available ? 'Impact estimate unavailable: no reliable 20–500 ms telemetry interval.'
                            : `${state.reward.impactEstimate.detected ? 'Possible impact' : 'No impact detected'} · Confidence ${(state.reward.impactEstimate.confidence * 100).toFixed(0)}% · Estimated severity ${(state.reward.impactEstimate.severity * 100).toFixed(0)}/100 · Excess Δv ${state.reward.impactEstimate.deltaV.toFixed(2)} m/s`}</p>
                        <p>Severity measures excess velocity change, reaching 100 at 15 m/s. It does not estimate vehicle damage. Values describe the latest scored transition.</p>
                    </div>
                </dl>
                <p className="driving-note">Stop ends the episode. Delayed or invalid observations pause learning while the session waits and retries. Learning resumes from fresh, matching inputs without using the interrupted sample.</p>
            </section>
            <ObservationInputDelays snapshot={inputs.delays} telemetryMaxAgeMs={config.runtime.telemetryMaxAgeMs}
                visionMaxAgeMs={config.runtime.visionMaxAgeMs} />
        </div>
    </section>;
}
