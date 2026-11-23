import React, { useCallback, useEffect, useRef, useState } from 'react';
import { OPERATION_COMPONENT_NAMES, useOptionalOperationComponentRefDirectory } from 'contexts/OperationComponentRefContext';
import { getVisualizationComponentName } from 'views/session-shared/visualization/visualization-component-names';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import type { LiveSessionHandle } from 'views/live-session/LiveSessionView';
import { SessionScreenFrameReader } from 'views/live-session/live-session-recording/session-screen-frame';
import type { LiveTrajectoryMapHandle } from 'views/live-session/LiveTrajectoryMap';
import { NEUTRAL, ObservationResult, readDrivingObservation, readDrivingTelemetry } from './driving-observation';
import { DrivingPolicy, DrivingTrainingDocument, emptyTrainingDocument } from './driving-policy';
import { cloneConfig, DEFAULT_DRIVING_CONFIG, MAX_EXPLORATION_GEAR } from './driving-model-config';
import { discardBrowserModels, drivingModelFiles, loadDrivingModel, saveDrivingModel, trackDrivingModelSave } from './driving-model-storage';
import { DrivingTrainingLoop, initialTrainingState } from './driving-training-loop';
import { DrivingControlOutput } from './driving-control-output';
import type { ControlSource } from './driving-control';
import { DrivingImpactMonitor } from './driving-impact-monitor';
import type { AccControlPreset, ControllerAction, VirtualController } from './virtual-controller';
import ManualDrivingControls, { ManualDrivingControlsHandle } from './ManualDrivingControls';
import ObservationInputDelays, { ObservationDelaySnapshot } from './ObservationInputDelays';
import RecordedFrameViewer, { RecordedDrivingFrame, RecordedFrameViewerHandle, RecordedReplayPosition } from './RecordedFrameViewer';
import { findRecordedTelemetry, readRecordedDrivingObservation, type RecordedObservationResult } from './recorded-driving-session';
import { loadRecordedDrivingReferences, RecordedDrivingReferences, recordedTrackKey } from './recorded-driving-reference';
import { RecordedDrivingReplay, RecordedReplayProgress } from './recorded-driving-replay';
import { selectLiveMapReference } from 'views/live-session/live-map-reference';
import './AiDrivingTraining.css';

const TEST_CONTROLS = [
    { key: 'throttle', label: 'throttle', value: 0.25 },
    { key: 'brake', label: 'brake', value: 0.25 },
    { key: 'steering', label: 'steering', value: 0.25 },
    { key: 'shiftDown', label: 'shift down', value: true },
    { key: 'shiftUp', label: 'shift up', value: true },
] as const;
const EXPLORATION_CONTROLS = [
    { key: 'pedalExplorationSigma', label: 'Throttle/brake', control: 'pedal' },
    { key: 'steeringExplorationSigma', label: 'Steering', control: 'steering' },
] as const;
const EXPLORATION_FREQUENCIES = [
    { key: 'pedalExplorationUpdateHz', label: 'Throttle/brake', control: 'pedal' },
    { key: 'steeringExplorationUpdateHz', label: 'Steering', control: 'steering' },
    { key: 'gearExplorationUpdateHz', label: 'Gear', control: 'gear' },
] as const;
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Self-driving training failed.';
type RecordingReferences = { sessionId: string; data?: RecordedDrivingReferences; error?: string };
const resolveRecordingReferences = async (session: RecordedDrivingFrame['session']): Promise<RecordingReferences> => {
    try { return { sessionId: session.id, data: await loadRecordedDrivingReferences(session) }; }
    catch (reason) { return { sessionId: session.id, error: `Could not load recorded training references. ${errorMessage(reason)}` }; }
};

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
    const sampledInputs = useRef(inputs);
    const [modelSteps, setModelSteps] = useState<number | null>(null);
    const [updateDiagnostics, setUpdateDiagnostics] = useState<Awaited<ReturnType<DrivingPolicy['trainingDiagnosticsAsync']>> | null>(null);
    const [measuringUpdate, setMeasuringUpdate] = useState(false);
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
    const [exploration, setExploration] = useState(() => ({
        pedalExplorationSigma: configuration.current.learning.pedalExplorationSigma,
        throttleExplorationSpeedLimitKmh: configuration.current.learning.throttleExplorationSpeedLimitKmh,
        steeringExplorationSigma: configuration.current.learning.steeringExplorationSigma,
        shiftTemperature: configuration.current.learning.shiftTemperature,
        maxExplorationGear: configuration.current.learning.maxExplorationGear,
        pedalExplorationUpdateHz: configuration.current.learning.pedalExplorationUpdateHz,
        steeringExplorationUpdateHz: configuration.current.learning.steeringExplorationUpdateHz,
        gearExplorationUpdateHz: configuration.current.learning.gearExplorationUpdateHz,
    }));
    const [testAction, setTestAction] = useState<ControllerAction>(NEUTRAL);
    const [inputSource, setInputSource] = useState<'live' | 'recorded'>('live');
    const inputSourceRef = useRef(inputSource);
    const [screenFrames] = useState(() => new SessionScreenFrameReader());
    const recordedViewer = useRef<RecordedFrameViewerHandle>(null);
    const [trainingPickerContainer, setTrainingPickerContainer] = useState<HTMLDivElement | null>(null);
    const recordedFrame = useRef<RecordedDrivingFrame | null>(null);
    const recordedReferences = useRef<RecordingReferences | null>(null);
    const [recordedSession, setRecordedSession] = useState<RecordedDrivingFrame['session'] | null>(null);
    const [referenceAttempt, setReferenceAttempt] = useState(0);
    const [viewerBusy, setViewerBusy] = useState(false);
    const [replayReady, setReplayReady] = useState(false);
    const [replaying, setReplaying] = useState(false);
    const [replayStatus, setReplayStatus] = useState('');
    const [annotationProgress, setAnnotationProgress] = useState<RecordedReplayPosition | 'preparing' | null>(null);
    const replayAbort = useRef<AbortController | null>(null);
    const running = state.mode !== 'idle';
    const finishing = state.mode === 'finishing';
    const aiActive = state.mode === 'training' || state.mode === 'evaluation';

    const save = (afterTraining?: Promise<void>): Promise<boolean> => {
        const destination = modelPath.current;
        if (!documentReady.current || !destination) return Promise.resolve(true);
        const current = policy.current;
        const generation = ++saveGeneration.current;
        const previous = saveTask.current;
        if (mounted.current) { setSaved(false); setSaving(true); }
        const task = (async () => {
            try {
                await previous;
                if (afterTraining) await afterTraining;
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
    const readRecorded = (frame: RecordedDrivingFrame | null, references = recordedReferences.current): RecordedObservationResult => {
        if (!frame) return { error: 'Waiting for a frame from replay or the annotation editor.' };
        if (!references || references.sessionId !== frame.session.id || !references.data) {
            return { error: references?.sessionId === frame.session.id && references.error
                ? references.error : 'Loading matching circuit maps and reference laps for this recording…' };
        }
        const row = findRecordedTelemetry(frame.session, frame.videoSeconds, frame.offsetSeconds,
            configuration.current.runtime.telemetryMaxAgeMs);
        const sample = row?.sample;
        const lap = selectLiveMapReference(references.data.laps, frame.session.game, sample?.Static_track,
            sample?.Static_car_model, sample?.Graphics_track_grip_status);
        const map = sample?.Static_track
            ? references.data.maps.get(recordedTrackKey(frame.session.game, sample.Static_track)) ?? null : null;
        if (row && !lap) return { error: 'No reference lap matches this recording’s game, car and circuit.' };
        if (row && !map) return { error: 'No saved circuit map matches this recording. Add its centerline in Circuit Maps.' };
        return readRecordedDrivingObservation(frame.session, frame.videoSeconds, frame.camera, lap,
            aggressionRef.current, configuration.current, map, frame.offsetSeconds);
    };
    const readLive = (): ObservationResult & { delays?: ObservationDelaySnapshot } => {
        const session = directoryRef.current?.findComponentRef<LiveSessionHandle>(OPERATION_COMPONENT_NAMES.LIVE_SESSION)?.current;
        const map = directoryRef.current?.findComponentRef<LiveTrajectoryMapHandle>(getVisualizationComponentName('live-trajectory-map'))?.current;
        const frame = screenFrames.read(session?.getScreenCaptureStream() ?? null);
        const live = liveTelemetryStore.getSnapshot();
        const checkedAt = Date.now(), started = performance.now();
        const result = readDrivingObservation(live, frame,
            map?.getReferenceLap() ?? null, aggressionRef.current, checkedAt, configuration.current, map?.getCircuitMap?.() ?? null);
        return { ...result, delays: { checkedAt, telemetryReceivedAt: live.receivedAt,
            capturedAt: frame?.capturedAt, pairedTelemetryReceivedAt: frame?.telemetry.receivedAt,
            captureMs: frame?.processingMs, publishedAt: frame?.publishedAt, captureIntervalMs: frame?.captureIntervalMs,
            observationMs: performance.now() - started, observationReady: Boolean(result.observation),
            modelTiming: loop.current?.getModelTimingSnapshot() } };
    };
    const read = (): ObservationResult & { delays?: ObservationDelaySnapshot } => {
        // The consumer and the panel share the same sampled observation.
        const result = inputSourceRef.current === 'recorded' ? readRecorded(recordedFrame.current) : readLive();
        sampledInputs.current = result;
        return result;
    };
    const refreshInputs = () => {
        const mode = loop.current?.state.mode;
        const consuming = replayAbort.current || mode === 'training' || mode === 'evaluation' || mode === 'finishing';
        // Preview while idle; active runs own sampling, even while awaiting work.
        const result = consuming ? sampledInputs.current : read();
        // Refresh profiling without acquiring another observation.
        setInputs(result.delays ? { ...result, delays: { ...result.delays, checkedAt: Date.now(),
            modelTiming: loop.current?.getModelTimingSnapshot() } } : result);
    };
    const updateRecordedFrame = useCallback((frame: RecordedDrivingFrame | null) => {
        recordedFrame.current = frame;
        // Seeking temporarily has no decoded frame; keep the session's references.
        if (frame) setRecordedSession(frame.session);
    }, []);
    useEffect(() => {
        if (!recordedSession || inputSource !== 'recorded') return;
        let cancelled = false;
        recordedReferences.current = { sessionId: recordedSession.id };
        void resolveRecordingReferences(recordedSession).then((references) => {
            if (!cancelled) recordedReferences.current = references;
        });
        return () => { cancelled = true; };
    }, [recordedSession, inputSource, referenceAttempt]);
    const changeAggression = (value: number) => {
        configuration.current.aggression = value;
        aggressionRef.current = value;
        setAggression(value);
        refreshInputs();
        save();
    };
    const changeExploration = (next: typeof exploration) => {
        if (!documentReady.current || !modelPath.current || running || replaying || viewerBusy || initializingGpu || fileBusy) return;
        Object.assign(configuration.current.learning, next);
        setExploration(next);
        void save();
    };

    const activateDocument = (document: DrivingTrainingDocument) => {
        const restored = document.model ? DrivingPolicy.restore(document) : null;
        saveGeneration.current++;
        policy.current?.dispose();
        policy.current = restored;
        lastSaveStep.current = restored?.steps ?? 0;
        documentReady.current = true;
        configuration.current = restored?.configuration ?? cloneConfig(document.configuration);
        aggressionRef.current = configuration.current.aggression;
        setAggression(aggressionRef.current);
        setExploration({ pedalExplorationSigma: configuration.current.learning.pedalExplorationSigma,
            throttleExplorationSpeedLimitKmh: configuration.current.learning.throttleExplorationSpeedLimitKmh,
            steeringExplorationSigma: configuration.current.learning.steeringExplorationSigma,
            shiftTemperature: configuration.current.learning.shiftTemperature,
            maxExplorationGear: configuration.current.learning.maxExplorationGear,
            pedalExplorationUpdateHz: configuration.current.learning.pedalExplorationUpdateHz,
            steeringExplorationUpdateHz: configuration.current.learning.steeringExplorationUpdateHz,
            gearExplorationUpdateHz: configuration.current.learning.gearExplorationUpdateHz });
        setModelSteps(restored?.steps ?? null);
        setUpdateDiagnostics(null); setMeasuringUpdate(false);
        setInputs(read());
    };

    const measureUpdate = async () => {
        const current = policy.current;
        if (!current || measuringUpdate) return;
        setMeasuringUpdate(true); setError('');
        try {
            const diagnostics = await current.trainingDiagnosticsAsync();
            if (mounted.current && policy.current === current) setUpdateDiagnostics(diagnostics);
        } catch (reason) {
            if (mounted.current && policy.current === current) setError(errorMessage(reason));
        } finally {
            if (mounted.current && policy.current === current) setMeasuringUpdate(false);
        }
    };

    useEffect(() => {
        mounted.current = true;
        let cancelled = false;
        const monitor = impactMonitor.current;
        const observeImpact = () => {
            const snapshot = liveTelemetryStore.getSnapshot();
            monitor.observe(snapshot);
            if (inputSourceRef.current === 'live') {
                const sample = readDrivingTelemetry(snapshot, Date.now(), configuration.current);
                if (sample) loop.current?.recordTelemetry(sample);
            }
        };
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
                const saveEvery = configuration.current.runtime.saveEverySteps;
                if (next.mode === 'idle' || (next.mode === 'training' && next.steps > 0
                    && Math.floor(next.steps / saveEvery) > Math.floor(Math.max(0, lastSaveStep.current) / saveEvery))) {
                    lastSaveStep.current = next.steps;
                    void save();
                }
            }, Date.now, () => manualControls.current?.read() ?? Promise.resolve(null),
            (previous, current) => monitor.between(previous, current));
        }
        const timer = setInterval(refreshInputs, 250);
        const releaseTest = () => {
            clearInterval(testTimer.current);
            testTimer.current = undefined;
            testMode.current = null;
            if (mounted.current) { setTestAction(NEUTRAL); setTestStatus(''); }
        };
        const emergencyStop = () => {
            startGeneration.current++;
            replayAbort.current?.abort();
            releaseTest();
            if (inputSourceRef.current === 'live') void loop.current?.stop('Emergency stop. Controller released.');
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
            replayAbort.current?.abort();
            screenFrames.stop();
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
            const liveLoop = inputSourceRef.current === 'live' ? loop.current : null;
            const afterTraining = liveLoop?.state.mode === 'finishing' ? liveLoop.finishTraining() : undefined;
            if (!afterTraining) void liveLoop?.stop();
            void api?.disconnect().catch(() => undefined);
            const current = policy.current;
            void save(afterTraining).finally(() => current?.dispose());
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
            if (inputSourceRef.current !== 'live' || !connected || testTimer.current || initializingGpu || fileBusy || !modelPath.current) return;
            const generation = ++startGeneration.current;
            setInitializingGpu(true);
            if (await saveTask.current === false) throw new Error('Save the model successfully before starting again.');
            if (!mounted.current || generation !== startGeneration.current) return;
            // Reconstruct from the selected file for every training/evaluation run.
            documentReady.current = false;
            const document = await loadDrivingModel(modelPath.current);
            if (!mounted.current || generation !== startGeneration.current) return;
            activateDocument(document);
            if (!train && (!policy.current || !policy.current.steps)) throw new Error('Train a model before evaluation.');
            if (!policy.current) {
                const input = read();
                if (!input.observation) throw new Error(input.error);
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
    const startRecorded = async (train: boolean) => {
        if (inputSourceRef.current !== 'recorded' || running || replayAbort.current || viewerBusy
            || initializingGpu || fileBusy || !replayReady || !modelPath.current || !recordedViewer.current) return;
        const abort = new AbortController();
        replayAbort.current = abort;
        const active = () => mounted.current && !abort.signal.aborted;
        setReplaying(true); setInitializingGpu(true); setError(''); setReplayStatus('Preparing recorded replay…');
        if (train) setAnnotationProgress('preparing');
        try {
            if (await saveTask.current === false) throw new Error('Save the model successfully before starting again.');
            if (!active()) return;
            const document = await loadDrivingModel(modelPath.current);
            if (!active()) return;
            activateDocument(document);
            if (!train && !policy.current?.steps) throw new Error('Train a model before evaluation.');
            let references: RecordingReferences | null = null;
            let replay: RecordedDrivingReplay | undefined;
            let progress: RecordedReplayProgress = { samples: 0, skipped: 0, pedalError: 0, steeringError: 0 };
            let savedAt = policy.current?.steps ?? 0;
            await recordedViewer.current!.replay(async (frame) => {
                if (!active()) return;
                const sessionId = frame.session.id;
                if (references?.sessionId !== sessionId) {
                    const preview = recordedReferences.current;
                    if (preview?.sessionId === sessionId && preview.data) references = preview;
                    else references = await resolveRecordingReferences(frame.session);
                }
                if (!active()) return;
                const result = readRecorded(frame, references);
                sampledInputs.current = result;
                setInputs(result);
                const observation = result.observation;
                if (!replay && observation && (!policy.current || policy.current.matches(observation))) {
                    if (!policy.current) policy.current = new DrivingPolicy(observation.context,
                        observation.features.length, Math.random, configuration.current);
                    configuration.current = policy.current.configuration;
                    await policy.current.initializeGpu();
                    if (!active()) return;
                    setInitializingGpu(false);
                    replay = new RecordedDrivingReplay(policy.current, train);
                    Object.assign(replay.progress, progress);
                    progress = replay.progress;
                }
                if (replay) {
                    if (frame.startsRange) replay.startRange();
                    await replay.consume(result, active);
                } else {
                    progress.skipped++;
                    progress.lastError = result.error || 'Recorded inputs do not match this model.';
                }
                if (!active()) return;
                const steps = policy.current?.steps ?? savedAt;
                setModelSteps(policy.current?.steps ?? null);
                if (train && replay) setSaved(false);
                setReplayStatus(`${train ? 'Learning from' : 'Evaluating'} recording${frame.recordingName ? ` · ${frame.recordingName}` : ''} · ${frame.videoSeconds.toFixed(1)} s · ${progress.samples} samples · ${progress.skipped} skipped`
                    + (train ? '' : ` · pedal MAE ${progress.pedalError.toFixed(3)} · steering MAE ${progress.steeringError.toFixed(3)}`)
                    + (progress.lastError ? ` · Last skipped frame: ${progress.lastError}` : ''));
                if (train && steps - savedAt >= configuration.current.runtime.saveEverySteps) {
                    if (!await save()) throw new Error('Recorded training stopped because the model could not be saved.');
                    savedAt = steps;
                }
            }, abort.signal, train ? (position) => { if (active()) setAnnotationProgress(position); } : undefined);
            if (active()) {
                if (!progress.samples) throw new Error(progress.lastError || 'No consecutive aligned samples were available for replay.');
                setReplayStatus((status) => `Completed. ${status}`);
            }
        } catch (reason) {
            if (reason instanceof Error && reason.name === 'AbortError') abort.abort();
            else if (active()) { setError(errorMessage(reason)); setReplayStatus('Recorded replay failed.'); }
        } finally {
            if (mounted.current) {
                if (abort.signal.aborted) setReplayStatus((status) => `Stopped. ${status}`);
                if (train) await save();
                if (mounted.current) {
                    setAnnotationProgress(null);
                    setReplaying(false); setInitializingGpu(false); setModelSteps(policy.current?.steps ?? null);
                }
            }
            if (replayAbort.current === abort) replayAbort.current = null;
        }
    };
    const changeInputSource = (source: 'live' | 'recorded') => {
        if (running || replayAbort.current || viewerBusy || initializingGpu || testTimer.current) return;
        inputSourceRef.current = source;
        if (source === 'recorded') screenFrames.stop();
        setInputSource(source); setError(''); setReplayStatus(''); setInputs(read());
    };
    const changeSource = (source: ControlSource) => {
        setError('');
        if (inputSourceRef.current !== 'live' || !connected || testTimer.current) return;
        try {
            if (source === 'manual' && loop.current?.state.mode === 'idle') loop.current.startManual();
            else loop.current?.setControlSource(source);
        } catch (reason) { setError(errorMessage(reason)); }
    };
    const chooseModel = async (mode: 'new' | 'open') => {
        if (running || replaying || viewerBusy || initializingGpu || fileBusy) return;
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
    const liveStartBlocker = fileBusy ? 'Wait for the model file to finish loading.'
        : !filePath ? 'Create or open a model file.'
        : initializingGpu ? 'Preparing the model and waiting for fresh live inputs.'
        : finishing ? 'Finishing the current training run.'
        : aiActive ? 'AI is already running. Stop it before starting another run.'
        : testTimer.current ? 'Stop the controller calibration test.'
        : !connected ? 'Connect the virtual wheel.'
        : !observation ? inputs.error
        : mismatched ? inputMismatches.join(' ')
        : null;
    const liveStartStatus = liveStartBlocker ? `Cannot start AI: ${liveStartBlocker}`
        : !modelSteps ? 'Ready to train. Run trained model becomes available after the model has learned at least one step.'
        : 'Ready to train or run the loaded model.';
    const recordedStartDisabled = !filePath || fileBusy || !replayReady || replaying || viewerBusy || initializingGpu;
    const trainingPosition = annotationProgress === 'preparing' ? null : annotationProgress;
    const trainingProgress = trainingPosition
        ? (trainingPosition.recordingNumber - 1 + (trainingPosition.totalFrames ? trainingPosition.processedFrames / trainingPosition.totalFrames : 1))
            / trainingPosition.recordingCount
        : undefined;
    const trainingProgressText = trainingPosition
        ? `Recording ${trainingPosition.recordingNumber} of ${trainingPosition.recordingCount} · ${trainingPosition.processedFrames.toLocaleString()} / ${trainingPosition.totalFrames.toLocaleString()} frames processed`
        : 'Preparing annotation training…';

    return <section className="driving-training" aria-label="AI Self-Driving Training">
        <header className="driving-training-header">
            <div><span className="driving-eyebrow">SIMULATOR LAB</span><h1>AI Self-Driving Training</h1>
                <p>{inputSource === 'live' ? 'Two drivers. One virtual wheel. Choose who controls the car.'
                    : 'Learn from recorded driving and inspect what the camera model responds to.'}</p></div>
            <button onClick={onOpenLiveSession}>Open Live Session</button>
        </header>
        <section className="driving-card driving-source-picker" aria-label="Observation input source">
            <label htmlFor="driving-input-source">AI observation input source</label>
            <select id="driving-input-source" value={inputSource}
                disabled={running || replaying || viewerBusy || initializingGpu || connecting || disconnecting || Boolean(testTimer.current)}
                onChange={(event) => changeInputSource(event.target.value as 'live' | 'recorded')}>
                <option value="live">Live Session</option>
                <option value="recorded">Recorded session</option>
            </select>
            <p>{inputSource === 'live' ? 'Use live telemetry and frames from the shared game screen.'
                : 'Choose a working folder to find annotated recordings, or create an annotation from a video and telemetry.'}</p>
        </section>
        <div className="driving-status" role="status"><span data-active={running || replaying} />{inputSource === 'live'
            ? state.status : replayStatus || 'Add or edit training annotations, or select saved annotations for offline training.'}</div>
        {error && <p className="driving-error" role="alert">{error}</p>}
        {inputSource === 'live' && <section className="driving-routing" aria-label="Shared control system">
            <div className="driving-route-source" data-active={state.source === 'manual'}>
                <span className="driving-eyebrow">HUMAN INPUT</span><strong>Human takeover</strong>
                <span>Wheel, pedals or gamepad</span>
            </div>
            <span className="driving-route-arrow" aria-hidden="true">→</span>
            <div className="driving-route-owner">
                <span className="driving-eyebrow">CONTROL OWNER</span>
                <strong aria-label="Current control owner">{testMode.current ? 'Calibration test' : state.source === 'manual' ? 'Human' : state.source === 'ai' ? 'AI' : 'No driver'}</strong>
                <span>{connected ? `vJoy ${deviceId} → Simulator` : 'Connect vJoy to drive'}</span>
                <button className="driving-stop" disabled={finishing || (!running && !testTimer.current && !initializingGpu)}
                    onClick={() => { startGeneration.current++; stopTest(); void loop.current?.finishTraining(); }}>Stop</button>
            </div>
            <span className="driving-route-arrow" aria-hidden="true">←</span>
            <div className="driving-route-source" data-active={state.source === 'ai'}>
                <span className="driving-eyebrow">AI INPUT</span><strong>Driving policy</strong>
                <span>{finishing ? 'Finishing training' : aiActive ? state.mode === 'training' ? 'Training enabled' : 'Evaluation · learning off' : 'Not started'}</span>
            </div>
            <p>One driver owns steering, pedals and shifts at a time. Human takeover stays active until you explicitly return control to AI. Stop releases every input. During training, it also finishes learning from the current run and saves the model.</p>
        </section>}
        <div className="driving-grid">
            {inputSource === 'recorded' && <RecordedFrameViewer ref={recordedViewer} configuration={config}
                policy={policy.current} disabled={replaying || initializingGpu || fileBusy} trainingPickerContainer={trainingPickerContainer}
                onFrame={updateRecordedFrame} onBusyChange={setViewerBusy} onReplayReadyChange={setReplayReady} />}
            {inputSource === 'live' && <ManualDrivingControls ref={manualControls} connected={connected} running={running} source={state.source}
                finishing={finishing} testing={Boolean(testMode.current)} aiActive={aiActive} aiReady={Boolean(observation && !mismatched)} onSourceChange={changeSource} />}
            <section className="driving-card driving-ai-input" aria-label="AI driver">
                <div className="driving-card-title"><h2>AI driver</h2><span>{inputSource === 'recorded' ? replaying ? 'Recorded replay' : 'Offline'
                    : finishing || (aiActive && !state.source) ? 'Finishing learning' : aiActive ? state.source === 'ai' ? 'Driving' : 'Human has control' : 'Not started'}</span></div>
                <p>{inputSource === 'live' ? 'Start AI training or evaluation when live inputs are ready. If you are already driving, you keep control until Return to AI.'
                    : 'Recorded training learns signed pedal intent, steering values plus single gear changes from aligned telemetry. Evaluation compares the fixed model with recorded controls. Gaps reset observation history.'}</p>
                <p>CNN–LSTM soft actor–critic · {config.network.hiddenUnits} recurrent units · continuous throttle/brake and steering, plus learned hold gear, shift up and shift down. Throttle/brake ranges from −1 (full brake) through 0 (both released) to 1 (full throttle).</p>
                <p>{config.memory.windowMs / 1000}-second observation history covers {fullObservationHistory ? 'all AI observation inputs' : 'the observation inputs selected in this saved model'} during training and evaluation, sampled in {config.memory.intervalMs} ms bins. Memory starts fresh after stopping, changing drivers or interrupted inputs.</p>
                {inputSource === 'live' && <p>During training, SAC uses up to {config.learning.sequenceSeconds} seconds of recorded rewards to learn how earlier controls affect later outcomes. The current plan keeps running as the model learns.</p>}
                <p>One action plan predicts {config.actions.planTimesSeconds.length} control steps from 0 to {planEnd} seconds. The AI smoothly blends the signed pedal command and steering between steps while computing. Gear shifts occur at their scheduled steps. The remaining plan continues when fresh inference agrees, and is replaced when it disagrees or expires. During training, each step that contributes to a command learns which controls are likely to earn more reward after penalties. Exploration adds random variation around those choices. Human driving teaches future steps from demonstrated actions.</p>
                <p className="driving-note">Each model JSON contains its inputs, memory, network, training settings and weights. Training and evaluation load that JSON each time they start.</p>
                {inputSource === 'live' && <p className="driving-note">Use the same model across games, cars and maps. After switching, stop and restart training or evaluation with fresh live inputs and a matching Live Map reference lap. Learned weights are preserved.</p>}
                {inputSource === 'live' && <fieldset className="driving-exploration"
                    disabled={!filePath || !documentReady.current || fileBusy || initializingGpu || running || replaying || viewerBusy}>
                    <legend>Training exploration</legend>
                    <div className="driving-aggression">
                        <div><label htmlFor="driving-throttle-exploration-speed-limit">Throttle exploration speed limit</label>
                            <output htmlFor="driving-throttle-exploration-speed-limit">{exploration.throttleExplorationSpeedLimitKmh} km/h</output></div>
                        <input id="driving-throttle-exploration-speed-limit" type="range" min="0"
                            max={DEFAULT_DRIVING_CONFIG.learning.throttleExplorationSpeedLimitKmh} step="1"
                            value={exploration.throttleExplorationSpeedLimitKmh} aria-valuetext={`${exploration.throttleExplorationSpeedLimitKmh} km/h`}
                            aria-describedby="driving-throttle-exploration-speed-limit-meaning"
                            onChange={(event) => changeExploration({ ...exploration, throttleExplorationSpeedLimitKmh: Number(event.target.value) })} />
                        <p id="driving-throttle-exploration-speed-limit-meaning">At or above this speed, exploration adds no extra throttle during training.</p>
                    </div>
                    {EXPLORATION_CONTROLS.map(({ key, label, control }) => <div className="driving-aggression" key={key}>
                        <div><label htmlFor={`driving-${control}-exploration`}>{label} exploration</label>
                            <output htmlFor={`driving-${control}-exploration`}>{Number(exploration[key].toPrecision(3))}</output></div>
                        <input id={`driving-${control}-exploration`} type="range" min={Math.min(0.001, exploration[key])}
                            max={Math.max(1, exploration[key])} step="0.001" value={exploration[key]}
                            aria-describedby={`driving-${control}-exploration-meaning`}
                            onChange={(event) => changeExploration({ ...exploration, [key]: Number(event.target.value) })} />
                        <p id={`driving-${control}-exploration-meaning`}>Higher values add stronger state-dependent variation to {control} choices, evolving smoothly with the observed driving state. Default: {DEFAULT_DRIVING_CONFIG.learning[key]}.</p>
                    </div>)}
                    <div className="driving-aggression">
                        <div><label htmlFor="driving-shift-temperature">Gear exploration</label>
                            <output htmlFor="driving-shift-temperature">{Number(exploration.shiftTemperature.toPrecision(3))}</output></div>
                        <input id="driving-shift-temperature" type="range" min="0.001" max="1" step="0.001" value={exploration.shiftTemperature}
                            aria-describedby="driving-shift-temperature-meaning"
                            onChange={(event) => changeExploration({ ...exploration, shiftTemperature: Number(event.target.value) })} />
                        <p id="driving-shift-temperature-meaning">Higher values try a wider mix of gear choices. Lower values favor the model’s preferred gear choice. Default: {DEFAULT_DRIVING_CONFIG.learning.shiftTemperature}.</p>
                    </div>
                    <div className="driving-output-options">
                        <label htmlFor="driving-max-exploration-gear">Maximum gear to explore</label>
                        <select id="driving-max-exploration-gear" value={exploration.maxExplorationGear}
                            aria-describedby="driving-max-exploration-gear-meaning"
                            onChange={(event) => changeExploration({ ...exploration, maxExplorationGear: Number(event.target.value) })}>
                            <option value={0}>Unlimited</option>
                            {Array.from({ length: MAX_EXPLORATION_GEAR }, (_, i) =>
                                <option key={i + 1} value={i + 1}>{i + 1}</option>)}
                        </select>
                    </div>
                    <p id="driving-max-exploration-gear-meaning">During live training, AI upshifts stop at the selected gear. Downshifts remain available.</p>
                    {EXPLORATION_FREQUENCIES.map(({ key, label, control }) => <div className="driving-aggression" key={key}>
                        <div><label htmlFor={`driving-${control}-exploration-frequency`}>{label} exploration update frequency</label>
                            <output htmlFor={`driving-${control}-exploration-frequency`}>{exploration[key]} Hz · every {Number((1 / exploration[key]).toPrecision(3))} s</output></div>
                        <input id={`driving-${control}-exploration-frequency`} type="range" min="0.1" max="10" step="0.1"
                            value={exploration[key]} aria-describedby={`driving-${control}-exploration-frequency-meaning`}
                            onChange={(event) => changeExploration({ ...exploration, [key]: Number(event.target.value) })} />
                        <p id={`driving-${control}-exploration-frequency-meaning`}>{control === 'gear'
                            ? 'How often a new gear exploration draw is sampled.'
                            : 'How often exploration weights refresh, blending smoothly between samples. Lower values keep exploration coherent longer while it responds to the driving state.'} Default: {DEFAULT_DRIVING_CONFIG.learning[key]} Hz.</p>
                    </div>)}
                    <button onClick={() => changeExploration({ pedalExplorationSigma: DEFAULT_DRIVING_CONFIG.learning.pedalExplorationSigma,
                        throttleExplorationSpeedLimitKmh: DEFAULT_DRIVING_CONFIG.learning.throttleExplorationSpeedLimitKmh,
                        steeringExplorationSigma: DEFAULT_DRIVING_CONFIG.learning.steeringExplorationSigma,
                        shiftTemperature: DEFAULT_DRIVING_CONFIG.learning.shiftTemperature,
                        maxExplorationGear: DEFAULT_DRIVING_CONFIG.learning.maxExplorationGear,
                        pedalExplorationUpdateHz: DEFAULT_DRIVING_CONFIG.learning.pedalExplorationUpdateHz,
                        steeringExplorationUpdateHz: DEFAULT_DRIVING_CONFIG.learning.steeringExplorationUpdateHz,
                        gearExplorationUpdateHz: DEFAULT_DRIVING_CONFIG.learning.gearExplorationUpdateHz })}>Reset exploration</button>
                    <p className="driving-note">Saved with this model. Stop training to adjust exploration for the next run. Evaluation uses the model’s preferred actions without random exploration.</p>
                </fieldset>}
                {inputSource === 'live' && <table className="driving-pedals" aria-label="Planned driving actions">
                    <caption>Action plan · 0–{planEnd} seconds</caption>
                    <thead><tr><th scope="col">Time</th><th scope="col">Throttle/brake</th><th scope="col">Steering</th><th scope="col">Shift</th></tr></thead>
                    <tbody>{state.plan ? state.plan.map(({ timeSeconds, action }) => <tr key={timeSeconds}>
                        <th scope="row">{timeSeconds.toFixed(1)} s</th>
                        <td>{action.pedal.toFixed(3)}</td>
                        <td>{action.steering.toFixed(3)}</td><td>{action.shiftUp ? 'Up' : action.shiftDown ? 'Down' : 'Hold'}</td>
                    </tr>) : <tr><td colSpan={4}>Waiting for AI control and fresh inputs.</td></tr>}</tbody>
                </table>}
                {inputSource === 'live' && <p className="driving-note">Disable automatic gears in the simulator and bind both shift buttons before training. The model learns gear choices from driving rewards. Human and AI shifts have no cooldown. Shifts use {config.actions.shiftPulseMs / 1000}-second presses; a new shift releases the previous press immediately. Downshifts are allowed in every gear, including neutral and reverse.</p>}
                <div className="driving-metrics"><div><strong>{modelSteps ?? 0}</strong><span>Learning steps</span></div>
                    {inputSource === 'live' && <div><strong>{state.episodeReward}</strong><span>Episode reward</span></div>}</div>
                <button disabled={!policy.current || measuringUpdate || initializingGpu || fileBusy}
                    onClick={() => { void measureUpdate(); }}>{measuringUpdate ? 'Measuring update…' : 'Measure last update'}</button>
                {updateDiagnostics && <div aria-label="Last training update">
                    <dl className="driving-details">
                        <dt>Gradient norm before clipping</dt><dd>{updateDiagnostics.gradientNorm.toExponential(3)}</dd>
                        <dt>Clip scale</dt><dd>{updateDiagnostics.clipScale.toFixed(4)}</dd>
                        <dt>Proposed update norm</dt><dd>{updateDiagnostics.updateNorm.toExponential(3)}</dd>
                        <dt>Base learning rate</dt><dd>{updateDiagnostics.learningRate.toExponential(3)}</dd>
                        <dt>Accumulated replay samples</dt><dd>{updateDiagnostics.samples}</dd>
                    </dl>
                    <p className="driving-note">Snapshot of the latest completed update. Weight bounds can reduce the applied update. A human or recorded training tick can apply several separately clipped updates.</p>
                    {!updateDiagnostics.finite && <p className="driving-error" role="alert">The update was skipped because its gradient was not finite.</p>}
                </div>}
                {updateDiagnostics === undefined && <p className="driving-note" role="status">No completed training update in this model session yet.</p>}
                {mismatched && <div className="driving-error" role="alert">
                    <p>The saved model’s input configuration does not match the live inputs.</p>
                    <ul>{inputMismatches.map((difference) => <li key={difference}>{difference}</li>)}</ul>
                    <p>Use matching live inputs, open a matching model file, or create a new model.</p>
                </div>}
                {inputSource === 'live' ? <>
                    <div className="driving-buttons">
                        <button className="driving-primary" disabled={Boolean(liveStartBlocker)} aria-describedby="driving-start-status" onClick={() => { void start(true); }}>Start training</button>
                        <button disabled={Boolean(liveStartBlocker) || !modelSteps} aria-describedby="driving-start-status" onClick={() => { void start(false); }}>Run trained model</button>
                    </div>
                    <p id="driving-start-status" className="driving-note" role="status" aria-label="AI start status">{liveStartStatus}</p>
                </> : <>
                    <div ref={setTrainingPickerContainer} />
                    <div className="driving-buttons" role="group" aria-label="Recorded training controls">
                        <button className="driving-primary" disabled={recordedStartDisabled}
                            onClick={() => { void startRecorded(true); }}>Train from annotations</button>
                        <button disabled={recordedStartDisabled || !modelSteps}
                            onClick={() => { void startRecorded(false); }}>Evaluate annotations</button>
                        <button className="driving-stop" disabled={!replaying} onClick={() => replayAbort.current?.abort()}>Stop replay</button>
                    </div>
                </>}
                {annotationProgress !== null && <div className="driving-annotation-progress">
                    <progress aria-label="Annotation training progress" aria-valuetext={trainingProgressText}
                        max={1} value={trainingProgress} />
                    <span>{trainingProgressText}</span>
                </div>}
                <p className="driving-model-path"><strong>Model file</strong><br />{filePath || 'Choose New model to select a save location, or Open model to continue training a saved file.'}</p>
                <div className="driving-buttons">
                    <button disabled={running || replaying || viewerBusy || initializingGpu || fileBusy} onClick={() => { void chooseModel('new'); }}>New model</button>
                    <button disabled={running || replaying || viewerBusy || initializingGpu || fileBusy} onClick={() => { void chooseModel('open'); }}>Open model</button>
                    <button disabled={!filePath || fileBusy || replaying || viewerBusy || initializingGpu || saving || !documentReady.current} onClick={() => { setError(''); void save(); }}>Save model</button>
                </div>
                <p className="driving-note" role="status">{saving ? 'Saving model…' : saved ? 'Model saved to file.' : 'Model not saved.'} The complete model saves to the selected file every {config.runtime.saveEverySteps} learning steps and when stopped.</p>
                {inputSource === 'live' ? <p className="driving-note">Training explores controls; use a practice session and reposition the car between episodes. A new policy starts untrained.</p>
                    : <p className="driving-note">Replay samples every {Math.max(100, config.runtime.controlIntervalMs)} ms of video. Pedal MAE measures error on the signed throttle/brake command. Steering MAE measures steering error. Both use normalized controls and do not measure image recognition accuracy. A new policy starts untrained.</p>}
            </section>
            {inputSource === 'live' && <section className="driving-card driving-shared-output" aria-label="Shared vJoy output">
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
            </section>}
            <section className="driving-card driving-live-inputs" aria-label="AI observation inputs">
                <h2>AI observation inputs</h2>
                <p>{inputSource === 'live' ? 'Start live telemetry, load a circuit map with a centerline in Live Map, then select Share game screen in the Live Session recording bar.'
                    : 'Video frames pair with the most recent telemetry sample at or before their timestamp. Training uses saved reference laps and circuit maps matching the recording.'}</p>
                {inputSource === 'recorded' && <button disabled={!recordedSession || replaying || viewerBusy || initializingGpu}
                    onClick={() => setReferenceAttempt((attempt) => attempt + 1)}>Reload recorded references</button>}
                <div className="driving-input-status" data-ready={Boolean(observation)}>{observation ? 'All model inputs ready' : inputs.error}</div>
                <div className="driving-aggression">
                    <div><label htmlFor="driving-aggression">Driving aggression</label><output htmlFor="driving-aggression">{aggression}</output></div>
                    <input id="driving-aggression" type="range" min="0" max="100" step="1" value={aggression} disabled={fileBusy || initializingGpu || replaying || viewerBusy}
                        aria-describedby="driving-aggression-meaning" onChange={(event) => changeAggression(Number(event.target.value))} />
                    <div className="driving-aggression-scale"><span>0 · Cautious</span><span>50 · Reference</span><span>100 · Aggressive</span></div>
                    <p id="driving-aggression-meaning">{aggression < 50
                        ? `Uses the same speed targets as Live Phrases at ${100 - 2 * aggression}% braking adjustment. 0 matches Live Phrases 100%; 50 follows the recorded reference.`
                        : aggression > 50 ? `Raises reference speed by ${aggression - 50}%. 50 follows the recorded reference; 100 targets 50% higher speed.`
                            : 'Reference uses the recorded speed targets (Live Phrases 0% braking adjustment).'}</p>
                    <button disabled={fileBusy || initializingGpu || replaying || viewerBusy} onClick={() => changeAggression(50)}>Reset to reference</button>
                </div>
                <dl className="driving-details">
                    <dt>Game / car</dt><dd>{observation ? `${observation.context.game} / ${observation.context.car}` : 'Waiting'}</dd>
                    <dt>Map</dt><dd>{observation?.context.map ?? 'Waiting'}</dd>
                    <dt>Centerline</dt><dd>{observation ? '50 steps behind · 50 steps ahead' : 'Waiting'}</dd>
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
                <p className="driving-note">The policy uses car-local velocity XYZ, g-force XYZ, yaw rate, speed, current gear, engine RPM, game, car, map, normalized position, current controls, camera pixels, relative centerline XYZ, aggression, reference speed and target speed. Centerline input covers 50 map steps behind and 50 ahead, relative to the centerline at the current lap position, wrapping at start/finish. Speed targets follow the Live Phrases calculation up to aggression 50, then increase reference speed linearly up to 50% at aggression 100. The policy generates the pedal commands. Aggression applies during training and evaluation; further training teaches its response. Evaluation does not update weights. Models saved with a different observation input set or aggression calculation need to be trained again.</p>
            </section>
            {inputSource === 'live' && <section className="driving-card" aria-label="Reward signal">
                <h2>Reward signal</h2>
                <dl className="driving-rewards">
                    <div><dt>Acceleration / deceleration</dt><dd>{state.reward?.timing ?? 0}</dd>
                        <p>The penalty grows with the size of the speed gap and the time spent away from the target. Reaching the target sooner reduces the cost.</p>
                        {state.reward && <p>Last scored at aggression {state.reward.speedComparison.aggression}. Signed speed {state.reward.speedComparison.measured.toFixed(1)} km/h, reference {state.reward.speedComparison.reference.toFixed(1)} km/h, target {state.reward.speedComparison.target.toFixed(1)} km/h. Scored speed gap {state.reward.speedComparison.speedGapTime.toFixed(2)} (km/h)·s. Measured acceleration {state.reward.speedComparison.measuredAcceleration.toFixed(2)} m/s².</p>}
                    </div>
                    <div><dt>Normalized position progress</dt><dd>{state.reward?.progress ?? 0}</dd></div>
                    <div><dt>Steering-change penalty</dt><dd>{state.reward?.steeringChange ?? 0}</dd>
                        <p>Turning the wheel in one direction costs zero, even through center. A penalty applies when you reverse direction or resume turning after a pause.</p>
                    </div>
                    <div><dt>Throttle-change penalty</dt><dd>{state.reward?.throttleChange ?? 0}</dd>
                        <p>Continuously pressing or releasing the throttle costs zero. A penalty applies when you reverse direction or resume moving the pedal after a pause.</p>
                    </div>
                    <div><dt>Brake-change penalty</dt><dd>{state.reward?.brakeChange ?? 0}</dd>
                        <p>Faster changes in brake position cost more.</p>
                    </div>
                    <div><dt>On-track reward</dt><dd>{state.reward?.onTrack ?? 0}</dd>
                        <p>Earns 1 point per second while moving forward on a confirmed track surface.</p>
                    </div>
                    <div><dt>Off-track penalty</dt><dd>{state.reward?.offTrack ?? 0}</dd>
                        <p aria-label="Off-track detection">{!state.reward ? 'Waiting for a scored transition.'
                            : state.reward.offTrackDetected === null ? 'Off-track detection unavailable: missing telemetry or matching track boundaries.'
                            : state.reward.offTrackDetected ? 'Off track' : 'On track'}</p>
                    </div>
                    <div><dt>Impact penalty</dt><dd>{state.reward?.impact ?? 0}</dd>
                        <p aria-label="Impact estimate">{!state.reward ? 'Waiting for two fresh motion samples.'
                            : !state.reward.impactEstimate.available ? 'Impact estimate unavailable: no reliable 20–500 ms telemetry interval.'
                            : `${state.reward.impactEstimate.detected ? 'Possible impact' : 'No impact detected'} · Confidence ${(state.reward.impactEstimate.confidence * 100).toFixed(0)}% · Estimated severity ${(state.reward.impactEstimate.severity * 100).toFixed(0)}/100 · Excess Δv ${state.reward.impactEstimate.deltaV.toFixed(2)} m/s`}</p>
                    </div>
                    <div><dt>Blocked-motion penalty</dt><dd>{state.reward?.blockedMotion ?? 0}</dd>
                        <p>Penalizes sustained throttle with little forward motion. It also flags possible wall riding when
                            steering is held to one side and the car repeatedly rebounds within two seconds without turning consistently.</p>
                    </div>
                </dl>
            </section>}
            {inputSource === 'live' && <ObservationInputDelays snapshot={inputs.delays} telemetryMaxAgeMs={config.runtime.telemetryMaxAgeMs}
                visionMaxAgeMs={config.runtime.visionMaxAgeMs} />}
        </div>
    </section>;
}
