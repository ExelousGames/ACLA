import React, { forwardRef, useCallback, useEffect, useId, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { NamedOperationComponentHandle, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { captureGameScreen, ScreenCaptureSource } from './screen-capture';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { DEPTH_INPUT_SIZE, MODEL_INPUT_RESOLUTIONS, VISION_INPUT_SIZE } from './vision-config';
import { CameraCalibration, DETECTION_TASKS, DetectionTask, TrackVisionAnalysis, TrackVisionDetection, TrackVisionFrame, VISION_MAX_AGE_MS } from './track-vision-types';
import { VISION_CONFIDENCE } from './semantic-scene';
import { analyzeTrackPositions } from './track-position-analysis';
import { reconstructTrack } from './track-reconstruction';
import { DEFAULT_CAMERA, validCalibration } from './camera-projection';
import TrackCalibration, { CameraGroundGrid } from './TrackCalibration';
import { drawVisionOverlay } from './vision-overlay';
import ReconstructedSceneView from './ReconstructedSceneView';
import BirdsEyeView from './BirdsEyeView';
import { reconstructScene } from './reconstructed-scene';
import { projectBirdsEyeScene } from './birds-eye-scene';
import { drawLabelDepths, filteredFrame, filteredMasks, PIPELINE_STEPS, PipelineStep } from './pipeline-visuals';
import PipelineDetails from './PipelineDetails';
import { createDepthMap, depthAtMouse, drawDepthMap, formatDepth } from './depth-map';
import './LiveTrackVision.css';

export interface TrackVisionHandle extends NamedOperationComponentHandle {
    getLatestDetection(): TrackVisionDetection | null;
    subscribeDetection(listener: () => void): () => void;
}

const message = (error: unknown) => error instanceof Error ? error.message : 'Vision detection failed.';
const GPU_RETRY_DELAY_MS = 3000;
type DetectorState = { status: 'loading' | 'ready' | 'retrying' | 'error'; error?: string; classNames?: string[] };

const LiveTrackVision = forwardRef<TrackVisionHandle, { name: string }>(({ name }, forwardedRef) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const previewRef = useRef<HTMLDialogElement>(null);
    const previewFrameRef = useRef<HTMLCanvasElement | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const models = useRef<Partial<Record<DetectionTask, TrackVisionModel>>>({});
    const modelQueue = useRef<Promise<void>>(Promise.resolve());
    const inferenceRef = useRef<Promise<unknown>>(Promise.resolve());
    const captureVersion = useRef(0);
    const modelVersion = useRef(0);
    const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    const latest = useRef<TrackVisionDetection | null>(null);
    const analysisExpiry = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    const [analysis, setAnalysis] = useState<TrackVisionAnalysis | null>(null);
    const listeners = useRef(new Set<() => void>());
    const [sources, setSources] = useState<ScreenCaptureSource[]>([]);
    const [sourceId, setSourceId] = useState('');
    const [retry, setRetry] = useState(0);
    const [inputSizes, setInputSizes] = useState({ segment: VISION_INPUT_SIZE, depth: DEPTH_INPUT_SIZE });
    const [detectors, setDetectors] = useState<Record<DetectionTask, DetectorState>>({
        segment: { status: 'loading' }, depth: { status: 'loading' },
    });
    const [captureState, setCaptureState] = useState<'idle' | 'starting' | 'active'>('idle');
    const [error, setError] = useState('');
    const [status, setStatus] = useState('Capture idle');
    const [hasFrame, setHasFrame] = useState(false);
    const [previewCapturedAt, setPreviewCapturedAt] = useState<number>();
    const [previewExpanded, setPreviewExpanded] = useState(false);
    const [step, setStep] = useState<PipelineStep>('capture');
    const pipelineId = useId();
    const [confidence, setConfidence] = useState(0.5);
    const [filterConfidence, setFilterConfidence] = useState(VISION_CONFIDENCE);
    const [displayLabel, setDisplayLabel] = useState('');
    const [cameraDraft, setCameraDraft] = useState(DEFAULT_CAMERA);
    const [calibration, setCalibration] = useState<CameraCalibration>();
    const [showCalibrationOnCapture, setShowCalibrationOnCapture] = useState(false);
    const cameraCalibration = useRef<CameraCalibration | undefined>(undefined);
    const [previewResult, setPreviewResult] = useState<TrackVisionDetection | null>(null);
    const [depthPointer, setDepthPointer] = useState<{ clientX: number; clientY: number } | null>(null);
    const depthMap = useMemo(() => step === 'depth-map' ? createDepthMap(previewResult) : null, [previewResult, step]);
    const depthRect = depthMap && depthPointer ? canvasRef.current?.getBoundingClientRect() : null;
    const hoveredDepth = depthMap && depthPointer && depthRect
        ? depthAtMouse(depthMap, depthRect, depthPointer.clientX, depthPointer.clientY) : null;
    const showFilteredMasks = step === 'filtering' || step === 'depth';
    const masks = useMemo(() => showFilteredMasks ? filteredMasks(previewResult) : [], [previewResult, showFilteredMasks]);
    const previewWidth = previewFrameRef.current?.width ?? 0, previewHeight = previewFrameRef.current?.height ?? 0;
    const previewCamera = useMemo(() => ({ ...cameraDraft, imageWidth: previewWidth, imageHeight: previewHeight }),
        [cameraDraft, previewWidth, previewHeight]);
    const options = useRef({ confidence, filterConfidence });
    options.current = { confidence, filterConfidence };

    const togglePreviewSize = () => {
        const preview = previewRef.current;
        if (!preview) return;
        // Keep the video and canvas mounted while moving the preview into the browser's top layer.
        preview.close();
        if (previewExpanded) preview.show();
        else preview.showModal();
        setPreviewExpanded((current) => !current);
    };

    useEffect(() => {
        const frame = previewFrameRef.current;
        const canvas = canvasRef.current;
        if (!frame || !canvas) return;
        canvas.width = frame.width;
        canvas.height = frame.height;
        const preview = canvas.getContext('2d');
        if (!preview) return;
        preview.drawImage(frame, 0, 0);
        if (!previewResult) return;
        if (step === 'segmentation') drawVisionOverlay(preview, previewResult, displayLabel);
        if (step === 'filtering') drawVisionOverlay(preview, filteredFrame(previewResult, masks));
        if (step === 'depth-map' && depthMap) drawDepthMap(preview, depthMap);
        if (step === 'depth') drawLabelDepths(preview, previewResult, masks);
    }, [previewResult, previewCapturedAt, step, displayLabel, masks, depthMap]);

    useEffect(() => { setDepthPointer(null); }, [step, previewExpanded]);

    useEffect(() => {
        if (detectors.segment.status === 'ready' && displayLabel && !detectors.segment.classNames?.includes(displayLabel)) {
            setDisplayLabel('');
        }
    }, [detectors.segment, displayLabel]);

    const publish = useCallback((result: TrackVisionFrame | null) => {
        clearTimeout(analysisExpiry.current);
        if (result) result = { ...result, filterConfidence: options.current.filterConfidence };
        const reconstruction = reconstructTrack(result);
        const geometry = reconstruction?.geometry ?? null;
        const scene = result?.detections.segment?.task === 'segment' ? analyzeTrackPositions(result, reconstruction) : null;
        const reconstructedScene = reconstructScene(result);
        const birdsEyeScene = projectBirdsEyeScene(reconstructedScene, result?.calibration);
        latest.current = result ? { ...result, reconstruction, reconstructedScene, birdsEyeScene, geometry, analysis: scene } : null;
        setPreviewResult(latest.current);
        if (!result) setDepthPointer(null);
        const remaining = result ? result.capturedAt + VISION_MAX_AGE_MS - Date.now() : 0;
        setAnalysis(remaining > 0 ? scene : null);
        if (scene && remaining > 0) analysisExpiry.current = setTimeout(() => setAnalysis(null), remaining);
        listeners.current.forEach((listener) => listener());
    }, []);
    const updateCalibration = (camera?: CameraCalibration) => {
        const frame = previewFrameRef.current;
        const result = latest.current;
        cameraCalibration.current = frame && validCalibration(camera) ? camera : undefined;
        setCalibration(cameraCalibration.current);
        if (result) publish({ ...result, calibration: cameraCalibration.current });
    };
    const handle = useMemo<TrackVisionHandle>(() => ({
        getComponentName: () => name,
        getLatestDetection: () => latest.current,
        subscribeDetection: (listener) => {
            listeners.current.add(listener);
            return () => { listeners.current.delete(listener); };
        },
    }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandle = useRef(handle);
    registeredHandle.current = handle;
    useRegisterOperationComponentRef(registeredHandle);

    const releaseCapture = useCallback(() => {
        captureVersion.current++;
        clearTimeout(timerRef.current);
        streamRef.current?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
        streamRef.current = null;
        previewFrameRef.current = null;
        setPreviewCapturedAt(undefined);
        cameraCalibration.current = undefined;
        setCalibration(undefined);
        setShowCalibrationOnCapture(false);
        if (videoRef.current) videoRef.current.srcObject = null;
        publish(null);
    }, [publish]);
    const stop = useCallback(() => {
        releaseCapture();
        setCaptureState('idle');
        setHasFrame(false);
        setStatus('Screen capture stopped.');
        const canvas = canvasRef.current;
        if (canvas && !canvas.hidden) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    }, [releaseCapture]);

    useEffect(() => () => {
        releaseCapture();
        modelVersion.current++;
        Object.values(models.current).forEach((model) => { void model.dispose().catch(() => undefined); });
        models.current = {};
    }, [releaseCapture]);

    const refreshSources = async () => {
        setError('');
        try {
            const available = await window.screenCapture!.listSources();
            setSources(available);
            setSourceId((current) => available.some(({ id }) => id === current) ? current : '');
            if (!available.length) setError('No capture sources found. Open the simulator and refresh.');
        } catch (reason) { setError(message(reason)); }
    };

    useEffect(() => {
        if (!window.screenCapture) return;
        const version = ++modelVersion.current;
        publish(null);
        const disposals: Promise<void>[] = [];
        for (const { id } of DETECTION_TASKS) {
            const model = models.current[id];
            if (model && model.inputSize !== inputSizes[id]) {
                delete models.current[id];
                disposals.push(model.dispose().catch(() => undefined));
            }
        }
        setDetectors(Object.fromEntries(DETECTION_TASKS.map(({ id }) => {
            const model = models.current[id];
            return [id, model ? {
                status: 'ready', classNames: model.classNames,
            } : { status: 'loading' }];
        })) as Record<DetectionTask, DetectorState>);
        // Serialize retries and resolution changes so only one session per model is loaded.
        modelQueue.current = modelQueue.current.then(async () => {
            await Promise.all(disposals);
            for (const { id } of DETECTION_TASKS) {
                if (version !== modelVersion.current) return;
                if (models.current[id]) continue;
                try {
                    const model = id === 'depth'
                        ? await TrackVisionModel.loadBuiltin(id, inputSizes[id])
                        : await TrackVisionModel.loadBackend(inputSizes[id]);
                    if (version !== modelVersion.current) { await model.dispose().catch(() => undefined); return; }
                    models.current[id] = model;
                    setDetectors((current) => ({ ...current, [id]: {
                        status: 'ready', classNames: model.classNames,
                    } }));
                } catch (reason) {
                    if (version === modelVersion.current) setDetectors((current) => ({ ...current, [id]: {
                        status: reason instanceof GpuInferenceError ? 'retrying' : 'error', error: message(reason),
                    } }));
                }
            }
        });
        // Retrying increments the version above; unmount cleanup invalidates it too.
    }, [retry, publish, inputSizes]);

    useEffect(() => {
        const active = Object.values(detectors);
        // Let the current load queue finish before scheduling another GPU attempt.
        if (active.some(({ status }) => status === 'loading') || !active.some(({ status }) => status === 'retrying')) return;
        const timer = setTimeout(() => setRetry((current) => current + 1), GPU_RETRY_DELAY_MS);
        return () => clearTimeout(timer);
    }, [detectors]);

    const start = async () => {
        releaseCapture();
        const version = captureVersion.current;
        setCaptureState('starting');
        setError('');
        setStatus('Waiting for screen selection…');
        try {
            const stream = await captureGameScreen(sourceId);
            if (version !== captureVersion.current) { stream.getTracks().forEach((track) => track.stop()); return; }
            streamRef.current = stream;
            stream.getVideoTracks().forEach((track) => { track.onended = stop; });
            const video = videoRef.current!;
            video.srcObject = stream;
            await video.play();
            await inferenceRef.current.catch(() => undefined);
            if (version !== captureVersion.current) return;
            setCaptureState('active');
            const frame = document.createElement('canvas');
            const tick = async () => {
                if (version !== captureVersion.current) return;
                const started = performance.now();
                try {
                    if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
                        frame.width = video.videoWidth;
                        frame.height = video.videoHeight;
                        const context = frame.getContext('2d');
                        if (!context) throw new Error('Screen preview is unavailable.');
                        context.drawImage(video, 0, 0);
                        const stackVersion = modelVersion.current;
                        const frameConfidence = options.current.confidence;
                        const result: TrackVisionFrame = { capturedAt: Date.now(), width: frame.width, height: frame.height, detections: {} };
                        const inference = (async () => {
                            if (!models.current.segment || !models.current.depth) return;
                            for (const { id } of DETECTION_TASKS) {
                                if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                const model = models.current[id]!;
                                try {
                                    if (id === 'depth') {
                                        const segment = result.detections.segment;
                                        // Preserve cockpit pixels for depth; their mask is used by downstream edge filtering.
                                        const region = segment?.task === 'segment' && segment.width > 0 && segment.height > 0
                                            ? { width: segment.width, height: segment.height, mask: new Uint8Array(segment.width * segment.height).fill(1) } : null;
                                        if (!region) throw new Error('Depth requires a valid segmentation grid.');
                                        result.detections.depth = await model.detect(frame, frameConfidence, region);
                                    } else result.detections[id] = await model.detect(frame, frameConfidence);
                                }
                                catch (reason) {
                                    if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                    delete models.current[id];
                                    void model.dispose().catch(() => undefined);
                                    setDetectors((current) => ({ ...current, [id]: {
                                        status: 'retrying', error: message(reason),
                                    } }));
                                    return;
                                }
                            }
                        })();
                        inferenceRef.current = inference;
                        await inference;
                        if (version !== captureVersion.current) return;
                        if (stackVersion !== modelVersion.current) {
                            timerRef.current = setTimeout(tick, 0);
                            return;
                        }
                        // Keep a clean copy for the camera calibration preview.
                        const previewFrame = previewFrameRef.current ?? document.createElement('canvas');
                        previewFrame.width = frame.width;
                        previewFrame.height = frame.height;
                        const previewContext = previewFrame.getContext('2d');
                        if (!previewContext) throw new Error('Screen preview is unavailable.');
                        previewContext.drawImage(frame, 0, 0);
                        previewFrameRef.current = previewFrame;
                        setPreviewCapturedAt(result.capturedAt);
                        // Read after inference so a pending frame cannot restore an old calibration.
                        if (cameraCalibration.current && (cameraCalibration.current.imageWidth !== frame.width || cameraCalibration.current.imageHeight !== frame.height)) {
                            cameraCalibration.current = undefined;
                            setCalibration(undefined);
                        }
                        result.calibration = cameraCalibration.current;
                        const complete = Boolean(result.detections.segment && result.detections.depth);
                        publish(complete ? result : null);
                        setHasFrame(true);
                        setStatus(complete ? DETECTION_TASKS.map(({ id, label }) => `${label} · ${Math.round(result.detections[id]!.inferenceMs)} ms`).join(' / ')
                            : 'Screen shared. Waiting for segmentation and depth.');
                    }
                    timerRef.current = setTimeout(tick, Math.max(0, 200 - (performance.now() - started)));
                } catch (reason) {
                    if (version !== captureVersion.current) return;
                    stop();
                    setError(message(reason));
                }
            };
            void tick();
        } catch (reason) {
            if (version !== captureVersion.current) return;
            stop();
            setError(reason instanceof Error && reason.name === 'NotAllowedError'
                ? 'Screen sharing was cancelled or denied. Share again and select your game window.' : message(reason));
        }
    };

    if (!window.screenCapture) {
        return <section className="track-vision" aria-label="Track Vision">
            <p role="alert">Track Vision is available only in the Electron desktop app.</p>
        </section>;
    }

    const activeStep = PIPELINE_STEPS.find((item) => item.id === step)!;
    const stepIndex = PIPELINE_STEPS.indexOf(activeStep);
    const isSceneStep = step === 'scene';
    const labels = previewResult?.detections.segment?.classNames ?? detectors.segment.classNames ?? [];

    return (
        <section className="track-vision" aria-label="Track Vision">
            <header className="track-vision__header">
                <div><span className="track-vision__eyebrow">VISION PIPELINE</span><h2>From capture to bird's-eye view</h2></div>
                <span className="track-vision__capture-state" data-active={captureState === 'active'}>
                    <i />{captureState === 'active' ? 'Capturing' : captureState === 'starting' ? 'Starting…' : 'Capture idle'}
                </span>
            </header>
            <div className="track-vision__controls">
                <select aria-label="Game window or screen" value={sourceId} disabled={captureState !== 'idle'} onChange={(event) => setSourceId(event.target.value)}>
                    <option value="">Choose a window or screen</option>
                    {sources.map(({ id, name: sourceName }) => <option key={id} value={id}>{sourceName}</option>)}
                </select>
                <button type="button" disabled={captureState !== 'idle'} onClick={() => void refreshSources()}>Refresh sources</button>
                {captureState === 'idle'
                    ? <button type="button" className="track-vision__start" disabled={!sourceId} onClick={() => void start()}>Share game screen</button>
                    : <button type="button" onClick={stop}>Stop capture</button>}
            </div>
            <div className="track-vision__tabs" role="tablist" aria-label="Vision pipeline steps">
                {PIPELINE_STEPS.map((item, index) => <button key={item.id} type="button" role="tab"
                    id={`${pipelineId}-${item.id}`} aria-controls={`${pipelineId}-panel`} aria-selected={step === item.id}
                    tabIndex={step === item.id ? 0 : -1} onClick={() => setStep(item.id)}
                    onKeyDown={(event) => {
                        const next = event.key === 'ArrowRight' ? (index + 1) % PIPELINE_STEPS.length
                            : event.key === 'ArrowLeft' ? (index + PIPELINE_STEPS.length - 1) % PIPELINE_STEPS.length
                                : event.key === 'Home' ? 0 : event.key === 'End' ? PIPELINE_STEPS.length - 1 : -1;
                        if (next < 0) return;
                        event.preventDefault();
                        setStep(PIPELINE_STEPS[next].id);
                        document.getElementById(`${pipelineId}-${PIPELINE_STEPS[next].id}`)?.focus();
                    }}><span aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>{item.label}</button>)}
            </div>
            <div className="track-vision__stage" role="tabpanel" id={`${pipelineId}-panel`}
                aria-labelledby={`${pipelineId}-${step}`} tabIndex={0}>
                <div className="track-vision__stage-heading">
                    <div><span className="track-vision__eyebrow">STEP {String(stepIndex + 1).padStart(2, '0')} / {String(PIPELINE_STEPS.length).padStart(2, '0')}</span>
                        <h3>{activeStep.title}</h3></div>
                    {hasFrame && <span className="track-vision__frame-size">{previewWidth} × {previewHeight}</span>}
                </div>
                <dialog ref={previewRef} open hidden={isSceneStep || step === 'birds-eye'} className={`track-vision__preview${hasFrame && step === 'calibration' && showCalibrationOnCapture ? ' track-vision__preview--calibrated' : ''}`}
                    aria-label="Capture preview" onCancel={(event) => { event.preventDefault(); togglePreviewSize(); }}>
                    <video ref={videoRef} muted playsInline hidden />
                    <canvas ref={canvasRef} aria-label="Captured game frame with vision detections" hidden={!hasFrame}
                        className={step === 'depth-map' ? 'track-vision__depth-map' : undefined}
                        onMouseMove={step === 'depth-map' ? (event) => setDepthPointer({ clientX: event.clientX, clientY: event.clientY }) : undefined}
                        onMouseLeave={() => setDepthPointer(null)} />
                    {hasFrame && hoveredDepth && depthRect && <span className="track-vision__depth-pointer" aria-label="Depth at mouse"
                        style={{ left: hoveredDepth.x, top: hoveredDepth.y,
                            transform: `translate(${hoveredDepth.x > depthRect.width / 2 ? 'calc(-100% - 12px)' : '12px'}, ${hoveredDepth.y > depthRect.height / 2 ? 'calc(-100% - 12px)' : '12px'})` }}>
                        {hoveredDepth.depth === null ? 'No valid depth' : formatDepth(hoveredDepth.depth, depthMap?.depth.scale, 2)}
                    </span>}
                    {hasFrame && step === 'calibration' && showCalibrationOnCapture && previewFrameRef.current && validCalibration(previewCamera)
                        && <CameraGroundGrid camera={previewCamera} applied={Boolean(calibration)} />}
                    {!hasFrame && <div className="track-vision__empty"><span className="track-vision__empty-icon" aria-hidden="true">▣</span><strong>No captured frame yet</strong></div>}
                    <div className="track-vision__preview-controls">
                        {previewExpanded && captureState !== 'idle' && <button type="button" onClick={stop}>Stop capture</button>}
                        <button type="button" aria-expanded={previewExpanded} onClick={togglePreviewSize}>
                            {previewExpanded ? 'Restore capture' : 'Expand capture'}
                        </button>
                    </div>
                </dialog>
                <div hidden={step !== 'calibration'}>
                    <TrackCalibration source={hasFrame ? previewFrameRef.current : null} draft={cameraDraft} applied={calibration}
                        showOnCapture={showCalibrationOnCapture} onToggleCapture={() => setShowCalibrationOnCapture((current) => !current)}
                        onChange={(draft) => { setCameraDraft(draft); updateCalibration(); }}
                        onApply={() => { const frame = previewFrameRef.current; if (frame) updateCalibration({ ...cameraDraft, imageWidth: frame.width, imageHeight: frame.height }); }}
                        onClear={() => updateCalibration()} />
                </div>
                <div hidden={step !== 'segmentation'}>
                    <div className="track-vision__controls">
                        <label>Display label
                            <select value={displayLabel} disabled={!detectors.segment.classNames?.length}
                                onChange={(event) => setDisplayLabel(event.target.value)}>
                                <option value="">All labels</option>
                                {detectors.segment.classNames?.map((label, index) => <option key={index} value={label}>{label}</option>)}
                            </select>
                        </label>
                        <label>Segmentation confidence {Math.round(confidence * 100)}%
                            <input aria-label="Segmentation confidence" type="range" min="0.1" max="0.95" step="0.05" value={confidence}
                                onChange={(event) => setConfidence(Number(event.target.value))} />
                        </label>
                    </div>
                </div>
                <PipelineDetails step={step} frame={previewResult} masks={masks} classNames={labels} filterConfidence={filterConfidence} depthMap={depthMap} />
                <div hidden={!isSceneStep}>
                    <ReconstructedSceneView scene={hasFrame ? previewResult?.reconstructedScene ?? null : null}
                        source={hasFrame ? previewFrameRef.current : null} capturedAt={previewCapturedAt} />
                </div>
                {step === 'birds-eye' && <BirdsEyeView scene={hasFrame ? previewResult?.birdsEyeScene ?? null : null}
                    hasReconstructedScene={hasFrame && Boolean(previewResult?.reconstructedScene)} capturedAt={previewResult?.capturedAt} />}
                <div hidden={!isSceneStep}>
                    <section className="track-vision__analysis" aria-label="Screen analysis">
                        <h3>Screen analysis</h3>
                        <dl>
                            <div><dt>Driver position</dt><dd aria-label="Driver position">{analysis?.driverPosition ? <>
                                <div>Left boundary: {analysis.driverPosition.leftBoundaryDistanceM.toFixed(1)} m</div>
                                <div>Right boundary: {analysis.driverPosition.rightBoundaryDistanceM.toFixed(1)} m</div>
                            </> : 'Unknown'}</dd>
                                {analysis?.driverPosition && <p className="track-vision__hint">Measured at visible track {analysis.driverPosition.referenceDistanceM.toFixed(1)} m ahead.</p>}
                            </div>
                            <div><dt>Opponents relative to driver</dt><dd aria-label="Opponent positions">{analysis?.opponents?.length
                                ? <ol className="track-vision__opponents">{analysis.opponents.map((opponent, index) => <li key={index}>
                                    {Math.abs(opponent.longitudinalOffsetM).toFixed(1)} m {opponent.longitudinalOffsetM >= 0 ? 'ahead' : 'behind'}
                                    {' · '}{Math.abs(opponent.lateralOffsetM) < 0.05 ? 'Aligned with driver'
                                        : `${Math.abs(opponent.lateralOffsetM).toFixed(1)} m ${opponent.lateralOffsetM < 0 ? 'left' : 'right'}`}
                                </li>)}</ol>
                                : analysis?.carAhead === 1 ? 'Individual positions unresolved'
                                    : analysis?.carAhead === 0 ? 'No opponent detected' : 'Unknown'}</dd></div>
                        </dl>
                    </section>
                </div>
            </div>
            <details className="track-vision__settings" open={DETECTION_TASKS.some(({ id }) => Boolean(detectors[id].error))}>
                <summary>Setting <span>{DETECTION_TASKS.map(({ id, label }) =>
                    `${label}: ${detectors[id].status}`).join(' · ')}</span></summary>
                <div className="track-vision__controls">
                    <label>Filtering confidence {Math.round(filterConfidence * 100)}%
                        <input aria-label="Filtering confidence" type="range" min="0.1" max="0.95" step="0.05" value={filterConfidence}
                            onChange={(event) => {
                                const value = Number(event.target.value);
                                options.current.filterConfidence = value;
                                setFilterConfidence(value);
                                if (latest.current) publish(latest.current);
                            }} />
                    </label>
                </div>
                <fieldset className="track-vision__stack">
                    <legend>Track models</legend>
                    {DETECTION_TASKS.map(({ id, label }) => <div className="track-vision__detector" key={id}>
                        <strong>{label}</strong>
                        <span className="track-vision__detector-state">{detectors[id].status === 'ready'
                            ? captureState === 'active' && DETECTION_TASKS.every(({ id }) => detectors[id].status === 'ready') ? 'Running' : 'Ready' : detectors[id].status === 'retrying' ? 'Retrying GPU…'
                                : detectors[id].status === 'error' ? 'Unavailable' : 'Loading…'}</span>
                        <label className="track-vision__resolution">Input resolution
                            <select aria-label={`${label} input resolution`} value={inputSizes[id]}
                                onChange={(event) => setInputSizes((current) => ({ ...current, [id]: Number(event.target.value) }))}>
                                {MODEL_INPUT_RESOLUTIONS[id].map(({ label, size }) =>
                                    <option key={size} value={size}>{label} · {size} × {size}</option>)}
                            </select>
                        </label>
                        {!!detectors[id].classNames?.length && <div className="track-vision__hint" aria-label="Model labels">Labels: {detectors[id].classNames!.join(', ')}</div>}
                        {detectors[id].status === 'ready' && <div className="track-vision__hint" aria-label={`${label} inference device`}>GPU acceleration active</div>}
                        {detectors[id].error && <div className="track-vision__error" role="alert">
                            {detectors[id].error} <button type="button" onClick={() => setRetry((current) => current + 1)}>Retry {label}</button>
                        </div>}
                    </div>)}
                </fieldset>
            </details>
            <div className="track-vision__status" role="status">{status}</div>
            {error && <div className="track-vision__error" role="alert">{error}</div>}
        </section>
    );
});

export default LiveTrackVision;
