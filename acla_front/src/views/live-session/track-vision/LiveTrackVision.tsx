import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { NamedOperationComponentHandle, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { captureGameScreen, ScreenCaptureSource } from './screen-capture';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { CameraCalibration, DETECTION_TASKS, DetectionTask, EnabledDetections, TrackVisionAnalysis, TrackVisionDetection, TrackVisionFrame, VISION_MAX_AGE_MS } from './track-vision-types';
import { analyzeTrackPositions, reconstructTrack } from './track-position-analysis';
import { DEFAULT_CAMERA, validCalibration } from './camera-projection';
import TrackCalibration, { CameraGroundGrid } from './TrackCalibration';
import TrackBoundaryCutoff from './TrackBoundaryCutoff';
import { drawVisionOverlay } from './vision-overlay';
import LocalTrackView from './LocalTrackView';
import './LiveTrackVision.css';

export interface TrackVisionHandle extends NamedOperationComponentHandle {
    getLatestDetection(): TrackVisionDetection | null;
    subscribeDetection(listener: () => void): () => void;
}

const message = (error: unknown) => error instanceof Error ? error.message : 'Vision detection failed.';
const GPU_RETRY_DELAY_MS = 3000;
type DetectorState = { status: 'off' | 'loading' | 'ready' | 'retrying' | 'error'; device?: string; error?: string; modelName?: string; classNames?: string[] };

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
    const [enabled, setEnabled] = useState<EnabledDetections>({ segment: true, depth: true });
    const [allowCpuFallback, setAllowCpuFallback] = useState(false);
    const [retry, setRetry] = useState(0);
    const [detectors, setDetectors] = useState<Record<DetectionTask, DetectorState>>({
        segment: { status: 'loading' }, depth: { status: 'loading' },
    });
    const [captureState, setCaptureState] = useState<'idle' | 'starting' | 'active'>('idle');
    const [error, setError] = useState('');
    const [status, setStatus] = useState('Share your game screen to run segmentation and depth estimation.');
    const [hasFrame, setHasFrame] = useState(false);
    const [previewExpanded, setPreviewExpanded] = useState(false);
    const [confidence, setConfidence] = useState(0.5);
    const [boundaryStartY, setBoundaryStartY] = useState(0.75);
    const [cameraDraft, setCameraDraft] = useState(DEFAULT_CAMERA);
    const [calibration, setCalibration] = useState<CameraCalibration>();
    const [showCalibrationOnCapture, setShowCalibrationOnCapture] = useState(false);
    const cameraCalibration = useRef<CameraCalibration | undefined>(undefined);
    const [previewResult, setPreviewResult] = useState<TrackVisionFrame | null>(null);
    const previewWidth = previewFrameRef.current?.width ?? 0, previewHeight = previewFrameRef.current?.height ?? 0;
    const previewCamera = useMemo(() => ({ ...cameraDraft, imageWidth: previewWidth, imageHeight: previewHeight }),
        [cameraDraft, previewWidth, previewHeight]);
    const previewScene = useMemo(() => previewResult ? reconstructTrack({ ...previewResult, calibration: previewCamera }) : null,
        [previewResult, previewCamera]);
    const options = useRef({ confidence, enabled, allowCpuFallback, boundaryStartY });
    options.current = { confidence, enabled, allowCpuFallback, boundaryStartY };

    const togglePreviewSize = () => {
        const preview = previewRef.current;
        if (!preview) return;
        // Keep the video and canvas mounted while moving the preview into the browser's top layer.
        preview.close();
        if (previewExpanded) preview.show();
        else preview.showModal();
        setPreviewExpanded((current) => !current);
    };

    const redrawPreview = useCallback((result: TrackVisionFrame) => {
        const frame = previewFrameRef.current;
        const canvas = canvasRef.current;
        if (!frame || !canvas) return;
        canvas.width = frame.width;
        canvas.height = frame.height;
        const preview = canvas.getContext('2d');
        if (!preview) throw new Error('Screen preview is unavailable.');
        preview.drawImage(frame, 0, 0);
        drawVisionOverlay(preview, result);
    }, []);

    const publish = useCallback((result: TrackVisionFrame | null) => {
        clearTimeout(analysisExpiry.current);
        const reconstruction = reconstructTrack(result);
        const geometry = reconstruction?.geometry ?? null;
        const scene = result?.detections.segment?.task === 'segment' ? analyzeTrackPositions(result, reconstruction) : null;
        latest.current = result ? { ...result, reconstruction, geometry, analysis: scene } : null;
        setPreviewResult(result);
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
    const updateBoundaryStart = (value: number) => {
        const next = Math.round(Math.max(0, Math.min(1, value)) * 100) / 100;
        options.current.boundaryStartY = next;
        setBoundaryStartY(next);
        if (latest.current) publish({ ...latest.current, boundaryStartY: next });
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
        setDetectors(Object.fromEntries(DETECTION_TASKS.map(({ id }) => {
            const current = models.current[id];
            const model = current?.executionProvider === 'wasm' && !allowCpuFallback ? undefined : current;
            return [id, !enabled[id] ? { status: 'off' } : model ? {
                status: 'ready', modelName: model.name, classNames: model.classNames, device: model.executionProvider === 'webgpu' ? 'GPU acceleration active'
                    : `CPU inference · ${model.fallbackReason || 'GPU acceleration unavailable.'}`,
            } : { status: 'loading' }];
        })) as Record<DetectionTask, DetectorState>);
        // Serialize loading so rapid toggles cannot leave duplicate model sessions alive.
        modelQueue.current = modelQueue.current.then(async () => {
            for (const { id } of DETECTION_TASKS) {
                if (version !== modelVersion.current) return;
                if (!enabled[id] || (!allowCpuFallback && models.current[id]?.executionProvider === 'wasm')) {
                    const previous = models.current[id];
                    delete models.current[id];
                    await previous?.dispose().catch(() => undefined);
                    if (version !== modelVersion.current) return;
                }
                if (!enabled[id]) continue;
                if (models.current[id]) continue;
                try {
                    const model = id === 'depth'
                        ? await TrackVisionModel.loadBuiltin(id, allowCpuFallback)
                        : await TrackVisionModel.loadBackend(allowCpuFallback);
                    if (version !== modelVersion.current) { await model.dispose().catch(() => undefined); return; }
                    models.current[id] = model;
                    setDetectors((current) => ({ ...current, [id]: {
                        status: 'ready', modelName: model.name, classNames: model.classNames, device: model.executionProvider === 'webgpu' ? 'GPU acceleration active'
                            : `CPU inference · ${model.fallbackReason || 'GPU acceleration unavailable.'}`,
                    } }));
                } catch (reason) {
                    if (version === modelVersion.current) setDetectors((current) => ({ ...current, [id]: {
                        status: !allowCpuFallback && reason instanceof GpuInferenceError ? 'retrying' : 'error', error: message(reason),
                    } }));
                }
            }
        });
        // Reconfiguration increments the version above; unmount cleanup invalidates it too.
    }, [enabled, allowCpuFallback, retry, publish]);

    useEffect(() => {
        const active = DETECTION_TASKS.filter(({ id }) => enabled[id]).map(({ id }) => detectors[id]);
        // Let the current load queue finish before scheduling another GPU attempt.
        if (active.some(({ status }) => status === 'loading') || !active.some(({ status }) => status === 'retrying')) return;
        const timer = setTimeout(() => setRetry((current) => current + 1), GPU_RETRY_DELAY_MS);
        return () => clearTimeout(timer);
    }, [detectors, enabled, allowCpuFallback]);

    const toggleDetection = (task: DetectionTask) => {
        // Remove stale overlays immediately; the next captured frame uses the new stack.
        modelVersion.current++;
        publish(null);
        setHasFrame(false);
        setEnabled((current) => ({ ...current, [task]: !current[task] }));
    };

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
                        const result: TrackVisionFrame = { capturedAt: Date.now(), width: frame.width, height: frame.height, detections: {} };
                        const inference = (async () => {
                            for (const { id } of DETECTION_TASKS) {
                                if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                const model = models.current[id];
                                if (!options.current.enabled[id] || !model || (!options.current.allowCpuFallback && model.executionProvider === 'wasm')) continue;
                                try { result.detections[id] = await model.detect(frame, options.current.confidence); }
                                catch (reason) {
                                    if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                    delete models.current[id];
                                    void model.dispose().catch(() => undefined);
                                    setDetectors((current) => ({ ...current, [id]: {
                                        status: model.executionProvider === 'webgpu' ? 'retrying' : 'error', error: message(reason),
                                    } }));
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
                        // Read after inference so a pending frame cannot restore an old calibration.
                        if (cameraCalibration.current && (cameraCalibration.current.imageWidth !== frame.width || cameraCalibration.current.imageHeight !== frame.height)) {
                            cameraCalibration.current = undefined;
                            setCalibration(undefined);
                        }
                        result.calibration = cameraCalibration.current;
                        result.boundaryStartY = options.current.boundaryStartY;
                        redrawPreview(result);
                        publish(result);
                        setHasFrame(true);
                        const completed = DETECTION_TASKS.filter(({ id }) => result.detections[id]);
                        setStatus(completed.length ? completed.map(({ id, label }) => `${label} · ${Math.round(result.detections[id]!.inferenceMs)} ms`).join(' / ')
                            : Object.values(options.current.enabled).some(Boolean) ? 'Screen shared. Waiting for enabled detectors.'
                                : 'Screen shared. Enable a detection to analyze the scene.');
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

    return (
        <section className="track-vision" aria-label="Track Vision">
            <fieldset className="track-vision__stack">
                <legend>Track models <span>Ultralytics</span></legend>
                <label className="track-vision__fallback">
                    <input type="checkbox" checked={allowCpuFallback} onChange={(event) => {
                        modelVersion.current++;
                        publish(null);
                        setHasFrame(false);
                        setAllowCpuFallback(event.target.checked);
                    }} />
                    Allow CPU fallback
                </label>
                <p className="track-vision__hint">{allowCpuFallback
                    ? 'Try GPU first, then use CPU if GPU inference is unavailable.'
                    : 'CPU fallback is off. Failed GPU inference retries automatically every 3 seconds.'}</p>
                {DETECTION_TASKS.map(({ id, label, description }) => <div className="track-vision__detector" key={id} data-enabled={enabled[id]}>
                    <label>
                        <input type="checkbox" aria-label={`Enable ${label}`} checked={enabled[id]} onChange={() => toggleDetection(id)} />
                        <span><strong>{label}</strong><small>{description}</small></span>
                    </label>
                    <span className="track-vision__detector-state">{!enabled[id] ? 'Off' : detectors[id].status === 'ready'
                        ? captureState === 'active' ? 'Running' : 'Ready' : detectors[id].status === 'retrying' ? 'Retrying GPU…'
                            : detectors[id].status === 'error' ? 'Unavailable' : 'Loading…'}</span>
                    {enabled[id] && detectors[id].modelName && <div className="track-vision__hint">Model: {detectors[id].modelName}</div>}
                    {enabled[id] && !!detectors[id].classNames?.length && <div className="track-vision__hint" aria-label="Model labels">Labels: {detectors[id].classNames!.join(', ')}</div>}
                    {enabled[id] && detectors[id].device && <div className="track-vision__hint" aria-label={`${label} inference device`}>{detectors[id].device}</div>}
                    {enabled[id] && detectors[id].error && <div className="track-vision__error" role="alert">
                        {detectors[id].error} <button type="button" onClick={() => setRetry((current) => current + 1)}>Retry {label}</button>
                    </div>}
                </div>)}
            </fieldset>
            <div className="track-vision__controls">
                <label>Segmentation confidence {Math.round(confidence * 100)}%
                    <input aria-label="Segmentation confidence" disabled={!enabled.segment} type="range" min="0.1" max="0.95" step="0.05" value={confidence}
                        onChange={(event) => setConfidence(Number(event.target.value))} />
                </label>
            </div>
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
            <dialog ref={previewRef} open className={`track-vision__preview${hasFrame && showCalibrationOnCapture ? ' track-vision__preview--calibrated' : ''}`}
                aria-label="Capture preview" onCancel={(event) => { event.preventDefault(); togglePreviewSize(); }}>
                <video ref={videoRef} muted playsInline hidden />
                <canvas ref={canvasRef} aria-label="Captured game frame with vision detections" hidden={!hasFrame} />
                {hasFrame && showCalibrationOnCapture && previewFrameRef.current && validCalibration(previewCamera)
                    && <CameraGroundGrid camera={previewCamera} applied={Boolean(calibration)} />}
                {hasFrame && <TrackBoundaryCutoff width={previewWidth} height={previewHeight}
                    value={boundaryStartY} onChange={updateBoundaryStart} />}
                {!hasFrame && <div className="track-vision__empty"><strong>See the full racing scene</strong><span>Share your simulator window and enable the detections you need.</span></div>}
                <div className="track-vision__preview-controls">
                    {previewExpanded && captureState !== 'idle' && <button type="button" onClick={stop}>Stop capture</button>}
                    <button type="button" aria-expanded={previewExpanded} onClick={togglePreviewSize}>
                        {previewExpanded ? 'Restore capture' : 'Expand capture'}
                    </button>
                </div>
            </dialog>
            <div className="track-vision__controls">
                <label>Boundary start {Math.round(boundaryStartY * 100)}% from top
                    <input aria-label="Boundary start" type="range" min="0" max="1" step="0.01" value={boundaryStartY}
                        onChange={(event) => updateBoundaryStart(Number(event.target.value))} />
                </label>
            </div>
            <p className="track-vision__hint">Drag the amber line above the hood or cockpit. Track edges are detected only above the line; the shaded area is excluded from boundary detection. Set to 100% to use the full frame.</p>
            <TrackCalibration source={hasFrame ? previewFrameRef.current : null} draft={cameraDraft} applied={calibration}
                showOnCapture={showCalibrationOnCapture} onToggleCapture={() => setShowCalibrationOnCapture((current) => !current)}
                onChange={(draft) => { setCameraDraft(draft); updateCalibration(); }}
                onApply={() => { const frame = previewFrameRef.current; if (frame) updateCalibration({ ...cameraDraft, imageWidth: frame.width, imageHeight: frame.height }); }}
                onClear={() => updateCalibration()} />
            <LocalTrackView frame={hasFrame ? previewResult : null} scene={previewScene} camera={previewCamera} applied={Boolean(calibration)} />
            <section className="track-vision__analysis" aria-label="Screen analysis">
                <h3>Screen analysis</h3>
                <dl>
                    <div><dt>Visible corner</dt><dd aria-label="Visible corner">{analysis?.cornerDirection ? `${analysis.cornerDirection === 'left' ? 'Left' : 'Right'}-hand corner` : 'Unknown'}</dd></div>
                    <div><dt>Driver position</dt><dd aria-label="Driver position">{analysis?.playerPosition ? analysis.playerPosition[0].toUpperCase() + analysis.playerPosition.slice(1) : 'Unknown'}</dd></div>
                    <div><dt>Opponent position</dt><dd aria-label="Opponent position">{analysis?.opponentPosition ? analysis.opponentPosition[0].toUpperCase() + analysis.opponentPosition.slice(1) : analysis?.carAhead === 1 ? 'Individual position unresolved' : analysis?.carAhead === 0 ? 'No opponent detected' : 'Unknown'}</dd></div>
                </dl>
                <p className="track-vision__hint">Positions use segmentation, estimated depth and camera position: inside, middle, or outside of the corner. Unclear or stale frames show unknown positions.</p>
                <p className="track-vision__hint">Track and car detections identify positions. The track mask defines the track edges. Car masks and depth reconstruct visible car surfaces. A car pack indicates grouped traffic; an individual car detection is needed for an opponent position.</p>
                <p className="track-vision__hint">Curb, grass, other, fence, sand, and Outfield asphalt road are excluded from the track surface. Analysis uses detections with confidence ≥ 65%. Labels ignore case and surrounding spaces.</p>
            </section>
            <div className="track-vision__status" role="status">{status}</div>
            {error && <div className="track-vision__error" role="alert">{error}</div>}
            <p className="track-vision__hint">Segmentation downloads from the backend and is saved on this device. Depth uses the bundled model. Frames and inference stay local.</p>
        </section>
    );
});

export default LiveTrackVision;
