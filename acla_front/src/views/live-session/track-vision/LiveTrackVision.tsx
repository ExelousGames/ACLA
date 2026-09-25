import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { NamedOperationComponentHandle, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { captureGameScreen, ScreenCaptureSource } from './screen-capture';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { DEFAULT_DEPTH_RANGE, DETECTION_TASKS, DetectionTask, EnabledDetections, PLAYER_TRACK_ROW, TrackVisionAnalysis, TrackVisionDetection, TrackVisionFrame, VISION_MAX_AGE_MS } from './track-vision-types';
import { analyzeTrackPositions } from './track-position-analysis';
import { drawVisionOverlay } from './vision-overlay';
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
    const carCenterCanvasRef = useRef<HTMLCanvasElement>(null);
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
    const [enabled, setEnabled] = useState<EnabledDetections>({ segment: true, depth: false });
    const [allowCpuFallback, setAllowCpuFallback] = useState(false);
    const [retry, setRetry] = useState(0);
    const [detectors, setDetectors] = useState<Record<DetectionTask, DetectorState>>({
        segment: { status: 'loading' }, depth: { status: 'off' },
    });
    const [captureState, setCaptureState] = useState<'idle' | 'starting' | 'active'>('idle');
    const [error, setError] = useState('');
    const [status, setStatus] = useState('Share your game screen to run segmentation and depth estimation.');
    const [hasFrame, setHasFrame] = useState(false);
    const [confidence, setConfidence] = useState(0.5);
    const [depthRange, setDepthRange] = useState(DEFAULT_DEPTH_RANGE);
    const [carCenterDraft, setCarCenterDraft] = useState(0.5);
    const [playerCenterX, setPlayerCenterX] = useState<number>();
    const carAlignment = useRef<{ x: number; width: number; height: number } | undefined>(undefined);
    const options = useRef({ confidence, enabled, depthRange, allowCpuFallback, carCenterDraft });
    options.current = { confidence, enabled, depthRange, allowCpuFallback, carCenterDraft };

    const redrawPreview = useCallback((result: TrackVisionFrame) => {
        const frame = previewFrameRef.current;
        const canvas = canvasRef.current;
        if (!frame || !canvas) return;
        canvas.width = frame.width;
        canvas.height = frame.height;
        const preview = canvas.getContext('2d');
        if (!preview) throw new Error('Screen preview is unavailable.');
        preview.drawImage(frame, 0, 0);
        drawVisionOverlay(preview, result, options.current.depthRange);
    }, []);

    const redrawCarCenter = useCallback(() => {
        const frame = previewFrameRef.current;
        const canvas = carCenterCanvasRef.current;
        if (!frame || !canvas) return;
        canvas.width = frame.width;
        canvas.height = frame.height;
        const preview = canvas.getContext('2d');
        if (!preview) return;
        // Mark the vehicle's centerline in capture coordinates. The draft is
        // not used for position analysis until the car reference is confirmed.
        const x = (carAlignment.current?.x ?? options.current.carCenterDraft) * frame.width;
        const y = PLAYER_TRACK_ROW * frame.height;
        preview.strokeStyle = carAlignment.current ? '#37efac' : '#ffbe57';
        preview.lineWidth = Math.max(2, frame.width / 500);
        preview.beginPath();
        preview.moveTo(x - frame.width * 0.02, y);
        preview.lineTo(x + frame.width * 0.02, y);
        preview.moveTo(x, y - frame.height * 0.035);
        preview.lineTo(x, y + frame.height * 0.035);
        preview.stroke();
    }, []);

    useEffect(() => {
        // Recolor the displayed frame immediately, even while the next inference is pending.
        if (latest.current) redrawPreview(latest.current);
    }, [depthRange, redrawPreview]);

    useEffect(() => {
        // Alignment stays responsive even when detector retries clear the latest result.
        redrawCarCenter();
    }, [carCenterDraft, playerCenterX, redrawCarCenter]);

    const publish = useCallback((result: TrackVisionFrame | null) => {
        clearTimeout(analysisExpiry.current);
        const scene = result?.detections.segment?.task === 'segment' ? analyzeTrackPositions(result) : null;
        latest.current = result ? { ...result, analysis: scene } : null;
        const remaining = result ? result.capturedAt + VISION_MAX_AGE_MS - Date.now() : 0;
        setAnalysis(remaining > 0 ? scene : null);
        if (scene && remaining > 0) analysisExpiry.current = setTimeout(() => setAnalysis(null), remaining);
        listeners.current.forEach((listener) => listener());
    }, []);
    const updateCarAlignment = (x?: number) => {
        const frame = previewFrameRef.current;
        const result = latest.current;
        carAlignment.current = x !== undefined && frame ? { x, width: frame.width, height: frame.height } : undefined;
        setPlayerCenterX(carAlignment.current?.x);
        if (result) publish({ ...result, playerCenterX: carAlignment.current?.x });
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
        carAlignment.current = undefined;
        setPlayerCenterX(undefined);
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
                        // Keep a clean copy so slider changes cannot accumulate overlays or mix frames.
                        const previewFrame = previewFrameRef.current ?? document.createElement('canvas');
                        previewFrame.width = frame.width;
                        previewFrame.height = frame.height;
                        const previewContext = previewFrame.getContext('2d');
                        if (!previewContext) throw new Error('Screen preview is unavailable.');
                        previewContext.drawImage(frame, 0, 0);
                        previewFrameRef.current = previewFrame;
                        // Read after inference: changing alignment during a pending
                        // frame must not let that frame restore an older reference.
                        if (carAlignment.current && (carAlignment.current.width !== frame.width || carAlignment.current.height !== frame.height)) {
                            carAlignment.current = undefined;
                            setPlayerCenterX(undefined);
                        }
                        result.playerCenterX = carAlignment.current?.x;
                        redrawPreview(result);
                        redrawCarCenter();
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
            {enabled.depth && <fieldset className="track-vision__depth">
                <legend>Depth range <span>Estimated meters</span></legend>
                <div className="track-vision__controls">
                    <label>Close · {depthRange.near} m
                        <input aria-label="Close depth" aria-valuetext={`${depthRange.near} meters or closer`} type="range"
                            min="0" max={depthRange.far - 0.5} step="0.5" value={depthRange.near}
                            onChange={(event) => setDepthRange((current) => ({ ...current, near: Math.min(Number(event.target.value), current.far - 0.5) }))} />
                    </label>
                    <label>Far · {depthRange.far} m
                        <input aria-label="Far depth" aria-valuetext={`${depthRange.far} meters or farther`} type="range"
                            min={depthRange.near + 0.5} max="200" step="0.5" value={depthRange.far}
                            onChange={(event) => setDepthRange((current) => ({ ...current, far: Math.max(Number(event.target.value), current.near + 0.5) }))} />
                    </label>
                    <button type="button" onClick={() => setDepthRange(DEFAULT_DEPTH_RANGE)}>Reset depth range</button>
                </div>
                <div className="track-vision__depth-meter" aria-hidden="true" />
                <div className="track-vision__legend"><span>Close ≤ {depthRange.near} m</span><span>Far ≥ {depthRange.far} m</span></div>
                <p className="track-vision__hint">Warm at or below Close, cool at or above Far. Adjust to tune the depth colors.</p>
            </fieldset>}
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
            <div className="track-vision__preview">
                <video ref={videoRef} muted playsInline hidden />
                <canvas ref={canvasRef} aria-label="Captured game frame with vision detections" hidden={!hasFrame} />
                <canvas ref={carCenterCanvasRef} className="track-vision__car-center" aria-label="Car center marker" hidden={!hasFrame} />
                {!hasFrame && <div className="track-vision__empty"><strong>See the full racing scene</strong><span>Share your simulator window and enable the detections you need.</span></div>}
            </div>
            <fieldset className="track-vision__alignment">
                <legend>Car center alignment</legend>
                <p className="track-vision__hint">Align the crosshair with your car's centerline at the marker height, using the visible nose or bonnet as the reference, then select Set car center. You can do this wherever the car is on the track. Leave alignment unset if the car's centerline is not visible.</p>
                <div className="track-vision__controls">
                    <label>Car center · {Math.round(carCenterDraft * 100)}% of image width
                        <input aria-label="Car center alignment" disabled={!hasFrame} type="range" min="0.05" max="0.95" step="0.005" value={carCenterDraft}
                            onChange={(event) => { setCarCenterDraft(Number(event.target.value)); updateCarAlignment(); }} />
                    </label>
                    <button type="button" disabled={!hasFrame} onClick={() => updateCarAlignment(carCenterDraft)}>Set car center</button>
                    <button type="button" disabled={playerCenterX === undefined} onClick={() => updateCarAlignment()}>Clear alignment</button>
                </div>
                <p className="track-vision__hint">{playerCenterX === undefined ? 'Alignment needed to identify the driver position.' : 'Car center set for this capture.'} Align again after changing car, camera, seat position, or field of view. Use a fixed forward view.</p>
            </fieldset>
            <section className="track-vision__analysis" aria-label="Screen analysis">
                <h3>Screen analysis</h3>
                <dl>
                    <div><dt>Visible corner</dt><dd aria-label="Visible corner">{analysis?.cornerDirection ? `${analysis.cornerDirection === 'left' ? 'Left' : 'Right'}-hand corner` : 'Unknown'}</dd></div>
                    <div><dt>Driver position</dt><dd aria-label="Driver position">{analysis?.playerPosition ? analysis.playerPosition[0].toUpperCase() + analysis.playerPosition.slice(1) : 'Unknown'}</dd></div>
                    <div><dt>Opponent position</dt><dd aria-label="Opponent position">{analysis?.opponentPosition ? analysis.opponentPosition[0].toUpperCase() + analysis.opponentPosition.slice(1) : analysis?.carAhead === 1 ? 'Individual position unresolved' : analysis?.carAhead === 0 ? 'No opponent detected' : 'Unknown'}</dd></div>
                </dl>
                <p className="track-vision__hint">Positions are estimated from the visible track edges: inside, middle, or outside of the corner. Unclear or stale frames show unknown positions.</p>
                <p className="track-vision__hint">Track and car detections identify positions. Left_boundary and right_boundary refine the track edges when visible. A car pack indicates grouped traffic; an individual car detection is needed for an opponent position.</p>
                <p className="track-vision__hint">Curb, grass, other, fence, sand, and Outfield asphalt road are excluded from the track surface. Analysis uses detections with confidence ≥ 65%. Labels ignore case and surrounding spaces.</p>
            </section>
            <div className="track-vision__status" role="status">{status}</div>
            {error && <div className="track-vision__error" role="alert">{error}</div>}
            <p className="track-vision__hint">Segmentation downloads from the backend and is saved on this device. Depth uses the bundled model. Frames and inference stay local.</p>
        </section>
    );
});

export default LiveTrackVision;
