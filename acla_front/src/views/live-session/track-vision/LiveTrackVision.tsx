import React, { forwardRef, useCallback, useEffect, useId, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { NamedOperationComponentHandle, useOptionalOperationComponentRefDirectory, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { getVisualizationComponentName } from 'views/session-shared/visualization/visualization-component-names';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { LiveTrajectoryMapHandle } from '../LiveTrajectoryMap';
import { useSessionScreenCapture } from '../live-session-recording/SessionScreenCapture';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { TrackPositionModel } from './track-position-model';
import { MODEL_INPUT_RESOLUTIONS } from './vision-config';
import { CameraCalibration, DETECTION_TASKS, TrackVisionDetection, TrackVisionFrame } from './track-vision-types';
import { capturePolicyCameraImage } from './policy-camera';
import { reconstructTrack } from './track-reconstruction';
import { validCalibration } from './camera-projection';
import { readTrackVisionSettings, saveTrackVisionSettings } from './track-vision-settings';
import TrackCalibration, { CameraGroundGrid } from './TrackCalibration';
import { drawVisionOverlay } from './vision-overlay';
import ReconstructedSceneView from './ReconstructedSceneView';
import BirdsEyeView from './BirdsEyeView';
import { reconstructScene } from './reconstructed-scene';
import { projectBirdsEyeScene } from './birds-eye-scene';
import { readVisionMotion, TrackHistory, VisionMotion } from './track-history';
import { liveTelemetryStore } from '../live-telemetry-store';
import { drawLabelDepths, filteredFrame, filteredMasks, PIPELINE_STEPS, PipelineStep } from './pipeline-visuals';
import PipelineDetails from './PipelineDetails';
import { createDepthMap, depthAtMouse, drawDepthMap, formatDepth } from './depth-map';
import { readDepthValues } from './vision-gpu-depth';
import { readSegmentInstances } from './vision-gpu-segments';
import { VisionFrameTimer, VisionTimingSnapshot } from './vision-timing';
import './LiveTrackVision.css';

export interface TrackVisionHandle extends NamedOperationComponentHandle {
    getLatestDetection(): TrackVisionDetection | null;
    /** Camera and position can be consumed without segmentation, depth or scene output. */
    getLatestDrivingDetection(): TrackVisionDetection | null;
    getTimingSnapshot(): VisionTimingSnapshot | null;
    subscribeDetection(listener: () => void): () => void;
    /** Request CPU depth for the current capture and while the consumer keeps polling. */
    requestDepthValues(): Promise<void>;
}

const message = (error: unknown) => error instanceof Error ? error.message : 'Vision detection failed.';
const GPU_RETRY_DELAY_MS = 3000;
const VISION_MODELS = [...DETECTION_TASKS, { id: 'position', label: 'Position' }] as const;
const disposeGpuFrame = (frame: TrackVisionFrame | null, next?: TrackVisionFrame | null) => {
    for (const { id } of DETECTION_TASKS) {
        const result = frame?.detections[id];
        if (result !== next?.detections[id]) void result?.gpu?.dispose().catch(() => undefined);
    }
};
type ModelTask = typeof VISION_MODELS[number]['id'];
type DetectorState = { status: 'loading' | 'ready' | 'retrying' | 'error' | 'disabled'; error?: string; classNames?: string[] };

const LiveTrackVision = forwardRef<TrackVisionHandle, { name: string }>(({ name }, forwardedRef) => {
    const directory = useOptionalOperationComponentRefDirectory();
    const directoryRef = useRef(directory);
    directoryRef.current = directory;
    const getLiveMap = useCallback(() => directoryRef.current?.findComponentRef<LiveTrajectoryMapHandle>(
        getVisualizationComponentName('live-trajectory-map'))?.current ?? null, []);
    const positionTelemetry = useRef<{ position: number; receivedAt: number; identity: string; game: string;
        track?: string; gameTime?: number; orientation?: [number, number, number] } | undefined>(undefined);
    const positionGeneration = useRef(0);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const previewRef = useRef<HTMLDialogElement>(null);
    const previewFrameRef = useRef<HTMLCanvasElement | null>(null);
    const stream = useSessionScreenCapture()?.stream ?? null;
    const models = useRef<{ segment?: TrackVisionModel; depth?: TrackVisionModel; position?: TrackPositionModel }>({});
    const modelQueue = useRef<Promise<void>>(Promise.resolve());
    const inferenceRef = useRef<Promise<unknown>>(Promise.resolve());
    const captureVersion = useRef(0);
    const modelVersion = useRef(0);
    const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    const latest = useRef<TrackVisionDetection | null>(null);
    const latestDriving = useRef<TrackVisionDetection | null>(null);
    const frameTimer = useRef<VisionFrameTimer | null>(null);
    const cpuDepthUntil = useRef(0);
    const trackHistory = useRef(new TrackHistory());
    const motion = useRef<VisionMotion | undefined>(undefined);
    const motionGeneration = useRef(0);
    const listeners = useRef(new Set<() => void>());
    const [retry, setRetry] = useState(0);
    const [savedSettings] = useState(readTrackVisionSettings);
    const [inputSizes, setInputSizes] = useState(savedSettings.inputSizes);
    const [segmentEnabled, setSegmentEnabled] = useState(savedSettings.segmentEnabled);
    const [depthEnabled, setDepthEnabled] = useState(savedSettings.depthEnabled);
    const [detectors, setDetectors] = useState<Record<ModelTask, DetectorState>>({
        segment: { status: savedSettings.segmentEnabled ? 'loading' : 'disabled' },
        depth: { status: savedSettings.depthEnabled ? 'loading' : 'disabled' },
        position: { status: 'loading' },
    });
    const [captureState, setCaptureState] = useState<'idle' | 'starting' | 'active'>('idle');
    const [error, setError] = useState('');
    const [status, setStatus] = useState('Capture idle');
    const [hasFrame, setHasFrame] = useState(false);
    const [previewCapturedAt, setPreviewCapturedAt] = useState<number>();
    const [previewExpanded, setPreviewExpanded] = useState(false);
    const [step, setStep] = useState<PipelineStep>('capture');
    const pipelineId = useId();
    const [confidence, setConfidence] = useState(savedSettings.confidence);
    const [filterConfidence, setFilterConfidence] = useState(savedSettings.filterConfidence);
    const [displayLabel, setDisplayLabel] = useState(savedSettings.displayLabel);
    const [cameraDraft, setCameraDraft] = useState(savedSettings.cameraDraft);
    const [calibration, setCalibration] = useState(savedSettings.calibration);
    const [showCalibrationOnCapture, setShowCalibrationOnCapture] = useState(savedSettings.showCalibrationOnCapture);
    const cameraCalibration = useRef(savedSettings.calibration);
    const [previewResult, setPreviewResult] = useState<TrackVisionDetection | null>(null);
    const [depthPointer, setDepthPointer] = useState<{ clientX: number; clientY: number } | null>(null);
    const depthMap = useMemo(() => depthEnabled && step === 'depth-map' ? createDepthMap(previewResult) : null, [previewResult, step, depthEnabled]);
    const depthRect = depthMap && depthPointer ? canvasRef.current?.getBoundingClientRect() : null;
    const hoveredDepth = depthMap && depthPointer && depthRect
        ? depthAtMouse(depthMap, depthRect, depthPointer.clientX, depthPointer.clientY) : null;
    const showFilteredMasks = step === 'filtering' || step === 'depth';
    const masks = useMemo(() => showFilteredMasks ? filteredMasks(previewResult) : [], [previewResult, showFilteredMasks]);
    const previewWidth = previewFrameRef.current?.width ?? 0, previewHeight = previewFrameRef.current?.height ?? 0;
    const previewCamera = useMemo(() => ({ ...cameraDraft, imageWidth: previewWidth, imageHeight: previewHeight }),
        [cameraDraft, previewWidth, previewHeight]);
    const needsCpuDepth = step === 'filtering' || step === 'depth' || (step === 'depth-map' && Boolean(depthPointer));
    const options = useRef({ confidence, filterConfidence, segmentEnabled, depthEnabled, needsCpuDepth });
    options.current = { confidence, filterConfidence, segmentEnabled, depthEnabled, needsCpuDepth };

    useEffect(() => {
        const depth = previewResult?.detections.depth;
        if (depth?.task !== 'depth' || !depth.gpu) return;
        let active = true;
        const reads: Promise<unknown>[] = [];
        if (needsCpuDepth && !depth.values) reads.push(readDepthValues(depth));
        if (step === 'depth-map' && !depth.gpu.rangeValues) reads.push(depth.gpu.readRange());
        if (reads.length) void Promise.all(reads).then(() => {
            if (active) setPreviewResult((current) => current?.detections.depth === depth ? { ...current } : current);
        }).catch((reason) => { if (active) setError(message(reason)); });
        return () => { active = false; };
    }, [previewResult, needsCpuDepth, step]);

    useEffect(() => {
        if (window.screenCapture) saveTrackVisionSettings({
            inputSizes, segmentEnabled, depthEnabled, confidence, filterConfidence, displayLabel, cameraDraft, calibration, showCalibrationOnCapture,
        });
    }, [inputSizes, segmentEnabled, depthEnabled, confidence, filterConfidence, displayLabel, cameraDraft, calibration, showCalibrationOnCapture]);

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
        if (step === 'depth' && depthEnabled) drawLabelDepths(preview, previewResult, masks);
    }, [previewResult, previewCapturedAt, step, displayLabel, masks, depthMap, depthEnabled]);

    useEffect(() => { setDepthPointer(null); }, [step, previewExpanded]);

    useEffect(() => {
        if (detectors.segment.status === 'ready' && displayLabel && !detectors.segment.classNames?.includes(displayLabel)) {
            setDisplayLabel('');
        }
    }, [detectors.segment, displayLabel]);

    const publish = useCallback((result: TrackVisionFrame | null, timing?: VisionFrameTimer, drivingFrame = result) => {
        disposeGpuFrame(latest.current, result);
        if (result) result = { ...result, filterConfidence: options.current.filterConfidence,
            motion: motion.current && result.motion?.generation === motionGeneration.current ? result.motion : undefined };
        timing?.start('track');
        const reconstruction = reconstructTrack(result);
        const geometry = reconstruction?.geometry ?? null;
        timing?.start('scene');
        const reconstructedScene = reconstructScene(result);
        timing?.start('birdsEye');
        const birdsEyeScene = trackHistory.current.update(result, projectBirdsEyeScene(reconstructedScene, result?.calibration));
        if (timing) {
            timing.finish(Boolean(result));
            if (result) result = { ...result, timing: timing.snapshot() };
        }
        latest.current = result ? { ...result, reconstruction, reconstructedScene, birdsEyeScene, geometry } : null;
        latestDriving.current = drivingFrame ? {
            capturedAt: drivingFrame.capturedAt, width: drivingFrame.width, height: drivingFrame.height,
            camera: drivingFrame.camera, telemetry: drivingFrame.telemetry, position: drivingFrame.position,
            timing: timing?.snapshot() ?? drivingFrame.timing, detections: {}, reconstruction: null, geometry: null,
        } : null;
        setPreviewResult(latest.current);
        if (!result) setDepthPointer(null);
        listeners.current.forEach((listener) => listener());
    }, []);
    const resetPositionHistory = useCallback(() => {
        positionGeneration.current++;
        models.current.position?.resetHistory();
        latestDriving.current = null;
        if (latest.current?.position) publish({ ...latest.current, position: undefined });
    }, [publish]);
    useEffect(() => liveTelemetryStore.subscribeEvents((event) => {
        const position = event.type === 'frame' ? event.sample.Graphics_normalized_car_position : undefined;
        const gameTime = event.type === 'frame' ? event.sample.Graphics_game_time : undefined;
        const angles = event.type === 'frame' ? [event.sample.Physics_heading, event.sample.Physics_pitch, event.sample.Physics_roll] : [];
        const orientation = angles.length === 3 && angles.every((value) => typeof value === 'number' && Number.isFinite(value))
            ? angles as [number, number, number] : undefined;
        const identity = event.type === 'frame' ? JSON.stringify([event.sessionGeneration, event.streamGeneration,
            event.update.game, event.sample.Static_track, event.sample.Static_car_model, event.sample.Graphics_session_index]) : '';
        const previous = positionTelemetry.current;
        if (event.type !== 'frame' || event.telemetryStatus !== ACC_STATUS.ACC_LIVE
            || typeof position !== 'number' || !Number.isFinite(position) || position < 0 || position > 1
            || (models.current.position?.requiresOrientation && !orientation)
            || (previous && (previous.identity !== identity || Date.now() - previous.receivedAt > 500
                || (gameTime !== undefined && previous.gameTime !== undefined && gameTime < previous.gameTime)))) {
            resetPositionHistory();
            positionTelemetry.current = undefined;
        }
        if (event.type === 'frame' && event.telemetryStatus === ACC_STATUS.ACC_LIVE
            && typeof position === 'number' && Number.isFinite(position) && position >= 0 && position <= 1) {
            positionTelemetry.current = { position, receivedAt: Date.now(), identity, game: event.update.game,
                track: event.sample.Static_track, gameTime, orientation };
        }
        const next = event.type === 'frame' ? readVisionMotion(event.sample, Date.now(), motionGeneration.current, event.update.game) : undefined;
        if (!next && (motion.current || event.type !== 'frame')) {
            motionGeneration.current++;
            motion.current = undefined;
            trackHistory.current.reset();
            if (latest.current) publish({ ...latest.current, motion: undefined });
        } else motion.current = next;
    }), [publish, resetPositionHistory]);
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
        getLatestDrivingDetection: () => latestDriving.current,
        getTimingSnapshot: () => frameTimer.current?.snapshot() ?? null,
        requestDepthValues: async () => {
            cpuDepthUntil.current = Date.now() + 1000;
            const frame = latest.current, depth = frame?.detections.depth;
            if (depth?.task !== 'depth' || depth.values) return;
            await readDepthValues(depth);
            if (latest.current === frame) {
                setPreviewResult((current) => current?.detections.depth === depth ? { ...current } : current);
                listeners.current.forEach((listener) => listener());
            }
        },
        subscribeDetection: (listener) => {
            listeners.current.add(listener);
            return () => { listeners.current.delete(listener); };
        },
    }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandle = useRef(handle);
    registeredHandle.current = handle;
    useRegisterOperationComponentRef(registeredHandle);

    const releaseFrames = useCallback(() => {
        captureVersion.current++;
        frameTimer.current = null;
        models.current.position?.resetHistory();
        clearTimeout(timerRef.current);
        previewFrameRef.current = null;
        setPreviewCapturedAt(undefined);
        if (videoRef.current) videoRef.current.srcObject = null;
        publish(null);
    }, [publish]);
    const stop = useCallback(() => {
        releaseFrames();
        setCaptureState('idle');
        setHasFrame(false);
        setStatus('Screen capture stopped.');
        const canvas = canvasRef.current;
        if (canvas && !canvas.hidden) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    }, [releaseFrames]);

    useEffect(() => () => {
        releaseFrames();
        modelVersion.current++;
        Object.values(models.current).forEach((model) => { void model.dispose().catch(() => undefined); });
        models.current = {};
    }, [releaseFrames]);

    useEffect(() => {
        if (!window.screenCapture) return;
        const version = ++modelVersion.current;
        publish(null);
        const enabled = { segment: segmentEnabled, depth: depthEnabled, position: true };
        const disposals: Promise<void>[] = [];
        for (const { id } of DETECTION_TASKS) {
            const model = models.current[id];
            if (model && (!enabled[id] || (id === 'depth' && model.inputSize !== inputSizes.depth))) {
                delete models.current[id];
                disposals.push(model.dispose().catch(() => undefined));
            }
        }
        setDetectors(Object.fromEntries(VISION_MODELS.map(({ id }) => {
            if (!enabled[id]) return [id, { status: 'disabled' }];
            const model = models.current[id];
            return [id, model ? {
                status: 'ready', classNames: id === 'position' ? undefined : models.current[id]?.classNames,
            } : { status: 'loading' }];
        })) as Record<ModelTask, DetectorState>);
        // Serialize retries, toggles and resolution changes so only one session per model is loaded.
        modelQueue.current = modelQueue.current.then(async () => {
            await Promise.all(disposals);
            for (const { id } of VISION_MODELS) {
                if (version !== modelVersion.current) return;
                if (!enabled[id]) continue;
                if (models.current[id]) continue;
                try {
                    const model = id === 'depth'
                        ? await TrackVisionModel.loadBuiltin(id, inputSizes[id])
                        : id === 'segment' ? await TrackVisionModel.loadBackend() : await TrackPositionModel.load();
                    if (version !== modelVersion.current) { await model.dispose().catch(() => undefined); return; }
                    if (id === 'depth') models.current.depth = model as TrackVisionModel;
                    else if (id === 'segment') models.current.segment = model as TrackVisionModel;
                    else models.current.position = model as TrackPositionModel;
                    setDetectors((current) => ({ ...current, [id]: {
                        status: 'ready', classNames: id === 'position' ? undefined : (model as TrackVisionModel).classNames,
                    } }));
                } catch (reason) {
                    if (version === modelVersion.current) setDetectors((current) => ({ ...current, [id]: {
                        status: reason instanceof GpuInferenceError ? 'retrying' : 'error', error: message(reason),
                    } }));
                }
            }
        });
        // Retrying increments the version above; unmount cleanup invalidates it too.
    }, [retry, publish, inputSizes, segmentEnabled, depthEnabled]);

    useEffect(() => {
        const active = Object.values(detectors);
        // Let the current load queue finish before scheduling another GPU attempt.
        if (active.some(({ status }) => status === 'loading') || !active.some(({ status }) => status === 'retrying')) return;
        const timer = setTimeout(() => setRetry((current) => current + 1), GPU_RETRY_DELAY_MS);
        return () => clearTimeout(timer);
    }, [detectors]);

    useEffect(() => {
        if (!stream) {
            stop();
            return;
        }
        releaseFrames();
        const version = captureVersion.current;
        setCaptureState('starting');
        setError('');
        setStatus('Starting session video preview…');
        const start = async () => {
            try {
                const video = videoRef.current!;
                video.srcObject = stream;
                await video.play();
                await inferenceRef.current.catch(() => undefined);
                if (version !== captureVersion.current) return;
                setCaptureState('active');
                const frame = document.createElement('canvas');
                let previousMap = getLiveMap();
                let previousCircuit = previousMap?.getCircuitMap();
                let previousReferenceLap = previousMap?.getReferenceLap();
                const tick = async () => {
                    if (version !== captureVersion.current) return;
                    const started = performance.now();
                    let unpublished: TrackVisionFrame | null = null;
                    try {
                        if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
                            const capturedAt = Date.now();
                            const previousCapture = frameTimer.current?.capturedAt;
                            const timing = new VisionFrameTimer(capturedAt, previousCapture);
                            frameTimer.current = timing;
                            timing.start('capture');
                            frame.width = video.videoWidth;
                            frame.height = video.videoHeight;
                            const context = frame.getContext('2d');
                            if (!context) throw new Error('Screen preview is unavailable.');
                            context.drawImage(video, 0, 0);
                            const stackVersion = modelVersion.current;
                            const frameConfidence = options.current.confidence;
                            const frameTasks = DETECTION_TASKS.filter(({ id }) => id === 'segment'
                                ? options.current.segmentEnabled : options.current.depthEnabled);
                            timing.start('camera');
                            const result: TrackVisionFrame = { capturedAt: Date.now(), motion: motion.current,
                                telemetry: liveTelemetryStore.getSnapshot(),
                                camera: capturePolicyCameraImage(frame),
                                width: frame.width, height: frame.height, detections: {} };
                            unpublished = result;
                            timing.capturedAt = result.capturedAt;
                            timing.start('references');
                            const telemetry = positionTelemetry.current;
                            const positionVersion = positionGeneration.current;
                            const liveMap = getLiveMap();
                            const circuitMap = liveMap?.getCircuitMap();
                            const referenceLap = liveMap?.getReferenceLap();
                            const references = telemetry && result.capturedAt - telemetry.receivedAt <= 500
                                ? liveMap?.getPositionReferences(telemetry.position) : null;
                            const positionModel = models.current.position;
                            if (!references || previousMap !== liveMap || previousCircuit !== circuitMap || previousReferenceLap !== referenceLap) {
                                positionModel?.resetHistory();
                            }
                            previousMap = liveMap;
                            previousCircuit = circuitMap;
                            previousReferenceLap = referenceLap;
                            const positionCurrent = () => positionVersion === positionGeneration.current
                                && liveMap === getLiveMap() && circuitMap === liveMap?.getCircuitMap()
                                && referenceLap === liveMap?.getReferenceLap() && positionModel === models.current.position;
                            const inference = (async () => {
                                await (async () => {
                                    if (frameTasks.some(({ id }) => !models.current[id])) return;
                                    for (const { id } of frameTasks) {
                                        if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                        const model = models.current[id]!;
                                        try {
                                            if (id === 'depth') {
                                                const segment = result.detections.segment;
                                                if (options.current.segmentEnabled
                                                    && (segment?.task !== 'segment' || !(segment.width > 0 && segment.height > 0))) {
                                                    throw new Error('Depth requires a valid segmentation grid.');
                                                }
                                                // Preserve cockpit pixels for depth; their mask is used by downstream edge filtering.
                                                timing.start('depth');
                                                result.detections.depth = await models.current.depth!.detect(frame, frameConfidence);
                                                const depth = result.detections.depth;
                                                if (depth.task === 'depth' && (options.current.needsCpuDepth || Date.now() < cpuDepthUntil.current
                                                    || (depth.scale !== 'relative' && cameraCalibration.current))) {
                                                    timing.start('depthReadback');
                                                    await readDepthValues(depth);
                                                }
                                            } else {
                                                timing.start('segment');
                                                result.detections.segment = await model.detect(frame, frameConfidence);
                                            }
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
                                    if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                    const segment = result.detections.segment;
                                    if (segment?.task === 'segment') {
                                        try {
                                            // Depth has already been submitted; mask mapping never holds the ORT queue.
                                            timing.start('segmentReadback');
                                            await readSegmentInstances(segment);
                                        } catch (reason) {
                                            if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                            const segmentModel = models.current.segment;
                                            delete models.current.segment;
                                            delete result.detections.segment;
                                            void segment.gpu?.dispose().catch(() => undefined);
                                            void segmentModel?.dispose().catch(() => undefined);
                                            setDetectors((current) => ({ ...current, segment: { status: 'retrying', error: message(reason) } }));
                                            return;
                                        }
                                    }
                                })();
                                if (positionModel && references && telemetry && positionCurrent()
                                    && (!positionModel.requiresOrientation || telemetry.orientation)
                                    && version === captureVersion.current && stackVersion === modelVersion.current) {
                                    try {
                                        timing.start('position');
                                        const position = await positionModel.detect(frame, references, {
                                            normalized_position: telemetry.position, game: telemetry.game,
                                            orientation: telemetry.orientation,
                                            map_name: [circuitMap?.circuit_name, circuitMap?.source_track_key, telemetry.track]
                                                .filter((value): value is string => typeof value === 'string' && Boolean(value)),
                                        });
                                        if (positionCurrent()) result.position = { ...position, references, normalizedPosition: telemetry.position,
                                            referenceLap: referenceLap ?? undefined };
                                    } catch (reason) {
                                        if (version !== captureVersion.current || stackVersion !== modelVersion.current) return;
                                        delete models.current.position;
                                        void positionModel.dispose().catch(() => undefined);
                                        setDetectors((current) => ({ ...current, position: { status: 'retrying', error: message(reason) } }));
                                    }
                                }
                                if (!positionCurrent()) {
                                    positionModel?.resetHistory();
                                    delete result.position;
                                }
                            })();
                            inferenceRef.current = inference;
                            await inference;
                            if (version !== captureVersion.current) return;
                            if (stackVersion !== modelVersion.current) {
                                timing.finish(false);
                                timerRef.current = setTimeout(tick, 0);
                                return;
                            }
                            // Keep a clean copy for the camera calibration preview.
                            timing.start('preview');
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
                            const complete = frameTasks.every(({ id }) => Boolean(result.detections[id]));
                            publish(complete ? result : null, timing, result);
                            if (complete) unpublished = null;
                            setHasFrame(true);
                            setStatus(complete ? frameTasks.map(({ id, label }) => `${label} · ${Math.round(result.detections[id]!.inferenceMs)} ms`).join(' / ') || 'Screen shared. Segmentation and depth are disabled.'
                                : `Screen shared. Waiting for ${frameTasks.map(({ label }) => label.toLowerCase()).join(' and ')}.`);
                        }
                        timerRef.current = setTimeout(tick, Math.max(0, 200 - (performance.now() - started)));
                    } catch (reason) {
                        if (version !== captureVersion.current) return;
                        stop();
                        setError(message(reason));
                    } finally { disposeGpuFrame(unpublished); }
                };
                void tick();
            } catch (reason) {
                if (version !== captureVersion.current) return;
                stop();
                setError(message(reason));
            }
        };
        void start();
        return releaseFrames;
    }, [stream, publish, releaseFrames, stop, getLiveMap]);

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
                    <video ref={videoRef} muted playsInline hidden onSeeking={resetPositionHistory} />
                    <canvas ref={canvasRef} aria-label="Captured game frame with vision detections" hidden={!hasFrame}
                        className={step === 'depth-map' ? 'track-vision__depth-map' : undefined}
                        onMouseMove={step === 'depth-map' ? (event) => setDepthPointer({ clientX: event.clientX, clientY: event.clientY }) : undefined}
                        onMouseLeave={() => setDepthPointer(null)} />
                    {hasFrame && hoveredDepth && depthRect && <span className="track-vision__depth-pointer" aria-label="Depth at mouse"
                        style={{ left: hoveredDepth.x, top: hoveredDepth.y,
                            transform: `translate(${hoveredDepth.x > depthRect.width / 2 ? 'calc(-100% - 12px)' : '12px'}, ${hoveredDepth.y > depthRect.height / 2 ? 'calc(-100% - 12px)' : '12px'})` }}>
                        {!depthMap?.depth.values ? 'Reading depth…' : hoveredDepth.depth === null ? 'No valid depth' : formatDepth(hoveredDepth.depth, depthMap?.depth.scale, 2)}
                    </span>}
                    {hasFrame && step === 'calibration' && showCalibrationOnCapture && previewFrameRef.current && validCalibration(previewCamera)
                        && <CameraGroundGrid camera={previewCamera} applied={Boolean(calibration)} />}
                    {!hasFrame && <div className="track-vision__empty"><span className="track-vision__empty-icon" aria-hidden="true">▣</span><strong>No captured frame yet</strong><span>Share a game window from the Session Recording bar.</span></div>}
                    <div className="track-vision__preview-controls">
                        <button type="button" aria-expanded={previewExpanded} onClick={togglePreviewSize}>
                            {previewExpanded ? 'Restore capture' : 'Expand capture'}
                        </button>
                    </div>
                </dialog>
                <PipelineDetails step={step} frame={previewResult} masks={masks} classNames={labels} filterConfidence={filterConfidence}
                    depthMap={depthMap} segmentEnabled={segmentEnabled} depthEnabled={depthEnabled} />
                <div hidden={!isSceneStep}>
                    <ReconstructedSceneView scene={hasFrame ? previewResult?.reconstructedScene ?? null : null}
                        source={hasFrame ? previewFrameRef.current : null} capturedAt={previewCapturedAt} />
                </div>
                {step === 'birds-eye' && <BirdsEyeView scene={hasFrame ? previewResult?.birdsEyeScene ?? null : null}
                    hasReconstructedScene={hasFrame && Boolean(previewResult?.reconstructedScene)} capturedAt={previewResult?.capturedAt}
                    position={hasFrame ? previewResult?.position : undefined} positionReady={detectors.position.status === 'ready'} />}
            </div>
            <div className="track-vision__status" role="status">{status}</div>
            {error && <div className="track-vision__error" role="alert">{error}</div>}
            <details className="track-vision__settings" aria-label="Track Vision settings" open={VISION_MODELS.some(({ id }) => Boolean(detectors[id].error))}>
                <summary>Settings <span>{VISION_MODELS.map(({ id, label }) =>
                    `${label}: ${detectors[id].status}`).join(' · ')}</span></summary>
                <TrackCalibration source={hasFrame ? previewFrameRef.current : null} draft={cameraDraft} applied={calibration}
                    showOnCapture={showCalibrationOnCapture} onToggleCapture={() => setShowCalibrationOnCapture((current) => !current)}
                    onChange={(draft) => { setCameraDraft(draft); updateCalibration(); }}
                    onApply={() => { const frame = previewFrameRef.current; if (frame) updateCalibration({ ...cameraDraft, imageWidth: frame.width, imageHeight: frame.height }); }}
                    onClear={() => updateCalibration()} />
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
                    {VISION_MODELS.map(({ id, label }) => <div className="track-vision__detector" key={id}>
                        <strong>{label}</strong>
                        <span className="track-vision__detector-state">{detectors[id].status === 'ready'
                            ? captureState === 'active' && DETECTION_TASKS.every(({ id }) => detectors[id].status === 'ready' || detectors[id].status === 'disabled') ? 'Running' : 'Ready' : detectors[id].status === 'retrying' ? 'Retrying GPU…'
                                : detectors[id].status === 'disabled' ? 'Disabled' : detectors[id].status === 'error' ? 'Unavailable' : 'Loading…'}</span>
                        {id !== 'depth' ? <div className="track-vision__hint">
                            {id === 'segment' ? 'track-segments' : 'track-position'}{models.current[id] && ` · ${models.current[id]!.inputSize} × ${models.current[id]!.inputSize}`}
                        </div> : <label className="track-vision__resolution">Input resolution
                            <select aria-label={`${label} input resolution`} value={inputSizes[id]} disabled={id === 'depth' && !depthEnabled}
                                onChange={(event) => setInputSizes((current) => ({ ...current, [id]: Number(event.target.value) }))}>
                                {MODEL_INPUT_RESOLUTIONS[id].map(({ label, size }) =>
                                    <option key={size} value={size}>{label} · {size} × {size}</option>)}
                            </select>
                        </label>}
                        {id !== 'position' && <label className="track-vision__model-toggle">
                            <input type="checkbox" checked={id === 'segment' ? segmentEnabled : depthEnabled}
                                onChange={(event) => (id === 'segment' ? setSegmentEnabled : setDepthEnabled)(event.target.checked)} />
                            Enable {id === 'segment' ? 'segmentation' : 'depth'} model
                        </label>}
                        {!!detectors[id].classNames?.length && <div className="track-vision__hint" aria-label="Model labels">Labels: {detectors[id].classNames!.join(', ')}</div>}
                        {detectors[id].status === 'ready' && <div className="track-vision__hint" aria-label={`${label} inference device`}>GPU acceleration active</div>}
                        {detectors[id].error && <div className="track-vision__error" role="alert">
                            {detectors[id].error} <button type="button" onClick={() => setRetry((current) => current + 1)}>Retry {label}</button>
                        </div>}
                    </div>)}
                </fieldset>
            </details>
        </section>
    );
});

export default LiveTrackVision;
