import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { isDesktopGame, type DesktopGame } from 'contexts/DesktopGameContext';
import { readVideoTelemetryStart, VIDEO_TIMING_BYTES } from 'data/session-video-timing';
import { capturePolicyCameraImage, PolicyCameraImage } from 'views/live-session/live-session-recording/policy-camera';
import type { DrivingModelConfig } from './driving-model-config';
import type { DrivingPolicy } from './driving-policy';
import { cameraChannelStats, DrivingCameraInspection } from './driving-camera-inspection';
import { findRecordedGearShifts, parseRecordedDrivingSessionStream, RecordedDrivingSession } from './recorded-driving-session';
import RecordedTimeline from './RecordedTimeline';
import RecordedTrainingRanges, { timestamp, TrainingAnnotation, TrainingRange, TrainingRangeDraft, TrainingSelection } from './RecordedTrainingRanges';
import './RecordedFrameViewer.css';

const MIN_PLAYBACK_SPEED = 0.25;
const MAX_PLAYBACK_SPEED = 2;
const PLAYBACK_SPEED_STEP = 0.25;

export interface RecordedReplayFrame {
    session: RecordedDrivingSession;
    videoSeconds: number;
    telemetrySeconds: number;
    recordingId?: string;
    recordingName?: string;
    startsRange?: boolean;
    clipLabel?: string;
}
export interface RecordedDrivingFrame extends RecordedReplayFrame {
    camera: PolicyCameraImage;
}
export interface RecordedReplayPosition {
    recordingNumber: number;
    recordingCount: number;
    processedFrames: number;
    totalFrames: number;
}
export interface RecordedFrameViewerHandle {
    countTrainingClips(): number;
    measure(consume: (frame: RecordedReplayFrame) => Promise<void>, signal: AbortSignal,
        onProgress?: (position: RecordedReplayPosition) => void, controlIntervalMs?: number): Promise<void>;
    replay(consume: (frame: RecordedDrivingFrame) => Promise<void>, signal: AbortSignal,
        onProgress?: (position: RecordedReplayPosition) => void, controlIntervalMs?: number): Promise<void>;
}
interface Props {
    configuration: DrivingModelConfig;
    policy: DrivingPolicy | null;
    disabled: boolean;
    trainingPickerContainer?: HTMLDivElement | null;
    onFrame: (frame: RecordedDrivingFrame | null) => void;
    onBusyChange?: (busy: boolean) => void;
    onReplayReadyChange?: (ready: boolean) => void;
    onReplaySelectionChange?: () => void;
}
interface Example { seconds: number; score: number; thumbnail: string }
interface Gallery { layer: number; examples: Example[][]; sampled: number; interval: number; complete: boolean }
interface SelectionState {
    selection: TrainingSelection;
    dirty?: boolean;
    error?: string;
    offsetEdit?: { telemetry: File | null; seconds: number };
}
interface ListedAnnotation extends TrainingAnnotation { folderPath?: string; imported?: boolean }
const FULL_RECORDING: TrainingSelection = { version: 1, ranges: null };
const videoFile = (file: File) => file.type.startsWith('video/') || /\.(mp4|webm|mkv|mov|m4v|avi|ogv|mpg|mpeg|3gp)$/i.test(file.name);
const telemetryFile = (file: File) => /\.jsonl$/i.test(file.name);
const inferRecordedGame = (name?: string): DesktopGame => {
    // Flat telemetry from the desktop writer and the local IBT importer identifies its game by filename.
    const fileName = name?.toLowerCase() ?? '';
    const game = /^(acc|ac|iracing)_\d+_[a-f0-9]{12}\.jsonl$/.exec(fileName)?.[1];
    if (isDesktopGame(game)) return game;
    return /^iracing_local_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\.jsonl$/.test(fileName) ? 'iracing' : 'acc';
};
const fileLabel = (file: File) => file.webkitRelativePath || file.name;
const folderRelativePath = (file: File) => fileLabel(file).replace(/^[^/]+\//, '');
const annotationLabel = (item: ListedAnnotation) => item.folderPath ?? item.name;
const sameFile = (a: File, b: File) => fileLabel(a) === fileLabel(b) && a.size === b.size && a.lastModified === b.lastModified;
const appendFiles = (existing: File[], added: File[]) => [...existing, ...added.filter((file) => !existing.some((other) => sameFile(file, other)))];
const folderVideosWithoutAnnotations = (files: File[], annotations: ListedAnnotation[]) => {
    const sidecarSuffix = /\.training-ranges\.json$/i;
    const annotated = new Set(files.filter((file) => sidecarSuffix.test(file.name))
        .map((file) => folderRelativePath(file).replace(sidecarSuffix, '')));
    for (const { folderPath, selection } of annotations) {
        if (folderPath) annotated.add(folderPath.replace(/[^/]+$/, () => selection.videoFileName));
    }
    return files.filter((file) => videoFile(file) && !annotated.has(folderRelativePath(file)));
};
const matchingTelemetry = (video: File, files: File[]) => {
    const label = fileLabel(video).toLowerCase();
    // Desktop recordings use <source>.jsonl.video-<timestamp>-<uuid>.mp4.
    const telemetry = label.replace(/\.jsonl\.video-\d+-[a-f0-9-]+\.(mp4|webm)$|\.[^./]+$/i, '.jsonl');
    const matches = files.filter((file) => fileLabel(file).toLowerCase() === telemetry);
    return matches.length === 1 ? matches[0] : null;
};
const abortError = () => new DOMException('Recording operation cancelled.', 'AbortError');
const checkAbort = (signal: AbortSignal) => { if (signal.aborted) throw abortError(); };
const message = (error: unknown) => error instanceof Error ? error.message : 'Could not read the recording.';
const readTelemetry = async (source: File | string, game: DesktopGame, signal: AbortSignal) => {
    checkAbort(signal);
    if (typeof source !== 'string') return parseRecordedDrivingSessionStream(source.stream(), game, signal);
    const response = await fetch(source, { signal });
    if (!response.ok || !response.body) throw new Error('The annotation’s telemetry file could not be read.');
    return parseRecordedDrivingSessionStream(response.body, game, signal);
};
const hasUsableVideoTiming = (session: RecordedDrivingSession, videoTiming?: number | null): videoTiming is number =>
    videoTiming != null && session.timestampField === 'Graphics_analyst_session_time';
const telemetryAtVideoStart = (session: RecordedDrivingSession, videoTiming?: number | null, offsetSeconds = 0) => {
    if (hasUsableVideoTiming(session, videoTiming)) return videoTiming;
    return session.startTimeSeconds - offsetSeconds;
};
const alignedRange = (duration: number, session: RecordedDrivingSession, telemetryStart: number): TrainingRange | null => {
    const start = Math.max(0, session.startTimeSeconds - telemetryStart);
    const end = Math.min(duration - 0.001, session.startTimeSeconds + session.durationSeconds - telemetryStart);
    return Number.isFinite(duration) && start <= end ? { start, end } : null;
};

function watchPresentedVideoFrames(player: HTMLVideoElement) {
    let frame: { position: number; mediaTime: number } | undefined;
    let callback: number | undefined;
    if (typeof player.requestVideoFrameCallback === 'function' && typeof player.cancelVideoFrameCallback === 'function') {
        const trackFrame: VideoFrameRequestCallback = (_, { mediaTime }) => {
            if (!player.seeking) frame = { position: player.currentTime, mediaTime };
            callback = player.requestVideoFrameCallback(trackFrame);
        };
        callback = player.requestVideoFrameCallback(trackFrame);
    }
    return {
        player,
        clear: () => { frame = undefined; },
        read: () => {
            const position = player.currentTime;
            return frame && (!player.paused || Math.abs(frame.position - position) < 0.0001) ? frame.mediaTime : position;
        },
        stop: () => { if (callback !== undefined) player.cancelVideoFrameCallback(callback); },
    };
}

function loadReplayVideo(video: HTMLVideoElement, url: string, signal: AbortSignal): Promise<void> {
    checkAbort(signal);
    return new Promise((resolve, reject) => {
        const finish = (error?: Error) => {
            clearTimeout(timeout);
            video.removeEventListener('loadeddata', loaded);
            video.removeEventListener('error', failed);
            signal.removeEventListener('abort', aborted);
            error ? reject(error) : resolve();
        };
        const loaded = () => finish();
        const failed = () => finish(new Error('The video could not be decoded. Try an MP4 or WebM recording.'));
        const aborted = () => finish(abortError());
        const timeout = setTimeout(() => finish(new Error('Timed out loading the video.')), 15000);
        video.addEventListener('loadeddata', loaded, { once: true });
        video.addEventListener('error', failed, { once: true });
        signal.addEventListener('abort', aborted, { once: true });
        video.preload = 'auto';
        video.src = url;
    });
}

function seekVideo(video: HTMLVideoElement, seconds: number, signal: AbortSignal,
    frames?: ReturnType<typeof watchPresentedVideoFrames> | null): Promise<void> {
    checkAbort(signal);
    if (!video.seeking && Math.abs(video.currentTime - seconds) < 0.0001 && video.readyState >= 2) return Promise.resolve();
    frames?.clear();
    return new Promise((resolve, reject) => {
        const finish = (error?: Error) => {
            clearTimeout(timeout);
            video.removeEventListener('seeked', seeked);
            video.removeEventListener('error', failed);
            signal.removeEventListener('abort', aborted);
            error ? reject(error) : resolve();
        };
        const seeked = () => finish();
        const failed = () => finish(new Error('The recorded video could not be decoded. Try an MP4 or WebM recording.'));
        const aborted = () => finish(abortError());
        const timeout = setTimeout(() => finish(new Error('Timed out seeking the recorded video.')), 15000);
        video.addEventListener('seeked', seeked, { once: true });
        video.addEventListener('error', failed, { once: true });
        signal.addEventListener('abort', aborted, { once: true });
        try { video.currentTime = seconds; } catch (error) { finish(error instanceof Error ? error : new Error(String(error))); }
    });
}

function originalFrame(video: HTMLVideoElement): HTMLCanvasElement {
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight || video.seeking) {
        throw new Error('Wait for the recorded video frame to finish loading.');
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Recorded frame pixels are unavailable.');
    context.drawImage(video, 0, 0);
    return canvas;
}

function thumbnail(source: HTMLCanvasElement): string {
    const canvas = document.createElement('canvas');
    canvas.width = 320; canvas.height = Math.max(1, Math.round(320 * source.height / source.width));
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Recorded frame preview is unavailable.');
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

const RecordedFrameViewer = forwardRef<RecordedFrameViewerHandle, Props>(function RecordedFrameViewer(props, ref) {
    const { onReplayReadyChange, onReplaySelectionChange } = props;
    const current = useRef(props); current.current = props;
    const video = useRef<HTMLVideoElement>(null);
    const mounted = useRef(true);
    const heatmap = useRef<HTMLCanvasElement>(null);
    const operation = useRef<AbortController | null>(null);
    const busyRef = useRef('');
    const [busy, setBusy] = useState('');
    const [editor, setEditor] = useState<'add' | { annotationId: string } | null>(null);
    const [annotations, setAnnotations] = useState<ListedAnnotation[]>([]);
    const [included, setIncluded] = useState<string[]>([]);
    const [annotationErrors, setAnnotationErrors] = useState<string[]>([]);
    const [trainingAnnotationErrors, setTrainingAnnotationErrors] = useState<string[]>([]);
    const [savedName, setSavedName] = useState('');
    const [pickedVideos, setPickedVideos] = useState<File[]>([]);
    const [pickedTelemetry, setPickedTelemetry] = useState<File[]>([]);
    const [workingFolder, setWorkingFolder] = useState<File[] | null>(null);
    const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
    const [telemetryOverrides, setTelemetryOverrides] = useState(new Map<File, File | null>());
    const [selections, setSelections] = useState(new Map<File | string, SelectionState>());
    const folderVideoCount = useMemo(() => workingFolder?.filter(videoFile).length ?? 0, [workingFolder]);
    const folderVideos = useMemo(() => folderVideosWithoutAnnotations(workingFolder ?? [], annotations), [annotations, workingFolder]);
    const videoFiles = useMemo(() => appendFiles(pickedVideos, folderVideos), [pickedVideos, folderVideos]);
    const telemetryFiles = useMemo(() => appendFiles(pickedTelemetry, workingFolder?.filter(telemetryFile) ?? []), [pickedTelemetry, workingFolder]);
    const videoSettings = selectedVideo ? selections.get(selectedVideo) : undefined;
    const adding = editor === 'add';
    const annotation = editor && editor !== 'add' ? annotations.find((item) => item.id === editor.annotationId) : undefined;
    const selectionKey = annotation?.id ?? (adding ? selectedVideo : null);
    const selectedSettings = selectionKey ? selections.get(selectionKey) : undefined;
    const draft = selectedSettings?.selection ?? annotation?.selection;
    const savedTelemetry = selectedVideo && (draft?.version === 2
        ? telemetryFiles.find((file) => fileLabel(file) === fileLabel(selectedVideo).replace(/[^/]+$/, () => draft.telemetryFileName))
        : matchingTelemetry(selectedVideo, telemetryFiles));
    const selectedTelemetry = (() => {
        if (!adding || !selectedVideo) return null;
        const override = telemetryOverrides.get(selectedVideo);
        if (override === null) return null;
        if (override && telemetryFiles.some((file) => sameFile(file, override))) return override;
        const allowSinglePair = draft?.version !== 2 && videoFiles.length === 1 && folderVideoCount === folderVideos.length;
        return savedTelemetry ?? (allowSinglePair && telemetryFiles.length === 1 ? telemetryFiles[0] : null);
    })();
    const offsetEdit = selectedSettings?.offsetEdit;
    const editedOffset = offsetEdit && (annotation || (selectedTelemetry && offsetEdit.telemetry && sameFile(selectedTelemetry, offsetEdit.telemetry)))
        ? offsetEdit.seconds : undefined;
    const savedPair = annotation || (selectedTelemetry && savedTelemetry && sameFile(selectedTelemetry, savedTelemetry));
    const previewOffset = editedOffset ?? (savedPair ? draft?.offsetSeconds : undefined);
    const previewVideo = annotation?.videoUrl ?? (adding ? selectedVideo : null);
    const previewTelemetry = annotation?.telemetryUrl ?? selectedTelemetry;
    const previewGame = draft?.game ?? inferRecordedGame(annotation?.selection.telemetryFileName ?? selectedTelemetry?.name);
    const timedLocalVideo = previewVideo && typeof previewVideo !== 'string' && /\.mp4$/i.test(previewVideo.name)
        && previewVideo.size >= VIDEO_TIMING_BYTES ? previewVideo : null;
    const [localVideoTiming, setLocalVideoTiming] = useState<{ file: File; start?: number | null } | null>(null);
    const videoTiming = annotation?.videoTelemetryStartSeconds
        ?? (timedLocalVideo && localVideoTiming?.file === timedLocalVideo ? localVideoTiming.start : undefined);
    const timingPending = Boolean(timedLocalVideo && localVideoTiming?.file !== timedLocalVideo);
    const [url, setUrl] = useState('');
    const [session, setSession] = useState<RecordedDrivingSession | null>(null);
    const [loadingTelemetry, setLoadingTelemetry] = useState(false);
    const { maxTransitionSeconds, telemetryMaxAgeMs } = props.configuration.runtime;
    const maxShiftSeconds = Math.min(0.5, maxTransitionSeconds, telemetryMaxAgeMs / 1000);
    const gearShifts = useMemo(() => session ? findRecordedGearShifts(session, maxShiftSeconds) : [], [session, maxShiftSeconds]);
    const telemetryStart = session ? telemetryAtVideoStart(session, videoTiming, previewOffset) : 0;
    const shiftVideoOffset = session ? session.startTimeSeconds - telemetryStart : 0;
    const sessionRef = useRef(session); sessionRef.current = session;
    const presentedFrames = useRef<ReturnType<typeof watchPresentedVideoFrames> | null>(null);
    const [time, setTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [error, setError] = useState('');
    const [inspection, setInspection] = useState<DrivingCameraInspection | null>(null);
    const [inspectedImage, setInspectedImage] = useState('');
    const [inspectedTime, setInspectedTime] = useState(0);
    const [layerIndex, setLayerIndex] = useState(0);
    const [channel, setChannel] = useState(0);
    const [interval, setIntervalSeconds] = useState(1);
    const [gallery, setGallery] = useState<Gallery | null>(null);
    const [progress, setProgress] = useState('');
    const locked = props.disabled || Boolean(busy) || loadingTelemetry || timingPending;
    const queued = useMemo(() => annotations.filter((item) => included.includes(item.id)), [annotations, included]);
    const replayReady = queued.length > 0;
    const folderPairCount = annotations.filter((item) => item.folderPath).length;
    const folderError = workingFolder && ((!pickedVideos.length && !folderVideoCount) || !telemetryFiles.length)
        ? 'Choose a folder containing a video and telemetry .jsonl file, or select the missing file individually.' : '';

    const changeBusy = (value: string) => {
        busyRef.current = value;
        if (mounted.current) { setBusy(value); current.current.onBusyChange?.(Boolean(value)); }
    };
    const resetInspection = () => { setInspection(null); setInspectedImage(''); setGallery(null); setProgress(''); };
    const publishFrame = () => {
        if (busyRef.current) return;
        const player = video.current;
        const recording = sessionRef.current;
        if (player) setTime(player.currentTime);
        if (timingPending || !recording || !player || player.readyState < 2 || player.seeking) {
            current.current.onFrame(null); return;
        }
        const frames = presentedFrames.current;
        const videoSeconds = frames?.player === player ? frames.read() : player.currentTime;
        try {
            current.current.onFrame({ session: recording, videoSeconds,
                telemetrySeconds: telemetryAtVideoStart(recording, videoTiming, previewOffset) + videoSeconds,
                camera: capturePolicyCameraImage(originalFrame(player)) });
        }
        catch (reason) { setError(message(reason)); current.current.onFrame(null); }
    };

    useEffect(() => {
        if (!adding || !selectedVideo || videoSettings) return;
        let active = true;
        const remember = (settings: SelectionState) => {
            if (active) setSelections((previous) => new Map(previous).set(selectedVideo, settings));
        };
        const bridge = window.recordedTrainingRanges;
        if (bridge) {
            void bridge.read(selectedVideo).then((selection) => remember({ selection: selection ?? FULL_RECORDING }))
                .catch((reason) => remember({ selection: FULL_RECORDING, error: `Could not load saved ranges. ${message(reason)}` }));
        } else remember({ selection: FULL_RECORDING });
        return () => { active = false; };
    }, [adding, selectedVideo, videoSettings]);
    useEffect(() => {
        setDuration(0); setTime(0); setError(''); resetInspection();
        current.current.onFrame(null);
        if (!previewVideo || typeof previewVideo === 'string') { setUrl(previewVideo || ''); return; }
        const next = URL.createObjectURL(previewVideo);
        setUrl(next);
        return () => URL.revokeObjectURL(next);
        // File identity owns the video URL and the matching inspection results.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [previewVideo]);
    useEffect(() => {
        if (!timedLocalVideo) return;
        let active = true;
        void timedLocalVideo.slice(-VIDEO_TIMING_BYTES).arrayBuffer().then((bytes) => {
            if (active) setLocalVideoTiming({ file: timedLocalVideo, start: readVideoTelemetryStart(new Uint8Array(bytes)) });
        }).catch((reason) => {
            if (!active) return;
            setLocalVideoTiming({ file: timedLocalVideo });
            setError(`Could not read video timing. ${message(reason)}`);
        });
        return () => { active = false; };
    }, [timedLocalVideo]);
    useEffect(() => {
        const player = video.current;
        if (!player) return;
        const frames = watchPresentedVideoFrames(player);
        presentedFrames.current = frames;
        return () => {
            presentedFrames.current = null;
            frames.stop();
        };
    }, [url]);
    useEffect(() => {
        const player = video.current;
        if (!player) return;
        player.defaultPlaybackRate = playbackSpeed;
        player.playbackRate = playbackSpeed;
    }, [url, playbackSpeed]);
    useEffect(() => {
        setSession(null); sessionRef.current = null; current.current.onFrame(null); setError('');
        resetInspection();
        if (!previewTelemetry) { setLoadingTelemetry(false); return; }
        const controller = new AbortController();
        const { signal } = controller;
        setLoadingTelemetry(true);
        void readTelemetry(previewTelemetry, previewGame, signal).then((recording) => {
            if (signal.aborted) return;
            setSession(recording);
        }).catch((reason) => { if (!signal.aborted) setError(message(reason)); })
            .finally(() => { if (!signal.aborted) setLoadingTelemetry(false); });
        return () => controller.abort();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [previewTelemetry, previewGame]);
    useEffect(() => {
        publishFrame();
        // Refresh the playhead when recorded timing or legacy annotation timing becomes available.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session, timingPending, videoTiming, previewOffset]);
    useEffect(() => {
        onReplayReadyChange?.(replayReady);
    }, [onReplayReadyChange, replayReady]);
    useEffect(() => { onReplaySelectionChange?.(); }, [onReplaySelectionChange, queued]);
    useEffect(() => {
        if (busyRef.current === 'Inspecting frame' || busyRef.current === 'Scanning recording') operation.current?.abort();
        resetInspection(); setLayerIndex(0); setChannel(0);
        // Model learning invalidates activation rankings from older weights.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [props.policy, props.policy?.steps]);
    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false; operation.current?.abort(); current.current.onFrame(null);
            current.current.onBusyChange?.(false); current.current.onReplayReadyChange?.(false);
        };
    }, []);

    const runOperation = async (label: string, work: (signal: AbortSignal) => Promise<void>, external?: AbortSignal) => {
        if (busyRef.current) throw new Error('Wait for the current recording operation to finish.');
        const controller = new AbortController();
        const abort = () => controller.abort();
        if (external?.aborted) throw abortError();
        external?.addEventListener('abort', abort, { once: true });
        operation.current = controller; changeBusy(label); setError('');
        video.current?.pause();
        try { await work(controller.signal); }
        catch (reason) { if (controller.signal.aborted) throw abortError(); throw reason; }
        finally {
            external?.removeEventListener('abort', abort);
            if (operation.current === controller) { operation.current = null; changeBusy(''); if (mounted.current) publishFrame(); }
        }
    };
    const walkReplay = async (label: string, consume: (frame: RecordedReplayFrame, player: HTMLVideoElement) => Promise<void>,
        signal: AbortSignal, onProgress?: (position: RecordedReplayPosition) => void,
        controlIntervalMs = current.current.configuration.runtime.controlIntervalMs) => {
            if (!replayReady) throw new Error('Select at least one training annotation with footage.');
            await runOperation(label, async (active) => {
                const step = Math.max(0.1, controlIntervalMs / 1000);
                for (let recordingIndex = 0; recordingIndex < queued.length; recordingIndex++) {
                    checkAbort(active);
                    const item = queued[recordingIndex];
                    const { telemetryUrl, videoUrl, videoTelemetryStartSeconds, selection } = item;
                    const { ranges, fullRecordingLabel } = selection;
                    const replayGame = selection.game ?? inferRecordedGame(selection.telemetryFileName);
                    const name = annotationLabel(item);
                    const preview = item.id === annotation?.id && url === videoUrl && sessionRef.current && duration ? video.current : null;
                    const player = preview ?? document.createElement('video');
                    const frames = preview ? presentedFrames.current : watchPresentedVideoFrames(player);
                    const previousTime = player.currentTime;
                    try {
                        player.pause();
                        const recording = (preview && previewGame === replayGame && sessionRef.current)
                            || await readTelemetry(telemetryUrl, replayGame, active);
                        checkAbort(active);
                        if (!preview) {
                            player.crossOrigin = 'anonymous';
                            await loadReplayVideo(player, videoUrl, active);
                        }
                        const telemetryStart = telemetryAtVideoStart(recording, videoTelemetryStartSeconds, selection.offsetSeconds);
                        const overlap = alignedRange(player.duration, recording, telemetryStart);
                        if (!overlap) throw new Error('The video and telemetry timelines do not overlap.');
                        const clips = (ranges ?? [{ ...overlap, label: fullRecordingLabel }]).map((range, index) => {
                            const start = Math.max(range.start, overlap.start), end = Math.min(range.end, overlap.end);
                            if (end - start < step - 1e-7) {
                                throw new Error(`Range ${index + 1} needs at least ${step} s of aligned video and telemetry.`);
                            }
                            const frameCount = Math.floor((end - start + 1e-7) / step) + 1;
                            return { start, end, frameCount, clipLabel: range.label };
                        });
                        const totalFrames = clips.reduce((total, clip) => total + clip.frameCount, 0);
                        let processedFrames = 0;
                        const reportProgress = () => onProgress?.({ recordingNumber: recordingIndex + 1,
                            recordingCount: queued.length, processedFrames, totalFrames });
                        reportProgress();
                        for (const { start, end, frameCount, clipLabel } of clips) {
                            for (let index = 0; index < frameCount; index++) {
                                checkAbort(active);
                                await seekVideo(player, Math.min(start + index * step, end), active, frames); checkAbort(active);
                                if (preview) setTime(player.currentTime);
                                const videoSeconds = frames?.read() ?? player.currentTime;
                                await consume({ session: recording, videoSeconds, telemetrySeconds: telemetryStart + videoSeconds,
                                    recordingId: item.id, recordingName: name, startsRange: index === 0,
                                    ...(clipLabel ? { clipLabel } : {}) }, player);
                                checkAbort(active);
                                processedFrames++;
                                reportProgress();
                            }
                        }
                        checkAbort(active);
                        if (preview) await seekVideo(preview, previousTime, active, frames);
                    } catch (reason) {
                        checkAbort(active);
                        throw new Error(`${name}: ${message(reason)}`);
                    } finally {
                        if (!preview) { frames?.stop(); player.removeAttribute('src'); player.load(); }
                    }
                }
            }, signal);
    };
    useImperativeHandle(ref, () => ({
        countTrainingClips: () => queued.reduce((count, { selection }) => count + (selection.ranges?.length ?? 1), 0),
        measure: (consume, signal, onProgress, controlIntervalMs) => walkReplay('Counting aligned samples', (frame) => consume(frame), signal, onProgress, controlIntervalMs),
        replay: (consume, signal, onProgress, controlIntervalMs) => walkReplay('Replaying recordings', (frame, player) => consume({ ...frame,
            camera: capturePolicyCameraImage(originalFrame(player)),
        }), signal, onProgress, controlIntervalMs),
    }));

    const reportError = (reason: unknown) => { if (mounted.current && !(reason instanceof Error && reason.name === 'AbortError')) setError(message(reason)); };
    const inspect = async (seconds?: number) => {
        const model = current.current.policy;
        if (!model || !video.current) return;
        try {
            await runOperation('Inspecting frame', async (signal) => {
                if (seconds !== undefined) await seekVideo(video.current!, seconds, signal);
                const source = originalFrame(video.current!), at = video.current!.currentTime;
                const result = await model.inspectCameraAsync(capturePolicyCameraImage(source)); checkAbort(signal);
                setInspection(result); setInspectedImage(source.toDataURL('image/jpeg', 0.9)); setInspectedTime(at);
                setLayerIndex((index) => Math.min(index, result.layers.length - 1));
            });
        } catch (reason) { reportError(reason); }
    };
    const scan = async () => {
        const model = current.current.policy, player = video.current;
        if (!model || !player || !Number.isFinite(player.duration) || player.duration <= 0) return;
        const selectedLayer = layerIndex, sampleInterval = interval, previousTime = player.currentTime, modelSteps = model.steps;
        let result: Gallery = { layer: selectedLayer, examples: [], sampled: 0, interval: sampleInterval, complete: false };
        try {
            await runOperation('Scanning recording', async (signal) => {
                const count = Math.ceil(player.duration / sampleInterval);
                setGallery(null);
                for (let index = 0; index < count; index++) {
                    checkAbort(signal); await seekVideo(player, index * sampleInterval, signal);
                    const source = originalFrame(player);
                    const response = await model.inspectCameraAsync(capturePolicyCameraImage(source)); checkAbort(signal);
                    const layer = response.layers[selectedLayer];
                    if (!layer) throw new Error('The selected camera layer is unavailable in this model.');
                    if (!result.examples.length) result.examples = Array.from({ length: layer.channels }, () => []);
                    let preview = '';
                    result.examples.forEach((examples, ch) => {
                        const score = cameraChannelStats(layer, ch).peak;
                        if (score <= 0 || (examples.length >= 3 && score <= examples[examples.length - 1].score)) return;
                        if (!preview) preview = thumbnail(source);
                        examples.push({ seconds: player.currentTime, score, thumbnail: preview });
                        examples.sort((a, b) => b.score - a.score || a.seconds - b.seconds);
                        examples.splice(3);
                    });
                    result.sampled++;
                    setProgress(`Scanned ${result.sampled} / ${count} frames`);
                }
                result.complete = true;
                await seekVideo(player, previousTime, signal);
            });
        } catch (reason) { reportError(reason); }
        finally { if (mounted.current && result.sampled && current.current.policy === model && model.steps === modelSteps) setGallery(result); }
    };

    const layer = inspection?.layers[layerIndex];
    const selectedChannel = layer ? Math.min(channel, layer.channels - 1) : 0;
    const stats = layer ? cameraChannelStats(layer, selectedChannel) : null;
    useEffect(() => {
        const canvas = heatmap.current;
        if (!canvas || !layer) return;
        const context = canvas.getContext('2d');
        if (!context) return;
        canvas.width = layer.width; canvas.height = layer.height;
        const pixels = context.createImageData(layer.width, layer.height);
        let maximum = 0;
        for (let i = selectedChannel; i < layer.values.length; i += layer.channels) maximum = Math.max(maximum, Math.abs(layer.values[i]));
        for (let pixel = 0; pixel < layer.width * layer.height; pixel++) {
            const value = maximum ? layer.values[pixel * layer.channels + selectedChannel] / maximum : 0;
            pixels.data[pixel * 4] = value > 0 ? Math.round(value * 255) : 0;
            pixels.data[pixel * 4 + 1] = Math.round(Math.abs(value) * 120);
            pixels.data[pixel * 4 + 2] = value < 0 ? Math.round(-value * 255) : 0;
            pixels.data[pixel * 4 + 3] = 255;
        }
        context.putImageData(pixels, 0, 0);
    }, [layer, selectedChannel]);

    const addVideos = (files: File[]) => {
        const added = appendFiles(videoFiles, files).slice(videoFiles.length);
        setPickedVideos([...pickedVideos, ...added]);
        setSavedName('');
        if (!selectedVideo) setSelectedVideo(videoFiles[0] ?? added[0] ?? null);
    };
    const chooseTelemetry = (files: FileList | null) => {
        const selected = Array.from(files ?? []).filter(telemetryFile);
        if (selectedVideo && selected.length === 1) {
            setTelemetryOverrides((previous) => new Map(previous).set(selectedVideo, selected[0]));
        }
        setPickedTelemetry(appendFiles(pickedTelemetry, selected));
    };
    const editSelection = (changes: Partial<TrainingRangeDraft>) => {
        if (!selectionKey || !draft || selectedSettings?.error) return;
        setSavedName('');
        const { offsetSeconds, ...selectionChanges } = changes;
        setSelections((previous) => {
            const prior = previous.get(selectionKey);
            return new Map(previous).set(selectionKey, {
                selection: { ...(prior?.selection ?? draft), ...selectionChanges }, dirty: true,
                offsetEdit: offsetSeconds === undefined ? prior?.offsetEdit : { telemetry: selectedTelemetry, seconds: offsetSeconds },
            });
        });
    };
    const openEditor = (next: typeof editor) => {
        setSavedName('');
        setEditor(next);
    };
    const rememberAnnotations = (loaded: ListedAnnotation[], mode: 'merge' | 'folder' | 'training' = 'merge') => {
        const replaceFolder = mode === 'folder';
        const existing = new Map(annotations.map((item) => [item.id, item]));
        const next = new Map(annotations.filter((item) => !replaceFolder || !item.folderPath || item.imported)
            .map((item) => [item.id, replaceFolder ? { ...item, folderPath: undefined } : item]));
        for (const item of loaded) {
            next.set(item.id, { ...next.get(item.id), ...item, imported: existing.get(item.id)?.imported || !item.folderPath });
        }
        const list = Array.from(next.values());
        const pickedIds = new Set(loaded.map(({ id }) => id));
        setAnnotations(list);
        setIncluded((previous) => list.filter(({ id, selection }) => selection.ranges?.length !== 0
            && (mode === 'training' ? pickedIds.has(id) : !existing.has(id) || previous.includes(id))).map(({ id }) => id));
        if (annotation && !next.has(annotation.id)) openEditor(null);
    };
    const chooseWorkingFolder = async (files: File[]) => {
        const applyFolder = (loaded: ListedAnnotation[]) => {
            setWorkingFolder(files);
            const videos = appendFiles(pickedVideos, folderVideosWithoutAnnotations(files, loaded));
            if (!selectedVideo || !videos.includes(selectedVideo)) {
                setSelectedVideo(videos.length === 1 ? videos[0] : null);
                if (adding) setSavedName('');
            }
        };
        const bridge = window.recordedTrainingRanges;
        if (!bridge) { applyFolder([]); return; }
        try {
            await runOperation('Searching working folder', async (signal) => {
                const jsonFiles = files.filter((file) => /\.json$/i.test(file.name));
                const results = jsonFiles.length ? await bridge.scan(jsonFiles) : [];
                checkAbort(signal);
                const loaded: ListedAnnotation[] = [], errors: string[] = [];
                results.forEach((result, index) => {
                    if (!result) return;
                    const folderPath = folderRelativePath(jsonFiles[index]);
                    if ('error' in result) errors.push(`${folderPath}: ${result.error}`);
                    else loaded.push({ ...result, folderPath });
                });
                applyFolder(loaded);
                rememberAnnotations(loaded, 'folder');
                setAnnotationErrors(errors);
            });
        } catch (reason) { reportError(reason); }
    };
    const openAnnotations = async (files: File[], purpose: 'edit' | 'training') => {
        const bridge = window.recordedTrainingRanges;
        if (!bridge || !files.length) return;
        const setErrors = purpose === 'training' ? setTrainingAnnotationErrors : setAnnotationErrors;
        try {
            await runOperation('Opening training annotations', async (signal) => {
                setErrors([]);
                const result = await Promise.allSettled(files.map((file) => bridge.open(file)));
                checkAbort(signal);
                const loaded: TrainingAnnotation[] = [], errors: string[] = [];
                result.forEach((item, index) => {
                    if (item.status === 'fulfilled') loaded.push(item.value);
                    else errors.push(`${files[index].name}: ${message(item.reason)}`);
                });
                rememberAnnotations(loaded, purpose === 'training' ? 'training' : 'merge');
                setErrors(errors);
                if (purpose === 'edit' && loaded.length) openEditor({ annotationId: loaded[0].id });
            });
        } catch (reason) { reportError(reason); }
    };
    const dirty = Boolean(selectedSettings?.dirty || (draft?.version === 2 && selectedTelemetry && draft.telemetryFileName !== selectedTelemetry.name));
    const canSave = Boolean(window.recordedTrainingRanges && session && draft && !timingPending
        && (annotation ? dirty : selectedVideo && selectedTelemetry && (dirty || draft.version === 1)));
    const saveSelection = async () => {
        const bridge = window.recordedTrainingRanges;
        if (!canSave || !bridge || !selectionKey || !draft) return;
        const fullRecordingLabel = draft.ranges === null ? draft.fullRecordingLabel?.trim() : undefined;
        const changes = { ranges: draft.ranges, ...(fullRecordingLabel ? { fullRecordingLabel } : {}),
            ...(draft.game !== undefined ? { game: draft.game } : {}),
            ...(editedOffset !== undefined ? { offsetSeconds: editedOffset } : {}) };
        const persist = annotation ? () => bridge.update(annotation.id, changes)
            : selectedVideo && selectedTelemetry ? () => bridge.write(selectedVideo, selectedTelemetry, changes) : null;
        if (!persist) return;
        try {
            await runOperation('Saving training annotation', async (signal) => {
                const saved = await persist();
                checkAbort(signal);
                setSelections((previous) => {
                    const next = new Map(previous);
                    if (annotation) next.delete(selectionKey);
                    else next.set(selectionKey, { selection: saved.selection });
                    return next;
                });
                setSavedName(saved.name);
                if (annotation) {
                    setAnnotations((previous) => previous.map((item) => item.id === saved.id ? { ...item, ...saved } : item));
                    if (saved.selection.ranges?.length === 0) setIncluded((previous) => previous.filter((id) => id !== saved.id));
                } else {
                    const folderPath = selectedVideo && workingFolder?.includes(selectedVideo)
                        ? folderRelativePath(selectedVideo).replace(/[^/]+$/, () => saved.name) : undefined;
                    rememberAnnotations([{ ...saved, folderPath }]);
                    if (folderPath) {
                        setSelectedVideo(null);
                        setEditor({ annotationId: saved.id });
                    }
                }
            });
        } catch (reason) { reportError(reason); }
    };
    const seek = (seconds: number) => {
        if (!video.current) return;
        video.current.pause();
        presentedFrames.current?.clear();
        video.current.currentTime = Math.min(seconds, Math.max(0, duration - 0.001));
        setTime(video.current.currentTime);
    };
    const layers = inspection?.layers ?? props.configuration.network.cameraLayers.map((item, index) => ({ ...item, index }));
    const trainingPicker = <div className="recorded-training-picker">
        <label className="recorded-setting">Select annotation files for training
            <input type="file" accept=".json" multiple disabled={props.disabled || Boolean(busy) || !window.recordedTrainingRanges}
                onChange={(event) => {
                    const files = Array.from(event.target.files ?? []);
                    event.target.value = '';
                    void openAnnotations(files, 'training');
                }} />
        </label>
        {trainingAnnotationErrors.length > 0 && <ul className="driving-error" role="alert">
            {trainingAnnotationErrors.map((item, index) => <li key={index}>{item}</li>)}
        </ul>}
        {annotations.length > 0 && <fieldset className="recorded-training-set" disabled={props.disabled || Boolean(busy)}>
            <legend>Annotations for training ({queued.length} selected)</legend>
            {annotations.map((item) => <label key={item.id}>
                <input type="checkbox" checked={included.includes(item.id)} disabled={item.selection.ranges?.length === 0}
                    onChange={(event) => setIncluded(event.target.checked ? [...included, item.id] : included.filter((id) => id !== item.id))} />
                <span>{annotationLabel(item)}<small>{item.selection.videoFileName} · {item.selection.telemetryFileName}</small>
                    <small>{item.selection.ranges === null ? 'Full aligned recording'
                        : item.selection.ranges.length ? `${item.selection.ranges.length} selected ranges` : 'No footage selected'}</small></span>
            </label>)}
        </fieldset>}
    </div>;
    return <section className="driving-card recorded-frame-viewer" aria-label="Training annotations">
        <div className="driving-card-title"><h2>Training annotations</h2></div>
        <label className="recorded-setting">Working folder
            <input type="file" multiple disabled={locked}
                {...{ webkitdirectory: '' } as React.InputHTMLAttributes<HTMLInputElement>}
                onChange={(event) => {
                    const files = Array.from(event.target.files ?? []);
                    event.target.value = '';
                    void chooseWorkingFolder(files);
                }} />
        </label>
        {workingFolder && window.recordedTrainingRanges && <p className="driving-note" role="status">
            {workingFolder[0]?.webkitRelativePath && `${workingFolder[0].webkitRelativePath.split('/')[0]} · `}
            {folderPairCount} annotated recording {folderPairCount === 1 ? 'pair' : 'pairs'} found
        </p>}
        <div className="driving-buttons">
            <button aria-pressed={adding} disabled={props.disabled || Boolean(busy)} onClick={() => openEditor('add')}>Add new training annotation</button>
        </div>
        <label className="recorded-setting">Edit saved annotations
            <input type="file" accept=".json" multiple disabled={props.disabled || Boolean(busy) || !window.recordedTrainingRanges}
                onChange={(event) => {
                    const files = Array.from(event.target.files ?? []);
                    event.target.value = '';
                    void openAnnotations(files, 'edit');
                }} />
        </label>
        <p className="driving-note">{window.recordedTrainingRanges
            ? 'Choose annotation JSON files. Keep each annotation with its video and telemetry JSONL.'
            : 'Open and save training annotations in the desktop app.'}</p>
        {annotationErrors.length > 0 && <ul className="driving-error" role="alert">
            {annotationErrors.map((item, index) => <li key={index}>{item}</li>)}
        </ul>}
        {annotations.length > 0 && <div className="driving-buttons recorded-edit-buttons" role="group" aria-label="Edit loaded annotations">
            {annotations.map((item) => <button key={item.id} disabled={props.disabled || Boolean(busy)}
                aria-pressed={annotation?.id === item.id} onClick={() => openEditor({ annotationId: item.id })}>
                Edit {annotationLabel(item)}
            </button>)}
        </div>}
        {props.trainingPickerContainer ? createPortal(trainingPicker, props.trainingPickerContainer) : trainingPicker}
        <div hidden={!adding}>
            {folderVideoCount > 0 && !folderVideos.length && <p className="driving-note">
                All videos in the working folder already have training annotations.
            </p>}
            <div className="recorded-file-inputs">
                <label>Recorded video<input type="file" accept="video/*,.mkv,.avi" multiple disabled={locked || !adding}
                    onChange={(event) => addVideos(Array.from(event.target.files ?? []).filter(videoFile))} /></label>
                <label>Telemetry JSONL<input type="file" accept=".jsonl" multiple disabled={locked || !adding}
                    onChange={(event) => chooseTelemetry(event.target.files)} /></label>
            </div>
            {(folderVideos.length > 0 || videoFiles.length > 1) && <label className="recorded-setting">Video in folder<select value={selectedVideo ? videoFiles.indexOf(selectedVideo) : ''} disabled={locked}
                onChange={(event) => setSelectedVideo(event.target.value === '' ? null : videoFiles[Number(event.target.value)])}>
                <option value="">Select a video to review</option>{videoFiles.map((file, index) => <option key={index} value={index}>{fileLabel(file)}</option>)}</select></label>}
            {telemetryFiles.length > 0 && (videoFiles.length > 1 || telemetryFiles.length > 1) && <label className="recorded-setting">
                Telemetry in folder<select value={selectedTelemetry ? telemetryFiles.findIndex((file) => sameFile(file, selectedTelemetry)) : ''}
                    disabled={locked || !selectedVideo}
                    onChange={(event) => {
                        if (selectedVideo) setTelemetryOverrides((previous) => new Map(previous).set(selectedVideo,
                            event.target.value === '' ? null : telemetryFiles[Number(event.target.value)]));
                    }}>
                    <option value="">Select telemetry</option>{telemetryFiles.map((file, index) => <option key={index} value={index}>{fileLabel(file)}</option>)}
                </select>
            </label>}
        </div>
        {annotation && <dl className="recorded-annotation-details">
            <div><dt>Video</dt><dd>{annotation.selection.videoFileName}</dd></div>
            <div><dt>Telemetry</dt><dd>{annotation.selection.telemetryFileName}</dd></div>
        </dl>}
        {(adding || annotation) && <>
            {savedName && <p role="status">Saved {savedName} beside the video.</p>}
            <label className="recorded-setting">Game for JSONL without game metadata
                <select value={draft?.game ?? ''} disabled={locked || !draft || Boolean(selectedSettings?.error)} onChange={(event) => {
                    const game = event.target.value;
                    editSelection({ game: isDesktopGame(game) ? game : null });
                }}>
                    <option value="">Automatic</option>
                    <option value="acc">Assetto Corsa Competizione</option><option value="ac">Assetto Corsa</option><option value="iracing">iRacing</option>
                </select>
            </label>
        </>}
        {loadingTelemetry && <p role="status">Reading telemetry…</p>}
        {session && <p>{session.game} · {session.rows.length.toLocaleString()} samples · {session.durationSeconds.toFixed(2)} seconds</p>}
        {(error || (adding && (selectedSettings?.error || folderError))) && <p className="driving-error" role="alert">
            {error || selectedSettings?.error || folderError}</p>}
        {busy && <p role="status">{busy}{progress ? ` · ${progress}` : ''}</p>}
        {url && <>
            <video key={`video:${url}`} ref={video} className="recorded-video" crossOrigin={annotation ? 'anonymous' : undefined}
                src={url} controls={!locked} preload="auto" aria-label="Original recorded video"
                onLoadedMetadata={(event) => setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0)}
                onRateChange={(event) => {
                    const rate = event.currentTarget.playbackRate;
                    if (rate >= MIN_PLAYBACK_SPEED && rate <= MAX_PLAYBACK_SPEED && rate % PLAYBACK_SPEED_STEP === 0) setPlaybackSpeed(rate);
                }}
                onLoadedData={publishFrame} onTimeUpdate={publishFrame} onSeeked={publishFrame}
                onSeeking={() => {
                    presentedFrames.current?.clear();
                    if (!busyRef.current) current.current.onFrame(null);
                }}
                onError={() => { setError('The recorded video could not be decoded. Try an MP4 or WebM recording.'); current.current.onFrame(null); }} />
            <RecordedTimeline key={url} time={time} duration={duration} disabled={locked || duration <= 0.001} seek={seek}
                session={timingPending ? null : session}
                telemetryStart={telemetryStart} />
            <label className="recorded-playback-speed">
                <span>Playback speed <span aria-hidden="true">· {playbackSpeed}×</span></span>
                <input type="range" min={MIN_PLAYBACK_SPEED} max={MAX_PLAYBACK_SPEED} step={PLAYBACK_SPEED_STEP}
                    value={playbackSpeed} aria-valuetext={`${playbackSpeed} times normal speed`} disabled={locked}
                    onChange={(event) => setPlaybackSpeed(event.target.valueAsNumber)} /></label>
            {draft && !selectedSettings?.error && <RecordedTrainingRanges key={annotation?.id ?? JSON.stringify([url, selectedTelemetry && fileLabel(selectedTelemetry)])}
                selection={{ ...draft, offsetSeconds: previewOffset }}
                duration={duration} time={time} step={Math.max(0.1, props.configuration.runtime.controlIntervalMs / 1000)}
                overlap={session && !timingPending ? alignedRange(duration, session, telemetryStart) : null}
                gearShifts={gearShifts} shiftVideoOffset={shiftVideoOffset}
                disabled={locked || !duration} dirty={dirty} canSave={canSave}
                playhead={() => video.current?.currentTime ?? time} seek={seek}
                change={({ ranges, fullRecordingLabel }) => editSelection({ ranges, fullRecordingLabel })}
                changeOffset={session && !timingPending && !hasUsableVideoTiming(session, videoTiming)
                    ? (offsetSeconds) => editSelection({ offsetSeconds }) : undefined}
                save={() => { void saveSelection(); }} />}
            <h3>Camera model inspection</h3>
            <p>Inspect the loaded driving model’s camera CNN. Feature maps show learned activations; they are diagnostic views, not a recognition accuracy score. {props.policy ? `Model: ${props.policy.steps.toLocaleString()} training steps.` : 'Open a model with camera weights to inspect it.'}</p>
            <div className="recorded-settings">
                <label>Feature layer<select value={layerIndex} disabled={locked} onChange={(event) => { setLayerIndex(Number(event.target.value)); setChannel(0); }}>
                    {layers.map((item, index) => <option key={index} value={index}>Layer {index + 1} · {item.channels} channels</option>)}</select></label>
                <label>Feature channel<select value={selectedChannel} disabled={locked || !layer} onChange={(event) => setChannel(Number(event.target.value))}>
                    {Array.from({ length: layer?.channels ?? 1 }, (_, index) => <option key={index} value={index}>Channel {index + 1}</option>)}</select></label>
            </div>
            <div className="driving-buttons"><button disabled={locked || !props.policy || !duration} onClick={() => { void inspect(); }}>Inspect current frame</button></div>
            {inspection && layer && <>
                <div className="recorded-feature-images">
                    <figure><img src={inspectedImage} alt={`Original recorded frame at ${timestamp(inspectedTime)}`} /><figcaption>Original image · {timestamp(inspectedTime)}</figcaption></figure>
                    <figure><canvas ref={heatmap} aria-label={`Layer ${layerIndex + 1}, channel ${selectedChannel + 1} activation map`} /><figcaption>Layer {layerIndex + 1} / channel {selectedChannel + 1} · {layer.width} × {layer.height}</figcaption></figure>
                </div>
                <p className="driving-note">Mean {stats?.mean.toFixed(5)} · Peak {stats?.peak.toFixed(5)}. Orange: positive activation; blue: negative activation. Brightness is scaled within this channel.</p>
            </>}
            <h3>Strongest real examples per channel</h3>
            <p>Scan the video for the three frames with the highest positive peak activation in each channel of the selected layer. Rankings cover sampled frames only.</p>
            <label className="recorded-setting">Sample every (seconds)<input type="number" min="0.1" max="60" step="0.1" value={interval} disabled={locked}
                onChange={(event) => { const value = event.target.valueAsNumber; if (Number.isFinite(value)) setIntervalSeconds(Math.max(0.1, Math.min(60, value))); }} /></label>
            <div className="driving-buttons"><button disabled={locked || !props.policy || !duration} onClick={() => { void scan(); }}>Find strongest examples</button>
                {(busy === 'Inspecting frame' || busy === 'Scanning recording') && <button className="driving-stop"
                    onClick={() => operation.current?.abort()}>Cancel inspection</button>}</div>
            {!busy && progress && <p role="status">{progress}</p>}
            {gallery && <>
                <p>{gallery.complete ? 'Scan complete' : 'Partial scan'} · Layer {gallery.layer + 1} · {gallery.sampled} frames sampled every {gallery.interval} seconds.</p>
                <div className="recorded-channel-gallery">{gallery.examples.map((examples, index) => <section key={index} aria-label={`Strongest examples for channel ${index + 1}`}>
                    <h4>Channel {index + 1}</h4>
                    {!examples.length ? <p>No positive activation in sampled frames.</p> : <div className="recorded-example-row">{examples.map((example) => <button key={example.seconds} disabled={locked}
                        onClick={() => { setLayerIndex(gallery.layer); setChannel(index); void inspect(example.seconds); }}>
                        <img src={example.thumbnail} alt={`Real example for channel ${index + 1} at ${timestamp(example.seconds)}`} />
                        <span>{timestamp(example.seconds)} · {example.score.toFixed(5)}</span>
                    </button>)}</div>}
                </section>)}</div>
            </>}
        </>}
    </section>;
});
export default RecordedFrameViewer;
