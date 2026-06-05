import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Badge, Box, Button, Card, Flex, Select, Slider, Text } from '@radix-ui/themes';
import { PauseIcon, PlayIcon, ReloadIcon } from '@radix-ui/react-icons';
import apiService from 'services/api.service';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { AnalysisContext } from '../../analysis-context';
import { VisualizationProps } from '../VisualizationRegistry';
import {
    CarPoint,
    getPlaybackFrameIndex,
    normalizeTelemetryFrames,
    parseTelemetryFrame,
    segmentVisiblePoints,
    TelemetryFrame,
    Vec3,
    VisibilitySample
} from './mapTelemetry';
import './MapVisualization.css';

type LoadState = {
    status: 'idle' | 'loading' | 'ready' | 'empty' | 'error';
    message?: string;
};

type SegmentClassificationSegment = {
    id?: string;
    labels: string[];
    start_index: number;
    end_index: number;
};

type SegmentClassificationResult = {
    status: string;
    session_id: string;
    samples_analyzed: number;
    segment_count: number;
    segments: SegmentClassificationSegment[];
};

type SegmentOverlayRun = {
    labels: string[];
    color: string;
    points: Vec3[];
};

type ProjectedPoint = Vec3 & {
    screenX: number;
    screenY: number;
    depth: number;
};

const AXES = ['x', 'y', 'z'] as const;
type AxisName = typeof AXES[number];
type AxisFlipState = Record<AxisName, boolean>;
type CameraMode = 'driver' | 'fit';

const LIVE_TRAIL_LIMIT = 900;
const RECORDED_RENDER_FRAME_LIMIT = 900;
const RECORDED_TELEMETRY_TIMEOUT_MS = 120000;
const MAX_PLAYBACK_DELTA_SECONDS = 0.25;
const FIT_ZOOM = 1;
const DRIVER_FOCUS_ZOOM = 2.8;
const MIN_ZOOM = 0.35;
const MAX_ZOOM = 24;
const PLAYER_COLOR = '#00e676';
const OPPONENT_COLORS = ['#29b6f6', '#ffca28', '#ef5350', '#ab47bc', '#ff8a65', '#26c6da'];
const SEGMENT_OVERLAY_COLORS = ['#ffca28', '#29b6f6', '#ef5350', '#ab47bc', '#ff8a65', '#26c6da', '#ec407a', '#66bb6a'];

const getCarColor = (carKey: string, isPlayer: boolean): string => {
    if (isPlayer) return PLAYER_COLOR;

    let hash = 0;
    for (let index = 0; index < carKey.length; index += 1) {
        hash = ((hash << 5) - hash + carKey.charCodeAt(index)) | 0;
    }

    return OPPONENT_COLORS[Math.abs(hash) % OPPONENT_COLORS.length];
};

const getSegmentColor = (segment: SegmentClassificationSegment, index: number): string => {
    const key = segment.labels.join('|') || segment.id || String(index);
    let hash = index;

    for (let charIndex = 0; charIndex < key.length; charIndex += 1) {
        hash = ((hash << 5) - hash + key.charCodeAt(charIndex)) | 0;
    }

    return SEGMENT_OVERLAY_COLORS[Math.abs(hash) % SEGMENT_OVERLAY_COLORS.length];
};

const getLastFrameIndex = (frames: TelemetryFrame[]): number => Math.max(0, frames.length - 1);

const formatTime = (seconds: number): string => {
    if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0';
    const minutes = Math.floor(seconds / 60);
    const wholeSeconds = Math.floor(seconds % 60).toString().padStart(2, '0');
    const tenths = Math.floor((seconds % 1) * 10);
    return `${minutes}:${wholeSeconds}.${tenths}`;
};

const getTrackFrames = (points?: { position_x: number; position_y: number }[]): TelemetryFrame[] => {
    if (!points || points.length === 0) return [];

    return [{
        time: 0,
        playerKey: 'track',
        cars: points.map((point, index) => ({
            key: 'track',
            id: 'track',
            slot: index,
            position: {
                x: point.position_x,
                y: 0,
                z: point.position_y
            }
        }))
    }];
};

const getBounds = (frames: TelemetryFrame[], trackFrames: TelemetryFrame[]) => {
    const positions: Vec3[] = [];
    frames.forEach((frame) => frame.cars.forEach((car) => positions.push(car.position)));
    trackFrames.forEach((frame) => frame.cars.forEach((car) => positions.push(car.position)));

    if (positions.length === 0) {
        return {
            minX: -100,
            maxX: 100,
            minY: 0,
            maxY: 20,
            minZ: -100,
            maxZ: 100,
            center: { x: 0, y: 0, z: 0 },
            span: 200
        };
    }

    const first = positions[0];
    let minX = first.x;
    let maxX = first.x;
    let minY = first.y;
    let maxY = first.y;
    let minZ = first.z;
    let maxZ = first.z;

    positions.forEach((point) => {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
        minZ = Math.min(minZ, point.z);
        maxZ = Math.max(maxZ, point.z);
    });

    const span = Math.max(maxX - minX, maxZ - minZ, maxY - minY, 1);

    return {
        minX,
        maxX,
        minY,
        maxY,
        minZ,
        maxZ,
        center: {
            x: (minX + maxX) / 2,
            y: (minY + maxY) / 2,
            z: (minZ + maxZ) / 2
        },
        span
    };
};

const MapVisualization: React.FC<VisualizationProps> = ({ width = '100%', height = '100%' }) => {
    const analysisContext = useContext(AnalysisContext);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const wrapperRef = useRef<HTMLDivElement | null>(null);
    const recordedCacheRef = useRef<Map<string, TelemetryFrame[]>>(new Map());
    const segmentClassificationCacheRef = useRef<Map<string, SegmentClassificationResult>>(new Map());
    const currentPlaybackTimeRef = useRef(0);
    const playbackRef = useRef<{ animationId: number | null; lastTick: number | null; elapsed: number }>({
        animationId: null,
        lastTick: null,
        elapsed: 0
    });

    const [canvasSize, setCanvasSize] = useState({ width: 800, height: 520 });
    const [liveFrames, setLiveFrames] = useState<TelemetryFrame[]>([]);
    const [recordedFrames, setRecordedFrames] = useState<TelemetryFrame[]>([]);
    const [loadState, setLoadState] = useState<LoadState>({ status: 'idle' });
    const [playbackIndex, setPlaybackIndex] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [cameraMode, setCameraMode] = useState<CameraMode>('driver');
    const [zoom, setZoom] = useState(DRIVER_FOCUS_ZOOM);
    const [axisFlip, setAxisFlip] = useState<AxisFlipState>({ x: false, y: false, z: false });
    const [segmentClassification, setSegmentClassification] = useState<SegmentClassificationResult | null>(null);
    const [segmentLoadState, setSegmentLoadState] = useState<LoadState>({ status: 'idle' });

    const selectedSessionId = analysisContext.sessionSelected?.SessionId || '';
    const isRecordedMode = Boolean(selectedSessionId);
    const isLiveMode = !isRecordedMode;
    const trackFrames = useMemo(() => getTrackFrames(analysisContext.sessionSelected?.points), [analysisContext.sessionSelected?.points]);
    const frames = isRecordedMode ? recordedFrames : liveFrames;
    const currentFrame = isRecordedMode ? recordedFrames[playbackIndex] : liveFrames[liveFrames.length - 1];
    const renderStartIndex = isRecordedMode
        ? Math.max(0, playbackIndex - RECORDED_RENDER_FRAME_LIMIT + 1)
        : 0;
    const visibleFrames = useMemo(() => (
        isRecordedMode ? recordedFrames.slice(renderStartIndex, playbackIndex + 1) : liveFrames
    ), [isRecordedMode, liveFrames, playbackIndex, recordedFrames, renderStartIndex]);
    const bounds = useMemo(() => (
        getBounds(frames, trackFrames)
    ), [frames, trackFrames]);
    const duration = recordedFrames.length > 1
        ? Math.max(0, recordedFrames[recordedFrames.length - 1].time - recordedFrames[0].time)
        : 0;
    const currentPlaybackTime = currentFrame && recordedFrames.length > 0
        ? Math.max(0, currentFrame.time - recordedFrames[0].time)
        : 0;
    const currentPlayerKey = currentFrame?.playerKey || 'slot:0';
    const driverCameraTarget = useMemo(() => (
        currentFrame?.cars.find((car) => car.key === currentPlayerKey)?.position || null
    ), [currentFrame, currentPlayerKey]);
    const segmentOverlayRuns = useMemo<SegmentOverlayRun[]>(() => {
        if (!isRecordedMode || !segmentClassification?.segments?.length) {
            return [];
        }

        return segmentClassification.segments
            .map((segment, index) => {
                const points: Vec3[] = [];

                visibleFrames.forEach((frame) => {
                    const sourceIndex = frame.sourceIndex;
                    if (sourceIndex === undefined || sourceIndex < segment.start_index || sourceIndex >= segment.end_index) {
                        return;
                    }

                    const playerKey = frame.playerKey || 'slot:0';
                    const playerCar = frame.cars.find((car) => car.key === playerKey) || frame.cars[0];
                    if (playerCar) {
                        points.push(playerCar.position);
                    }
                });

                return {
                    labels: segment.labels,
                    color: getSegmentColor(segment, index),
                    points
                };
            })
            .filter((run) => run.points.length > 1);
    }, [isRecordedMode, segmentClassification, visibleFrames]);
    const activeSegmentLabels = useMemo(() => {
        if (!currentFrame || !segmentClassification?.segments?.length || currentFrame.sourceIndex === undefined) {
            return [];
        }

        const activeSegments = segmentClassification.segments.filter((segment) => (
            currentFrame.sourceIndex !== undefined
            && currentFrame.sourceIndex >= segment.start_index
            && currentFrame.sourceIndex < segment.end_index
        ));

        return Array.from(new Set(activeSegments.flatMap((segment) => segment.labels)));
    }, [currentFrame, segmentClassification]);

    useEffect(() => {
        currentPlaybackTimeRef.current = currentPlaybackTime;
        if (!isPlaying) {
            playbackRef.current.elapsed = currentPlaybackTime;
        }
    }, [currentPlaybackTime, isPlaying]);

    useEffect(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper) return;

        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            const nextWidth = Math.max(1, Math.floor(entry.contentRect.width));
            const nextHeight = Math.max(1, Math.floor(entry.contentRect.height));
            setCanvasSize({ width: nextWidth, height: nextHeight });
        });

        observer.observe(wrapper);
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        if (!isLiveMode || !analysisContext.liveData || typeof analysisContext.liveData !== 'object') {
            return;
        }

        if (analysisContext.TelemetryDataLiveStatus !== ACC_STATUS.ACC_LIVE) {
            return;
        }

        setLiveFrames((previous) => {
            const parsed = parseTelemetryFrame(analysisContext.liveData as Record<string, any>, previous.length);
            if (!parsed) return previous;

            const next = [...previous, parsed];
            return next.length > LIVE_TRAIL_LIMIT ? next.slice(next.length - LIVE_TRAIL_LIMIT) : next;
        });
    }, [analysisContext.TelemetryDataLiveStatus, analysisContext.liveData, isLiveMode]);

    useEffect(() => {
        if (!isRecordedMode) {
            setRecordedFrames([]);
            setPlaybackIndex(0);
            setIsPlaying(false);
            setLoadState({ status: 'idle' });
            return;
        }

        const cached = recordedCacheRef.current.get(selectedSessionId);
        if (cached) {
            setRecordedFrames(cached);
            setPlaybackIndex(getLastFrameIndex(cached));
            setLoadState(cached.length > 0 ? { status: 'ready' } : { status: 'empty', message: 'No telemetry rows were found for this session.' });
            return;
        }

        let cancelled = false;

        const loadRecordedTelemetry = async () => {
            setIsPlaying(false);
            setPlaybackIndex(0);
            setRecordedFrames([]);
            setLoadState({ status: 'loading', message: 'Loading recorded telemetry from backend...' });

            try {
                const initResponse = await apiService.post<any>('/racing-session/download/init', {
                    sessionId: selectedSessionId
                }, { timeout: RECORDED_TELEMETRY_TIMEOUT_MS });
                const initData = initResponse.data;
                const metadata = Array.isArray(initData?.sessionMetadata)
                    ? initData.sessionMetadata.find((session: any) => session.sessionId === selectedSessionId)
                    : null;

                if (!metadata) {
                    throw new Error('Selected session was not returned by the backend download initializer.');
                }

                const chunkCount = Math.max(1, Number(metadata.chunkCount) || 1);
                const rawFrames: TelemetryFrame[] = [];
                let rowOffset = 0;

                for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex += 1) {
                    const chunkResponse = await apiService.post<any>('/racing-session/download/chunk', {
                        downloadId: initData.downloadId,
                        sessionId: selectedSessionId,
                        trackName: analysisContext.mapSelected || metadata.map || '',
                        carName: metadata.car_name || analysisContext.sessionSelected?.car || '',
                        chunkIndex
                    }, { timeout: RECORDED_TELEMETRY_TIMEOUT_MS });

                    if (cancelled) return;

                    const body = chunkResponse.data;
                    const chunkRows: Record<string, any>[] = Array.isArray(body) ? body : Array.isArray(body?.data) ? body.data : [];
                    for (let rowIndex = 0; rowIndex < chunkRows.length; rowIndex += 1) {
                        const frame = parseTelemetryFrame(chunkRows[rowIndex], rowOffset + rowIndex);
                        if (frame) rawFrames.push(frame);
                    }
                    rowOffset += chunkRows.length;
                    setLoadState({
                        status: 'loading',
                        message: `Loading recorded telemetry ${chunkIndex + 1}/${chunkCount}...`
                    });
                }

                const parsed = normalizeTelemetryFrames(rawFrames);

                if (cancelled) return;

                recordedCacheRef.current.set(selectedSessionId, parsed);
                setRecordedFrames(parsed);
                setPlaybackIndex(getLastFrameIndex(parsed));
                setLoadState(parsed.length > 0
                    ? { status: 'ready' }
                    : { status: 'empty', message: 'No drawable trajectory data was found in this session.' });
            } catch (error: any) {
                if (cancelled) return;
                setLoadState({
                    status: 'error',
                    message: error?.message || 'Failed to load recorded telemetry.'
                });
            }
        };

        void loadRecordedTelemetry();

        return () => {
            cancelled = true;
        };
    }, [analysisContext.mapSelected, analysisContext.sessionSelected?.car, isRecordedMode, selectedSessionId]);

    useEffect(() => {
        if (!isRecordedMode || !selectedSessionId) {
            setSegmentClassification(null);
            setSegmentLoadState({ status: 'idle' });
            return;
        }

        const cached = segmentClassificationCacheRef.current.get(selectedSessionId);
        if (cached) {
            setSegmentClassification(cached);
            setSegmentLoadState(cached.segment_count > 0
                ? { status: 'ready' }
                : { status: 'empty', message: 'AI analysis found no classified segments.' });
        } else {
            setSegmentClassification(null);
            setSegmentLoadState({ status: 'idle' });
        }
    }, [isRecordedMode, selectedSessionId]);

    const handleRunSegmentClassification = useCallback(async () => {
        if (!isRecordedMode || !selectedSessionId || segmentLoadState.status === 'loading') {
            return;
        }

        const cached = segmentClassificationCacheRef.current.get(selectedSessionId);
        if (cached) {
            setSegmentClassification(cached);
            setSegmentLoadState(cached.segment_count > 0
                ? { status: 'ready' }
                : { status: 'empty', message: 'AI analysis found no classified segments.' });
            return;
        }

        setSegmentLoadState({ status: 'loading', message: 'Running AI segment analysis...' });
        setSegmentClassification(null);

        try {
            const response = await apiService.post<SegmentClassificationResult>('/racing-session/segment-classification', {
                session_id: selectedSessionId
            }, { timeout: RECORDED_TELEMETRY_TIMEOUT_MS });

            const result = response.data;
            const normalizedResult: SegmentClassificationResult = {
                ...result,
                segments: Array.isArray(result?.segments) ? result.segments : [],
                segment_count: Number(result?.segment_count) || 0,
                samples_analyzed: Number(result?.samples_analyzed) || 0,
                session_id: result?.session_id || selectedSessionId,
                status: result?.status || 'success'
            };

            segmentClassificationCacheRef.current.set(selectedSessionId, normalizedResult);
            setSegmentClassification(normalizedResult);
            setSegmentLoadState(normalizedResult.segment_count > 0
                ? { status: 'ready' }
                : { status: 'empty', message: 'AI analysis found no classified segments.' });
        } catch (error: any) {
            setSegmentLoadState({
                status: 'error',
                message: error?.data?.message || error?.message || 'Failed to run AI segment analysis.'
            });
        }
    }, [isRecordedMode, selectedSessionId, segmentLoadState.status]);

    useEffect(() => {
        const playbackState = playbackRef.current;

        if (playbackState.animationId !== null) {
            cancelAnimationFrame(playbackState.animationId);
            playbackState.animationId = null;
        }

        if (!isRecordedMode || !isPlaying || recordedFrames.length < 2) {
            playbackState.lastTick = null;
            playbackState.elapsed = currentPlaybackTimeRef.current;
            return;
        }

        let active = true;
        playbackState.lastTick = null;

        const tick = (now: number) => {
            if (!active) return;

            const state = playbackRef.current;
            const deltaSeconds = state.lastTick === null
                ? 0
                : Math.min((now - state.lastTick) / 1000, MAX_PLAYBACK_DELTA_SECONDS) * playbackSpeed;
            state.lastTick = now;
            state.elapsed += deltaSeconds;

            const nextIndex = getPlaybackFrameIndex(recordedFrames, state.elapsed);

            if (nextIndex === -1) {
                setPlaybackIndex(recordedFrames.length - 1);
                setIsPlaying(false);
                state.animationId = null;
                return;
            }

            setPlaybackIndex(nextIndex);
            state.animationId = requestAnimationFrame(tick);
        };

        playbackState.animationId = requestAnimationFrame(tick);

        return () => {
            active = false;
            if (playbackState.animationId !== null) {
                cancelAnimationFrame(playbackState.animationId);
                playbackState.animationId = null;
            }
        };
    }, [isPlaying, isRecordedMode, playbackSpeed, recordedFrames]);

    const projectPoint = useCallback((point: Vec3, size = canvasSize): ProjectedPoint => {
        const center = cameraMode === 'driver' && driverCameraTarget ? driverCameraTarget : bounds.center;
        const dx = (point.x - center.x) * (axisFlip.x ? -1 : 1);
        const dz = (point.z - center.z) * (axisFlip.z ? -1 : 1);
        const padding = Math.max(24, Math.min(size.width, size.height) * 0.06);
        const spanX = Math.max(bounds.maxX - bounds.minX, 1);
        const spanZ = Math.max(bounds.maxZ - bounds.minZ, 1);
        const scaleX = Math.max(1, size.width - padding * 2) / spanX;
        const scaleZ = Math.max(1, size.height - padding * 2) / spanZ;
        const scale = Math.min(scaleX, scaleZ) * zoom;

        return {
            ...point,
            screenX: size.width / 2 + dx * scale,
            screenY: size.height / 2 - dz * scale,
            depth: point.y * (axisFlip.y ? -1 : 1)
        };
    }, [axisFlip, bounds, cameraMode, canvasSize, driverCameraTarget, zoom]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const context = canvas.getContext('2d');
        if (!context) return;

        const ratio = window.devicePixelRatio || 1;
        canvas.width = canvasSize.width * ratio;
        canvas.height = canvasSize.height * ratio;
        canvas.style.width = `${canvasSize.width}px`;
        canvas.style.height = `${canvasSize.height}px`;
        context.setTransform(ratio, 0, 0, ratio, 0, 0);

        const widthPx = canvasSize.width;
        const heightPx = canvasSize.height;
        context.clearRect(0, 0, widthPx, heightPx);

        const background = context.createLinearGradient(0, 0, widthPx, heightPx);
        background.addColorStop(0, '#070b10');
        background.addColorStop(0.55, '#0d151b');
        background.addColorStop(1, '#050608');
        context.fillStyle = background;
        context.fillRect(0, 0, widthPx, heightPx);

        const drawLine = (points: Vec3[], color: string, width: number, alpha = 1, dash?: number[]) => {
            if (points.length < 2) return;
            context.save();
            context.globalAlpha = alpha;
            context.strokeStyle = color;
            context.lineWidth = width;
            context.lineCap = 'round';
            context.lineJoin = 'round';
            if (dash) context.setLineDash(dash);
            context.beginPath();
            points.forEach((point, index) => {
                const projected = projectPoint(point);
                if (index === 0) {
                    context.moveTo(projected.screenX, projected.screenY);
                } else {
                    context.lineTo(projected.screenX, projected.screenY);
                }
            });
            context.stroke();
            context.restore();
        };

        const drawSegments = (segments: Vec3[][], color: string, width: number, alpha = 1, dash?: number[]) => {
            segments.forEach((segment) => drawLine(segment, color, width, alpha, dash));
        };

        const drawGrid = () => {
            const margin = bounds.span * 0.16;
            const minX = bounds.minX - margin;
            const maxX = bounds.maxX + margin;
            const minZ = bounds.minZ - margin;
            const maxZ = bounds.maxZ + margin;
            const gridCount = 10;

            context.save();
            context.lineWidth = 1;
            context.strokeStyle = 'rgba(255, 255, 255, 0.08)';
            for (let index = 0; index <= gridCount; index += 1) {
                const x = minX + ((maxX - minX) * index) / gridCount;
                const z = minZ + ((maxZ - minZ) * index) / gridCount;
                drawLine([{ x, y: 0, z: minZ }, { x, y: 0, z: maxZ }], 'rgba(255,255,255,0.08)', 1);
                drawLine([{ x: minX, y: 0, z }, { x: maxX, y: 0, z }], 'rgba(255,255,255,0.08)', 1);
            }
            context.restore();
        };

        drawGrid();

        if (trackFrames.length > 0) {
            const trackSegments = segmentVisiblePoints(trackFrames[0].cars.map((car) => ({
                position: car.position,
                visible: true
            })));
            drawSegments(trackSegments, 'rgba(255,255,255,0.13)', 18, 0.8);
            drawSegments(trackSegments, 'rgba(0,0,0,0.5)', 12, 0.9);
            drawSegments(trackSegments, 'rgba(255,255,255,0.34)', 2, 0.8, [8, 16]);
        }

        const grouped = new Map<string, { car: CarPoint; samples: VisibilitySample[] }>();
        visibleFrames.forEach((frame) => {
            frame.cars.forEach((car) => {
                const existing = grouped.get(car.key);
                const sample = {
                    position: car.position,
                    visible: true
                };

                if (existing) {
                    existing.samples.push(sample);
                    existing.car = car;
                } else {
                    grouped.set(car.key, { car, samples: [sample] });
                }
            });
        });

        const playerKey = currentFrame?.playerKey || visibleFrames.find((frame) => frame.playerKey)?.playerKey || 'slot:0';
        const groupedEntries = Array.from(grouped.entries()).sort(([keyA], [keyB]) => {
            if (keyA === playerKey) return 1;
            if (keyB === playerKey) return -1;
            return keyA.localeCompare(keyB);
        });

        groupedEntries.forEach(([key, item]) => {
            const isPlayer = key === playerKey;
            const color = getCarColor(key, isPlayer);
            const segments = segmentVisiblePoints(item.samples);
            const shadowSegments = segments.map((segment) => segment.map((point) => ({ ...point, y: 0 })));
            const tailSegments = segmentVisiblePoints(item.samples.slice(-80));

            drawSegments(shadowSegments, 'rgba(0,0,0,0.4)', isPlayer ? 8 : 5, isPlayer ? 0.42 : 0.24);
            drawSegments(segments, color, isPlayer ? 5 : 3, isPlayer ? 0.95 : 0.7);
            drawSegments(tailSegments, '#ffffff', isPlayer ? 1.4 : 0.8, isPlayer ? 0.55 : 0.25);
        });

        segmentOverlayRuns.forEach((run) => {
            drawLine(run.points, 'rgba(0,0,0,0.66)', 12, 0.7);
            drawLine(run.points, run.color, 7, 0.9);
            drawLine(run.points, 'rgba(255,255,255,0.76)', 1.5, 0.72);
        });

        const currentCars = currentFrame?.cars || [];
        currentCars
            .map((car) => ({
                car,
                projected: projectPoint(car.position),
                isPlayer: car.key === (currentFrame?.playerKey || 'slot:0')
            }))
            .sort((a, b) => a.projected.depth - b.projected.depth)
            .forEach(({ car, projected, isPlayer }, index) => {
                const color = getCarColor(car.key, isPlayer);
                const markerSize = isPlayer ? 8 : 6;
                context.save();
                context.translate(projected.screenX, projected.screenY);
                context.fillStyle = color;
                context.strokeStyle = 'rgba(255,255,255,0.9)';
                context.lineWidth = 1.5;
                context.beginPath();
                context.arc(0, 0, markerSize * 0.72, 0, Math.PI * 2);
                context.fill();
                context.stroke();
                context.restore();

                if (isPlayer || index < 5) {
                    context.save();
                    context.font = '11px monospace';
                    context.fillStyle = 'rgba(235,255,245,0.86)';
                    context.fillText(isPlayer ? 'DRIVER' : `OPP ${car.slot}`, projected.screenX + 10, projected.screenY - 10);
                    context.restore();
                }
            });
    }, [bounds, canvasSize, currentFrame, projectPoint, segmentOverlayRuns, trackFrames, visibleFrames]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const handleWheel = (event: WheelEvent) => {
            event.preventDefault();
            setCameraMode('driver');
            setZoom((previousZoom) => {
                const nextZoom = event.deltaY > 0 ? previousZoom / 1.08 : previousZoom * 1.08;
                return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nextZoom));
            });
        };

        canvas.addEventListener('wheel', handleWheel, { passive: false });
        return () => canvas.removeEventListener('wheel', handleWheel);
    }, []);

    const resetPlayback = () => {
        setIsPlaying(false);
        setPlaybackIndex(0);
        playbackRef.current.elapsed = 0;
        playbackRef.current.lastTick = null;
    };

    const handleScrub = (value: number[]) => {
        const nextIndex = Math.max(0, Math.min(recordedFrames.length - 1, value[0] || 0));
        setPlaybackIndex(nextIndex);
        playbackRef.current.elapsed = recordedFrames[nextIndex]
            ? recordedFrames[nextIndex].time - recordedFrames[0].time
            : 0;
        playbackRef.current.lastTick = null;
    };

    const togglePlayback = () => {
        if (recordedFrames.length < 2) return;
        if (playbackIndex >= recordedFrames.length - 1) {
            setPlaybackIndex(0);
            playbackRef.current.elapsed = 0;
        } else {
            playbackRef.current.elapsed = currentPlaybackTime;
        }
        playbackRef.current.lastTick = null;
        setIsPlaying((previous) => !previous);
    };

    const toggleAxisFlip = (axis: AxisName) => {
        setAxisFlip((previous) => ({
            ...previous,
            [axis]: !previous[axis]
        }));
    };

    const focusDriver = () => {
        setCameraMode('driver');
        setZoom(DRIVER_FOCUS_ZOOM);
    };

    const fitTrack = () => {
        setCameraMode('fit');
        setZoom(FIT_ZOOM);
    };

    return (
        <Card className="map-visualization-card" style={{ width, height }}>
            <Box ref={wrapperRef} className="map-visualization">
                <canvas
                    ref={canvasRef}
                    className="map-visualization__canvas"
                />

                <div className="map-visualization__hud map-visualization__hud--top">
                    <Flex align="center" gap="2" wrap="wrap">
                        <Badge color={isRecordedMode ? 'blue' : 'green'} variant="soft">
                            {isRecordedMode ? 'Recorded Telemetry' : 'Live Telemetry'}
                        </Badge>
                        <Text size="1" className="map-visualization__metric">
                            {frames.length.toLocaleString()} samples
                        </Text>
                        <Text size="1" className="map-visualization__metric">
                            {Math.max(0, (currentFrame?.cars.length || 1) - 1)} opponents
                        </Text>
                        {isRecordedMode && (
                            <Button
                                size="1"
                                variant="soft"
                                onClick={handleRunSegmentClassification}
                                disabled={!selectedSessionId || segmentLoadState.status === 'loading'}
                            >
                                {segmentLoadState.status === 'loading' ? 'Analyzing...' : 'Run AI Analysis'}
                            </Button>
                        )}
                    </Flex>
                </div>

                {isRecordedMode && segmentLoadState.status !== 'idle' && (
                    <div className="map-visualization__hud map-visualization__hud--segments">
                        <Flex direction="column" gap="2">
                            <Flex align="center" gap="2" wrap="wrap">
                                <Badge
                                    color={segmentLoadState.status === 'error' ? 'red' : segmentLoadState.status === 'empty' ? 'gray' : 'amber'}
                                    variant="soft"
                                >
                                    {segmentLoadState.status === 'loading'
                                        ? 'AI analyzing'
                                        : segmentLoadState.status === 'error'
                                            ? 'AI analysis failed'
                                            : `${segmentClassification?.segment_count ?? 0} AI segments`}
                                </Badge>
                                {segmentLoadState.status === 'ready' && activeSegmentLabels.length > 0 && (
                                    <Text size="1" className="map-visualization__metric">
                                        Active: {activeSegmentLabels.join(', ')}
                                    </Text>
                                )}
                            </Flex>
                            {segmentLoadState.message && (
                                <Text size="1" className="map-visualization__segment-message">
                                    {segmentLoadState.message}
                                </Text>
                            )}
                            {segmentLoadState.status === 'ready' && segmentClassification?.segments?.length ? (
                                <Flex gap="2" wrap="wrap" className="map-visualization__segment-legend">
                                    {segmentClassification.segments.slice(0, 6).map((segment, index) => (
                                        <span key={segment.id || `${segment.start_index}-${segment.end_index}`} className="map-visualization__segment-legend-item">
                                            <span
                                                className="map-visualization__segment-swatch"
                                                style={{ backgroundColor: getSegmentColor(segment, index) }}
                                            />
                                            {segment.labels.join(', ') || 'Unlabeled'}
                                        </span>
                                    ))}
                                </Flex>
                            ) : null}
                        </Flex>
                    </div>
                )}

                <div className="map-visualization__hud map-visualization__hud--camera">
                    <Flex align="center" gap="2" justify="end" wrap="wrap">
                        <Flex align="center" gap="1" className="map-visualization__axis-flips">
                            {AXES.map((axis) => (
                                <Button
                                    key={axis}
                                    size="1"
                                    variant={axisFlip[axis] ? 'solid' : 'soft'}
                                    color={axisFlip[axis] ? 'orange' : undefined}
                                    aria-pressed={axisFlip[axis]}
                                    aria-label={`Flip ${axis.toUpperCase()} axis`}
                                    className="map-visualization__axis-button"
                                    onClick={() => toggleAxisFlip(axis)}
                                >
                                    {axis.toUpperCase()}
                                </Button>
                            ))}
                        </Flex>
                        <Button
                            size="1"
                            variant={cameraMode === 'driver' ? 'solid' : 'soft'}
                            onClick={focusDriver}
                        >
                            Driver
                        </Button>
                        <Button size="1" variant={cameraMode === 'fit' ? 'solid' : 'soft'} onClick={fitTrack}>
                            Fit
                        </Button>
                    </Flex>
                </div>

                {isRecordedMode && (
                    <div className="map-visualization__player">
                        <Flex align="center" gap="2" className="map-visualization__player-row">
                            <Button size="2" variant="soft" onClick={togglePlayback} disabled={recordedFrames.length < 2}>
                                {isPlaying ? <PauseIcon /> : <PlayIcon />}
                            </Button>
                            <Button size="2" variant="ghost" onClick={resetPlayback} disabled={recordedFrames.length < 2}>
                                <ReloadIcon />
                            </Button>
                            <Text size="1" className="map-visualization__time">
                                {formatTime(currentPlaybackTime)} / {formatTime(duration)}
                            </Text>
                            <Select.Root value={String(playbackSpeed)} onValueChange={(value) => setPlaybackSpeed(Number(value))}>
                                <Select.Trigger className="map-visualization__speed" />
                                <Select.Content>
                                    <Select.Item value="0.5">0.5x</Select.Item>
                                    <Select.Item value="1">1x</Select.Item>
                                    <Select.Item value="2">2x</Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </Flex>
                        <Slider
                            value={[playbackIndex]}
                            min={0}
                            max={Math.max(0, recordedFrames.length - 1)}
                            step={1}
                            disabled={recordedFrames.length < 2}
                            onValueChange={handleScrub}
                        />
                    </div>
                )}

                {isRecordedMode && loadState.status !== 'ready' && (
                    <div className="map-visualization__state">
                        <Text size="2" weight="bold">
                            {loadState.status === 'loading' ? 'Loading telemetry' : loadState.status === 'error' ? 'Telemetry unavailable' : 'No telemetry'}
                        </Text>
                        <Text size="1">{loadState.message || 'Select a recorded backend session to replay telemetry.'}</Text>
                    </div>
                )}

                {isLiveMode && liveFrames.length === 0 && (
                    <div className="map-visualization__state">
                        <Text size="2" weight="bold">Waiting for live telemetry</Text>
                        <Text size="1">Start recording a live ACC session to draw driver and opponent trajectories.</Text>
                    </div>
                )}
            </Box>
        </Card>
    );
};

export default MapVisualization;
