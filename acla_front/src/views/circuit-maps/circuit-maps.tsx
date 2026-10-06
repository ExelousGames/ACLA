import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertDialog, Badge, Box, Button, Flex, Heading, Select, Spinner, Tabs, Text, TextField } from '@radix-ui/themes';
import { CheckIcon, Cross2Icon, PauseIcon, PlayIcon, PlusIcon, ReloadIcon, TrashIcon } from '@radix-ui/react-icons';
import apiService from 'services/api.service';
import { fetchCircuitMapById, fetchCircuitMapList, normalizeCircuitMap } from 'services/circuitMapService';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
import { readLocalTelemetry } from 'views/session-shared/read-local-telemetry';
import {
    OPERATION_COMPONENT_NAMES,
    useOptionalOperationComponentSnapshot,
} from 'contexts/OperationComponentRefContext';
import type { LiveSessionRuntime } from 'views/live-session/live-session-types';
import {
    liveTelemetryStore,
    useCurrentTelemetry,
    useLiveTelemetrySelector,
    useTelemetryStatus,
} from 'views/live-session/live-telemetry-store';
import {
    CIRCUIT_MAP_CAPTURE_MODES,
    CIRCUIT_MAP_GAMES,
    CircuitMapBinSample,
    CircuitMapCaptureMode,
    CircuitMapGame,
    CircuitMapSamplesByMode,
    CircuitMapSummaryDto
} from './circuit-map-types';
import {
    CIRCUIT_MAP_BIN_RESOLUTION,
    cloneSamplesByMode,
    countCircuitMapSamples,
    extractCircuitMapCaptureSample,
    getCircuitMapBin,
    getCircuitMapDrawSegments,
    getCircuitMapName,
    getCircuitMapTrackKey,
    mergeCircuitMapSamples,
    upsertCaptureModeSample
} from './circuit-map-utils';
import './circuit-maps.css';

type LoadState = 'idle' | 'loading' | 'ready' | 'error';
type CircuitMapView = 'bounded' | 'centerline';
type SelectedPoint = { mode: CircuitMapCaptureMode; bin: number } | null;
type ProjectedPoint = { screenX: number; screenY: number; sample: CircuitMapBinSample; mode: CircuitMapCaptureMode };

const VIEW_CAPTURE_MODES = {
    bounded: CIRCUIT_MAP_CAPTURE_MODES.filter(({ value }) => value !== 'middle_line'),
    centerline: CIRCUIT_MAP_CAPTURE_MODES.filter(({ value }) => value === 'middle_line')
};

const MODE_COLORS: Record<CircuitMapCaptureMode, string> = {
    left_boundary: '#29b6f6',
    middle_line: '#ce93d8',
    right_boundary: '#ffca28',
    pit_lane: '#66bb6a'
};

const EMPTY_SAMPLES: CircuitMapSamplesByMode = {
    left_boundary: [],
    middle_line: [],
    right_boundary: [],
    pit_lane: []
};

const formatModeLabel = (mode: CircuitMapCaptureMode): string => (
    CIRCUIT_MAP_CAPTURE_MODES.find((option) => option.value === mode)?.label || mode
);

const toNumber = (value: string, fallback: number): number => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

const getSamplesForMode = (samplesByMode: CircuitMapSamplesByMode, mode: CircuitMapCaptureMode): CircuitMapBinSample[] => (
    samplesByMode[mode] || []
);

const CircuitMaps = () => {
    const liveSession = useOptionalOperationComponentSnapshot<LiveSessionRuntime>(
        OPERATION_COMPONENT_NAMES.LIVE_SESSION,
    );
    const currentTelemetry = useCurrentTelemetry();
    const telemetryGame = useLiveTelemetrySelector((snapshot) => snapshot.game);
    const telemetryStatus = useTelemetryStatus();
    const { refreshCircuitMaps, upsertCachedCircuitMap, removeCachedCircuitMap } = useCircuitMaps();
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const canvasWrapRef = useRef<HTMLDivElement | null>(null);
    const projectedPointsRef = useRef<ProjectedPoint[]>([]);
    const lastCaptureSignatureRef = useRef('');
    const mapLoadRequestRef = useRef(0);
    const listLoadRequestRef = useRef(0);
    const importRef = useRef<AbortController | null>(null);

    const [game, setGame] = useState<CircuitMapGame>('acc');
    const [mapList, setMapList] = useState<CircuitMapSummaryDto[]>([]);
    const [listState, setListState] = useState<LoadState>('idle');
    const [error, setError] = useState<string | null>(null);
    const [selectedMapId, setSelectedMapId] = useState<string | null>(null);
    const [circuitName, setCircuitName] = useState('');
    const [sourceTrackKey, setSourceTrackKey] = useState<string | null>(null);
    const [samplesByMode, setSamplesByMode] = useState<CircuitMapSamplesByMode>(EMPTY_SAMPLES);
    const [mapView, setMapView] = useState<CircuitMapView>('bounded');
    const [captureMode, setCaptureMode] = useState<CircuitMapCaptureMode>('left_boundary');
    const [isCapturing, setIsCapturing] = useState(false);
    const [selectedPoint, setSelectedPoint] = useState<SelectedPoint>(null);
    const [canvasSize, setCanvasSize] = useState({ width: 900, height: 620 });
    const [manualNormalized, setManualNormalized] = useState('0');
    const [manualX, setManualX] = useState('0');
    const [manualZ, setManualZ] = useState('0');
    const [isSaving, setIsSaving] = useState(false);
    const [isMapLoading, setIsMapLoading] = useState(false);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [deleteError, setDeleteError] = useState<string | null>(null);
    const [isImporting, setIsImporting] = useState(false);
    const [importStatus, setImportStatus] = useState('');

    const isAcc = game === 'acc';
    const isIRacing = game === 'iracing';
    const canImportIRacing = Boolean(window.electronAPI?.importLocalIRacingTelemetry);
    const isAccLive = isAcc && telemetryGame === game && telemetryStatus === ACC_STATUS.ACC_LIVE;
    const sampleCount = countCircuitMapSamples(samplesByMode);
    const visibleModes = VIEW_CAPTURE_MODES[mapView];
    const visibleSampleCount = visibleModes.reduce((total, { value }) => total + getSamplesForMode(samplesByMode, value).length, 0);
    const currentTrackKey = (telemetryGame === game ? getCircuitMapTrackKey(currentTelemetry) : null)
        || (liveSession?.sessionGame === game ? getCircuitMapTrackKey(liveSession.staticData) : null);
    const liveCapture = useMemo(() => (
        isAccLive ? extractCircuitMapCaptureSample(currentTelemetry) : null
    ), [currentTelemetry, isAccLive]);

    const loadMapList = useCallback(async (nextGame: CircuitMapGame = game) => {
        const requestId = ++listLoadRequestRef.current;
        setListState('loading');
        setError(null);

        try {
            const maps = await fetchCircuitMapList(nextGame);
            if (requestId !== listLoadRequestRef.current) return;
            setMapList(maps);
            setListState('ready');
        } catch (loadError: any) {
            if (requestId !== listLoadRequestRef.current) return;
            setMapList([]);
            setListState('error');
            setError(loadError?.data?.message || loadError?.message || 'Unable to load circuit maps.');
        }
    }, [game]);

    const cancelImport = useCallback(() => {
        importRef.current?.abort();
        importRef.current = null;
        setIsImporting(false);
        setImportStatus('');
    }, []);

    useEffect(() => () => { importRef.current?.abort(); }, []);

    const changeMapView = (value: string) => {
        if (isImporting) return;
        const nextView = value as CircuitMapView;
        setIsCapturing(false);
        lastCaptureSignatureRef.current = '';
        setSelectedPoint(null);
        setImportStatus('');
        setMapView(nextView);
        setCaptureMode(nextView === 'centerline' ? 'middle_line' : 'left_boundary');
    };

    useEffect(() => {
        cancelImport();
        mapLoadRequestRef.current += 1;
        setIsMapLoading(false);
        setSelectedMapId(null);
        setCircuitName('');
        setSourceTrackKey(null);
        setSamplesByMode(cloneSamplesByMode(EMPTY_SAMPLES));
        setSelectedPoint(null);
        setIsCapturing(false);
        void loadMapList(game);
    }, [cancelImport, game, loadMapList]);

    useEffect(() => {
        if (!isAcc || selectedMapId || circuitName) {
            return;
        }

        const name = getCircuitMapName(currentTrackKey, game);
        if (name) {
            setCircuitName(name);
            setSourceTrackKey(currentTrackKey);
        }
    }, [circuitName, currentTrackKey, game, isAcc, selectedMapId]);

    useEffect(() => {
        const wrapper = canvasWrapRef.current;
        if (!wrapper) return;

        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            setCanvasSize({
                width: Math.max(1, Math.floor(entry.contentRect.width)),
                height: Math.max(1, Math.floor(entry.contentRect.height))
            });
        });

        observer.observe(wrapper);
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        return liveTelemetryStore.subscribeEvents((event) => {
            if (event.type === 'session-reset') {
                lastCaptureSignatureRef.current = '';
                return;
            }
            if (
                event.type !== 'frame'
                || !isCapturing
                || !isAcc
                || event.update.game !== game
                || event.telemetryStatus !== ACC_STATUS.ACC_LIVE
            ) return;

            const liveCapture = extractCircuitMapCaptureSample(event.sample);
            if (!liveCapture) return;
            const signature = `${liveCapture.bin}:${liveCapture.position.x}:${liveCapture.position.y}:${liveCapture.position.z}`;
            if (signature === lastCaptureSignatureRef.current) return;

            lastCaptureSignatureRef.current = signature;
            setSamplesByMode((previous) => upsertCaptureModeSample(previous, captureMode, liveCapture));
        }, { replayLatest: true });
    }, [captureMode, game, isAcc, isCapturing]);

    const loadMap = useCallback(async (mapId: string) => {
        cancelImport();
        const requestId = ++mapLoadRequestRef.current;
        setIsMapLoading(true);
        setSelectedMapId(mapId);
        setError(null);
        setIsCapturing(false);
        setSelectedPoint(null);

        try {
            const map = await fetchCircuitMapById(mapId, game);
            if (requestId !== mapLoadRequestRef.current) return;
            setCircuitName(map.circuit_name);
            setSourceTrackKey(map.source_track_key || null);
            setSamplesByMode(cloneSamplesByMode(map.samples));
            upsertCachedCircuitMap(map);
        } catch (loadError: any) {
            if (requestId !== mapLoadRequestRef.current) return;
            setError(loadError?.data?.message || loadError?.message || 'Unable to load circuit map.');
        } finally {
            if (requestId === mapLoadRequestRef.current) setIsMapLoading(false);
        }
    }, [cancelImport, game, upsertCachedCircuitMap]);

    const resetForNewMap = useCallback(() => {
        cancelImport();
        mapLoadRequestRef.current += 1;
        setIsMapLoading(false);
        setIsCapturing(false);
        setSelectedMapId(null);
        setSamplesByMode(cloneSamplesByMode(EMPTY_SAMPLES));
        setSelectedPoint(null);
        if (isAcc) {
            setCircuitName(getCircuitMapName(currentTrackKey, game));
            setSourceTrackKey(currentTrackKey);
        } else {
            setCircuitName('');
            setSourceTrackKey(null);
        }
    }, [cancelImport, currentTrackKey, game, isAcc]);

    const importIRacingFile = async () => {
        if (importRef.current || !isIRacing || isMapLoading || isSaving || isDeleting) return;
        const controller = new AbortController();
        importRef.current = controller;
        setIsImporting(true);
        setError(null);
        setImportStatus('Opening and converting iRacing telemetry...');
        setSelectedPoint(null);
        let convertedPath: string | undefined;
        try {
            const imported = await window.electronAPI.importLocalIRacingTelemetry();
            if (!imported) {
                if (!controller.signal.aborted) setImportStatus('');
                return;
            }
            convertedPath = imported.filePath;
            if (controller.signal.aborted) return;
            let importedTrackKey: string | null = null;
            let nextSamples = getSamplesForMode(samplesByMode, captureMode);
            let capturedRows = 0;
            const updatedAt = new Date().toISOString();
            await readLocalTelemetry(imported.filePath, game, controller.signal, (count) => {
                setImportStatus(`Reading telemetry: ${count.toLocaleString()} / ${imported.rowCount.toLocaleString()}`);
            }, (rows) => {
                for (const row of rows) {
                    const trackKey = getCircuitMapTrackKey(row);
                    if (!trackKey) continue;
                    if ((importedTrackKey && trackKey !== importedTrackKey)
                        || (sampleCount > 0 && sourceTrackKey && trackKey !== sourceTrackKey)) {
                        throw new Error('This recording is from a different circuit. Create a New Map to import it.');
                    }
                    importedTrackKey = trackKey;
                }
                const result = mergeCircuitMapSamples(nextSamples, rows, updatedAt);
                nextSamples = result.samples;
                capturedRows += result.capturedRows;
            });
            if (controller.signal.aborted) return;
            if (!capturedRows) {
                throw new Error('This .ibt file contains no usable driver coordinates and lap positions. Try a completed recording with GPS telemetry.');
            }
            if (!importedTrackKey) {
                throw new Error('This recording contains no track name in its telemetry.');
            }
            setSamplesByMode((previous) => ({ ...previous, [captureMode]: nextSamples }));
            setCircuitName((previous) => previous || getCircuitMapName(importedTrackKey, game));
            setSourceTrackKey(importedTrackKey);
            setImportStatus(`${imported.fileName}: imported ${capturedRows.toLocaleString()} driver samples into ${formatModeLabel(captureMode)}.`);
        } catch (cause) {
            if (!controller.signal.aborted) {
                setError(cause instanceof Error ? cause.message : 'Could not import the .ibt file.');
                setImportStatus('');
            }
        } finally {
            if (convertedPath) await window.electronAPI.deleteTempFile(convertedPath).catch(() => undefined);
            if (importRef.current === controller) {
                importRef.current = null;
                if (!controller.signal.aborted) setIsImporting(false);
            }
        }
    };

    const deleteMap = async () => {
        if (!selectedMapId || isDeleting || isSaving || isMapLoading || listState === 'loading') return;
        setIsDeleting(true);
        setDeleteError(null);
        try {
            try {
                await apiService.delete(`/circuit-map/${encodeURIComponent(selectedMapId)}`);
            } catch (deleteError: any) {
                if (deleteError?.status !== 404 || deleteError?.data?.message !== 'Circuit map not found') {
                    throw deleteError;
                }
            }
            removeCachedCircuitMap(selectedMapId);
            setMapList((previous) => previous.filter((map) => map.id !== selectedMapId));
            resetForNewMap();
            setError(null);
            setDeleteDialogOpen(false);
        } catch {
            setDeleteError('Could not remove this map. Please try again.');
        } finally {
            setIsDeleting(false);
        }
    };

    const saveMap = useCallback(async () => {
        const trimmedName = circuitName.trim();
        if (!trimmedName) {
            setError('Circuit name is required.');
            return;
        }

        const payload = {
            game,
            circuit_name: trimmedName,
            source_track_key: sourceTrackKey,
            resolution: CIRCUIT_MAP_BIN_RESOLUTION,
            samples: samplesByMode
        };

        setIsSaving(true);
        setError(null);

        try {
            let savedMapId = selectedMapId;
            if (selectedMapId) {
                await apiService.put(`/circuit-map/${encodeURIComponent(selectedMapId)}`, payload);
            } else {
                const response = await apiService.post<any>('/circuit-map', payload);
                const nextId = String(response.data?.id ?? response.data?.map_id ?? '');
                if (nextId) {
                    savedMapId = nextId;
                    setSelectedMapId(nextId);
                }
            }

            if (savedMapId) {
                upsertCachedCircuitMap(normalizeCircuitMap({
                    id: savedMapId,
                    ...payload,
                    sample_count: sampleCount
                }, game));
            }

            await refreshCircuitMaps(game);
            await loadMapList(game);
        } catch (saveError: any) {
            setError(saveError?.data?.message || saveError?.message || 'Unable to save circuit map.');
        } finally {
            setIsSaving(false);
        }
    }, [
        circuitName,
        game,
        loadMapList,
        refreshCircuitMaps,
        sampleCount,
        samplesByMode,
        selectedMapId,
        sourceTrackKey,
        upsertCachedCircuitMap
    ]);

    const setSelectedSample = useCallback((updater: (sample: CircuitMapBinSample) => CircuitMapBinSample | null) => {
        if (!selectedPoint) return;

        setSamplesByMode((previous) => {
            const samples = getSamplesForMode(previous, selectedPoint.mode);
            const index = samples.findIndex((sample) => sample.bin === selectedPoint.bin);
            if (index < 0) return previous;

            const nextSample = updater(samples[index]);
            const nextSamples = nextSample
                ? [...samples.slice(0, index), nextSample, ...samples.slice(index + 1)]
                : [...samples.slice(0, index), ...samples.slice(index + 1)];

            return {
                ...previous,
                [selectedPoint.mode]: nextSamples
            };
        });

        if (!updater) {
            setSelectedPoint(null);
        }
    }, [selectedPoint]);

    const deleteSelectedPoint = useCallback(() => {
        if (!selectedPoint) return;

        setSamplesByMode((previous) => ({
            ...previous,
            [selectedPoint.mode]: getSamplesForMode(previous, selectedPoint.mode).filter((sample) => sample.bin !== selectedPoint.bin)
        }));
        setSelectedPoint(null);
    }, [selectedPoint]);

    const addManualPoint = useCallback(() => {
        const normalizedPosition = Number(manualNormalized);
        const bin = getCircuitMapBin(normalizedPosition);
        if (!manualNormalized.trim() || bin === null) {
            setError('Normalized position is required and must be a number from 0 to 1.');
            return;
        }
        setError(null);
        const sample: CircuitMapBinSample = {
            bin,
            normalized_position: normalizedPosition,
            x: toNumber(manualX, 0),
            y: 0,
            z: toNumber(manualZ, 0),
            sample_count: 1,
            updated_at: new Date().toISOString(),
            locked: true
        };

        setSamplesByMode((previous) => {
            const samples = getSamplesForMode(previous, captureMode).filter((item) => item.bin !== bin);
            return {
                ...previous,
                [captureMode]: [...samples, sample].sort((a, b) => a.bin - b.bin)
            };
        });
        setSelectedPoint({ mode: captureMode, bin });
    }, [captureMode, manualNormalized, manualX, manualZ]);

    const getCanvasProjection = useCallback(() => {
        const points: { x: number; z: number }[] = [];
        visibleModes.forEach(({ value }) => {
            getSamplesForMode(samplesByMode, value).forEach((sample) => points.push({ x: sample.x, z: sample.z }));
        });
        if (liveCapture) {
            points.push({ x: liveCapture.position.x, z: liveCapture.position.z });
        }

        if (points.length === 0) {
            points.push({ x: -100, z: -100 }, { x: 100, z: 100 });
        }

        let minX = points[0].x;
        let maxX = points[0].x;
        let minZ = points[0].z;
        let maxZ = points[0].z;

        points.forEach((point) => {
            minX = Math.min(minX, point.x);
            maxX = Math.max(maxX, point.x);
            minZ = Math.min(minZ, point.z);
            maxZ = Math.max(maxZ, point.z);
        });

        const padding = 42;
        const spanX = Math.max(1, maxX - minX);
        const spanZ = Math.max(1, maxZ - minZ);
        const usableWidth = Math.max(1, canvasSize.width - padding * 2);
        const usableHeight = Math.max(1, canvasSize.height - padding * 2);
        const scale = Math.min(usableWidth / spanX, usableHeight / spanZ);
        const centerX = (minX + maxX) / 2;
        const centerZ = (minZ + maxZ) / 2;
        // Canvas Y grows downward: keep that Z flip for ACC and undo it for iRacing.
        const zDirection = isIRacing ? -1 : 1;

        return {
            project: (x: number, z: number) => ({
                screenX: canvasSize.width / 2 + (x - centerX) * scale,
                screenY: canvasSize.height / 2 + (z - centerZ) * scale * zDirection
            })
        };
    }, [canvasSize, isIRacing, liveCapture, samplesByMode, visibleModes]);

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
        context.clearRect(0, 0, canvasSize.width, canvasSize.height);

        const { project } = getCanvasProjection();
        const projectedPoints: ProjectedPoint[] = [];

        context.fillStyle = '#070b10';
        context.fillRect(0, 0, canvasSize.width, canvasSize.height);

        context.save();
        context.strokeStyle = 'rgba(255,255,255,0.07)';
        context.lineWidth = 1;
        for (let index = 0; index <= 8; index += 1) {
            const x = (canvasSize.width * index) / 8;
            const y = (canvasSize.height * index) / 8;
            context.beginPath();
            context.moveTo(x, 0);
            context.lineTo(x, canvasSize.height);
            context.moveTo(0, y);
            context.lineTo(canvasSize.width, y);
            context.stroke();
        }
        context.restore();

        visibleModes.forEach(({ value }) => {
            const samples = getSamplesForMode(samplesByMode, value);
            const drawSegments = getCircuitMapDrawSegments(samples, value);

            drawSegments.forEach((segment) => {
                if (segment.length < 2) {
                    return;
                }

                context.save();
                context.strokeStyle = MODE_COLORS[value];
                context.lineWidth = 3;
                context.globalAlpha = 0.82;
                context.beginPath();
                segment.forEach((sample, index) => {
                    const point = project(sample.x, sample.z);
                    if (index === 0) context.moveTo(point.screenX, point.screenY);
                    else context.lineTo(point.screenX, point.screenY);
                });
                context.stroke();
                context.restore();
            });

            samples.forEach((sample) => {
                const point = project(sample.x, sample.z);
                projectedPoints.push({ ...point, sample, mode: value });
                const isSelected = selectedPoint?.mode === value && selectedPoint.bin === sample.bin;

                context.save();
                context.fillStyle = MODE_COLORS[value];
                context.strokeStyle = isSelected ? '#ffffff' : sample.locked ? 'rgba(255,255,255,0.72)' : 'rgba(0,0,0,0.7)';
                context.lineWidth = isSelected ? 3 : 1.5;
                context.beginPath();
                context.arc(point.screenX, point.screenY, isSelected ? 6 : 4, 0, Math.PI * 2);
                context.fill();
                context.stroke();
                context.restore();
            });
        });

        if (mapView === 'centerline') {
            const startPoint = projectedPoints.reduce<ProjectedPoint | null>((closest, point) => (
                !closest || point.sample.normalized_position < closest.sample.normalized_position ? point : closest
            ), null);

            if (startPoint) {
                context.save();
                context.strokeStyle = '#00e676';
                context.lineWidth = 3;
                context.beginPath();
                context.arc(startPoint.screenX, startPoint.screenY, 10, 0, Math.PI * 2);
                context.stroke();
                context.fillStyle = '#00e676';
                context.font = 'bold 12px monospace';
                context.textAlign = 'center';
                context.fillText('START', startPoint.screenX, startPoint.screenY - 18);
                context.restore();
            }
        }

        if (liveCapture) {
            const point = project(liveCapture.position.x, liveCapture.position.z);
            context.save();
            context.fillStyle = '#ffffff';
            context.strokeStyle = '#00e676';
            context.lineWidth = 3;
            context.beginPath();
            context.arc(point.screenX, point.screenY, 7, 0, Math.PI * 2);
            context.fill();
            context.stroke();
            context.restore();
        }

        if (visibleSampleCount === 0 && !liveCapture) {
            context.save();
            context.fillStyle = 'rgba(235,255,245,0.74)';
            context.font = '12px monospace';
            context.textAlign = 'center';
            context.fillText(mapView === 'centerline' ? 'NO CENTERLINE SAMPLES' : 'NO BOUNDED MAP SAMPLES', canvasSize.width / 2, canvasSize.height / 2);
            context.restore();
        }

        projectedPointsRef.current = projectedPoints;
    }, [canvasSize, getCanvasProjection, liveCapture, mapView, visibleSampleCount, samplesByMode, selectedPoint, visibleModes]);

    const getPointerPosition = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        return {
            screenX: event.clientX - rect.left,
            screenY: event.clientY - rect.top
        };
    }, []);

    const handlePointerDown = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
        if (isImporting) return;
        const pointer = getPointerPosition(event);
        const nearest = projectedPointsRef.current.reduce<{ point: ProjectedPoint | null; distance: number }>((closest, point) => {
            const distance = Math.hypot(point.screenX - pointer.screenX, point.screenY - pointer.screenY);
            if (distance < closest.distance) {
                return { point, distance };
            }
            return closest;
        }, { point: null, distance: 12 }).point;

        if (!nearest) {
            setSelectedPoint(null);
            return;
        }

        setSelectedPoint({ mode: nearest.mode, bin: nearest.sample.bin });
    }, [getPointerPosition, isImporting]);

    const selectedSample = useMemo(() => {
        if (!selectedPoint) return null;
        return getSamplesForMode(samplesByMode, selectedPoint.mode).find((sample) => sample.bin === selectedPoint.bin) || null;
    }, [samplesByMode, selectedPoint]);

    const captureButton = isCapturing ? (
        <Button color="amber" variant="soft" onClick={() => setIsCapturing(false)}>
            <PauseIcon />
            Pause Capture
        </Button>
    ) : (
        <Button color="green" disabled={!isAccLive || !liveCapture} onClick={() => setIsCapturing(true)}>
            <PlayIcon />
            Start Capture
        </Button>
    );

    return (
        <Tabs.Root className="circuit-maps" value={mapView} onValueChange={changeMapView}>
            <aside className="circuit-maps__sidebar">
                <div className="circuit-maps__section">
                    <Heading size="5">Circuit Maps</Heading>
                    <Text size="2" className="circuit-maps__muted">Global map builder</Text>
                </div>

                <div className="circuit-maps__section">
                    <Text className="circuit-maps__label">Game</Text>
                    <Select.Root value={game} onValueChange={(value) => setGame(value as CircuitMapGame)}>
                        <Select.Trigger />
                        <Select.Content>
                            {CIRCUIT_MAP_GAMES.map((option) => (
                                <Select.Item key={option.value} value={option.value}>{option.label}</Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                </div>

                <div className="circuit-maps__section">
                    <Flex align="center" justify="between" gap="2">
                        <Text className="circuit-maps__label">Circuit</Text>
                        <Button size="1" variant="soft" onClick={() => void loadMapList(game)}>
                            <ReloadIcon />
                            Refresh
                        </Button>
                    </Flex>

                    <TextField.Root
                        placeholder="Circuit name"
                        value={circuitName}
                        onChange={(event) => setCircuitName(event.target.value)}
                    />

                    <Button variant="outline" onClick={resetForNewMap}>
                        <PlusIcon />
                        New Map
                    </Button>

                    <div className="circuit-maps__map-list">
                        {listState === 'loading' ? (
                            <Flex align="center" gap="2"><Spinner size="1" /><Text size="2">Loading maps</Text></Flex>
                        ) : mapList.length === 0 ? (
                            <Text size="2" className="circuit-maps__muted">No global maps found.</Text>
                        ) : mapList.map((map) => (
                            <button
                                key={map.id}
                                type="button"
                                className={`circuit-maps__map-button${selectedMapId === map.id ? ' circuit-maps__map-button--active' : ''}`}
                                onClick={() => void loadMap(map.id)}
                            >
                                <span className="circuit-maps__map-name">{map.circuit_name}</span>
                                <Badge color={map.game === 'acc' ? 'green' : 'gray'}>{map.game.toUpperCase()}</Badge>
                            </button>
                        ))}
                    </div>
                </div>

                <div className="circuit-maps__section">
                    <Text className="circuit-maps__label">Capture Mode</Text>
                    <Select.Root value={captureMode} disabled={isImporting} onValueChange={(value) => setCaptureMode(value as CircuitMapCaptureMode)}>
                        <Select.Trigger />
                        <Select.Content>
                            {visibleModes.map((option) => (
                                <Select.Item key={option.value} value={option.value}>{option.label}</Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>

                    <Text size="2" className="circuit-maps__muted">
                        {mapView === 'centerline'
                            ? 'Middle line only. Each point requires a normalized lap position from 0 to 1.'
                            : 'Left boundary, right boundary, and pit lane.'}
                    </Text>

                    {isAcc ? (
                        <Flex align="center" gap="2" wrap="wrap">
                            {captureButton}
                            <Badge color={isAccLive ? 'green' : 'gray'}>{isAccLive ? 'ACC Live' : 'ACC Offline'}</Badge>
                        </Flex>
                    ) : (
                        <Flex direction="column" gap="2">
                            <Text size="2" className="circuit-maps__muted">
                                Import your recorded driving path into {formatModeLabel(captureMode)}. Exit the car in iRacing first to finish recording.
                            </Text>
                            <Button onClick={() => void importIRacingFile()} disabled={!canImportIRacing || isImporting || isMapLoading || isSaving || isDeleting}>
                                {isImporting ? <Spinner size="1" /> : <PlusIcon />}
                                {isImporting ? 'Importing...' : 'Open .ibt file'}
                            </Button>
                            {!canImportIRacing && <Text size="2">Open the desktop app to import local iRacing .ibt files.</Text>}
                            {importStatus && <Text role="status" size="2">{importStatus}</Text>}
                        </Flex>
                    )}
                </div>

                <div className="circuit-maps__section">
                    <Text className="circuit-maps__label">Manual Point</Text>
                    <div className="circuit-maps__manual-grid">
                        <TextField.Root
                            aria-label="Normalized position 0-1"
                            placeholder="Normalized position 0-1"
                            type="number"
                            min="0"
                            max="1"
                            step="any"
                            required
                            value={manualNormalized}
                            onChange={(event) => setManualNormalized(event.target.value)}
                        />
                        <TextField.Root placeholder="X" value={manualX} onChange={(event) => setManualX(event.target.value)} />
                        <TextField.Root placeholder="Z" value={manualZ} onChange={(event) => setManualZ(event.target.value)} />
                    </div>
                    <Button variant="soft" onClick={addManualPoint} disabled={isImporting}>
                        <PlusIcon />
                        Add Point
                    </Button>
                </div>

                <div className="circuit-maps__section">
                    <Text className="circuit-maps__label">Samples</Text>
                    <div className="circuit-maps__mode-grid">
                        {visibleModes.map((mode) => (
                            <div key={mode.value} className="circuit-maps__mode-row">
                                <Text size="2">
                                    <span
                                        className="circuit-maps__swatch"
                                        style={{ background: MODE_COLORS[mode.value] }}
                                    />
                                    {mode.label}
                                </Text>
                                <Text size="2" className="circuit-maps__muted">{getSamplesForMode(samplesByMode, mode.value).length}</Text>
                            </div>
                        ))}
                    </div>
                </div>

                {error ? (
                    <div className="circuit-maps__section">
                        <Text role="alert" size="2" className="circuit-maps__error">{error}</Text>
                    </div>
                ) : null}
            </aside>

            <main className="circuit-maps__stage">
                <div className="circuit-maps__toolbar">
                    <div className="circuit-maps__toolbar-main">
                        <Text size="2" className="circuit-maps__title">{circuitName || 'Unsaved Circuit Map'}</Text>
                        <Text size="1" className="circuit-maps__muted">
                            {visibleSampleCount.toLocaleString()} samples
                            {sourceTrackKey ? ` / ${sourceTrackKey}` : ''}
                        </Text>
                    </div>

                    <div className="circuit-maps__controls">
                        {isAcc ? (
                            <div className={`circuit-maps__status${isCapturing ? ' circuit-maps__status--capture' : isAccLive ? ' circuit-maps__status--live' : ''}`}>
                                <span className="circuit-maps__status-dot" />
                                <Text size="1">
                                    {isCapturing
                                        ? `Capturing ${formatModeLabel(captureMode)}`
                                        : !isAccLive
                                            ? 'Waiting for telemetry'
                                            : liveCapture ? 'Live telemetry' : 'Waiting for coordinates and normalized position'}
                                </Text>
                            </div>
                        ) : null}
                        <AlertDialog.Root open={deleteDialogOpen} onOpenChange={(open) => {
                            if (isDeleting) return;
                            setDeleteDialogOpen(open);
                            setDeleteError(null);
                        }}>
                            <AlertDialog.Trigger>
                                <Button color="red" variant="soft" disabled={!selectedMapId || isMapLoading || listState === 'loading' || isSaving || isDeleting || isImporting}>
                                    <TrashIcon />
                                    Remove Map
                                </Button>
                            </AlertDialog.Trigger>
                            <AlertDialog.Content maxWidth="450px" onEscapeKeyDown={(event) => {
                                if (isDeleting) event.preventDefault();
                            }}>
                                <AlertDialog.Title>Remove circuit map?</AlertDialog.Title>
                                <AlertDialog.Description size="2">
                                    Permanently remove “{mapList.find((map) => map.id === selectedMapId)?.circuit_name || circuitName}” and all samples in its bounded and centerline maps?
                                    {' '}This global map will be removed for everyone. This cannot be undone.
                                </AlertDialog.Description>
                                {deleteError && (
                                    <Box mt="3">
                                        <Text role="alert" size="2" color="red">{deleteError}</Text>
                                    </Box>
                                )}
                                <Flex gap="3" mt="4" justify="end">
                                    <AlertDialog.Cancel>
                                        <Button variant="soft" color="gray" disabled={isDeleting}>Cancel</Button>
                                    </AlertDialog.Cancel>
                                    <Button color="red" disabled={isDeleting} onClick={() => void deleteMap()}>
                                        {isDeleting ? 'Removing...' : 'Remove map'}
                                    </Button>
                                </Flex>
                            </AlertDialog.Content>
                        </AlertDialog.Root>
                        <Button onClick={() => void saveMap()} disabled={isSaving || isMapLoading || isDeleting || isImporting || !circuitName.trim()}>
                            {isSaving ? <Spinner size="1" /> : <CheckIcon />}
                            Save
                        </Button>
                    </div>
                </div>

                <Tabs.List className="circuit-maps__tabs" aria-label="Circuit map type">
                    <Tabs.Trigger value="bounded" disabled={isImporting}>Bounded Map</Tabs.Trigger>
                    <Tabs.Trigger value="centerline" disabled={isImporting}>Centerline Map</Tabs.Trigger>
                </Tabs.List>

                <Tabs.Content value={mapView} ref={canvasWrapRef} className="circuit-maps__canvas-wrap">
                    <canvas
                        ref={canvasRef}
                        className="circuit-maps__canvas"
                        aria-label={mapView === 'centerline' ? 'Centerline map' : 'Bounded map'}
                        onPointerDown={handlePointerDown}
                    />

                    {selectedSample && selectedPoint ? (
                        <div className="circuit-maps__selection">
                            <Badge color="green">{formatModeLabel(selectedPoint.mode)}</Badge>
                            <Text size="1">Bin {selectedSample.bin}</Text>
                            {mapView === 'centerline' && <Text size="1">Normalized position {selectedSample.normalized_position}</Text>}
                            <Text size="1">Samples {selectedSample.sample_count}</Text>
                            <Button size="1" variant="soft" onClick={() => setSelectedSample((sample) => ({ ...sample, locked: !sample.locked }))}>
                                {selectedSample.locked ? <Cross2Icon /> : <CheckIcon />}
                                {selectedSample.locked ? 'Unlock' : 'Lock'}
                            </Button>
                            <Button size="1" color="red" variant="soft" onClick={deleteSelectedPoint}>
                                <TrashIcon />
                                Delete
                            </Button>
                        </div>
                    ) : null}
                </Tabs.Content>
            </main>
        </Tabs.Root>
    );
};

export default CircuitMaps;
