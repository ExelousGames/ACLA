import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Badge, Box, Button, Flex, Heading, Select, Spinner, Text, TextField } from '@radix-ui/themes';
import { CheckIcon, Cross2Icon, PauseIcon, PlayIcon, PlusIcon, ReloadIcon, TrashIcon } from '@radix-ui/react-icons';
import apiService from 'services/api.service';
import { ACC_STATUS, ACCMemoeryTracks } from 'data/live-analysis/live-map-data';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import {
    CIRCUIT_MAP_CAPTURE_MODES,
    CIRCUIT_MAP_GAMES,
    CircuitMapBinSample,
    CircuitMapCaptureMode,
    CircuitMapDto,
    CircuitMapGame,
    CircuitMapSamplesByMode,
    CircuitMapSummaryDto
} from './circuit-map-types';
import {
    CIRCUIT_MAP_BIN_RESOLUTION,
    cloneSamplesByMode,
    countCircuitMapSamples,
    extractAccCaptureSample,
    upsertCaptureModeSample
} from './circuit-map-utils';
import './circuit-maps.css';

type LoadState = 'idle' | 'loading' | 'ready' | 'error';
type SelectedPoint = { mode: CircuitMapCaptureMode; bin: number } | null;
type DragState = { mode: CircuitMapCaptureMode; bin: number } | null;
type ProjectedPoint = { screenX: number; screenY: number; sample: CircuitMapBinSample; mode: CircuitMapCaptureMode };

const MODE_COLORS: Record<CircuitMapCaptureMode, string> = {
    left_boundary: '#29b6f6',
    right_boundary: '#ffca28',
    pit_lane: '#66bb6a'
};

const EMPTY_SAMPLES: CircuitMapSamplesByMode = {
    left_boundary: [],
    right_boundary: [],
    pit_lane: []
};

const normalizeMapList = (data: any): CircuitMapSummaryDto[] => {
    const rows = Array.isArray(data) ? data : Array.isArray(data?.list) ? data.list : [];
    return rows
        .map((row: any): CircuitMapSummaryDto | null => {
            const id = String(row.id ?? row.map_id ?? row.MapId ?? '');
            const circuitName = String(row.circuit_name ?? row.name ?? row.map_name ?? '');
            const game = row.game === 'other' ? 'other' : 'acc';
            if (!id || !circuitName) return null;

            return {
                id,
                game,
                circuit_name: circuitName,
                source_track_key: row.source_track_key ?? null,
                updated_at: row.updated_at ?? null,
                sample_count: Number(row.sample_count ?? 0)
            };
        })
        .filter((row: CircuitMapSummaryDto | null): row is CircuitMapSummaryDto => row !== null);
};

const normalizeMap = (data: any, fallbackGame: CircuitMapGame): CircuitMapDto => {
    const rawSamples = data?.samples || {};
    return {
        id: String(data?.id ?? data?.map_id ?? ''),
        game: data?.game === 'other' ? 'other' : fallbackGame,
        circuit_name: String(data?.circuit_name ?? data?.name ?? ''),
        source_track_key: data?.source_track_key ?? null,
        updated_at: data?.updated_at ?? null,
        sample_count: Number(data?.sample_count ?? countCircuitMapSamples(rawSamples)),
        resolution: Number(data?.resolution ?? CIRCUIT_MAP_BIN_RESOLUTION),
        samples: {
            left_boundary: Array.isArray(rawSamples.left_boundary) ? rawSamples.left_boundary : [],
            right_boundary: Array.isArray(rawSamples.right_boundary) ? rawSamples.right_boundary : [],
            pit_lane: Array.isArray(rawSamples.pit_lane) ? rawSamples.pit_lane : []
        }
    };
};

const getAccTrackKey = (liveData: any, staticData: any): string | null => (
    liveData?.Static_track
    || liveData?.Static?.track
    || staticData?.track
    || null
);

const getAccCircuitName = (trackKey: string | null): string => {
    if (!trackKey) return '';
    return ACCMemoeryTracks.get(trackKey) || trackKey;
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
    const analysisContext = useContext(AnalysisContext);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const canvasWrapRef = useRef<HTMLDivElement | null>(null);
    const projectedPointsRef = useRef<ProjectedPoint[]>([]);
    const dragStateRef = useRef<DragState>(null);
    const liveSequenceRef = useRef(0);
    const lastCaptureSignatureRef = useRef('');

    const [game, setGame] = useState<CircuitMapGame>('acc');
    const [mapList, setMapList] = useState<CircuitMapSummaryDto[]>([]);
    const [listState, setListState] = useState<LoadState>('idle');
    const [error, setError] = useState<string | null>(null);
    const [selectedMapId, setSelectedMapId] = useState<string | null>(null);
    const [circuitName, setCircuitName] = useState('');
    const [sourceTrackKey, setSourceTrackKey] = useState<string | null>(null);
    const [samplesByMode, setSamplesByMode] = useState<CircuitMapSamplesByMode>(EMPTY_SAMPLES);
    const [captureMode, setCaptureMode] = useState<CircuitMapCaptureMode>('left_boundary');
    const [isCapturing, setIsCapturing] = useState(false);
    const [selectedPoint, setSelectedPoint] = useState<SelectedPoint>(null);
    const [canvasSize, setCanvasSize] = useState({ width: 900, height: 620 });
    const [manualNormalized, setManualNormalized] = useState('0');
    const [manualX, setManualX] = useState('0');
    const [manualZ, setManualZ] = useState('0');
    const [isSaving, setIsSaving] = useState(false);

    const isAcc = game === 'acc';
    const isAccLive = isAcc && analysisContext.TelemetryDataLiveStatus === ACC_STATUS.ACC_LIVE;
    const sampleCount = countCircuitMapSamples(samplesByMode);
    const currentAccTrackKey = getAccTrackKey(analysisContext.liveData, analysisContext.recordedSessioStaticsData);
    const liveCapture = useMemo(() => (
        isAccLive && analysisContext.liveData && typeof analysisContext.liveData === 'object'
            ? extractAccCaptureSample(analysisContext.liveData as Record<string, any>, liveSequenceRef.current)
            : null
    ), [analysisContext.liveData, isAccLive]);

    const loadMapList = useCallback(async (nextGame: CircuitMapGame = game) => {
        setListState('loading');
        setError(null);

        try {
            const response = await apiService.get<any>('/circuit-map/list', { game: nextGame });
            setMapList(normalizeMapList(response.data));
            setListState('ready');
        } catch (loadError: any) {
            setMapList([]);
            setListState('error');
            setError(loadError?.data?.message || loadError?.message || 'Unable to load circuit maps.');
        }
    }, [game]);

    useEffect(() => {
        setSelectedMapId(null);
        setCircuitName('');
        setSourceTrackKey(null);
        setSamplesByMode(cloneSamplesByMode(EMPTY_SAMPLES));
        setSelectedPoint(null);
        setIsCapturing(false);
        void loadMapList(game);
    }, [game, loadMapList]);

    useEffect(() => {
        if (!isAcc || selectedMapId || circuitName) {
            return;
        }

        const accName = getAccCircuitName(currentAccTrackKey);
        if (accName) {
            setCircuitName(accName);
            setSourceTrackKey(currentAccTrackKey);
        }
    }, [circuitName, currentAccTrackKey, isAcc, selectedMapId]);

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
        if (!isCapturing || !isAccLive || !liveCapture) {
            return;
        }

        const signature = `${liveCapture.bin}:${liveCapture.position.x}:${liveCapture.position.y}:${liveCapture.position.z}`;
        if (signature === lastCaptureSignatureRef.current) {
            return;
        }

        lastCaptureSignatureRef.current = signature;
        liveSequenceRef.current += 1;
        setSamplesByMode((previous) => upsertCaptureModeSample(previous, captureMode, liveCapture));
    }, [captureMode, isAccLive, isCapturing, liveCapture]);

    const loadMap = useCallback(async (mapId: string) => {
        setSelectedMapId(mapId);
        setError(null);
        setIsCapturing(false);
        setSelectedPoint(null);

        try {
            const response = await apiService.get<any>(`/circuit-map/${encodeURIComponent(mapId)}`);
            const map = normalizeMap(response.data, game);
            setCircuitName(map.circuit_name);
            setSourceTrackKey(map.source_track_key || null);
            setSamplesByMode(cloneSamplesByMode(map.samples));
        } catch (loadError: any) {
            setError(loadError?.data?.message || loadError?.message || 'Unable to load circuit map.');
        }
    }, [game]);

    const resetForNewMap = useCallback(() => {
        setSelectedMapId(null);
        setSamplesByMode(cloneSamplesByMode(EMPTY_SAMPLES));
        setSelectedPoint(null);
        if (isAcc) {
            const accName = getAccCircuitName(currentAccTrackKey);
            setCircuitName(accName);
            setSourceTrackKey(currentAccTrackKey);
        } else {
            setCircuitName('');
            setSourceTrackKey(null);
        }
    }, [currentAccTrackKey, isAcc]);

    const saveMap = useCallback(async () => {
        const trimmedName = circuitName.trim();
        if (!trimmedName) {
            setError('Circuit name is required.');
            return;
        }

        const payload = {
            game,
            circuit_name: trimmedName,
            source_track_key: isAcc ? sourceTrackKey : null,
            resolution: CIRCUIT_MAP_BIN_RESOLUTION,
            samples: samplesByMode
        };

        setIsSaving(true);
        setError(null);

        try {
            if (selectedMapId) {
                await apiService.put(`/circuit-map/${encodeURIComponent(selectedMapId)}`, payload);
            } else {
                const response = await apiService.post<any>('/circuit-map', payload);
                const nextId = String(response.data?.id ?? response.data?.map_id ?? '');
                if (nextId) {
                    setSelectedMapId(nextId);
                }
            }
            await loadMapList(game);
        } catch (saveError: any) {
            setError(saveError?.data?.message || saveError?.message || 'Unable to save circuit map.');
        } finally {
            setIsSaving(false);
        }
    }, [circuitName, game, isAcc, loadMapList, samplesByMode, selectedMapId, sourceTrackKey]);

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
        const normalizedPosition = Math.min(1, Math.max(0, toNumber(manualNormalized, 0)));
        const bin = Math.min(CIRCUIT_MAP_BIN_RESOLUTION - 1, Math.floor(normalizedPosition * CIRCUIT_MAP_BIN_RESOLUTION));
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
        Object.values(samplesByMode).forEach((samples) => {
            samples?.forEach((sample) => points.push({ x: sample.x, z: sample.z }));
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

        return {
            project: (x: number, z: number) => ({
                screenX: canvasSize.width / 2 + (x - centerX) * scale,
                screenY: canvasSize.height / 2 + (z - centerZ) * scale
            }),
            unproject: (screenX: number, screenY: number) => ({
                x: centerX + (screenX - canvasSize.width / 2) / scale,
                z: centerZ + (screenY - canvasSize.height / 2) / scale
            })
        };
    }, [canvasSize, liveCapture, samplesByMode]);

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

        CIRCUIT_MAP_CAPTURE_MODES.forEach(({ value }) => {
            const samples = getSamplesForMode(samplesByMode, value);
            if (samples.length > 1) {
                context.save();
                context.strokeStyle = MODE_COLORS[value];
                context.lineWidth = 3;
                context.globalAlpha = 0.82;
                context.beginPath();
                samples.forEach((sample, index) => {
                    const point = project(sample.x, sample.z);
                    if (index === 0) context.moveTo(point.screenX, point.screenY);
                    else context.lineTo(point.screenX, point.screenY);
                });
                context.stroke();
                context.restore();
            }

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

        if (sampleCount === 0 && !liveCapture) {
            context.save();
            context.fillStyle = 'rgba(235,255,245,0.74)';
            context.font = '12px monospace';
            context.textAlign = 'center';
            context.fillText('NO CIRCUIT SAMPLES', canvasSize.width / 2, canvasSize.height / 2);
            context.restore();
        }

        projectedPointsRef.current = projectedPoints;
    }, [canvasSize, getCanvasProjection, liveCapture, sampleCount, samplesByMode, selectedPoint]);

    const getPointerPosition = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        return {
            screenX: event.clientX - rect.left,
            screenY: event.clientY - rect.top
        };
    }, []);

    const handlePointerDown = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
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

        const nextSelection = { mode: nearest.mode, bin: nearest.sample.bin };
        setSelectedPoint(nextSelection);
        dragStateRef.current = nextSelection;
        event.currentTarget.setPointerCapture(event.pointerId);
    }, [getPointerPosition]);

    const handlePointerMove = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
        const dragState = dragStateRef.current;
        if (!dragState) return;

        const pointer = getPointerPosition(event);
        const { unproject } = getCanvasProjection();
        const nextPosition = unproject(pointer.screenX, pointer.screenY);

        setSamplesByMode((previous) => {
            const samples = getSamplesForMode(previous, dragState.mode);
            return {
                ...previous,
                [dragState.mode]: samples.map((sample) => sample.bin === dragState.bin
                    ? {
                        ...sample,
                        x: nextPosition.x,
                        z: nextPosition.z,
                        locked: true,
                        updated_at: new Date().toISOString()
                    }
                    : sample)
            };
        });
    }, [getCanvasProjection, getPointerPosition]);

    const handlePointerUp = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
        dragStateRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
    }, []);

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
        <Button color="green" disabled={!isAccLive} onClick={() => setIsCapturing(true)}>
            <PlayIcon />
            Start Capture
        </Button>
    );

    return (
        <div className="circuit-maps">
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
                    <Select.Root value={captureMode} onValueChange={(value) => setCaptureMode(value as CircuitMapCaptureMode)}>
                        <Select.Trigger />
                        <Select.Content>
                            {CIRCUIT_MAP_CAPTURE_MODES.map((option) => (
                                <Select.Item key={option.value} value={option.value}>{option.label}</Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>

                    {isAcc ? (
                        <Flex align="center" gap="2" wrap="wrap">
                            {captureButton}
                            <Badge color={isAccLive ? 'green' : 'gray'}>{isAccLive ? 'ACC Live' : 'ACC Offline'}</Badge>
                        </Flex>
                    ) : (
                        <Badge color="gray">Manual Edit</Badge>
                    )}
                </div>

                <div className="circuit-maps__section">
                    <Text className="circuit-maps__label">Manual Point</Text>
                    <div className="circuit-maps__manual-grid">
                        <TextField.Root placeholder="Normalized position 0-1" value={manualNormalized} onChange={(event) => setManualNormalized(event.target.value)} />
                        <TextField.Root placeholder="X" value={manualX} onChange={(event) => setManualX(event.target.value)} />
                        <TextField.Root placeholder="Z" value={manualZ} onChange={(event) => setManualZ(event.target.value)} />
                    </div>
                    <Button variant="soft" onClick={addManualPoint}>
                        <PlusIcon />
                        Add Point
                    </Button>
                </div>

                <div className="circuit-maps__section">
                    <Text className="circuit-maps__label">Samples</Text>
                    <div className="circuit-maps__mode-grid">
                        {CIRCUIT_MAP_CAPTURE_MODES.map((mode) => (
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
                        <Text size="2" className="circuit-maps__error">{error}</Text>
                    </div>
                ) : null}
            </aside>

            <main className="circuit-maps__stage">
                <div className="circuit-maps__toolbar">
                    <div className="circuit-maps__toolbar-main">
                        <Text size="2" className="circuit-maps__title">{circuitName || 'Unsaved Circuit Map'}</Text>
                        <Text size="1" className="circuit-maps__muted">
                            {sampleCount.toLocaleString()} samples
                            {isAcc && sourceTrackKey ? ` / ${sourceTrackKey}` : ''}
                        </Text>
                    </div>

                    <div className="circuit-maps__controls">
                        {isAcc ? (
                            <div className={`circuit-maps__status${isCapturing ? ' circuit-maps__status--capture' : isAccLive ? ' circuit-maps__status--live' : ''}`}>
                                <span className="circuit-maps__status-dot" />
                                <Text size="1">
                                    {isCapturing ? `Capturing ${formatModeLabel(captureMode)}` : isAccLive ? 'Live telemetry' : 'Waiting for telemetry'}
                                </Text>
                            </div>
                        ) : null}
                        <Button onClick={() => void saveMap()} disabled={isSaving || !circuitName.trim()}>
                            {isSaving ? <Spinner size="1" /> : <CheckIcon />}
                            Save
                        </Button>
                    </div>
                </div>

                <Box ref={canvasWrapRef} className="circuit-maps__canvas-wrap">
                    <canvas
                        ref={canvasRef}
                        className="circuit-maps__canvas"
                        onPointerDown={handlePointerDown}
                        onPointerMove={handlePointerMove}
                        onPointerUp={handlePointerUp}
                        onPointerCancel={handlePointerUp}
                    />

                    {selectedSample && selectedPoint ? (
                        <div className="circuit-maps__selection">
                            <Badge color="green">{formatModeLabel(selectedPoint.mode)}</Badge>
                            <Text size="1">Bin {selectedSample.bin}</Text>
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
                </Box>
            </main>
        </div>
    );
};

export default CircuitMaps;
