import React, { forwardRef, useCallback, useContext, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { Badge, Box, Button, Card, Flex, Text } from '@radix-ui/themes';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
import { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import { Vec3 } from 'views/session-shared/visualization/charts/mapTelemetry';
import { LiveSessionContext } from './LiveSessionContext';
import { useLiveTelemetrySelector } from './live-telemetry-store';
import { getLiveMapCars, getLiveMapMiddleLine } from './live-map-data';
import 'views/session-shared/visualization/charts/MapVisualization.css';
import { NamedOperationComponentHandle, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';

const PLAYER_COLOR = '#00e676';
const OPPONENT_COLORS = ['#29b6f6', '#ffca28', '#ef5350', '#ab47bc', '#ff8a65', '#26c6da'];

type CameraMode = 'driver' | 'fit';

const getCarColor = (key: string, isPlayer: boolean): string => {
    if (isPlayer) return PLAYER_COLOR;
    let hash = 0;
    for (let index = 0; index < key.length; index += 1) {
        hash = ((hash << 5) - hash + key.charCodeAt(index)) | 0;
    }
    return OPPONENT_COLORS[Math.abs(hash) % OPPONENT_COLORS.length];
};

const getBounds = (points: Vec3[]) => {
    if (points.length === 0) return { minX: -100, maxX: 100, minZ: -100, maxZ: 100, center: { x: 0, y: 0, z: 0 } };
    const xs = points.map((point) => point.x);
    const zs = points.map((point) => point.z);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    return {
        minX,
        maxX,
        minZ,
        maxZ,
        center: { x: (minX + maxX) / 2, y: 0, z: (minZ + maxZ) / 2 },
    };
};

const drawPolyline = (
    context: CanvasRenderingContext2D,
    points: Vec3[],
    project: (point: Vec3) => { x: number; y: number },
    color: string,
    width: number,
) => {
    if (points.length < 2) return;
    context.beginPath();
    points.forEach((point, index) => {
        const projected = project(point);
        if (index === 0) context.moveTo(projected.x, projected.y);
        else context.lineTo(projected.x, projected.y);
    });
    context.strokeStyle = color;
    context.lineWidth = width;
    context.lineJoin = 'round';
    context.lineCap = 'round';
    context.stroke();
};

export interface LiveTrajectoryMapHandle extends NamedOperationComponentHandle {
    focusDriver(): void;
    fitTrack(): void;
}

interface LiveTrajectoryMapProps {
    name: string;
    width?: string | number;
    height?: string | number;
}

const LiveTrajectoryMap = forwardRef<LiveTrajectoryMapHandle, LiveTrajectoryMapProps>(({
    name,
    width = '100%',
    height = '100%',
}, forwardedRef) => {
    const liveSession = useContext(LiveSessionContext);
    const { currentTelemetry, telemetryStatus, game: telemetryGame } = useLiveTelemetrySelector((snapshot) => snapshot);
    const { getCircuitMapByTrack } = useCircuitMaps();
    const mapLookupRef = useRef(getCircuitMapByTrack);
    mapLookupRef.current = getCircuitMapByTrack;
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const wrapperRef = useRef<HTMLDivElement | null>(null);
    const [canvasSize, setCanvasSize] = useState({ width: 800, height: 520 });
    const [mapResult, setMapResult] = useState<{
        key: string;
        map: CircuitMapDto | null;
        status: 'loading' | 'ready' | 'error';
    } | null>(null);
    const [cameraMode, setCameraMode] = useState<CameraMode>('fit');
    const [zoom, setZoom] = useState(1);
    const [flipX, setFlipX] = useState(false);
    const [flipZ, setFlipZ] = useState(false);
    const handle = useMemo<LiveTrajectoryMapHandle>(() => ({
        getComponentName: () => name,
        focusDriver: () => {
            setCameraMode('driver');
            setZoom(1);
        },
        fitTrack: () => {
            setCameraMode('fit');
            setZoom(1);
        },
    }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandleRef = useRef(handle);
    registeredHandleRef.current = handle;
    useRegisterOperationComponentRef(registeredHandleRef);

    const game = liveSession.sessionGame ?? telemetryGame;
    const track = liveSession.staticData.Static_track ?? currentTelemetry.Static_track;
    const trackKey = game === 'acc' ? getAccTelemetryTrackKey(track) || track?.trim() : track?.trim();
    const mapKey = `${game}:${trackKey}`;
    const circuitMap = mapResult?.key === mapKey ? mapResult.map : null;
    const mapStatus = mapResult?.key === mapKey ? mapResult.status : 'loading';
    const middleLine = useMemo(() => getLiveMapMiddleLine(circuitMap), [circuitMap]);
    const live = telemetryStatus === ACC_STATUS.ACC_LIVE;
    const cars = useMemo(() => live ? getLiveMapCars(currentTelemetry, middleLine) : [], [currentTelemetry, live, middleLine]);
    const bounds = useMemo(() => getBounds(middleLine), [middleLine]);
    const playerPosition = cars.find((car) => car.isPlayer)?.position;

    useEffect(() => {
        let cancelled = false;
        if (!trackKey || (game !== 'acc' && game !== 'iracing')) {
            setMapResult(null);
            return;
        }
        setMapResult({ key: mapKey, map: null, status: 'loading' });
        // Cache/list updates change the provider callback, but must not restart this download.
        void mapLookupRef.current(game, trackKey).then((map) => {
            if (!cancelled) setMapResult({ key: mapKey, map, status: 'ready' });
        }).catch(() => {
            if (!cancelled) setMapResult({ key: mapKey, map: null, status: 'error' });
        });
        return () => { cancelled = true; };
    }, [game, mapKey, trackKey]);

    useEffect(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper) return;
        const observer = new ResizeObserver(([entry]) => {
            if (!entry) return;
            setCanvasSize({
                width: Math.max(1, Math.floor(entry.contentRect.width)),
                height: Math.max(1, Math.floor(entry.contentRect.height)),
            });
        });
        observer.observe(wrapper);
        return () => observer.disconnect();
    }, []);

    const project = useCallback((point: Vec3) => {
        const center = cameraMode === 'driver' && playerPosition ? playerPosition : bounds.center;
        const padding = Math.max(28, Math.min(canvasSize.width, canvasSize.height) * 0.08);
        const spanX = Math.max(bounds.maxX - bounds.minX, 1);
        const spanZ = Math.max(bounds.maxZ - bounds.minZ, 1);
        const fitScale = Math.min(
            Math.max(1, canvasSize.width - padding * 2) / spanX,
            Math.max(1, canvasSize.height - padding * 2) / spanZ,
        );
        const scale = fitScale * (cameraMode === 'driver' ? 2.8 : 1) * zoom;
        return {
            x: canvasSize.width / 2 + (point.x - center.x) * scale * (flipX ? -1 : 1),
            y: canvasSize.height / 2 + (point.z - center.z) * scale * (flipZ ? -1 : 1) * (game === 'iracing' ? -1 : 1),
        };
    }, [bounds, cameraMode, canvasSize, playerPosition, flipX, flipZ, game, zoom]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.floor(canvasSize.width * ratio);
        canvas.height = Math.floor(canvasSize.height * ratio);
        canvas.style.width = `${canvasSize.width}px`;
        canvas.style.height = `${canvasSize.height}px`;
        const context = canvas.getContext('2d');
        if (!context) return;
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, canvasSize.width, canvasSize.height);

        const gradient = context.createRadialGradient(
            canvasSize.width / 2,
            canvasSize.height / 2,
            0,
            canvasSize.width / 2,
            canvasSize.height / 2,
            Math.max(canvasSize.width, canvasSize.height) * 0.7,
        );
        gradient.addColorStop(0, 'rgba(0, 230, 118, 0.045)');
        gradient.addColorStop(1, 'rgba(6, 7, 13, 0)');
        context.fillStyle = gradient;
        context.fillRect(0, 0, canvasSize.width, canvasSize.height);

        if (middleLine.length > 1) {
            const loop = [...middleLine, middleLine[0]];
            drawPolyline(context, loop, project, 'rgba(77, 82, 91, 0.7)', 12);
            drawPolyline(context, loop, project, 'rgba(255,255,255,.7)', 2);
        }
        cars.forEach((car) => {
            const point = project(car.position);
            context.beginPath();
            context.arc(point.x, point.y, car.isPlayer ? 6 : 4, 0, Math.PI * 2);
            context.fillStyle = getCarColor(car.key, car.isPlayer);
            context.shadowColor = context.fillStyle;
            context.shadowBlur = car.isPlayer ? 12 : 5;
            context.fill();
            context.shadowBlur = 0;
        });
    }, [canvasSize, cars, middleLine, project]);

    const stateMessage = !trackKey || !game
        ? ['Waiting for circuit', 'Start live telemetry to load the circuit map.']
        : mapStatus === 'loading'
            ? ['Loading circuit map', 'Downloading the circuit middle line.']
            : mapStatus === 'error'
                ? ['Unable to load circuit map', 'The circuit map could not be downloaded.']
                : middleLine.length < 2
                    ? ['Circuit middle line unavailable', 'Add a middle line for this circuit in Circuit Maps.']
                    : cars.length === 0
                        ? ['Waiting for live positions', 'Car markers appear when normalized lap positions are available.']
                        : null;

    return (
        <Card className="map-visualization-card live-trajectory-map" style={{ width, height }} data-testid="live-trajectory-map">
            <Box ref={wrapperRef} className="map-visualization">
                <canvas ref={canvasRef} className="map-visualization__canvas" role="img" aria-label="Live circuit map" />
                <div className={`map-visualization__hud map-visualization__hud--top ${live ? 'map-visualization__hud--live' : 'map-visualization__hud--standby'}`}>
                    <Flex align="center" gap="2" wrap="wrap">
                        <Badge color={live ? 'green' : 'gray'} variant="soft">{live ? 'Live Telemetry' : 'Telemetry Standby'}</Badge>
                        <Text size="1" className="map-visualization__metric">{circuitMap?.circuit_name || track || 'Live Map'}</Text>
                        <Text size="1" className="map-visualization__metric">{cars.filter((car) => !car.isPlayer).length} opponents</Text>
                    </Flex>
                </div>
                <div className="map-visualization__hud map-visualization__hud--camera">
                    <Flex align="center" gap="2" justify="end" wrap="wrap">
                        <Button size="1" variant={flipX ? 'solid' : 'soft'} onClick={() => setFlipX((value) => !value)}>X</Button>
                        <Button size="1" variant={flipZ ? 'solid' : 'soft'} onClick={() => setFlipZ((value) => !value)}>Z</Button>
                        <Button size="1" variant={cameraMode === 'driver' ? 'solid' : 'soft'} onClick={() => { setCameraMode('driver'); setZoom(1); }}>Driver</Button>
                        <Button size="1" variant={cameraMode === 'fit' ? 'solid' : 'soft'} onClick={() => { setCameraMode('fit'); setZoom(1); }}>Fit</Button>
                        <Button size="1" variant="soft" onClick={() => setZoom((value) => Math.min(6, value * 1.25))}>+</Button>
                        <Button size="1" variant="soft" onClick={() => setZoom((value) => Math.max(0.35, value / 1.25))}>−</Button>
                    </Flex>
                </div>
                {stateMessage ? (
                    <div className="map-visualization__state">
                        <Text size="2" weight="bold">{stateMessage[0]}</Text>
                        <Text size="1">{stateMessage[1]}</Text>
                    </div>
                ) : null}
            </Box>
        </Card>
    );
});

LiveTrajectoryMap.displayName = 'LiveTrajectoryMap';

export default LiveTrajectoryMap;
