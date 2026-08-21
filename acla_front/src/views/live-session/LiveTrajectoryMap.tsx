import React, { forwardRef, useCallback, useContext, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { Badge, Box, Button, Card, Flex, Text } from '@radix-ui/themes';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
import { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import {
    buildCircuitTrackLayout,
    CircuitTrackLayout,
    EMPTY_CIRCUIT_TRACK_LAYOUT,
    getAccTelemetryTrackKey,
} from 'views/lap-analysis/visualization/charts/circuitTrackLayout';
import { parseTelemetryFrame, TelemetryFrame, Vec3 } from 'views/lap-analysis/visualization/charts/mapTelemetry';
import { LiveSessionContext } from './LiveSessionContext';
import 'views/lap-analysis/visualization/charts/MapVisualization.css';
import { NamedAiToolComponentHandle, useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';

const LIVE_TRAIL_LIMIT = 900;
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

const getBounds = (frames: TelemetryFrame[], track: CircuitTrackLayout) => {
    const points = [...track.allPoints];
    frames.forEach((frame) => frame.cars.forEach((car) => points.push(car.position)));
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

export interface LiveTrajectoryMapHandle extends NamedAiToolComponentHandle {
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
    const { getCircuitMapByTrack } = useCircuitMaps();
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const wrapperRef = useRef<HTMLDivElement | null>(null);
    const telemetrySequenceRef = useRef(0);
    const [canvasSize, setCanvasSize] = useState({ width: 800, height: 520 });
    const [frames, setFrames] = useState<TelemetryFrame[]>([]);
    const [circuitMap, setCircuitMap] = useState<CircuitMapDto | null>(null);
    const [cameraMode, setCameraMode] = useState<CameraMode>('driver');
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
    useRegisterAiToolComponentRef(registeredHandleRef);

    const trackKey = useMemo(() => getAccTelemetryTrackKey(
        liveSession.currentTelemetry?.Static_track,
        liveSession.staticData.Static_track,
    ), [liveSession.currentTelemetry, liveSession.staticData.Static_track]);
    const trackLayout = useMemo(() => circuitMap ? buildCircuitTrackLayout(circuitMap) : EMPTY_CIRCUIT_TRACK_LAYOUT, [circuitMap]);
    const currentFrame = frames[frames.length - 1];
    const bounds = useMemo(() => getBounds(frames, trackLayout), [frames, trackLayout]);

    useEffect(() => {
        if (liveSession.telemetryStatus !== ACC_STATUS.ACC_LIVE) return;
        const parsed = parseTelemetryFrame(liveSession.currentTelemetry, telemetrySequenceRef.current);
        if (!parsed) return;
        telemetrySequenceRef.current += 1;
        setFrames((previous) => {
            const next = [...previous, parsed];
            return next.length > LIVE_TRAIL_LIMIT ? next.slice(-LIVE_TRAIL_LIMIT) : next;
        });
    }, [liveSession.currentTelemetry, liveSession.telemetryStatus]);

    useEffect(() => {
        let cancelled = false;
        if (!trackKey) {
            setCircuitMap(null);
            return;
        }
        void getCircuitMapByTrack('acc', trackKey).then((map) => {
            if (!cancelled) setCircuitMap(map);
        });
        return () => { cancelled = true; };
    }, [getCircuitMapByTrack, trackKey]);

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
        const playerPosition = currentFrame?.cars.find((car) => car.key === currentFrame.playerKey)?.position;
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
            y: canvasSize.height / 2 + (point.z - center.z) * scale * (flipZ ? -1 : 1),
        };
    }, [bounds, cameraMode, canvasSize, currentFrame, flipX, flipZ, zoom]);

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

        trackLayout.surface.forEach((polygon) => {
            if (polygon.length < 3) return;
            context.beginPath();
            polygon.forEach((point, index) => {
                const projected = project(point);
                if (index === 0) context.moveTo(projected.x, projected.y);
                else context.lineTo(projected.x, projected.y);
            });
            context.closePath();
            context.fillStyle = 'rgba(77, 82, 91, 0.42)';
            context.fill();
        });
        trackLayout.leftBoundary.forEach((line) => drawPolyline(context, line, project, '#29b6f6', 2));
        trackLayout.rightBoundary.forEach((line) => drawPolyline(context, line, project, '#ffca28', 2));
        trackLayout.centerLine.forEach((line) => drawPolyline(context, line, project, 'rgba(255,255,255,.18)', 1));
        trackLayout.pitLane.forEach((line) => drawPolyline(context, line, project, '#66bb6a', 2));

        const trajectories = new Map<string, Vec3[]>();
        frames.forEach((frame) => frame.cars.forEach((car) => {
            const points = trajectories.get(car.key) || [];
            points.push(car.position);
            trajectories.set(car.key, points);
        }));
        const playerKey = currentFrame?.playerKey || 'slot:0';
        trajectories.forEach((points, key) => drawPolyline(context, points, project, getCarColor(key, key === playerKey), key === playerKey ? 3 : 1.5));
        currentFrame?.cars.forEach((car) => {
            const point = project(car.position);
            context.beginPath();
            context.arc(point.x, point.y, car.key === playerKey ? 6 : 4, 0, Math.PI * 2);
            context.fillStyle = getCarColor(car.key, car.key === playerKey);
            context.shadowColor = context.fillStyle;
            context.shadowBlur = car.key === playerKey ? 12 : 5;
            context.fill();
            context.shadowBlur = 0;
        });
    }, [canvasSize, currentFrame, frames, project, trackLayout]);

    const live = liveSession.telemetryStatus === ACC_STATUS.ACC_LIVE;
    return (
        <Card className="map-visualization-card live-trajectory-map" style={{ width, height }} data-testid="live-trajectory-map">
            <Box ref={wrapperRef} className="map-visualization">
                <canvas ref={canvasRef} className="map-visualization__canvas" />
                <div className="map-visualization__hud map-visualization__hud--top">
                    <Flex align="center" gap="2" wrap="wrap">
                        <Badge color={live ? 'green' : 'gray'} variant="soft">{live ? 'Live Telemetry' : 'Telemetry Standby'}</Badge>
                        <Text size="1" className="map-visualization__metric">{frames.length.toLocaleString()} visible samples</Text>
                        <Text size="1" className="map-visualization__metric">{Math.max(0, (currentFrame?.cars.length || 1) - 1)} opponents</Text>
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
                {frames.length === 0 ? (
                    <div className="map-visualization__state">
                        <Text size="2" weight="bold">Waiting for current telemetry</Text>
                        <Text size="1">Live trajectory data appears here when ACC is running.</Text>
                    </div>
                ) : null}
            </Box>
        </Card>
    );
});

LiveTrajectoryMap.displayName = 'LiveTrajectoryMap';

export default LiveTrajectoryMap;
