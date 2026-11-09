import React, { forwardRef, useCallback, useContext, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { Badge, Box, Button, Card, Checkbox, Flex, Text } from '@radix-ui/themes';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { useCircuitMaps } from 'contexts/CircuitMapsContext';
import { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import { Vec3 } from 'views/session-shared/visualization/charts/mapTelemetry';
import { LiveSessionContext } from './LiveSessionContext';
import { useLiveTelemetrySelector } from './live-telemetry-store';
import { getLiveMapCars, getLiveMapMiddleLine, getLiveMapPosition } from './live-map-data';
import { loadTopLapReferenceModel, TopLapReferenceEntry } from 'services/topLapReferenceService';
import { getLiveMapReferenceHit, getLiveMapReferenceLine, getLiveMapReferencePedal, getLiveMapReferenceSpeed, selectLiveMapReference } from './live-map-reference';
import 'views/session-shared/visualization/charts/MapVisualization.css';
import { NamedOperationComponentHandle, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';

const PLAYER_COLOR = '#00e676';
const REFERENCE_COLOR = '#e879f9';
const REFERENCE_BRAKE_COLOR = '#ef5350';
const REFERENCE_THROTTLE_COLOR = '#00e676';
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

const getBounds = (points: Vec3[], verticalAxis: 'y' | 'z') => {
    if (points.length === 0) return { minX: -100, maxX: 100, minVertical: -100, maxVertical: 100, center: { x: 0, y: 0, z: 0 } };
    let minX = Infinity, maxX = -Infinity, minVertical = Infinity, maxVertical = -Infinity;
    points.forEach((point) => {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minVertical = Math.min(minVertical, point[verticalAxis]);
        maxVertical = Math.max(maxVertical, point[verticalAxis]);
    });
    const center = { x: (minX + maxX) / 2, y: 0, z: 0 };
    center[verticalAxis] = (minVertical + maxVertical) / 2;
    return {
        minX,
        maxX,
        minVertical,
        maxVertical,
        center,
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

const drawReferencePedal = (
    context: CanvasRenderingContext2D,
    points: Vec3[],
    values: number[],
    project: (point: Vec3) => { x: number; y: number },
    color: string,
    width: number,
) => {
    if (points.length < 2 || values.length !== points.length) return;
    context.lineWidth = width;
    context.lineJoin = 'round';
    context.lineCap = 'round';
    for (let index = 1; index < points.length; index += 1) {
        if (values[index - 1] === 0 && values[index] === 0) continue;
        const start = project(points[index - 1]);
        const end = project(points[index]);
        const gradient = context.createLinearGradient(start.x, start.y, end.x, end.y);
        // Interpolate pedal strength along the same segment as the reference line.
        gradient.addColorStop(0, `${color}${Math.round(values[index - 1] * 255).toString(16).padStart(2, '0')}`);
        gradient.addColorStop(1, `${color}${Math.round(values[index] * 255).toString(16).padStart(2, '0')}`);
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.strokeStyle = gradient;
        context.stroke();
    }
};

export interface LiveTrajectoryMapHandle extends NamedOperationComponentHandle {
    focusDriver(): void;
    fitTrack(): void;
    getCircuitMap(): CircuitMapDto | null;
    getReferenceLap(): TopLapReferenceEntry | null;
    getPositionReferences(position: number): { reference: Vec3 } | null;
    subscribeCircuitMap(listener: () => void): () => void;
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
    const [flipVertical, setFlipVertical] = useState(false);
    const [referenceModel, setReferenceModel] = useState<TopLapReferenceEntry[]>([]);
    const [referenceStatus, setReferenceStatus] = useState<'loading' | 'ready' | 'error'>('loading');
    const [referenceRequest, setReferenceRequest] = useState(0);
    const [showReferenceBrake, setShowReferenceBrake] = useState(false);
    const [showReferenceThrottle, setShowReferenceThrottle] = useState(false);
    const [hoverPointer, setHoverPointer] = useState<{ clientX: number; clientY: number } | null>(null);
    const circuitMapRef = useRef<CircuitMapDto | null>(null);
    const referenceLapRef = useRef<TopLapReferenceEntry | null>(null);
    const positionLookupRef = useRef<LiveTrajectoryMapHandle['getPositionReferences']>(() => null);
    const mapListenersRef = useRef(new Set<() => void>());
    const handle = useMemo<LiveTrajectoryMapHandle>(() => ({
        getComponentName: () => name,
        getCircuitMap: () => circuitMapRef.current,
        getReferenceLap: () => referenceLapRef.current,
        getPositionReferences: (position) => positionLookupRef.current(position),
        subscribeCircuitMap: (listener) => {
            mapListenersRef.current.add(listener);
            return () => { mapListenersRef.current.delete(listener); };
        },
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
    // Stored ACC lines use X/Y for the track plane and Z for altitude.
    const verticalAxis = game === 'acc' ? 'y' : 'z';
    const track = liveSession.staticData.Static_track ?? currentTelemetry.Static_track;
    const car = liveSession.staticData.Static_car_model ?? currentTelemetry.Static_car_model;
    const grip = currentTelemetry.Graphics_track_grip_status;
    const trackKey = game === 'acc' ? getAccTelemetryTrackKey(track) || track?.trim() : track?.trim();
    const mapKey = `${game}:${trackKey}`;
    const circuitMap = mapResult?.key === mapKey ? mapResult.map : null;
    circuitMapRef.current = circuitMap;
    const mapStatus = mapResult?.key === mapKey ? mapResult.status : 'loading';
    const middleLine = useMemo(() => getLiveMapMiddleLine(circuitMap), [circuitMap]);
    const canLoadReference = Boolean(track && car && (game === 'acc' || game === 'iracing'));
    const referenceLap = useMemo(() => selectLiveMapReference(referenceModel, game, track, car, grip), [referenceModel, game, track, car, grip]);
    const referenceLine = useMemo(() => getLiveMapReferenceLine(referenceLap), [referenceLap]);
    const referenceBrake = useMemo(() => getLiveMapReferencePedal(referenceLap, 'brake'), [referenceLap]);
    const referenceThrottle = useMemo(() => getLiveMapReferencePedal(referenceLap, 'throttle'), [referenceLap]);
    const referenceSpeed = useMemo(() => getLiveMapReferenceSpeed(referenceLap), [referenceLap]);
    const hasReferenceBrake = referenceLine.length > 1 && referenceBrake.length === referenceLine.length;
    const hasReferenceThrottle = referenceLine.length > 1 && referenceThrottle.length === referenceLine.length;
    const referencePoints = useMemo(() => referenceLine.map((point, index) => ({
        ...point, normalized_position: referenceLap!.x[index],
    })), [referenceLine, referenceLap]);
    referenceLapRef.current = referenceLap;
    positionLookupRef.current = (position) => {
        const reference = getLiveMapPosition(referencePoints, position);
        return reference ? { reference } : null;
    };
    const live = telemetryStatus === ACC_STATUS.ACC_LIVE;
    const cars = useMemo(() => live ? getLiveMapCars(currentTelemetry, middleLine) : [], [currentTelemetry, live, middleLine]);
    const bounds = useMemo(() => getBounds([...middleLine, ...referenceLine], verticalAxis), [middleLine, referenceLine, verticalAxis]);
    const playerPosition = cars.find((car) => car.isPlayer)?.position;

    useEffect(() => { mapListenersRef.current.forEach((listener) => listener()); }, [circuitMap, referenceLap]);

    useEffect(() => {
        if (!canLoadReference) return;
        let cancelled = false;
        setReferenceStatus('loading');
        void loadTopLapReferenceModel().then((model) => {
            if (cancelled) return;
            setReferenceModel(model);
            setReferenceStatus('ready');
        }).catch(() => {
            if (!cancelled) setReferenceStatus('error');
        });
        return () => { cancelled = true; };
    }, [canLoadReference, referenceRequest]);

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
        const spanVertical = Math.max(bounds.maxVertical - bounds.minVertical, 1);
        const fitScale = Math.min(
            Math.max(1, canvasSize.width - padding * 2) / spanX,
            Math.max(1, canvasSize.height - padding * 2) / spanVertical,
        );
        const scale = fitScale * (cameraMode === 'driver' ? 2.8 : 1) * zoom;
        return {
            x: canvasSize.width / 2 + (point.x - center.x) * scale * (flipX ? -1 : 1),
            y: canvasSize.height / 2 + (point[verticalAxis] - center[verticalAxis]) * scale * (flipVertical ? -1 : 1) * (game === 'iracing' ? -1 : 1),
        };
    }, [bounds, cameraMode, canvasSize, playerPosition, flipX, flipVertical, game, verticalAxis, zoom]);

    const projectedReferenceLine = useMemo(() => referenceLine.map(project), [referenceLine, project]);
    const referenceHover = useMemo(() => {
        if (!hoverPointer || !canvasRef.current || !referenceLap) return null;
        const rect = canvasRef.current.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        const pointer = {
            x: (hoverPointer.clientX - rect.left) * canvasSize.width / rect.width,
            y: (hoverPointer.clientY - rect.top) * canvasSize.height / rect.height,
        };
        if (pointer.x < 0 || pointer.x > canvasSize.width || pointer.y < 0 || pointer.y > canvasSize.height) return null;
        const hit = getLiveMapReferenceHit(projectedReferenceLine, pointer);
        if (!hit) return null;
        const interpolate = (values: number[]) => values.length === referenceLine.length
            ? values[hit.index] + (values[hit.index + 1] - values[hit.index]) * hit.fraction : null;
        return { ...hit, position: interpolate(referenceLap.x), brake: interpolate(referenceBrake), throttle: interpolate(referenceThrottle), speed: interpolate(referenceSpeed) };
    }, [hoverPointer, canvasSize, projectedReferenceLine, referenceLap, referenceLine, referenceBrake, referenceThrottle, referenceSpeed]);

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
        drawPolyline(context, referenceLine, project, REFERENCE_COLOR, 3);
        // Keep throttle visible around the narrower brake trace when both pedals overlap.
        if (showReferenceThrottle) drawReferencePedal(context, referenceLine, referenceThrottle, project, REFERENCE_THROTTLE_COLOR, 7);
        if (showReferenceBrake) drawReferencePedal(context, referenceLine, referenceBrake, project, REFERENCE_BRAKE_COLOR, 3);
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
    }, [canvasSize, cars, middleLine, project, referenceLine, referenceBrake, referenceThrottle, showReferenceBrake, showReferenceThrottle]);

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
                <canvas ref={canvasRef} className="map-visualization__canvas" role="img" aria-label="Live circuit map"
                    onMouseMove={(event) => setHoverPointer({ clientX: event.clientX, clientY: event.clientY })}
                    onMouseLeave={() => setHoverPointer(null)} />
                {referenceHover ? (
                    <>
                        <div className="map-visualization__reference-hover-point" aria-hidden="true"
                            style={{ left: referenceHover.x, top: referenceHover.y }} />
                        <div className="map-visualization__reference-tooltip" role="tooltip" style={{
                            left: Math.max(8, Math.min(referenceHover.x + 12, canvasSize.width - 188)),
                            top: referenceHover.y,
                            transform: referenceHover.y > canvasSize.height / 2 ? 'translateY(calc(-100% - 12px))' : 'translateY(12px)',
                        }}>
                            <Text size="1" weight="bold">Reference lap · {((referenceHover.position ?? 0) * 100).toFixed(1)}%</Text>
                            <Text size="1">
                                Speed: {referenceHover.speed === null ? 'Unavailable' : `${referenceHover.speed.toFixed(1)} km/h`}
                            </Text>
                            <Text size="1" style={{ color: REFERENCE_BRAKE_COLOR }}>
                                Brake: {referenceHover.brake === null ? 'Unavailable' : `${(referenceHover.brake * 100).toFixed(1)}%`}
                            </Text>
                            <Text size="1" style={{ color: REFERENCE_THROTTLE_COLOR }}>
                                Throttle: {referenceHover.throttle === null ? 'Unavailable' : `${(referenceHover.throttle * 100).toFixed(1)}%`}
                            </Text>
                        </div>
                    </>
                ) : null}
                <div className={`map-visualization__hud map-visualization__hud--top ${live ? 'map-visualization__hud--live' : 'map-visualization__hud--standby'}`}>
                    <Flex align="center" gap="2" wrap="wrap">
                        <Badge color={live ? 'green' : 'gray'} variant="soft">{live ? 'Live Telemetry' : 'Telemetry Standby'}</Badge>
                        <Text size="1" className="map-visualization__metric">{circuitMap?.circuit_name || track || 'Live Map'}</Text>
                        <Text size="1" className="map-visualization__metric">{cars.filter((car) => !car.isPlayer).length} opponents</Text>
                        <Text size="1" style={{ color: referenceLine.length > 1 ? REFERENCE_COLOR : undefined }} role="status">
                            {!canLoadReference ? 'Reference lap: waiting for car and circuit'
                                : referenceStatus === 'loading' ? 'Loading reference lap model…'
                                    : referenceStatus === 'error' ? 'Reference lap unavailable'
                                        : !referenceLap ? 'No reference lap for this car and circuit'
                                            : referenceLine.length < 2 ? 'Reference lap has no racing line'
                                                : `Reference lap · ${referenceLap.car} · grip ${referenceLap.avg_grip_int}`}
                        </Text>
                        {canLoadReference && referenceStatus === 'error' ? (
                            <Button size="1" variant="soft" onClick={() => setReferenceRequest((value) => value + 1)}>Retry reference lap</Button>
                        ) : null}
                    </Flex>
                </div>
                <div className="map-visualization__hud map-visualization__hud--camera">
                    <Flex align="center" gap="2" justify="end" wrap="wrap">
                        <Button size="1" variant={flipX ? 'solid' : 'soft'} onClick={() => setFlipX((value) => !value)}>X</Button>
                        <Button size="1" variant={flipVertical ? 'solid' : 'soft'} onClick={() => setFlipVertical((value) => !value)}>{verticalAxis.toUpperCase()}</Button>
                        <Button size="1" variant={cameraMode === 'driver' ? 'solid' : 'soft'} onClick={() => { setCameraMode('driver'); setZoom(1); }}>Driver</Button>
                        <Button size="1" variant={cameraMode === 'fit' ? 'solid' : 'soft'} onClick={() => { setCameraMode('fit'); setZoom(1); }}>Fit</Button>
                        <Button size="1" variant="soft" onClick={() => setZoom((value) => Math.min(6, value * 1.25))}>+</Button>
                        <Button size="1" variant="soft" onClick={() => setZoom((value) => Math.max(0.35, value / 1.25))}>−</Button>
                    </Flex>
                </div>
                <div className="map-visualization__hud map-visualization__hud--reference-pedals" role="group" aria-label="Reference lap pedals">
                    <Text as="p" size="1" mb="1" color="gray">Hover the reference line to inspect speed, brake and throttle.</Text>
                    <Flex align="center" gap="3" wrap="wrap">
                        <Text as="label" size="1" style={{ color: REFERENCE_BRAKE_COLOR }}>
                            <Flex align="center" gap="2">
                                <Checkbox color="red" checked={showReferenceBrake} disabled={!hasReferenceBrake}
                                    onCheckedChange={(checked) => setShowReferenceBrake(checked === true)} aria-label="Reference brake" />
                                Reference brake{!hasReferenceBrake ? ' (unavailable)' : ''}
                            </Flex>
                        </Text>
                        <Text as="label" size="1" style={{ color: REFERENCE_THROTTLE_COLOR }}>
                            <Flex align="center" gap="2">
                                <Checkbox color="green" checked={showReferenceThrottle} disabled={!hasReferenceThrottle}
                                    onCheckedChange={(checked) => setShowReferenceThrottle(checked === true)} aria-label="Reference throttle" />
                                Reference throttle{!hasReferenceThrottle ? ' (unavailable)' : ''}
                            </Flex>
                        </Text>
                    </Flex>
                    {(showReferenceBrake && hasReferenceBrake) || (showReferenceThrottle && hasReferenceThrottle) ? (
                        <Text as="p" size="1" mt="1" color="gray">
                            Brighter = more input{showReferenceBrake && hasReferenceBrake && showReferenceThrottle && hasReferenceThrottle
                                ? ' · Brake inside, throttle outside' : ''}
                        </Text>
                    ) : null}
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
