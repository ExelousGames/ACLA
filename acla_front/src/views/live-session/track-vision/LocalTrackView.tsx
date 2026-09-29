import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { CameraCalibration, GroundPoint, LocalTrackScene, TrackBoundaryPoint, TrackSceneMemory, TrackVisionFrame } from './track-vision-types';
import { VISION_MAX_AGE_MS } from './track-vision-types';
import { createCameraProjection, validCalibration } from './camera-projection';
import { reconstructDistanceGrid } from './track-position-analysis';
import { createLocalOverviewCamera, DEFAULT_OVERVIEW_ORBIT } from './local-overview-camera';
import { SCENE_MEMORY_MAX_AGE_MS } from './scene-memory';

const carCorners = (car: LocalTrackScene['cars'][number]) => [0, 1, 2, 3, 4, 5, 6, 7].map((bits) => ({
    x: bits & 1 ? car.max.x : car.min.x, y: bits & 2 ? car.max.y : car.min.y, z: bits & 4 ? car.max.z : car.min.z,
}));

export default function LocalTrackView({ frame, scene, camera, applied, memory }: {
    frame: TrackVisionFrame | null; scene: LocalTrackScene | null; camera: CameraCalibration; applied: boolean;
    memory?: TrackSceneMemory | null;
}) {
    const [now, setNow] = useState(Date.now);
    const [overview, setOverview] = useState(true);
    const [orbit, setOrbit] = useState(DEFAULT_OVERVIEW_ORBIT);
    const drag = useRef<{ pointerId: number; x: number; y: number } | null>(null);
    const orbitHintId = useId();
    const rotate = (yaw: number, pitch: number) => setOrbit((previous) => ({
        yawDeg: (previous.yawDeg + yaw) % 360,
        pitchDeg: Math.max(-85, Math.min(85, previous.pitchDeg + pitch)),
    }));
    const endDrag = (event: React.PointerEvent<SVGSVGElement>) => {
        if (drag.current?.pointerId !== event.pointerId) return;
        drag.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    };
    useEffect(() => {
        setNow(Date.now());
        const remaining = frame ? frame.capturedAt + VISION_MAX_AGE_MS - Date.now() : 0;
        const timer = setTimeout(() => setNow(Date.now()), Math.max(0, remaining));
        return () => clearTimeout(timer);
    }, [frame]);
    const distanceGrid = useMemo(() => frame ? reconstructDistanceGrid({ ...frame, calibration: camera }) : [], [frame, camera]);
    const captureView = validCalibration(camera) ? createCameraProjection(camera) : null;
    const fresh = frame && frame.capturedAt + VISION_MAX_AGE_MS > Math.max(now, Date.now());
    const visible = fresh && captureView ? scene : null;
    const visibleMemory = visible && applied && memory?.capturedAt === frame?.capturedAt ? memory : null;
    const viewCamera = overview && captureView ? createLocalOverviewCamera([
        { x: 0, y: 0, z: 0 }, { x: 0, y: 3, z: 0 },
        ...(visible?.leftBoundary ?? []), ...(visible?.rightBoundary ?? []),
        ...(visible?.cars.flatMap(carCorners) ?? []),
        ...(visibleMemory?.points ?? []),
    ], orbit) : camera;
    const view = captureView ? createCameraProjection(viewCamera) : null;
    // Normalize SVG units for readable labels; camera view retains the capture aspect ratio.
    const viewWidth = 800, viewHeight = view ? viewWidth * viewCamera.imageHeight / viewCamera.imageWidth : 0;
    const project = (point: GroundPoint) => {
        const pixel = view?.localToImage(point);
        return pixel ? { x: pixel.u * viewWidth, y: pixel.v * viewHeight } : null;
    };
    const path = (points: GroundPoint[]) => {
        let connected = false;
        return points.map((point) => {
            const pixel = project(point);
            if (!pixel) { connected = false; return ''; }
            const command = connected ? 'L' : 'M';
            connected = true;
            return `${command}${pixel.x.toFixed(2)},${pixel.y.toFixed(2)}`;
        }).join(' ');
    };
    const origin = project({ x: 0, y: 0, z: 0 });
    const edgePath = (points: TrackBoundaryPoint[], estimated: boolean) => {
        const runs: GroundPoint[][] = [];
        let run: GroundPoint[] = [];
        for (let i = 1; i < points.length; i++) {
            const previous = points[i - 1], point = points[i];
            if (Boolean(previous.estimated || point.estimated) === estimated) {
                if (!run.length) run.push(previous);
                run.push(point);
            } else if (run.length) { runs.push(run); run = []; }
        }
        return [...runs, run].map(path).filter(Boolean).join(' ');
    };
    const estimatedCount = [...(visible?.leftBoundary ?? []), ...(visible?.rightBoundary ?? [])].filter((point) => point.estimated).length;
    const geometry = visible?.geometry;
    const gridLabels: Array<{ x: number; y: number; width: number }> = [];
    const status = !frame ? 'Share a driving view to reconstruct the scene.' : !fresh ? 'Waiting for a fresh frame.'
        : !frame.detections.segment || !frame.detections.depth ? 'Enable segmentation and depth; both results are needed for local 3D.'
            : !view || !scene ? 'Enter valid camera settings and wait for valid depth.'
                : !scene.leftBoundary.length && !scene.rightBoundary.length && !scene.cars.length ? 'No supported track edges or cars in this frame.'
                    : `${scene.leftBoundary.length} left / ${scene.rightBoundary.length} right edge points · ${scene.cars.length} car detections reconstructed.`
                        + (estimatedCount ? ` ${estimatedCount} edge points estimated behind traffic.` : '');
    return <section className="track-vision__reconstruction" aria-label="Local 3D reconstruction">
        <h3>Local 3D · track edges and cars</h3>
        <span className="track-vision__hint">{applied ? 'Calibration applied' : 'Draft camera preview'}</span>
        <div className="track-vision__controls" role="group" aria-label="Local 3D viewpoint">
            {[true, false].map((value) => <button key={String(value)} type="button" className="track-vision__capture-toggle"
                aria-pressed={overview === value} onClick={() => { drag.current = null; setOverview(value); }}>
                {value ? '3D overview' : 'Camera view'}
            </button>)}
            {overview && <button type="button" onClick={() => setOrbit(DEFAULT_OVERVIEW_ORBIT)}>Reset view</button>}
        </div>
        {overview && <p id={orbitHintId} className="track-vision__hint">Drag to rotate the 3D world. Use arrow keys when focused; Home resets the view.</p>}
        {view && <svg className={`track-vision__local-scene${overview ? ' track-vision__local-scene--orbit' : ''}`}
            viewBox={`0 0 ${viewWidth} ${viewHeight}`} role="group" aria-label="Perspective 3D track edges and cars"
            tabIndex={overview ? 0 : undefined} aria-describedby={overview ? orbitHintId : undefined}
            onPointerDown={(event) => {
                if (!overview || event.button !== 0 || drag.current) return;
                event.preventDefault();
                event.currentTarget.focus();
                event.currentTarget.setPointerCapture(event.pointerId);
                drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
            }}
            onPointerMove={(event) => {
                const previous = drag.current;
                if (!overview || !previous || previous.pointerId !== event.pointerId) return;
                rotate((previous.x - event.clientX) * 0.4, (event.clientY - previous.y) * 0.4);
                drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
            }}
            onPointerUp={endDrag} onPointerCancel={endDrag} onLostPointerCapture={endDrag}
            onKeyDown={(event) => {
                if (!overview) return;
                switch (event.key) {
                    case 'ArrowLeft': rotate(5, 0); break;
                    case 'ArrowRight': rotate(-5, 0); break;
                    case 'ArrowUp': rotate(0, -5); break;
                    case 'ArrowDown': rotate(0, 5); break;
                    case 'Home': setOrbit(DEFAULT_OVERVIEW_ORBIT); break;
                    default: return;
                }
                event.preventDefault();
            }}>
            <g aria-label="Rolling scene memory">
                {visibleMemory?.points.map((point, index) => {
                    const pixel = project(point);
                    const color = { road: '#6d919a', roadside: '#8d9368', 'left-edge': '#37efac', 'right-edge': '#57b9ff' }[point.surface];
                    const age = Math.max(0, visibleMemory.capturedAt - point.lastSeenAt);
                    return pixel ? <circle key={index} cx={pixel.x} cy={pixel.y} r={point.surface.endsWith('edge') ? 1.8 : 1.2}
                        fill={color} opacity={(0.3 + Math.min(point.observations, 4) * 0.12) * (1 - age / SCENE_MEMORY_MAX_AGE_MS)} /> : null;
                })}
            </g>
            <g aria-label="Depth distance grid">
                {visible && distanceGrid.map(({ distanceM, segments }) => {
                    const label = segments.flat().map(project).filter((point): point is { x: number; y: number } =>
                        Boolean(point && point.x >= 0 && point.x <= viewWidth && point.y >= 0 && point.y <= viewHeight))
                        .sort((a, b) => a.x - b.x)[0];
                    const bounds = label && { x: label.x, y: Math.max(12, label.y - 4), width: `${distanceM} m`.length * 6 };
                    const showLabel = bounds && gridLabels.every((other) => Math.abs(bounds.y - other.y) >= 14
                        || bounds.x >= other.x + other.width + 4 || other.x >= bounds.x + bounds.width + 4);
                    if (showLabel) gridLabels.push(bounds);
                    return <g key={distanceM}><path d={segments.map(path).join(' ')} fill="none" stroke="#ffffff25" />
                        {showLabel && <text x={bounds.x} y={bounds.y}>{distanceM} m</text>}</g>;
                })}
            </g>
            <g aria-label="Reconstructed track edges" fill="none" strokeWidth="2">
                {visible && <>
                    <path aria-label="Observed left track edge" d={edgePath(visible.leftBoundary, false)} stroke="#37efac" />
                    <path aria-label="Observed right track edge" d={edgePath(visible.rightBoundary, false)} stroke="#57b9ff" />
                </>}
            </g>
            <g aria-label="Estimated track edges" fill="none" strokeWidth="2" strokeDasharray="5 4" opacity="0.65">
                {visible && <>
                    <path aria-label="Estimated left track edge" d={edgePath(visible.leftBoundary, true)} stroke="#37efac" />
                    <path aria-label="Estimated right track edge" d={edgePath(visible.rightBoundary, true)} stroke="#57b9ff" />
                </>}
            </g>
            <g aria-label="Reconstructed cars">
                {visible?.cars.slice().sort((a, b) => b.center.y - a.center.y).map((car, i) => {
                    const label = project({ ...car.center, z: car.max.z + 0.5 });
                    const corners = carCorners(car);
                    return <g key={i} fill={car.pack ? '#cf9fff' : '#ffbe57'}>
                        {car.points.map((point, j) => {
                            const pixel = project(point);
                            return pixel ? <circle key={j} cx={pixel.x} cy={pixel.y} r="1.3" opacity="0.7" /> : null;
                        })}
                        {corners.flatMap((point, index) => [1, 2, 4].filter((bit) => !(index & bit)).map((bit) => <path
                            key={`${index}-${bit}`} d={path([point, corners[index | bit]])} fill="none" stroke="currentColor"
                            style={{ color: car.pack ? '#cf9fff' : '#ffbe57' }} opacity="0.6" />))}
                        {label && <text x={label.x} y={label.y}>{car.pack ? 'Car pack' : 'Car'} · {car.center.y.toFixed(1)} m</text>}
                    </g>;
                })}
            </g>
            <path d={path([{ x: 0, y: 0, z: 0 }, { x: 0, y: 3, z: 0 }])} stroke="#ffbe57" strokeWidth="4" />
            {origin && <text x={origin.x + (overview ? 14 : 6)} y={origin.y + (overview ? 20 : 0)}>Your car</text>}
        </svg>}
        <p aria-label="Reconstruction status">{status}</p>
        <p aria-label="Scene memory status" className="track-vision__hint">{!applied ? 'Apply camera calibration to build scene memory.'
            : !visibleMemory ? 'Waiting for a fresh scene for memory.'
                : `${visibleMemory.points.length} static memory points · ${visibleMemory.reason}${visibleMemory.status === 'aligned'
                    ? ` ${visibleMemory.inliers}/${visibleMemory.matchedFeatures} features · ${visibleMemory.alignmentErrorM!.toFixed(2)} m fit error.` : ''}`}</p>
        <p className="track-vision__hint">Green / blue: track edges · dashed edges: estimated behind traffic · muted points: road and roadside memory · amber: cars · purple: car packs. X right, Y forward, Z up. Distances and visible car surfaces are estimated from monocular depth.</p>
        <p aria-label="Road fit status">{!fresh ? 'Waiting for a fresh frame.' : geometry
            ? `Road observed from ${geometry.referenceY.toFixed(1)} to ${Math.min(geometry.left.maxY, geometry.right.maxY).toFixed(1)} m ahead.`
            : 'Road geometry unresolved — both edges need reliable segmentation and depth.'}</p>
        {geometry && <dl className="track-vision__geometry">
            <div><dt>Width at {geometry.referenceY.toFixed(1)} m</dt><dd>{geometry.trackWidthM.toFixed(2)} m</dd></div>
            <div><dt>Car axis offset from road center</dt><dd>{geometry.lateralOffsetM.toFixed(2)} m (right +)</dd></div>
            <div><dt>Road heading</dt><dd>{geometry.headingDeg.toFixed(1)}° (right +)</dd></div>
        </dl>}
    </section>;
}
