import React, { useEffect, useMemo, useState } from 'react';
import type { CameraCalibration, GroundPoint, LocalTrackScene, TrackVisionFrame } from './track-vision-types';
import { VISION_MAX_AGE_MS } from './track-vision-types';
import { createCameraProjection, validCalibration } from './camera-projection';
import { reconstructDistanceGrid } from './track-position-analysis';
import TrackBoundaryCutoff, { MAX_BOUNDARY_START_M, MIN_BOUNDARY_START_M } from './TrackBoundaryCutoff';

export default function LocalTrackView({ frame, scene, camera, applied, boundaryStartDistanceM = 5, onBoundaryStartChange }: {
    frame: TrackVisionFrame | null; scene: LocalTrackScene | null; camera: CameraCalibration; applied: boolean;
    boundaryStartDistanceM?: number; onBoundaryStartChange?: (value: number) => void;
}) {
    const [now, setNow] = useState(Date.now);
    useEffect(() => {
        setNow(Date.now());
        const remaining = frame ? frame.capturedAt + VISION_MAX_AGE_MS - Date.now() : 0;
        const timer = setTimeout(() => setNow(Date.now()), Math.max(0, remaining));
        return () => clearTimeout(timer);
    }, [frame]);
    const distanceGrid = useMemo(() => frame ? reconstructDistanceGrid({ ...frame, calibration: camera }) : [], [frame, camera]);
    const view = validCalibration(camera) ? createCameraProjection(camera) : null;
    // Normalize SVG units for readable labels while retaining the capture aspect ratio.
    const viewWidth = 800, viewHeight = view ? viewWidth * camera.imageHeight / camera.imageWidth : 0;
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
    const fresh = frame && frame.capturedAt + VISION_MAX_AGE_MS > Math.max(now, Date.now());
    const visible = fresh && view ? scene : null;
    const geometry = visible?.geometry;
    const startY = camera.forwardOffsetM + boundaryStartDistanceM;
    const starts = [visible?.leftBoundary[0], visible?.rightBoundary[0]].map((point) =>
        point && Math.abs(point.y - startY) < 0.0001 ? point : null);
    const gridLabels: Array<{ x: number; y: number; width: number }> = [];
    const status = !frame ? 'Share a driving view to reconstruct the scene.' : !fresh ? 'Waiting for a fresh frame.'
        : !frame.detections.segment || !frame.detections.depth ? 'Enable segmentation and depth; both results are needed for local 3D.'
            : !view || !scene ? 'Enter valid camera settings and wait for valid depth.'
                : !scene.leftBoundary.length && !scene.rightBoundary.length && !scene.cars.length ? 'No supported track edges or cars in this frame.'
                    : `${scene.leftBoundary.length} left / ${scene.rightBoundary.length} right edge points · ${scene.cars.length} car detections reconstructed.`;
    return <section className="track-vision__reconstruction" aria-label="Local 3D reconstruction">
        <h3>Local 3D · track edges and cars</h3>
        <span className="track-vision__hint">{applied ? 'Calibration applied' : 'Draft camera preview'}</span>
        {view && <svg className="track-vision__local-scene" viewBox={`0 0 ${viewWidth} ${viewHeight}`} role="group" aria-label="Perspective 3D track edges and cars">
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
                    <path d={path(visible.leftBoundary)} stroke="#37efac" />
                    <path d={path(visible.rightBoundary)} stroke="#57b9ff" />
                </>}
            </g>
            <g aria-label="Reconstructed cars">
                {visible?.cars.slice().sort((a, b) => b.center.y - a.center.y).map((car, i) => {
                    const label = project({ ...car.center, z: car.max.z + 0.5 });
                    const corners = [0, 1, 2, 3, 4, 5, 6, 7].map((bits) => ({
                        x: bits & 1 ? car.max.x : car.min.x, y: bits & 2 ? car.max.y : car.min.y, z: bits & 4 ? car.max.z : car.min.z,
                    }));
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
            {origin && <text x={origin.x + 6} y={origin.y}>Your car</text>}
            {frame && onBoundaryStartChange && <TrackBoundaryCutoff camera={camera} width={viewWidth} value={boundaryStartDistanceM}
                left={starts[0]} right={starts[1]} onChange={onBoundaryStartChange} />}
        </svg>}
        {onBoundaryStartChange && <>
            <div className="track-vision__controls">
                <label>Boundary start {boundaryStartDistanceM.toFixed(1)} m forward from camera
                    <input aria-label="Boundary start" type="range" min={MIN_BOUNDARY_START_M} max={MAX_BOUNDARY_START_M}
                        step="0.1" value={boundaryStartDistanceM} onChange={(event) => onBoundaryStartChange(Number(event.target.value))} />
                </label>
            </div>
            <p className="track-vision__hint">Drag the amber line in local 3D to start detection beyond the hood or cockpit. The line sets a shared forward distance from the camera for both edges.</p>
            <dl className="track-vision__geometry" aria-label="Boundary starting positions">
                {starts.map((point, index) => <div key={index}><dt>{index ? 'Right' : 'Left'} start (X, Y, Z)</dt>
                    <dd>{point ? `${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)} m` : 'Unobserved at start line'}</dd></div>)}
            </dl>
        </>}
        <p aria-label="Reconstruction status">{status}</p>
        <p className="track-vision__hint">Green / blue: track edges · amber: cars · purple: car packs. X right, Y forward, Z up. Distances and visible car surfaces are estimated from monocular depth.</p>
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
