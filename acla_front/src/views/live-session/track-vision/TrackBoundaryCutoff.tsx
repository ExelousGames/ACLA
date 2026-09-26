import React, { useRef } from 'react';
import { createCameraProjection } from './camera-projection';
import type { CameraCalibration, GroundPoint } from './track-vision-types';

export const MIN_BOUNDARY_START_M = 0.5;
export const MAX_BOUNDARY_START_M = 100;

/** A cross-track guide in local 3D, projected with the reconstruction camera. */
export default function TrackBoundaryCutoff({ camera, width, value, left, right, onChange }: {
    camera: CameraCalibration; width: number; value: number;
    left: GroundPoint | null; right: GroundPoint | null; onChange: (value: number) => void;
}) {
    const groupRef = useRef<SVGGElement>(null);
    const drag = useRef<{ pointerId: number; z: number } | null>(null);
    const projection = createCameraProjection(camera);
    const height = width * camera.imageHeight / camera.imageWidth;
    const startY = camera.forwardOffsetM + value;
    const start = left ?? { x: camera.lateralOffsetM - 12, y: startY, z: right?.z ?? 0 };
    const end = right ?? { x: camera.lateralOffsetM + 12, y: startY, z: left?.z ?? 0 };
    const project = (point: GroundPoint) => {
        const pixel = projection.localToImage(point);
        return pixel ? { x: pixel.u * width, y: pixel.v * height } : null;
    };
    // Sampling clips portions behind a yawed camera without dropping the visible line.
    const pixels = Array.from({ length: 81 }, (_, i) => project({ x: start.x + (end.x - start.x) * i / 80,
        y: startY, z: start.z + (end.z - start.z) * i / 80 })).filter((point): point is { x: number; y: number } => Boolean(point));
    const path = pixels.map((point, i) => `${i ? 'L' : 'M'}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ');
    const visible = pixels.filter(({ x, y }) => x >= 0 && x <= width && y >= 0 && y <= height);
    const label = visible[Math.floor(visible.length / 2)];
    const change = (distance: number) => onChange(Math.max(MIN_BOUNDARY_START_M, Math.min(MAX_BOUNDARY_START_M, distance)));
    const rayFromPointer = (clientX: number, clientY: number) => {
        const bounds = groupRef.current!.ownerSVGElement!.getBoundingClientRect();
        const scale = Math.min(bounds.width / width, bounds.height / height);
        if (!scale) return null;
        const u = (clientX - bounds.left - (bounds.width - width * scale) / 2) / (width * scale);
        const v = (clientY - bounds.top - (bounds.height - height * scale) / 2) / (height * scale);
        return projection.imageToLocal(Math.max(0, Math.min(1, u)), Math.max(0, Math.min(1, v)), 1);
    };
    const updateFromPointer = (clientX: number, clientY: number) => {
        const ray = rayFromPointer(clientX, clientY);
        if (!ray || !drag.current) return;
        const scale = (drag.current.z - camera.heightM) / (ray.z - camera.heightM);
        if (!Number.isFinite(scale) || scale <= 0) return;
        change(scale * (ray.y - camera.forwardOffsetM));
    };
    return <g ref={groupRef} className="track-vision__boundary-handle" role="slider" tabIndex={0} aria-label="Boundary start line"
        aria-orientation="vertical" aria-valuemin={MIN_BOUNDARY_START_M} aria-valuemax={MAX_BOUNDARY_START_M} aria-valuenow={value}
        aria-valuetext={`${value.toFixed(1)} meters forward from camera; track edges detected beyond this line`}
        onPointerDown={(event) => {
            if (event.button !== 0) return;
            const ray = rayFromPointer(event.clientX, event.clientY);
            if (!ray) return;
            const scale = value / (ray.y - camera.forwardOffsetM);
            if (!Number.isFinite(scale) || scale <= 0) return;
            event.preventDefault();
            event.currentTarget.focus();
            event.currentTarget.setPointerCapture(event.pointerId);
            // Grab at the current Y plane so an elevated road does not jump to Z=0 during dragging.
            drag.current = { pointerId: event.pointerId, z: camera.heightM + scale * (ray.z - camera.heightM) };
        }}
        onPointerMove={(event) => {
            if (drag.current?.pointerId === event.pointerId) updateFromPointer(event.clientX, event.clientY);
        }}
        onPointerUp={(event) => {
            if (drag.current?.pointerId !== event.pointerId) return;
            updateFromPointer(event.clientX, event.clientY);
            drag.current = null;
            event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => { drag.current = null; }}
        onLostPointerCapture={() => { drag.current = null; }}
        onKeyDown={(event) => {
            const next = event.key === 'ArrowUp' || event.key === 'ArrowRight' ? value + 0.5
                : event.key === 'ArrowDown' || event.key === 'ArrowLeft' ? value - 0.5
                    : event.key === 'Home' ? MIN_BOUNDARY_START_M : event.key === 'End' ? MAX_BOUNDARY_START_M : null;
            if (next === null) return;
            event.preventDefault();
            change(next);
        }}>
        <path d={path} fill="none" stroke="transparent" strokeWidth="24" vectorEffect="non-scaling-stroke" />
        <path d={path} fill="none" stroke="#ffbe57" strokeWidth="2" strokeDasharray="8 5" vectorEffect="non-scaling-stroke" />
        {[left, right].map((point, index) => {
            const pixel = point && project(point);
            return pixel ? <circle key={index} aria-label={`${index ? 'Right' : 'Left'} boundary start`} cx={pixel.x} cy={pixel.y}
                r="4" fill={index ? '#57b9ff' : '#37efac'} /> : null;
        })}
        {label && <text x={Math.max(110, Math.min(width - 110, label.x))} y={Math.max(16, label.y - 10)} textAnchor="middle">
            Boundary start · {value.toFixed(1)} m · drag
        </text>}
    </g>;
}
