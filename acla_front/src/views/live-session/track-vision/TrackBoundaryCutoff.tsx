import React, { useRef } from 'react';

/** A horizontal image-row limit for boundary detection, independent of camera pose and depth. */
export default function TrackBoundaryCutoff({ width, height, value, onChange }: {
    width: number; height: number; value: number; onChange: (value: number) => void;
}) {
    const groupRef = useRef<SVGGElement>(null);
    const drag = useRef<{ pointerId: number; offsetV: number } | null>(null);
    const y = value * height;
    const path = `M0,${y} L${width},${y}`;
    const change = (v: number) => onChange(Math.max(0, Math.min(1, v)));
    const rowFromPointer = (clientY: number) => {
        const bounds = groupRef.current!.ownerSVGElement!.getBoundingClientRect();
        const scale = Math.min(bounds.width / width, bounds.height / height);
        if (!scale) return null;
        return (clientY - bounds.top - (bounds.height - height * scale) / 2) / (height * scale);
    };
    const updateFromPointer = (clientY: number) => {
        const v = rowFromPointer(clientY);
        if (v !== null && drag.current) change(v - drag.current.offsetV);
    };
    const endDrag = () => { drag.current = null; };
    return <g ref={groupRef} className="track-vision__boundary-handle"
        role="slider" tabIndex={0} aria-label="Boundary start line" aria-orientation="vertical"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}
        aria-valuetext={`${Math.round(value * 100)}% from top of capture; boundary detection scans upward`}
        onPointerDown={(event) => {
            if (event.button !== 0) return;
            const v = rowFromPointer(event.clientY);
            if (v === null) return;
            event.preventDefault();
            event.currentTarget.focus();
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = { pointerId: event.pointerId, offsetV: v - value };
        }}
        onPointerMove={(event) => {
            if (drag.current?.pointerId === event.pointerId) updateFromPointer(event.clientY);
        }}
        onPointerUp={(event) => {
            if (drag.current?.pointerId !== event.pointerId) return;
            updateFromPointer(event.clientY);
            endDrag();
            event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={endDrag}
        onLostPointerCapture={endDrag}
        onKeyDown={(event) => {
            const next = event.key === 'ArrowUp' || event.key === 'ArrowLeft' ? value - 0.01
                : event.key === 'ArrowDown' || event.key === 'ArrowRight' ? value + 0.01
                    : event.key === 'Home' ? 0 : event.key === 'End' ? 1 : null;
            if (next === null) return;
            event.preventDefault();
            change(next);
        }}>
        <path d={path} fill="none" stroke="transparent" strokeWidth="24" vectorEffect="non-scaling-stroke" />
        <path d={path} fill="none" stroke="#ffbe57" strokeWidth="2" strokeDasharray="8 5" vectorEffect="non-scaling-stroke" />
        <text x={width / 2} y={Math.max(16, y - 10)} textAnchor="middle">
            Boundary detection start · drag
        </text>
    </g>;
}
