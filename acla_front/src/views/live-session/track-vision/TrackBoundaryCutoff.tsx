import React, { useRef } from 'react';

/** Uses the same contained source-image coordinates as the capture canvas. */
export default function TrackBoundaryCutoff({ width, height, value, onChange }: {
    width: number; height: number; value: number; onChange: (value: number) => void;
}) {
    const svgRef = useRef<SVGSVGElement>(null);
    const dragPointer = useRef<number | null>(null);
    const updateFromPointer = (clientY: number) => {
        const bounds = svgRef.current!.getBoundingClientRect();
        const imageHeight = Math.min(bounds.width / width, bounds.height / height) * height;
        if (!imageHeight) return;
        const top = bounds.top + (bounds.height - imageHeight) / 2;
        onChange(Math.max(0, Math.min(1, (clientY - top) / imageHeight)));
    };
    const y = value * height;
    return <svg ref={svgRef} className="track-vision__boundary-cutoff" viewBox={`0 0 ${width} ${height}`}>
        <rect x="0" y={y} width={width} height={height - y} fill="#ffbe5714" />
        <g className="track-vision__boundary-handle" role="slider" tabIndex={0} aria-label="Boundary start line"
            aria-orientation="vertical" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}
            aria-valuetext={`${Math.round(value * 100)}% from top; track edges detected above this line`}
            onPointerDown={(event) => {
                if (event.button !== 0) return;
                event.preventDefault();
                event.currentTarget.focus();
                event.currentTarget.setPointerCapture(event.pointerId);
                dragPointer.current = event.pointerId;
                updateFromPointer(event.clientY);
            }}
            onPointerMove={(event) => {
                if (dragPointer.current === event.pointerId) updateFromPointer(event.clientY);
            }}
            onPointerUp={(event) => {
                if (dragPointer.current !== event.pointerId) return;
                updateFromPointer(event.clientY);
                dragPointer.current = null;
                event.currentTarget.releasePointerCapture(event.pointerId);
            }}
            onPointerCancel={() => { dragPointer.current = null; }}
            onLostPointerCapture={() => { dragPointer.current = null; }}
            onKeyDown={(event) => {
                const next = event.key === 'ArrowUp' || event.key === 'ArrowLeft' ? value - 0.01
                    : event.key === 'ArrowDown' || event.key === 'ArrowRight' ? value + 0.01
                        : event.key === 'Home' ? 0 : event.key === 'End' ? 1 : null;
                if (next === null) return;
                event.preventDefault();
                onChange(Math.max(0, Math.min(1, next)));
            }}>
            <line x1="0" x2={width} y1={y} y2={y} stroke="transparent" strokeWidth="24" vectorEffect="non-scaling-stroke" />
            <line x1="0" x2={width} y1={y} y2={y} stroke="#ffbe57" strokeWidth="2" strokeDasharray="8 5" vectorEffect="non-scaling-stroke" />
            <text x="12" y={Math.max(24, y - 12)} fontSize={Math.max(18, width / 65)} fill="#ffbe57"
                stroke="#090d13" strokeWidth="3" paintOrder="stroke">Boundary start · drag</text>
        </g>
    </svg>;
}
