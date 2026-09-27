import React from 'react';
import TrackBoundaryCutoff from './TrackBoundaryCutoff';

export default function CaptureBoundaryStart({ width, height, value, onChange }: {
    width: number; height: number; value: number; onChange: (value: number) => void;
}) {
    if (width <= 0 || height <= 0) return null;
    const viewWidth = 800, viewHeight = viewWidth * height / width;
    return <>
        <svg className="track-vision__boundary-overlay" viewBox={`0 0 ${viewWidth} ${viewHeight}`}
            role="group" aria-label="Capture boundary start">
            <TrackBoundaryCutoff width={viewWidth} height={viewHeight} value={value} onChange={onChange} />
        </svg>
        <div className="track-vision__controls track-vision__boundary-controls">
            <label>Detection start {Math.round(value * 100)}% from top
                <input aria-label="Boundary start" type="range" min="0" max="100" step="1"
                    value={Math.round(value * 100)} onChange={(event) => onChange(Number(event.target.value) / 100)} />
            </label>
        </div>
    </>;
}
