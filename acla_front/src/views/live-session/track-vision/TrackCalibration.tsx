import React, { useMemo } from 'react';
import type { CameraCalibration, CameraParameters } from './track-vision-types';
import { createCameraProjection, validCalibration } from './camera-projection';

const controls: Array<{ key: keyof CameraParameters; label: string; min: number; max: number; step: number }> = [
    { key: 'heightM', label: 'Camera height (m)', min: 0.2, max: 5, step: 0.05 },
    { key: 'pitchDeg', label: 'Pitch down (°)', min: -15, max: 45, step: 0.5 },
    { key: 'horizontalFovDeg', label: 'Horizontal field of view (°)', min: 20, max: 150, step: 1 },
    { key: 'yawDeg', label: 'Yaw right (°)', min: -45, max: 45, step: 0.5 },
    { key: 'lateralOffsetM', label: 'Camera right of car center (m)', min: -3, max: 3, step: 0.05 },
    { key: 'forwardOffsetM', label: 'Camera ahead of car origin (m)', min: -5, max: 5, step: 0.05 },
];
/** Reference ground grid for checking the camera pose; it does not generate scene geometry. */
export function CameraGroundGrid({ camera, applied }: { camera: CameraCalibration; applied: boolean }) {
    if (!validCalibration(camera)) return null;
    const projection = createCameraProjection(camera);
    const line = (points: Array<{ x: number; y: number }>) => points.map((point) => projection.localToImage({ ...point, z: 0 }))
        .filter((point): point is { u: number; v: number } => Boolean(point && point.u >= 0 && point.u <= 1 && point.v >= 0 && point.v <= 1))
        .map(({ u, v }) => `${u * camera.imageWidth},${v * camera.imageHeight}`).join(' ');
    return <svg className="track-vision__ground-grid" viewBox={`0 0 ${camera.imageWidth} ${camera.imageHeight}`}
        aria-label="Projected ground grid" role="img" style={{ color: applied ? '#37efac' : '#ffbe57' }}>
        {[-6, -3, 0, 3, 6].map((x) => <polyline key={`x${x}`} points={line(Array.from({ length: 120 }, (_, i) => ({ x, y: 1 + i * 0.5 })))}
            fill="none" stroke="currentColor" strokeWidth={x === 0 ? 2 : 1} vectorEffect="non-scaling-stroke" />)}
        {[5, 10, 20, 30, 40, 60].map((y) => <polyline key={`y${y}`} points={line(Array.from({ length: 81 }, (_, i) => ({ x: -12 + i * 0.3, y })))}
            fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" />)}
    </svg>;
}

export default function TrackCalibration({ source, draft, applied, showOnCapture, onToggleCapture, onChange, onApply, onClear }: {
    source: HTMLCanvasElement | null;
    draft: CameraParameters;
    applied?: CameraCalibration;
    showOnCapture: boolean;
    onToggleCapture: () => void;
    onChange: (value: CameraParameters) => void;
    onApply: () => void;
    onClear: () => void;
}) {
    const width = source?.width ?? 0, height = source?.height ?? 0;
    const camera = useMemo(() => ({ ...draft, imageWidth: width, imageHeight: height }), [draft, width, height]);
    const valid = validCalibration(camera);
    return <fieldset className="track-vision__alignment">
        <legend>Camera position</legend>
        <p className="track-vision__hint">Enter camera height, angles, horizontal FOV and position relative to your car. A left-seat camera has a negative right offset. Apply these settings to place depth estimates in local 3D.</p>
        <div className="track-vision__camera-controls">
            {controls.map(({ key, label, ...bounds }) => <label key={key}>{label}
                <input aria-label={label} type="number" {...bounds} value={Number.isFinite(draft[key]) ? draft[key] : ''}
                    onChange={(event) => onChange({ ...draft, [key]: event.target.value === '' ? NaN : Number(event.target.value) })} />
            </label>)}
        </div>
        <div className="track-vision__controls">
            <button type="button" className="track-vision__capture-toggle" aria-pressed={showOnCapture}
                disabled={!showOnCapture && (!source || !valid)} onClick={onToggleCapture}>
                {showOnCapture ? 'Disable on capture' : 'Enable on capture'}
            </button>
            <button type="button" disabled={!source || !valid} onClick={onApply}>Apply camera calibration</button>
            <button type="button" disabled={!applied} onClick={onClear}>Clear calibration</button>
            <span>{applied ? 'Calibration applied for this capture.' : 'Draft preview · apply calibration to enable positions.'}</span>
        </div>
        {source && !valid && <p className="track-vision__error">Enter valid camera values within the displayed input limits.</p>}
        <p className="track-vision__hint">Enable on capture shows a reference ground grid. Use a fixed camera with zero roll. Reapply after changing capture, resolution, camera or FOV.</p>
    </fieldset>;
}
