import React, { useEffect, useState } from 'react';
import type { BirdsEyeScene } from './birds-eye-scene';
import { GroundPoint, VISION_MAX_AGE_MS } from './track-vision-types';

export default function BirdsEyeView({ scene: ground, hasReconstructedScene, capturedAt }: {
    scene: BirdsEyeScene | null;
    hasReconstructedScene: boolean;
    capturedAt?: number;
}) {
    const [now, setNow] = useState(Date.now);
    useEffect(() => {
        setNow(Date.now());
        if (capturedAt === undefined) return;
        const timer = setTimeout(() => setNow(Date.now()), Math.max(0, capturedAt + VISION_MAX_AGE_MS - Date.now()));
        return () => clearTimeout(timer);
    }, [capturedAt]);
    const stale = capturedAt !== undefined && now >= capturedAt + VISION_MAX_AGE_MS;
    const points = ground ? [...ground.leftBoundary.flat(), ...ground.rightBoundary.flat(), ...ground.centerline.flat(),
        ...ground.cars.map(({ position }) => position)] : [];
    let minX = -10, maxX = 10, maxY = 20;
    for (const { x, y } of points) {
        minX = Math.min(minX, x - 2); maxX = Math.max(maxX, x + 2); maxY = Math.max(maxY, y + 3);
    }
    const minY = -5, scale = Math.min(620 / (maxX - minX), 370 / (maxY - minY));
    const map = ({ x, y }: Pick<GroundPoint, 'x' | 'y'>) => ({
        x: 360 + (x - (minX + maxX) / 2) * scale, y: 230 - (y - (minY + maxY) / 2) * scale,
    });
    const driver = map({ x: 0, y: 0 });
    const rows = Array.from({ length: Math.floor(maxY / 10) + 1 }, (_, i) => i * 10);
    const hasEdges = Boolean(ground && (ground.leftBoundary.length || ground.rightBoundary.length));
    return <section className="track-vision__reconstruction" aria-label="Bird's-eye view">
        <div className="track-vision__legend" aria-label="Bird's-eye view legend">
            <span><i style={{ background: '#37efac' }} />Left track boundary</span>
            <span><i style={{ background: '#57b9ff' }} />Right track boundary</span>
            <span><i style={{ background: '#f4f7ff' }} />Your car</span>
            <span><i style={{ background: '#ffbe57' }} />Car</span>
            <span><i style={{ background: '#ce87ff' }} />Car pack</span>
        </div>
        <div className="track-vision__scene" style={{ aspectRatio: '720 / 460' }}>
            {ground ? <svg viewBox="0 0 720 460" role="img" aria-label="Top-down track boundaries and cars">
                <title>Estimated top-down scene. Your car is at the origin, with forward pointing up.</title>
                <g stroke="#ffffff12" fill="#a4adbb" fontSize="11">
                    {rows.map((y) => <g key={y}>
                        <line x1={map({ x: minX, y }).x} x2={map({ x: maxX, y }).x} y1={map({ x: 0, y }).y} y2={map({ x: 0, y }).y} />
                        <text x="12" y={map({ x: 0, y }).y + 4} stroke="none">{y} m</text>
                    </g>)}
                    <line x1={driver.x} x2={driver.x} y1={map({ x: 0, y: maxY }).y} y2={map({ x: 0, y: minY }).y} strokeDasharray="3 6" />
                </g>
                {(['leftBoundary', 'rightBoundary', 'centerline'] as const).map((side) => <g key={side}
                    aria-label={side === 'leftBoundary' ? 'Top-down left boundary' : side === 'rightBoundary' ? 'Top-down right boundary' : 'Top-down middle line'}
                    fill="none" stroke={side === 'leftBoundary' ? '#37efac' : side === 'rightBoundary' ? '#57b9ff' : '#f4f7ff'}
                    strokeWidth={side === 'centerline' ? 1.5 : 3} strokeDasharray={side === 'centerline' ? '6 5' : undefined}
                    strokeLinecap="round" strokeLinejoin="round">
                    {ground[side].map((line, index) => <polyline key={index} points={line.map((point) => {
                        const { x, y } = map(point); return `${x.toFixed(2)},${y.toFixed(2)}`;
                    }).join(' ')} />)}
                </g>)}
                <g aria-label="Top-down traffic">
                    {ground.cars.map(({ position, pack, confidence }, index) => {
                        const { x, y } = map(position), color = pack ? '#ce87ff' : '#ffbe57';
                        const label = `${pack ? 'Car pack' : 'Car'} ${index + 1}`;
                        return <g key={index} aria-label={label}>
                            <title>{label} · {Math.round(confidence * 100)}% confidence · estimated {position.y.toFixed(1)} m ahead, {Math.abs(position.x).toFixed(1)} m {position.x < 0 ? 'left' : 'right'}</title>
                            <circle cx={x} cy={y} r={pack ? 9 : 6} fill={color} fillOpacity="0.25" stroke={color}
                                strokeWidth="2" strokeDasharray={pack ? '3 2' : undefined} />
                            <text x={x > 600 ? x - 12 : x + 12} y={y + 4} textAnchor={x > 600 ? 'end' : 'start'}
                                fill={color} fontSize="11" stroke="#090d13" strokeWidth="3" paintOrder="stroke">{label}</text>
                        </g>;
                    })}
                </g>
                <g aria-label="Your car at the origin" transform={`translate(${driver.x},${driver.y})`}>
                    <title>Your car · origin · forward points up</title>
                    <rect x="-7" y="-11" width="14" height="22" rx="4" fill="#f4f7ff" stroke="#090d13" strokeWidth="2" />
                    <path d="M -4,-3 L 0,-7 L 4,-3" fill="none" stroke="#090d13" strokeWidth="2" />
                    <text x={driver.x > 600 ? -13 : 13} y="4" textAnchor={driver.x > 600 ? 'end' : 'start'}
                        fill="#f4f7ff" fontSize="12" stroke="#090d13" strokeWidth="3" paintOrder="stroke">Your car</text>
                </g>
                <text x="704" y="24" textAnchor="end" fill="#a4adbb" fontSize="12">↑ Forward</text>
            </svg> : <div className="track-vision__empty"><strong>{hasReconstructedScene ? 'Apply camera calibration in Camera position' : 'Waiting for reconstructed scene'}</strong></div>}
        </div>
        <p className="track-vision__hint" aria-label="Bird's-eye view status">{!hasReconstructedScene ? 'Waiting for scene.' : !ground
            ? 'Set and apply the camera position to construct the top-down view.'
            : stale ? 'Showing last frame (stale).'
                : hasEdges ? 'Track boundaries and cars from the reconstructed scene.'
                    : ground.cars.length ? 'Detected traffic; no projectable track boundaries.' : 'No projectable track boundaries or traffic.'}</p>
        {ground && <p className="track-vision__hint">Flat-road estimate from camera calibration. Your car is the origin; distances and traffic positions are approximate.</p>}
        {!!ground?.unplacedCars && <p className="track-vision__hint">{ground.unplacedCars} car / car-pack detection(s) could not be placed on the ground.</p>}
    </section>;
}
