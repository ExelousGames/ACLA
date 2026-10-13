import React, { useEffect, useRef, useState } from 'react';
import type { ReconstructedScene } from './reconstructed-scene';
import { VISION_MAX_AGE_MS } from './track-vision-types';

export default function ReconstructedSceneView({ scene, source, capturedAt }: {
    scene: ReconstructedScene | null;
    source: HTMLCanvasElement | null;
    capturedAt?: number;
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const width = source?.width || scene?.width || 1280;
    const height = source?.height || scene?.height || 720;
    const [now, setNow] = useState(Date.now);
    useEffect(() => {
        const context = canvasRef.current?.getContext('2d');
        if (!context) return;
        context.clearRect(0, 0, width, height);
        if (source) context.drawImage(source, 0, 0);
    }, [source, capturedAt, width, height]);
    useEffect(() => {
        setNow(Date.now());
        if (capturedAt === undefined) return;
        const timer = setTimeout(() => setNow(Date.now()), Math.max(0, capturedAt + VISION_MAX_AGE_MS - Date.now()));
        return () => clearTimeout(timer);
    }, [capturedAt]);
    const hasEdges = Boolean(scene && (scene.leftBoundary.length || scene.rightBoundary.length));
    const fontSize = Math.max(12, width / 90);
    const stale = capturedAt !== undefined && now >= capturedAt + VISION_MAX_AGE_MS;
    return <section className="track-vision__reconstruction" aria-label="Reconstructed scene">
        <div className="track-vision__legend" aria-label="Reconstructed scene legend">
            <span><i style={{ background: '#37efac55' }} />Track ribbon</span>
            <span><i style={{ background: '#37efac' }} />Left track boundary</span>
            <span><i style={{ background: '#57b9ff' }} />Right track boundary</span>
            <span><i style={{ background: '#f4f7ff' }} />Track middle line</span>
            <span><i style={{ background: '#ffbe57' }} />Car</span>
            <span><i style={{ background: '#ce87ff' }} />Car pack</span>
        </div>
        <div className="track-vision__scene" style={{ aspectRatio: `${width} / ${height}` }}>
            <canvas ref={canvasRef} width={width} height={height} hidden={!source}
                aria-label="Captured window scene" />
            <svg viewBox={`0 0 ${width} ${height}`}
                role="img" aria-label="2D reconstructed scene">
                <title>Fitted track ribbons, boundaries, middle line, cars and car packs in camera image space</title>
                {scene && <g aria-label="Track ribbons">
                    {scene.ribbons.map(({ pairs }, index) => <g key={index} aria-label={`Track ribbon ${index + 1}`}>
                        <polygon fill="#37efac" fillOpacity="0.08" points={[
                            ...pairs.map(({ left }) => left), ...pairs.map(({ right }) => right).reverse(),
                        ].map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')} />
                        {pairs.map(({ left, right }, pairIndex) => <g key={pairIndex}>
                            <line x1={left.x} y1={left.y} x2={right.x} y2={right.y}
                                stroke="#b8ffdf" strokeOpacity="0.25" strokeWidth="1" vectorEffect="non-scaling-stroke" />
                            <circle cx={left.x} cy={left.y} r={width / 500} fill="#37efac" />
                            <circle cx={right.x} cy={right.y} r={width / 500} fill="#57b9ff" />
                        </g>)}
                    </g>)}
                </g>}
                {scene && (['leftBoundary', 'rightBoundary'] as const).map((side) => <g key={side}
                    aria-label={side === 'leftBoundary' ? 'Left track boundary' : 'Right track boundary'}
                    fill="none" stroke={side === 'leftBoundary' ? '#37efac' : '#57b9ff'} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    {scene[side].map((line, index) => <polyline key={index} vectorEffect="non-scaling-stroke"
                        points={line.map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')} />)}
                </g>)}
                {scene && <g aria-label="Track middle line" fill="none" stroke="#f4f7ff" strokeWidth="2"
                    strokeDasharray="6 4" strokeLinecap="round" strokeLinejoin="round">
                    {scene.centerline.map((line, index) => <polyline key={index} vectorEffect="non-scaling-stroke"
                        points={line.map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')} />)}
                </g>}
                {scene && scene.cars.length > 0 && <g aria-label="Reconstructed cars">
                    {scene.cars.map(({ box: [left, top, right, bottom], pack, confidence }, index) => {
                        const color = pack ? '#ce87ff' : '#ffbe57';
                        const label = `${pack ? 'Car pack' : 'Car'} · ${Math.round(confidence * 100)}%`;
                        return <g key={index} aria-label={`${label} confidence`}>
                            <title>{label} confidence</title>
                            <rect x={left} y={top} width={right - left} height={bottom - top}
                                fill={color} fillOpacity="0.12" stroke={color} strokeWidth="2"
                                strokeDasharray={pack ? '6 4' : undefined} vectorEffect="non-scaling-stroke" />
                            <text x={Math.min(left + 4, Math.max(4, width - label.length * fontSize * 0.65))}
                                y={Math.min(height - 4, top + fontSize + 4)} fontSize={fontSize}
                                fill={color} stroke="#090d13" strokeWidth="3" strokeLinejoin="round" paintOrder="stroke">
                                {label}
                            </text>
                        </g>;
                    })}
                </g>}
            </svg>
        </div>
        <p className="track-vision__hint" aria-label="Reconstructed scene status">{!scene
            ? 'Waiting for scene.'
            : stale ? 'Showing last frame (stale).'
                : !hasEdges ? scene.cars.length ? 'Detected cars and car packs in 2D.'
                    : 'No visible track boundaries or traffic.'
                    : 'Track ribbons fitted to detected edges · 50 point pairs per section.'}</p>
    </section>;
}
