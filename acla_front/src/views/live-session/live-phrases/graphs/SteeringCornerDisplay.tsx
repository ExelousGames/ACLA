import React from 'react';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import { isOverlayNonEmptyString, isOverlayRecord } from 'views/floating-chat/overlay-renderer-validation';
import type { SteeringCornerOverlaySnapshot, SteeringCornerPoint } from '../steering-corner-overlay';
import analystStyles from 'views/session-shared/driver-expert-comparison/DriverExpertComparisonGraph.module.css';
import './SteeringCornerDisplay.css';

const WIDTH = 480, HEIGHT = 280, PADDING = 28;

export const SteeringCornerDisplay = ({ snapshot }: { snapshot: SteeringCornerOverlaySnapshot }) => {
    const gridId = React.useId();
    const { referenceLine, leftBoundary, rightBoundary } = snapshot;
    const points = [...referenceLine, ...leftBoundary, ...rightBoundary];
    const xs = points.map(({ x }) => x), ys = points.map(({ y }) => y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = Math.min((WIDTH - PADDING * 2) / Math.max(1, maxX - minX),
        (HEIGHT - PADDING * 2) / Math.max(1, maxY - minY));
    const project = ({ x, y }: SteeringCornerPoint) => ({
        x: WIDTH / 2 + (x - (minX + maxX) / 2) * scale,
        y: HEIGHT / 2 - (y - (minY + maxY) / 2) * scale,
    });
    const path = (line: SteeringCornerPoint[]) => line.map((point) => {
        const { x, y } = project(point);
        return `${x},${y}`;
    }).join(' ');
    const entry = project(referenceLine[0]), exit = project(referenceLine[referenceLine.length - 1]);
    const missing = [!leftBoundary.length && 'Left boundary', !rightBoundary.length && 'Right boundary'].filter(Boolean);
    return (
        <section className={`${analystStyles.root} steering-corner-display`} aria-label={snapshot.title}>
            <header className={`${analystStyles.header} steering-corner-display__header`}>
                <div className={analystStyles.titleBlock}>
                    <span className="steering-corner-display__eyebrow">Incoming corner · {snapshot.circuitName}</span>
                    <h2 className={analystStyles.title}>{snapshot.title}</h2>
                </div>
                <span className={`${analystStyles.replayReadout} steering-corner-display__status`}>
                    <span className={`${analystStyles.statusDot} ${analystStyles.statusDotComplete}`} aria-hidden="true" />
                    <span className={analystStyles.replayStatus}>Steering active</span>
                </span>
            </header>
            <div className={analystStyles.trackPanel}>
                <svg className="steering-corner-display__map" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
                    aria-label="Incoming corner reference line and track boundaries">
                    <defs>
                        <pattern id={gridId} width="24" height="24" patternUnits="userSpaceOnUse">
                            <path d="M 24 0 H 0 V 24" fill="none" stroke="#abb0b8" strokeOpacity="0.08" />
                        </pattern>
                    </defs>
                    <rect width={WIDTH} height={HEIGHT} fill={`url(#${gridId})`} />
                    {leftBoundary.length > 0 && rightBoundary.length > 0 && <polygon
                        className="steering-corner-display__road" points={path([...leftBoundary, ...rightBoundary.slice().reverse()])} />}
                    <polyline className="steering-corner-display__left" points={path(leftBoundary)} aria-label="Left boundary" />
                    <polyline className="steering-corner-display__right" points={path(rightBoundary)} aria-label="Right boundary" />
                    <polyline className="steering-corner-display__reference" points={path(referenceLine)} aria-label="Reference line" />
                    <circle cx={entry.x} cy={entry.y} r="4" fill="#80b1ff" />
                    <text x={entry.x} y={entry.y + 17} textAnchor="middle">Entry</text>
                    <circle cx={exit.x} cy={exit.y} r="4" fill="#e9f4ff" />
                    <text x={exit.x} y={exit.y - 10} textAnchor="middle">Exit</text>
                </svg>
            </div>
            <div className="steering-corner-display__legend" aria-label="Map legend">
                <span data-line="reference">Reference line</span>
                <span data-line="left">Left boundary</span>
                <span data-line="right">Right boundary</span>
            </div>
            {missing.length > 0 && <p className="steering-corner-display__missing" role="status">{missing.join(' and ')} unavailable in Circuit Map.</p>}
        </section>
    );
};

const isLine = (value: unknown): value is SteeringCornerPoint[] => Array.isArray(value)
    && (value.length === 0 || value.length >= 2)
    && value.every((point) => isOverlayRecord(point) && typeof point.x === 'number' && Number.isFinite(point.x)
        && typeof point.y === 'number' && Number.isFinite(point.y));

export const steeringCornerOverlayRenderer: AiOverlayRenderer<SteeringCornerOverlaySnapshot> = {
    componentType: 'live_steering_corner',
    validateSnapshot: (snapshot): snapshot is SteeringCornerOverlaySnapshot => isOverlayRecord(snapshot)
        && isOverlayNonEmptyString(snapshot.title) && typeof snapshot.circuitName === 'string'
        && isLine(snapshot.referenceLine) && snapshot.referenceLine.length >= 2
        && isLine(snapshot.leftBoundary) && isLine(snapshot.rightBoundary),
    renderOverlay: (snapshot, status) => status === 'folded' ? snapshot.title : <SteeringCornerDisplay snapshot={snapshot} />,
    dimensions: {
        expanded: { width: 532, height: 410 },
        folded: { width: 360, height: 58 },
        focus: { width: 532, height: 410 },
    },
};
