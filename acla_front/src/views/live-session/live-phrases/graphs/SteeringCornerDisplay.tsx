import React from 'react';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import { isOverlayNonEmptyString, isOverlayRecord } from 'views/floating-chat/overlay-renderer-validation';
import type { SteeringCornerOverlaySnapshot, SteeringCornerPoint } from '../steering-corner-overlay';
import { createSteeringCornerCamera } from './steering-corner-camera';
import analystStyles from 'views/session-shared/driver-expert-comparison/DriverExpertComparisonGraph.module.css';
import './SteeringCornerDisplay.css';

const WIDTH = 480, HEIGHT = 280;

export const SteeringCornerDisplay = ({ snapshot }: { snapshot: SteeringCornerOverlaySnapshot }) => {
    const { referenceLine, leftBoundary, rightBoundary } = snapshot;
    const { project, path, horizon } = createSteeringCornerCamera(referenceLine, WIDTH, HEIGHT);
    const entry = project(referenceLine[0]), exit = project(referenceLine[referenceLine.length - 1]);
    const visible = (point: SteeringCornerPoint | undefined): point is SteeringCornerPoint => !!point
        && point.x >= 16 && point.x <= WIDTH - 16 && point.y >= 20 && point.y <= HEIGHT - 20;
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
                <svg className="steering-corner-display__map" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img"
                    aria-label="Vehicle view of incoming corner reference line and track boundaries">
                    <rect className="steering-corner-display__ground" y={horizon} width={WIDTH} height={HEIGHT - horizon} />
                    <line className="steering-corner-display__horizon" x1="0" x2={WIDTH} y1={horizon} y2={horizon} />
                    <text className="steering-corner-display__view-label" x="14" y="22">Vehicle view</text>
                    {leftBoundary.length > 0 && rightBoundary.length > 0 && <path
                        className="steering-corner-display__road" d={path([...leftBoundary, ...rightBoundary.slice().reverse()], true)} />}
                    <path className="steering-corner-display__left" d={path(leftBoundary)} aria-label="Left boundary" />
                    <path className="steering-corner-display__right" d={path(rightBoundary)} aria-label="Right boundary" />
                    <path className="steering-corner-display__reference" d={path(referenceLine)} aria-label="Reference line" />
                    {visible(entry) && <>
                        <circle cx={entry.x} cy={entry.y} r="4" fill="#80b1ff" />
                        <text x={entry.x} y={entry.y + 17} textAnchor="middle">Entry</text>
                    </>}
                    {visible(exit) && <>
                        <circle cx={exit.x} cy={exit.y} r="3" fill="#e9f4ff" />
                        <text x={exit.x} y={exit.y - 10} textAnchor="middle">Exit</text>
                    </>}
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
