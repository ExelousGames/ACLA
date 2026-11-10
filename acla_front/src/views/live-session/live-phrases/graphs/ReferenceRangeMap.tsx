import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { referenceRangeSegments, type ReferenceCornerBounds } from '../phrase-reference-guidance';
import './ReferenceRangeMap.css';

const WIDTH = 560, HEIGHT = 300, PADDING = 28;
const MIN_ZOOM = 0.35, MAX_ZOOM = 6, ZOOM_STEP = 1.25;
const INITIAL_VIEW = { x: 0, y: 0, zoom: 1 };

const mapPoint = (svg: SVGSVGElement, clientX: number, clientY: number) => {
    const rect = svg.getBoundingClientRect();
    const scale = Math.min(rect.width / WIDTH, rect.height / HEIGHT);
    return {
        x: (clientX - rect.left - (rect.width - WIDTH * scale) / 2) / scale,
        y: (clientY - rect.top - (rect.height - HEIGHT * scale) / 2) / scale,
    };
};

const ReferenceRangeMap = ({ reference, corners }: { reference: TopLapReferenceEntry | null; corners?: readonly ReferenceCornerBounds[] }) => {
    const svgRef = useRef<SVGSVGElement>(null);
    const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
    const [view, setView] = useState(INITIAL_VIEW);
    const [isDragging, setIsDragging] = useState(false);
    const stopDragging = useCallback(() => {
        const pointerId = dragRef.current?.pointerId;
        dragRef.current = null;
        setIsDragging(false);
        if (pointerId !== undefined && svgRef.current?.hasPointerCapture(pointerId)) {
            svgRef.current.releasePointerCapture(pointerId);
        }
    }, []);
    const resetView = useCallback(() => {
        stopDragging();
        setView(INITIAL_VIEW);
    }, [stopDragging]);
    const zoomAt = useCallback((factor: number, point = { x: WIDTH / 2, y: HEIGHT / 2 }) => {
        setView((current) => {
            const zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, current.zoom * factor));
            const ratio = zoom / current.zoom;
            return { x: point.x - (point.x - current.x) * ratio, y: point.y - (point.y - current.y) * ratio, zoom };
        });
    }, []);

    const map = useMemo(() => {
        const ranges = referenceRangeSegments(reference, corners);
        if (!ranges || !ranges.segments.length) return null;
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        ranges.segments.forEach(({ start, end }) => [start, end].forEach(({ x, y }) => {
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }));
        const scale = Math.min((WIDTH - 2 * PADDING) / Math.max(1, maxX - minX),
            (HEIGHT - 2 * PADDING) / Math.max(1, maxY - minY));
        const project = ({ x, y }: { x: number; y: number }) => ({
            x: WIDTH / 2 + (x - (minX + maxX) / 2) * scale,
            y: HEIGHT / 2 - (y - (minY + maxY) / 2) * scale,
        });
        const paths = ranges.segments.map((segment) => {
            const a = project(segment.start), b = project(segment.end);
            return { ...segment, path: `M${a.x.toFixed(2)},${a.y.toFixed(2)}L${b.x.toFixed(2)},${b.y.toFixed(2)}` };
        });
        return {
            line: paths.map(({ path }) => path).join(' '),
            deceleration: paths.filter(({ deceleration }) => deceleration !== undefined && deceleration > 0).map(({ path }) => path).join(' '),
            direction: paths.filter(({ changesDirection }) => changesDirection).map(({ path }) => path).join(' '),
            start: project(ranges.segments[0].start),
            speedAvailable: ranges.speedAvailable,
        };
    }, [reference, corners]);

    // Keep the viewport when only the aggression-dependent speed ranges change.
    useEffect(() => resetView(), [map?.line, reference?.game, reference?.track, resetView]);

    const hasMap = Boolean(map);
    useEffect(() => {
        const svg = svgRef.current;
        if (!svg) return;
        const handleWheel = (event: WheelEvent) => {
            event.preventDefault();
            if (event.deltaY === 0) return;
            zoomAt(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP, mapPoint(svg, event.clientX, event.clientY));
        };
        // React's delegated wheel listener is passive; this also prevents sidebar scrolling.
        svg.addEventListener('wheel', handleWheel, { passive: false });
        return () => svg.removeEventListener('wheel', handleWheel);
    }, [hasMap, zoomAt]);

    const finishDrag = (event: React.PointerEvent<SVGSVGElement>) => {
        if (dragRef.current?.pointerId === event.pointerId) stopDragging();
    };

    return (
        <section className="reference-range-map" aria-label="Deceleration and direction ranges">
            <h3>Deceleration &amp; direction ranges</h3>
            {map ? <>
                <div className="reference-range-map__controls" role="group" aria-label="Reference range map controls">
                    <button type="button" aria-label="Zoom out" disabled={view.zoom <= MIN_ZOOM} onClick={() => zoomAt(1 / ZOOM_STEP)}>−</button>
                    <span aria-label="Zoom level">{Math.round(view.zoom * 100)}%</span>
                    <button type="button" aria-label="Zoom in" disabled={view.zoom >= MAX_ZOOM} onClick={() => zoomAt(ZOOM_STEP)}>+</button>
                    <button type="button" onClick={resetView}>Fit map</button>
                </div>
                <svg ref={svgRef} className="reference-range-map__track" data-dragging={isDragging}
                    viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" tabIndex={0}
                    aria-label={`Deceleration and direction ranges for ${reference!.track}`}
                    onPointerDown={(event) => {
                        if (event.button !== 0 || dragRef.current) return;
                        event.preventDefault();
                        event.currentTarget.focus();
                        dragRef.current = { pointerId: event.pointerId, ...mapPoint(event.currentTarget, event.clientX, event.clientY) };
                        event.currentTarget.setPointerCapture(event.pointerId);
                        setIsDragging(true);
                    }}
                    onPointerMove={(event) => {
                        const drag = dragRef.current;
                        if (!drag || drag.pointerId !== event.pointerId) return;
                        const point = mapPoint(event.currentTarget, event.clientX, event.clientY);
                        const dx = point.x - drag.x, dy = point.y - drag.y;
                        dragRef.current = { pointerId: event.pointerId, ...point };
                        setView((current) => ({ ...current, x: current.x + dx, y: current.y + dy }));
                    }}
                    onPointerUp={finishDrag} onPointerCancel={finishDrag} onLostPointerCapture={finishDrag}
                    onKeyDown={(event) => {
                        const directions: Record<string, [number, number]> = { ArrowLeft: [28, 0], ArrowRight: [-28, 0], ArrowUp: [0, 28], ArrowDown: [0, -28] };
                        const pan = directions[event.key];
                        if (pan) setView((current) => ({ ...current, x: current.x + pan[0], y: current.y + pan[1] }));
                        else if (event.key === '+' || event.key === '=') zoomAt(ZOOM_STEP);
                        else if (event.key === '-') zoomAt(1 / ZOOM_STEP);
                        else if (event.key === 'Home' || event.key === '0') resetView();
                        else return;
                        event.preventDefault();
                    }}>
                    <title>{reference!.track} reference ranges</title>
                    <desc>Orange highlights decreasing reference speed. Blue highlights direction changes. Both colors remain visible where ranges overlap. Scroll to zoom and drag to move. Arrow keys pan, plus and minus zoom, and Home fits the map.</desc>
                    <g data-testid="reference-range-map-viewport" transform={`translate(${view.x} ${view.y}) scale(${view.zoom})`}>
                        <path className="reference-range-map__line" d={map.line} aria-label="Reference line" />
                        <path className="reference-range-map__deceleration" d={map.deceleration} aria-label="Deceleration ranges" />
                        <path className="reference-range-map__direction" d={map.direction} aria-label="Direction-change ranges" />
                        <circle className="reference-range-map__start" cx={map.start.x} cy={map.start.y} r="4" />
                        <text x={map.start.x} y={map.start.y - 12} textAnchor="middle">Lap start</text>
                    </g>
                </svg>
                <div className="reference-range-map__legend" aria-label="Reference range map legend">
                    <span data-line="reference">Reference line</span>
                    <span data-line="deceleration">Deceleration</span>
                    <span data-line="direction">Direction change</span>
                </div>
                <p className="live-phrases__hint">Scroll to zoom. Drag to move. Ranges follow the selected driving aggression. Both colors appear where ranges overlap.</p>
                {!map.speedAvailable && <p className="live-phrases__hint" role="status">Reference speed unavailable. Deceleration ranges cannot be shown.</p>}
            </> : <p className="live-phrases__hint" role="status">Open Live Map with a reference lap to see deceleration and direction ranges.</p>}
        </section>
    );
};

export default React.memo(ReferenceRangeMap);
