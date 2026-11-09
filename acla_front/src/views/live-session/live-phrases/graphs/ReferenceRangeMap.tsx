import React, { useMemo } from 'react';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { referenceRangeSegments } from '../phrase-reference-guidance';
import './ReferenceRangeMap.css';

const WIDTH = 560, HEIGHT = 300, PADDING = 28;

const ReferenceRangeMap = ({ reference }: { reference: TopLapReferenceEntry | null }) => {
    const map = useMemo(() => {
        const ranges = referenceRangeSegments(reference);
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
    }, [reference]);

    return (
        <section className="reference-range-map" aria-label="Deceleration and direction ranges">
            <h3>Deceleration &amp; direction ranges</h3>
            {map ? <>
                <svg className="reference-range-map__track" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
                    aria-label={`Deceleration and direction ranges for ${reference!.track}`}>
                    <title>{reference!.track} reference ranges</title>
                    <desc>Orange highlights decreasing reference speed. Blue highlights direction changes. Both colors remain visible where ranges overlap.</desc>
                    <path className="reference-range-map__line" d={map.line} aria-label="Reference line" />
                    <path className="reference-range-map__deceleration" d={map.deceleration} aria-label="Deceleration ranges" />
                    <path className="reference-range-map__direction" d={map.direction} aria-label="Direction-change ranges" />
                    <circle className="reference-range-map__start" cx={map.start.x} cy={map.start.y} r="4" />
                    <text x={map.start.x} y={map.start.y - 12} textAnchor="middle">Lap start</text>
                </svg>
                <div className="reference-range-map__legend" aria-label="Reference range map legend">
                    <span data-line="reference">Reference line</span>
                    <span data-line="deceleration">Deceleration</span>
                    <span data-line="direction">Direction change</span>
                </div>
                <p className="live-phrases__hint">Ranges follow the selected driving aggression. Both colors appear where ranges overlap.</p>
                {!map.speedAvailable && <p className="live-phrases__hint" role="status">Reference speed unavailable. Deceleration ranges cannot be shown.</p>}
            </> : <p className="live-phrases__hint" role="status">Open Live Map with a reference lap to see deceleration and direction ranges.</p>}
        </section>
    );
};

export default React.memo(ReferenceRangeMap);
