import React from 'react';
import type { DriverExpertComparisonLabelGroup } from './DriverExpertComparisonGraph';
import styles from './DriverExpertComparisonGraph.module.css';

export interface DriverExpertComparisonLabelRange {
    label: string;
    startIndex: number;
    /** Exclusive original telemetry index. */
    endIndex: number;
    category?: DriverExpertComparisonLabelGroup['category'];
}

interface Point { x: number; y: number }
interface TrackPoint extends Point { svgX: number; svgY: number }

export interface RoadsideLabelGeometry extends DriverExpertComparisonLabelRange {
    start?: TrackPoint;
    end?: TrackPoint;
}

export const isComparisonLabelRange = (value: unknown): value is DriverExpertComparisonLabelRange => {
    if (!value || typeof value !== 'object') return false;
    const range = value as DriverExpertComparisonLabelRange;
    return typeof range.label === 'string' && Boolean(range.label.trim())
        && Number.isInteger(range.startIndex) && range.startIndex >= 0
        && Number.isInteger(range.endIndex) && range.endIndex > range.startIndex
        && (range.category === undefined || ['mistakes', 'expert', 'recovery'].includes(range.category));
};

export const buildRoadsideLabelGeometry = (
    ranges: readonly DriverExpertComparisonLabelRange[],
    stream: readonly { sourceIndex?: number; trajectory?: Point }[],
    project: (point: Point | undefined) => TrackPoint | undefined,
): RoadsideLabelGeometry[] => {
    const indexed = stream.filter((point): point is { sourceIndex: number; trajectory?: Point } => (
        point.sourceIndex !== undefined && Number.isFinite(point.sourceIndex)
    ));
    if (!indexed.length) return [];
    const atIndex = (index: number): TrackPoint | undefined => {
        const right = indexed.findIndex((point) => point.sourceIndex >= index);
        if (right < 0) return undefined;
        const next = indexed[right];
        if (next.sourceIndex === index) return project(next.trajectory);
        const previous = indexed[right - 1];
        if (!previous?.trajectory || !next.trajectory) return undefined;
        const ratio = (index - previous.sourceIndex) / (next.sourceIndex - previous.sourceIndex);
        return project({
            x: previous.trajectory.x + (next.trajectory.x - previous.trajectory.x) * ratio,
            y: previous.trajectory.y + (next.trajectory.y - previous.trajectory.y) * ratio,
        });
    };
    return ranges.filter(isComparisonLabelRange).flatMap((range) => {
        const start = Math.max(range.startIndex, indexed[0].sourceIndex);
        const end = Math.min(range.endIndex, indexed[indexed.length - 1].sourceIndex);
        if (start > end || start >= range.endIndex) return [];
        return [{ ...range, start: atIndex(start), end: atIndex(end) }];
    });
};

const wrapLabel = (label: string): string[] => {
    const lines: string[] = [];
    for (const word of label.match(/\S{1,20}/g) ?? []) {
        const last = lines.length - 1;
        if (last >= 0 && lines[last].length + word.length < 20) lines[last] += ` ${word}`;
        else lines.push(word);
    }
    return lines;
};

// World-space distances keep fading independent of camera depth and viewport size.
const APPROACH_FADE_DISTANCE = 180;
const RELEASE_FADE_DISTANCE = 20;

export const DriverExpertComparisonRoadSigns: React.FC<{
    ranges: readonly RoadsideLabelGeometry[];
    sourceIndex?: number;
    driverPosition?: TrackPoint;
    camera: { project: (point: TrackPoint | undefined) => (TrackPoint & { perspectiveScale?: number }) | undefined };
    viewportWidth: number;
    viewportHeight: number;
    scale?: number;
}> = ({ ranges, sourceIndex, driverPosition, camera, viewportWidth, viewportHeight, scale = 1 }) => {
    const isFollowing = (range: RoadsideLabelGeometry) => sourceIndex !== undefined
        && sourceIndex >= range.startIndex && sourceIndex < range.endIndex;
    const active = ranges.filter(isFollowing).sort((a, b) => a.startIndex - b.startIndex);
    const labels = active.filter((range, index) => active.findIndex((other) => (
        other.label.trim() === range.label.trim() && other.category === range.category
    )) === index);
    // Waiting and released signs stay at fixed trajectory points. Only the active
    // bundle shares the player's moving world position; no screen position is stored.
    const signs = [
        ...(labels.length ? [{ key: 'following', labels, position: driverPosition, state: 'following' }] : []),
        ...ranges.flatMap((range, index) => {
            if (isFollowing(range)) return [];
            const released = sourceIndex !== undefined && sourceIndex >= range.endIndex;
            return [{
                key: `${range.category}-${range.label}-${range.startIndex}-${range.endIndex}-${index}`,
                labels: [range],
                position: released ? range.end : range.start,
                state: released ? 'released' : 'waiting',
            }];
        }),
    ];
    // Paint distant signs behind nearby signs.
    const projectedSigns = signs.map((sign) => ({ ...sign, anchor: camera.project(sign.position) }))
        .sort((a, b) => (b.anchor?.perspectiveScale ?? 1) - (a.anchor?.perspectiveScale ?? 1));
    const annotations = projectedSigns.flatMap((sign) => {
        const { anchor } = sign;
        // Let track-bound signs enter and leave the camera view with the road.
        if (!anchor || (anchor.perspectiveScale ?? 1) <= 0 || anchor.svgX < 0 || anchor.svgX > viewportWidth
            || anchor.svgY < 0 || anchor.svgY > viewportHeight) return [];
        const distance = driverPosition && sign.position
            ? Math.hypot(sign.position.x - driverPosition.x, sign.position.y - driverPosition.y) : 0;
        const fadeDistance = sign.state === 'released' ? RELEASE_FADE_DISTANCE : APPROACH_FADE_DISTANCE;
        const fadeProgress = Math.min(1, distance / fadeDistance);
        const opacity = 1 - fadeProgress * fadeProgress * (3 - 2 * fadeProgress);
        const signScale = scale * (anchor.perspectiveScale ?? 1);
        let height = 8 * signScale;
        const rows = sign.labels.map((range, index) => {
            const lines = wrapLabel(range.label);
            const y = height;
            const rowHeight = (lines.length * 16 + (index < sign.labels.length - 1 ? 4 : 12)) * signScale;
            height += rowHeight;
            return { ...range, lines, y, height: rowHeight };
        });
        height += 8 * signScale;
        const width = 160 * signScale;
        const gap = 18 * signScale;
        const postHeight = 64 * signScale;
        const board = { x: anchor.svgX - width - gap, y: anchor.svgY - height - postHeight };

        const description = sign.labels.map((range) => range.label.trim()).join(', ');
        const category = sign.labels.every((range) => range.category === sign.labels[0].category)
            ? sign.labels[0].category : undefined;
        return [<g key={sign.key} className={styles.roadSign} data-category={category ?? 'other'}
            data-state={sign.state} data-anchor-x={anchor.svgX} data-anchor-y={anchor.svgY}
            data-world-x={sign.position?.x} data-world-y={sign.position?.y}
            data-perspective-scale={anchor.perspectiveScale ?? 1}
            style={{ opacity }}
            aria-label={description} data-testid="comparison-road-sign">
            <path className={styles.roadSignPost} aria-hidden="true" data-testid="comparison-label-sign-post"
                style={{ strokeWidth: signScale }}
                d={`M ${anchor.svgX} ${anchor.svgY} L ${board.x + width - 10 * signScale} ${board.y + height}`} />
            <g transform={`translate(${board.x} ${board.y})`} aria-label={description} data-testid="comparison-label-sign">
                <title>{description}</title>
                <rect className={styles.roadSignBoard} width={width} height={height} rx={12 * signScale}
                    style={{ strokeWidth: 0.65 * signScale }} data-testid="comparison-label-sign-board" />
                {rows.map((row) => <g key={`${row.category}-${row.label.trim()}`}
                    className={styles.roadSign} data-category={row.category ?? 'other'}>
                    <circle className={styles.roadSignDot} cx={15 * signScale} cy={row.y + 14 * signScale}
                        r={2.5 * signScale} aria-hidden="true" />
                    <text className={styles.roadSignLabel} x={26 * signScale} y={row.y + 17 * signScale} style={{ fontSize: 10 * signScale }}>
                        {row.lines.map((line, index) => <tspan key={index} x={26 * signScale} dy={index ? 16 * signScale : 0}>{line}</tspan>)}
                    </text>
                </g>)}
            </g>
        </g>];
    });
    return annotations.length ? <g aria-label="Road signs on Driver trajectory" data-testid="comparison-road-signs">
        {annotations.reverse()}
    </g> : null;
};
