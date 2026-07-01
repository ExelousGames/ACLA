import React, { useMemo } from 'react';
import { CircuitMapBinSample, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';

export type AiMapSectionSelection = {
    start?: number;
    end?: number;
    label?: string;
};

export type AiMapDisplayPayload = {
    status: 'ready' | 'unavailable';
    map?: CircuitMapDto | null;
    requestedMap?: string;
    title?: string;
    note?: string;
    reason?: string;
    section?: AiMapSectionSelection;
};

type ProjectedPoint = {
    x: number;
    y: number;
};

const VIEWBOX_WIDTH = 720;
const VIEWBOX_HEIGHT = 360;
const VIEWBOX_PAD = 28;

const clamp01 = (value: unknown): number | undefined => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return undefined;
    return Math.max(0, Math.min(1, parsed));
};

const sampleToPoint = (sample: CircuitMapBinSample) => ({
    x: Number(sample.x),
    y: Number(sample.z),
    normalizedPosition: Number(sample.normalized_position),
});

const isFinitePoint = (point: ReturnType<typeof sampleToPoint>) => (
    Number.isFinite(point.x)
    && Number.isFinite(point.y)
    && Number.isFinite(point.normalizedPosition)
);

const toPolyline = (
    samples: CircuitMapBinSample[] | undefined,
    project: (sample: CircuitMapBinSample) => ProjectedPoint,
) => (samples || [])
    .filter((sample) => isFinitePoint(sampleToPoint(sample)))
    .sort((a, b) => a.bin - b.bin)
    .map(project)
    .map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(' ');

const midpointSample = (
    left: CircuitMapBinSample,
    right: CircuitMapBinSample,
): CircuitMapBinSample => ({
    ...left,
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2,
    z: (left.z + right.z) / 2,
});

const buildCenterlineSamples = (map: CircuitMapDto): CircuitMapBinSample[] => {
    const rightByBin = new Map((map.samples.right_boundary || []).map((sample) => [sample.bin, sample]));
    return [...(map.samples.left_boundary || [])]
        .sort((a, b) => a.bin - b.bin)
        .map((left) => {
            const right = rightByBin.get(left.bin);
            return right ? midpointSample(left, right) : null;
        })
        .filter((sample): sample is CircuitMapBinSample => sample !== null);
};

const isWithinSection = (sample: CircuitMapBinSample, section?: AiMapSectionSelection): boolean => {
    const start = clamp01(section?.start);
    const end = clamp01(section?.end);
    const position = clamp01(sample.normalized_position);

    if (position === undefined) return false;
    if (start === undefined && end === undefined) return false;
    if (start === undefined) return position <= (end as number);
    if (end === undefined) return position >= start;
    if (start <= end) return position >= start && position <= end;
    return position >= start || position <= end;
};

const pathFromPoints = (points: ProjectedPoint[]) => points
    .map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(' ');

const formatPercent = (value?: number) => (
    value === undefined ? null : `${Math.round(value * 100)}%`
);

type AiMapToolDisplayProps = {
    display: AiMapDisplayPayload;
    surface?: 'chat' | 'pill';
};

const AiMapToolDisplay: React.FC<AiMapToolDisplayProps> = ({ display, surface = 'chat' }) => {
    const renderData = useMemo(() => {
        const map = display.map;
        if (!map || display.status !== 'ready') return null;

        const samples = [
            ...(map.samples.left_boundary || []),
            ...(map.samples.right_boundary || []),
            ...(map.samples.pit_lane || []),
        ].map(sampleToPoint).filter(isFinitePoint);

        if (samples.length < 2) return null;

        const minX = Math.min(...samples.map((point) => point.x));
        const maxX = Math.max(...samples.map((point) => point.x));
        const minY = Math.min(...samples.map((point) => point.y));
        const maxY = Math.max(...samples.map((point) => point.y));
        const spanX = Math.max(1, maxX - minX);
        const spanY = Math.max(1, maxY - minY);
        const scale = Math.min(
            (VIEWBOX_WIDTH - VIEWBOX_PAD * 2) / spanX,
            (VIEWBOX_HEIGHT - VIEWBOX_PAD * 2) / spanY,
        );
        const drawWidth = spanX * scale;
        const drawHeight = spanY * scale;
        const offsetX = (VIEWBOX_WIDTH - drawWidth) / 2;
        const offsetY = (VIEWBOX_HEIGHT - drawHeight) / 2;

        const project = (sample: CircuitMapBinSample): ProjectedPoint => {
            const point = sampleToPoint(sample);
            return {
                x: offsetX + (point.x - minX) * scale,
                y: offsetY + drawHeight - ((point.y - minY) * scale),
            };
        };

        const centerlineSamples = buildCenterlineSamples(map);
        const highlightPoints = centerlineSamples
            .filter((sample) => isWithinSection(sample, display.section))
            .map(project);

        return {
            left: toPolyline(map.samples.left_boundary, project),
            right: toPolyline(map.samples.right_boundary, project),
            pit: toPolyline(map.samples.pit_lane, project),
            center: pathFromPoints(centerlineSamples.map(project)),
            highlight: pathFromPoints(highlightPoints),
            hasHighlight: highlightPoints.length > 1,
        };
    }, [display]);

    const sectionStart = formatPercent(clamp01(display.section?.start));
    const sectionEnd = formatPercent(clamp01(display.section?.end));
    const sectionLabel = display.section?.label
        || ([sectionStart, sectionEnd].filter(Boolean).length === 2
            ? `${sectionStart} to ${sectionEnd}`
            : 'Selected section');

    if (!renderData) {
        return (
            <div className={`ai-chat__map-card ai-chat__map-card--empty ai-chat__map-card--${surface}`}>
                <div className="ai-chat__map-head">
                    <span>{display.title || 'Map'}</span>
                </div>
                <div className="ai-chat__map-empty">Map is not available</div>
                {display.reason && (
                    <div className="ai-chat__map-note">{display.reason}</div>
                )}
            </div>
        );
    }

    return (
        <div className={`ai-chat__map-card ai-chat__map-card--${surface}`}>
            <div className="ai-chat__map-head">
                <span>{display.title || display.map?.circuit_name || 'Map'}</span>
                {display.map?.circuit_name && (
                    <b>{display.map.circuit_name}</b>
                )}
            </div>
            <svg className="ai-chat__map-svg" viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`} role="img" aria-label="Circuit map">
                <rect x="0" y="0" width={VIEWBOX_WIDTH} height={VIEWBOX_HEIGHT} rx="8" className="ai-chat__map-bg" />
                {renderData.left && <polyline points={renderData.left} className="ai-chat__map-boundary" />}
                {renderData.right && <polyline points={renderData.right} className="ai-chat__map-boundary" />}
                {renderData.pit && <polyline points={renderData.pit} className="ai-chat__map-pit" />}
                {renderData.center && <polyline points={renderData.center} className="ai-chat__map-center" />}
                {renderData.hasHighlight && <polyline points={renderData.highlight} className="ai-chat__map-highlight" />}
            </svg>
            <div className="ai-chat__map-foot">
                <span>{renderData.hasHighlight ? sectionLabel : 'Full map'}</span>
                {display.note && <span>{display.note}</span>}
            </div>
        </div>
    );
};

export default AiMapToolDisplay;
