import React from 'react';
import type { AmodalMask } from './amodal-masks';
import { labelDepthRange, labelDepths, PipelineStep } from './pipeline-visuals';
import type { TrackVisionFrame } from './track-vision-types';
import { createDepthMap, formatDepth } from './depth-map';

import { VISION_DEPTH_COLORS, VISION_LABEL_COLORS as COLORS } from './vision-colors';

export default function PipelineDetails({ step, frame, masks, classNames, filterConfidence, depthMap }: {
    step: PipelineStep; frame: TrackVisionFrame | null; masks: AmodalMask[]; classNames: string[];
    depthMap: ReturnType<typeof createDepthMap>;
    filterConfidence: number;
}) {
    const segment = frame?.detections.segment;
    const depth = frame?.detections.depth;
    const scale = depth?.task === 'depth' ? depth.scale : undefined;
    const relative = scale === 'relative';
    const reading = (value: number | null) => formatDepth(value, scale);
    const instances = segment?.task === 'segment' ? segment.instances : [];
    const rows = labelDepths(masks);
    const measured = rows.filter((row) => row.samples);
    const { near, far } = labelDepthRange(masks);
    if (step === 'segmentation') return <div className="track-vision__legend" aria-label="Segmentation label legend">
        {classNames.map((label, classId) => <span key={classId}>
            <i style={{ background: COLORS[classId % COLORS.length] }} />{label}
            <b>{instances.filter((item) => item.classId === classId).length}</b>
        </span>)}
        {!classNames.length && <p className="track-vision__hint">Waiting for segmentation.</p>}
    </div>;
    if (step === 'filtering') return <div className="track-vision__filter-details">
        <dl className="track-vision__metrics">
            <div><dt>Detected masks</dt><dd>{frame ? instances.length : '—'}</dd></div>
            <div><dt>Retained masks</dt><dd>{frame ? masks.length : '—'}</dd></div>
            <div><dt>Labels with depth</dt><dd>{frame ? new Set(measured.map((row) => row.classId)).size : '—'}</dd></div>
        </dl>
        <ul className="track-vision__filter-list" aria-label="Applied filters">
            <li><strong>Confidence ≥ {Math.round(filterConfidence * 100)}%</strong></li>
            <li><strong>Car interior retained</strong></li>
            <li><strong>Overlaps resolved by depth</strong></li>
            <li><strong>Valid depth only</strong></li>
        </ul>
        {!segment && <p className="track-vision__hint">Waiting for segmentation.</p>}
    </div>;
    if (step === 'depth-map') return <div className="track-vision__depth-details">
        <div className="track-vision__depth-scale" aria-label="Depth map color scale">
            <span>Near {reading(depthMap?.near ?? null)}</span><i style={{ background: `linear-gradient(90deg, ${VISION_DEPTH_COLORS.join(', ')})` }} /><span>Far {reading(depthMap?.far ?? null)}</span>
        </div>
        {!depthMap && <p className="track-vision__hint">Waiting for depth.</p>}
        {depthMap && depthMap.near === null && <p className="track-vision__hint">No valid depth in this frame.</p>}
    </div>;
    if (step !== 'depth') return null;
    return <div className="track-vision__depth-details">
        <div className="track-vision__depth-scale" aria-label="Label depth color scale">
            <span>Near {reading(near)}</span><i style={{ background: `linear-gradient(90deg, ${VISION_DEPTH_COLORS.join(', ')})` }} /><span>Far {reading(far)}</span>
        </div>
        <div className="track-vision__table-wrap">
            <table className="track-vision__depth-table">
                <caption>Depth per mask after filtering · {relative ? 'relative depth (unitless)' : 'estimated camera-axis distance'}</caption>
                <thead><tr><th scope="col">Mask</th><th scope="col">Median</th><th scope="col">Near / far</th><th scope="col">Depth pixels</th></tr></thead>
                <tbody>{rows.map((row) => <tr key={row.maskIndex}>
                    <th scope="row">{row.label} #{row.instance}</th><td>{reading(row.median)}</td>
                    <td>{row.samples ? `${reading(row.near)} / ${reading(row.far)}` : 'No valid depth'}</td><td>{row.samples.toLocaleString()}</td>
                </tr>)}</tbody>
            </table>
        </div>
        {!rows.length && <p className="track-vision__hint">No retained masks yet.</p>}
        {!frame?.detections.depth && <p className="track-vision__hint">Waiting for depth.</p>}
    </div>;
}
