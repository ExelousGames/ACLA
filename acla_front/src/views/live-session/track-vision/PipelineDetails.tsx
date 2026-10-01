import React from 'react';
import type { AmodalMask } from './amodal-masks';
import { labelDepthRange, labelDepths, PipelineStep } from './pipeline-visuals';
import { VISION_CONFIDENCE } from './semantic-scene';
import type { TrackVisionFrame } from './track-vision-types';
import type { createDepthMap } from './depth-map';

import { VISION_DEPTH_COLORS, VISION_LABEL_COLORS as COLORS } from './vision-colors';
const meters = (value: number | null) => value === null ? '—' : `${value.toFixed(1)} m`;

export default function PipelineDetails({ step, frame, masks, classNames, confidence, depthMap }: {
    step: PipelineStep; frame: TrackVisionFrame | null; masks: AmodalMask[]; classNames: string[]; confidence: number;
    depthMap: ReturnType<typeof createDepthMap>;
}) {
    const segment = frame?.detections.segment;
    const instances = segment?.task === 'segment' ? segment.instances : [];
    const rows = labelDepths(masks);
    const measured = rows.filter((row) => row.samples);
    const { near, far } = labelDepthRange(masks);
    if (step === 'segmentation') return <div className="track-vision__legend" aria-label="Segmentation label legend">
        {classNames.map((label, classId) => <span key={classId}>
            <i style={{ background: COLORS[classId % COLORS.length] }} />{label}
            <b>{instances.filter((item) => item.classId === classId).length}</b>
        </span>)}
        {!classNames.length && <p className="track-vision__hint">Model labels will appear when segmentation is ready.</p>}
    </div>;
    if (step === 'filtering') return <div className="track-vision__filter-details">
        <dl className="track-vision__metrics">
            <div><dt>Detected masks</dt><dd>{frame ? instances.length : '—'}</dd></div>
            <div><dt>Retained masks</dt><dd>{frame ? masks.length : '—'}</dd></div>
            <div><dt>Labels with depth</dt><dd>{frame ? new Set(measured.map((row) => row.classId)).size : '—'}</dd></div>
        </dl>
        <ul className="track-vision__filter-list" aria-label="Applied filters">
            <li><strong>Confidence ≥ {Math.round(VISION_CONFIDENCE * 100)}%</strong><span>Reconstruction keeps accepted detections above this threshold. Detection is currently set to {Math.round(confidence * 100)}%.</span></li>
            <li><strong>Car interior retained</strong><span>Cockpit labels and depth stay available. The final scene uses this mask to remove false track outline edges.</span></li>
            <li><strong>Overlaps resolved by depth</strong><span>Visible pixels belong to the supported foreground mask. Only supported hidden sections are completed.</span></li>
            <li><strong>Valid depth only</strong><span>Distances must be finite, greater than 0 and at most 200 m. Image padding and missing depth are excluded.</span></li>
        </ul>
        <p className="track-vision__hint">Colors identify retained labels. Only track labels define the left and right boundaries in the reconstructed scene.</p>
        {!segment && <p className="track-vision__hint">Waiting for segmentation to show the filtered masks.</p>}
    </div>;
    if (step === 'depth-map') return <div className="track-vision__depth-details">
        <div className="track-vision__depth-scale" aria-label="Depth map color scale">
            <span>Near {meters(depthMap?.near ?? null)}</span><i style={{ background: `linear-gradient(90deg, ${VISION_DEPTH_COLORS.join(', ')})` }} /><span>Far {meters(depthMap?.far ?? null)}</span>
        </div>
        <p className="track-vision__hint">Move the mouse over the depth map to inspect estimated camera-axis distance in meters. Colors run from red (near) to violet (far) across the entire frame, including unlabeled areas. Dark areas have no valid depth.</p>
        {!depthMap && <p className="track-vision__hint">Waiting for depth. Enable Depth and Segmentation and share a driving view.</p>}
        {depthMap && depthMap.near === null && <p className="track-vision__hint">No valid depth in this frame.</p>}
    </div>;
    if (step !== 'depth') return null;
    return <div className="track-vision__depth-details">
        <div className="track-vision__depth-scale" aria-label="Label depth color scale">
            <span>Near {meters(near)}</span><i style={{ background: `linear-gradient(90deg, ${VISION_DEPTH_COLORS.join(', ')})` }} /><span>Far {meters(far)}</span>
        </div>
        <div className="track-vision__table-wrap">
            <table className="track-vision__depth-table">
                <caption>Depth per mask after filtering · estimated camera-axis distance</caption>
                <thead><tr><th scope="col">Mask</th><th scope="col">Median</th><th scope="col">Near / far</th><th scope="col">Depth pixels</th></tr></thead>
                <tbody>{rows.map((row) => <tr key={row.maskIndex}>
                    <th scope="row">{row.label} #{row.instance}</th><td>{meters(row.median)}</td>
                    <td>{row.samples ? `${meters(row.near)} / ${meters(row.far)}` : 'No valid depth'}</td><td>{row.samples.toLocaleString()}</td>
                </tr>)}</tbody>
            </table>
        </div>
        {!rows.length && <p className="track-vision__hint">No retained masks yet. Enable segmentation and share a driving view.</p>}
        {!frame?.detections.depth && <p className="track-vision__hint">Waiting for depth. Enable Depth to estimate distances for the retained labels.</p>}
        <p className="track-vision__hint">Numbered mask labels in the preview match the table rows for the current frame. Medians and ranges use observed pixels after filtering; predicted hidden pixels are excluded from the table. Depth colors run from red (near) to violet (far).</p>
    </div>;
}
