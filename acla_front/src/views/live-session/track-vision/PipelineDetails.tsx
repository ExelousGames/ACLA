import React from 'react';
import type { AmodalMask } from './amodal-masks';
import { labelDepths, PipelineStep } from './pipeline-visuals';
import { VISION_CONFIDENCE } from './semantic-scene';
import type { TrackVisionFrame } from './track-vision-types';

import { VISION_LABEL_COLORS as COLORS } from './vision-colors';
const meters = (value: number | null) => value === null ? '—' : `${value.toFixed(1)} m`;

export default function PipelineDetails({ step, frame, masks, classNames, confidence }: {
    step: PipelineStep; frame: TrackVisionFrame | null; masks: AmodalMask[]; classNames: string[]; confidence: number;
}) {
    const segment = frame?.detections.segment;
    const instances = segment?.task === 'segment' ? segment.instances : [];
    const rows = labelDepths(masks);
    const measured = rows.filter((row) => row.samples);
    const near = measured.length ? Math.min(...measured.map((row) => row.near!)) : null;
    const far = measured.length ? Math.max(...measured.map((row) => row.far!)) : null;
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
            <div><dt>Labels with depth</dt><dd>{frame ? measured.length : '—'}</dd></div>
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
    if (step !== 'depth') return null;
    return <div className="track-vision__depth-details">
        <div className="track-vision__depth-scale" aria-label="Label depth color scale">
            <span>Near {meters(near)}</span><i /><span>Far {meters(far)}</span>
        </div>
        <div className="track-vision__table-wrap">
            <table className="track-vision__depth-table">
                <caption>Depth after filtering · estimated camera-axis distance</caption>
                <thead><tr><th scope="col">Label</th><th scope="col">Masks</th><th scope="col">Median</th><th scope="col">Near / far</th><th scope="col">Depth pixels</th></tr></thead>
                <tbody>{rows.map((row) => <tr key={row.classId}>
                    <th scope="row">{row.label}</th><td>{row.instances}</td><td>{meters(row.median)}</td>
                    <td>{row.samples ? `${meters(row.near)} / ${meters(row.far)}` : 'No valid depth'}</td><td>{row.samples.toLocaleString()}</td>
                </tr>)}</tbody>
            </table>
        </div>
        {!rows.length && <p className="track-vision__hint">No retained labels yet. Enable segmentation and share a driving view.</p>}
        {!frame?.detections.depth && <p className="track-vision__hint">Waiting for depth. Enable Depth to estimate distances for the retained labels.</p>}
        <p className="track-vision__hint">Medians and ranges use observed pixels after filtering; predicted hidden pixels are excluded from the table. Depth colors run from amber (near) to blue (far).</p>
    </div>;
}
