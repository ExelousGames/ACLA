import React from 'react';
import type { BaselineCollectionTag } from './BaselineCollection';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import { isOverlayRecord } from 'views/floating-chat/overlay-renderer-validation';
import './baseline-collection.css';

type BaselineProgressDisplayProps = {
    tag: BaselineCollectionTag;
    surface?: 'panel' | 'pill';
};

const BaselineProgressDisplay: React.FC<BaselineProgressDisplayProps> = ({
    tag,
    surface = 'panel',
}) => {
    const progress = Math.max(0, Math.min(100, Math.round(tag.progress_percent)));

    return (
        <div
            className={`ai-chat__baseline-progress ai-chat__baseline-progress--${surface}`}
            aria-label="Baseline collection progress"
        >
            <div className="ai-chat__baseline-progress-head">
                <span>BASELINE</span>
                <span>{progress}%</span>
            </div>
            <div
                className="ai-chat__baseline-progress-track"
                role="progressbar"
                aria-valuenow={progress}
                aria-valuemin={0}
                aria-valuemax={100}
            >
                <div
                    className="ai-chat__baseline-progress-fill"
                    style={{ width: `${progress}%` }}
                />
            </div>
            <div className="ai-chat__baseline-progress-detail">
                {tag.detail}
            </div>
        </div>
    );
};

export const baselineProgressDisplayOverlayRenderer: AiOverlayRenderer<BaselineCollectionTag> = {
    componentType: 'baseline_progress',
    validateSnapshot: (snapshot): snapshot is BaselineCollectionTag => (
        isOverlayRecord(snapshot)
        && ['waiting_for_start', 'collecting', 'complete'].includes(String(snapshot.status))
        && typeof snapshot.progress_percent === 'number'
        && Number.isFinite(snapshot.progress_percent)
        && typeof snapshot.detail === 'string'
    ),
    renderOverlay: (snapshot, status) => status === 'folded'
        ? `Baseline ${Math.round(snapshot.progress_percent)}% - ${snapshot.detail}`
        : <BaselineProgressDisplay tag={snapshot} surface="pill" />,
    dimensions: {
        expanded: { width: 420, height: 136 },
        folded: { width: 300, height: 58 },
    },
};

export default BaselineProgressDisplay;
