import React from 'react';
import type { BaselineCollectionTag } from './BaselineCollectionTracker';

type BaselineProgressDisplayProps = {
    tag: BaselineCollectionTag;
    surface?: 'chat' | 'pill';
};

const BaselineProgressDisplay: React.FC<BaselineProgressDisplayProps> = ({
    tag,
    surface = 'chat',
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

export default BaselineProgressDisplay;
