import React from 'react';
import type { BaselineCollectionTag } from './BaselineCollection';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import { isOverlayRecord } from 'views/floating-chat/overlay-renderer-validation';
import './baseline-collection.css';

type BaselineProgressDisplayProps = {
    tag: BaselineCollectionTag | null;
    surface?: 'panel' | 'pill';
    action?: React.ReactNode;
};

type BaselineStage = 'start' | 'recording' | 'finish';
type BaselineStageState = 'complete' | 'active' | 'upcoming';

const StageIcon = ({ stage }: { stage: BaselineStage }) => {
    if (stage === 'start') {
        return (
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8.25 6.5 17 12l-8.75 5.5z" />
            </svg>
        );
    }

    if (stage === 'recording') {
        return (
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M5 12h2.25l1.6-4.25L12 16.5l2.2-6 1.35 1.5H19" />
            </svg>
        );
    }

    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m6.75 12.25 3.2 3.2 7.3-7.3" />
        </svg>
    );
};

const BaselineProgressDisplay: React.FC<BaselineProgressDisplayProps> = ({
    tag,
    surface = 'panel',
    action,
}) => {
    const progress = Math.max(0, Math.min(100, Math.round(tag?.progress_percent ?? 0)));
    const activeStage = !tag
        ? 0
        : tag.status === 'complete'
            ? 2
            : 1;
    const statusLabel = !tag
        ? 'Ready'
        : tag.status === 'waiting_for_start'
            ? 'Armed'
            : tag.status === 'collecting'
                ? 'Recording'
                : 'Captured';
    const stages: Array<{
        key: BaselineStage;
        label: string;
        title: string;
        detail: string;
    }> = [
        {
            key: 'start',
            label: 'Starting stage',
            title: 'Start collection',
            detail: !tag
                ? 'Capture one complete, clean lap as your reference.'
                : 'Collection started and the telemetry trigger is armed.',
        },
        {
            key: 'recording',
            label: 'Recording stage',
            title: tag?.status === 'collecting' ? 'Recording baseline' : 'Record baseline',
            detail: !tag
                ? 'Recording begins at the next start / finish crossing.'
                : tag.status === 'waiting_for_start'
                    ? 'Waiting for the start / finish line.'
                    : tag.status === 'collecting'
                        ? tag.detail
                        : 'Full-lap telemetry has been captured.',
        },
        {
            key: 'finish',
            label: 'End stage',
            title: tag?.status === 'complete' ? 'Baseline ready' : 'Finish collection',
            detail: tag?.status === 'complete'
                ? 'The recorded lap is ready for performance analysis. Only one baseline recording is held at a time; starting another replaces it.'
                : 'Collection ends automatically after one full lap. Only one baseline recording is held at a time; starting another replaces it.',
        },
    ];

    const getStageState = (index: number): BaselineStageState => {
        if (index < activeStage) return 'complete';
        if (index === activeStage) return 'active';
        return 'upcoming';
    };

    return (
        <div
            className={`baseline-timeline baseline-timeline--${surface}`}
            aria-label="Baseline collection progress"
        >
            {surface === 'panel' && (
                <div className="baseline-timeline__header">
                    <span className="baseline-timeline__eyebrow">Baseline run</span>
                    <span
                        className={`baseline-timeline__status baseline-timeline__status--${statusLabel.toLowerCase()}`}
                    >
                        <span aria-hidden="true" />
                        {statusLabel}
                    </span>
                </div>
            )}

            <div className="baseline-timeline__stages">
                {stages.map((stage, index) => {
                    if (surface === 'pill' && index !== activeStage) return null;

                    const state = getStageState(index);
                    const showProgress = stage.key === 'recording'
                        && tag?.status !== 'complete'
                        && Boolean(tag);
                    const showAction = index === activeStage && Boolean(action);

                    return (
                        <section
                            className={`baseline-timeline__stage baseline-timeline__stage--${state}`}
                            data-stage={stage.key}
                            data-state={state}
                            key={stage.key}
                        >
                            <div className="baseline-timeline__rail" aria-hidden="true">
                                <span className="baseline-timeline__node">
                                    <StageIcon stage={stage.key} />
                                </span>
                            </div>
                            <div className="baseline-timeline__content">
                                <div className="baseline-timeline__stage-heading">
                                    <div>
                                        {surface === 'panel' && (
                                            <span className="baseline-timeline__stage-label">{stage.label}</span>
                                        )}
                                        <h4>{stage.title}</h4>
                                    </div>
                                    {showProgress && (
                                        <span
                                            className="baseline-timeline__percent"
                                            role="progressbar"
                                            aria-label="Recorded lap progress"
                                            aria-valuenow={progress}
                                            aria-valuemin={0}
                                            aria-valuemax={100}
                                        >
                                            {progress}%
                                        </span>
                                    )}
                                </div>
                                <p>{stage.detail}</p>
                                {showAction && <div className="baseline-timeline__action">{action}</div>}
                            </div>
                        </section>
                    );
                })}
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
        expanded: { width: 420, height: 280 },
        folded: { width: 300, height: 58 },
    },
};

export default BaselineProgressDisplay;
