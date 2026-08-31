import React from 'react';
import type {
    GoalDeterminationResult,
    GoalSnapshot,
    GoalStepSnapshot,
} from './Goal';

export type GoalOverlayDisplayProps = {
    snapshot: GoalSnapshot;
};

type GoalOverlayActivity =
    | { kind: 'step'; step: GoalStepSnapshot; index: number }
    | { kind: 'determination'; result: GoalDeterminationResult }
    | { kind: 'complete' }
    | { kind: 'missed' }
    | { kind: 'error'; message: string };

const getActivity = (snapshot: GoalSnapshot): GoalOverlayActivity => {
    const runningStepIndex = snapshot.steps.findIndex((step) => step.status === 'running');
    if (runningStepIndex >= 0) {
        return {
            kind: 'step',
            step: snapshot.steps[runningStepIndex],
            index: runningStepIndex,
        };
    }

    const failedStepIndex = snapshot.steps.findIndex((step) => step.status === 'error');
    if (failedStepIndex >= 0) {
        return {
            kind: 'step',
            step: snapshot.steps[failedStepIndex],
            index: failedStepIndex,
        };
    }

    if (
        snapshot.determination_result
        && snapshot.determination_result.status !== 'pending'
        && snapshot.status !== 'achieved'
        && snapshot.status !== 'missed'
    ) {
        return { kind: 'determination', result: snapshot.determination_result };
    }

    if (snapshot.status === 'achieved') return { kind: 'complete' };
    if (snapshot.status === 'missed') return { kind: 'missed' };
    return { kind: 'error', message: snapshot.error || 'Waiting for the next step' };
};

const getCompletedCount = (snapshot: GoalSnapshot): number => (
    snapshot.steps.filter((step) => step.status === 'completed').length
);

const getOutcomeMetric = (snapshot: GoalSnapshot): string | null => {
    if (!snapshot.determination || snapshot.actual === null) return null;
    return `${snapshot.actual} ${snapshot.determination.operator} ${snapshot.determination.target}`;
};

export const getGoalOverlaySummary = (snapshot: GoalSnapshot): string => {
    const activity = getActivity(snapshot);
    if (activity.kind === 'step') return activity.step.title;
    if (activity.kind === 'determination') return 'Checking goal result';
    if (activity.kind === 'complete') return `Goal achieved: ${snapshot.name}`;
    if (activity.kind === 'missed') return `Goal not met: ${snapshot.name}`;
    return activity.message;
};

export const GoalOverlayDisplay: React.FC<GoalOverlayDisplayProps> = ({ snapshot }) => {
    const activity = getActivity(snapshot);
    const completedCount = getCompletedCount(snapshot);
    const progress = snapshot.steps.length > 0
        ? Math.round((completedCount / snapshot.steps.length) * 100)
        : 0;
    const outcomeMetric = getOutcomeMetric(snapshot);

    let activityLabel = 'Goal runner';
    let activityTitle = snapshot.name;
    let activityMeta: string | null = null;
    let activityError: string | null = null;

    if (activity.kind === 'step') {
        const isError = activity.step.status === 'error';
        activityLabel = isError ? 'Step needs attention' : 'Current step';
        activityTitle = activity.step.title;
        activityMeta = [
            `Step ${activity.index + 1} of ${snapshot.steps.length}`,
            activity.step.attempts > 1 ? `Attempt ${activity.step.attempts}` : null,
        ].filter(Boolean).join(' · ');
        activityError = activity.step.error;
    } else if (activity.kind === 'determination') {
        activityLabel = activity.result.status === 'error'
            ? 'Check needs attention'
            : 'Current step';
        activityTitle = 'Checking goal result';
        activityMeta = activity.result.attempt > 1
            ? `Final check · Attempt ${activity.result.attempt}`
            : 'Final check';
        activityError = activity.result.error || null;
    } else if (activity.kind === 'complete') {
        activityLabel = 'Run complete';
        activityTitle = 'Goal achieved';
        activityMeta = outcomeMetric;
    } else if (activity.kind === 'missed') {
        activityLabel = 'Goal not met';
        activityTitle = 'Preparing another run';
        activityMeta = outcomeMetric;
    } else {
        activityLabel = snapshot.status === 'error' ? 'Goal stopped' : 'Goal runner';
        activityTitle = activity.message;
        activityError = snapshot.error || null;
    }

    return (
        <section
            className={`goal-overlay goal-overlay--${snapshot.status}`}
            aria-label="Goal runner overlay"
            aria-live="polite"
            data-testid="goal-overlay"
        >
            <header className="goal-overlay__header">
                <div className="goal-overlay__identity">
                    <span className="goal-overlay__pulse" aria-hidden="true" />
                    <span>Goal runner</span>
                </div>
                <span className="goal-overlay__progress-count">
                    {completedCount}/{snapshot.steps.length}
                </span>
            </header>

            <div className="goal-overlay__goal" title={snapshot.name}>{snapshot.name}</div>

            <div className="goal-overlay__activity">
                <span className="goal-overlay__activity-marker" aria-hidden="true">
                    <span />
                </span>
                <div className="goal-overlay__activity-copy">
                    <span className="goal-overlay__activity-label">{activityLabel}</span>
                    <strong className="goal-overlay__activity-title">{activityTitle}</strong>
                    {activityMeta && (
                        <span className="goal-overlay__activity-meta">{activityMeta}</span>
                    )}
                    {activityError && (
                        <span className="goal-overlay__activity-error">{activityError}</span>
                    )}
                </div>
            </div>

            <div className="goal-overlay__progress" aria-hidden="true">
                <span style={{ width: `${progress}%` }} />
            </div>
        </section>
    );
};

export default GoalOverlayDisplay;
