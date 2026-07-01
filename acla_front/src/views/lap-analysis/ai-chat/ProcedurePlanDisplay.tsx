import React from 'react';
import type { ProcedurePlan } from './ai-chat-plan';

type ProcedurePlanDisplayProps = {
    plan: ProcedurePlan;
    surface?: 'chat' | 'pill';
    onClear?: () => void;
};

const getProcedurePlanRequestMeta = (request: ProcedurePlan['requests'][number]): string => {
    const parts = [
        request.type,
        request.status,
    ].filter((part): part is string => Boolean(part));
    return parts.join(' - ');
};

const ProcedurePlanDisplay: React.FC<ProcedurePlanDisplayProps> = ({
    plan,
    surface = 'chat',
    onClear,
}) => {
    const requests = surface === 'pill'
        ? plan.requests.slice(Math.max(0, plan.currentStep - 1), plan.currentStep + 2)
        : plan.requests;

    return (
        <div className={`ai-chat__plan ai-chat__plan--${surface}`} aria-label="Procedure plan">
            <div className="ai-chat__plan-head">
                <div>
                    <span className="ai-chat__plan-kicker">PLAN</span>
                    <div className="ai-chat__plan-goal">{plan.goal}</div>
                </div>
                {onClear && surface === 'chat' && (
                    <button
                        type="button"
                        className="ai-chat__plan-clear"
                        onClick={onClear}
                        title="Dismiss the visible plan"
                        aria-label="Dismiss the visible plan"
                    >
                        x
                    </button>
                )}
            </div>
            <ul className="ai-chat__plan-list">
                {requests.map((request) => {
                    const index = plan.requests.indexOf(request);
                    const isActive = index === plan.currentStep;
                    const isDone = index < plan.currentStep;
                    const meta = getProcedurePlanRequestMeta(request);
                    return (
                        <li
                            key={`${index}-${request.type}-${request.title}`}
                            className={[
                                'ai-chat__plan-step',
                                isActive ? 'ai-chat__plan-step--active' : '',
                                isDone ? 'ai-chat__plan-step--done' : '',
                            ].filter(Boolean).join(' ')}
                        >
                            <span className="ai-chat__plan-step-dot" aria-hidden="true" />
                            <span className="ai-chat__plan-step-text">
                                <span>{request.title}</span>
                                {meta && (
                                    <span className="ai-chat__plan-step-meta">{meta}</span>
                                )}
                                {request.detail && surface === 'chat' && (
                                    <span className="ai-chat__plan-step-detail">{request.detail}</span>
                                )}
                            </span>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
};

export default ProcedurePlanDisplay;
