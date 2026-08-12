import React from 'react';
import type { TaskStartFunction } from './task-start-function';
import { AiToolComponentBase } from './AiToolComponentBase';

export const PROCEDURE_PLAN_STEP_STATUSES = [
    'pending',
    'running',
    'complete',
    'blocked',
    'failed',
    'skipped',
] as const;

export type ProcedurePlanStepStatus = typeof PROCEDURE_PLAN_STEP_STATUSES[number];

export type ProcedurePlanRequestSnapshot = {
    type: string;
    title: string;
    name?: string;
    status: ProcedurePlanStepStatus;
    detail?: string;
    method?: string;
    url?: string;
    payload?: unknown;
};

export type ProcedurePlanRequest = ProcedurePlanRequestSnapshot & {
    taskStart: TaskStartFunction;
};

export type ProcedurePlanSnapshot = {
    goal: string;
    requests: ProcedurePlanRequestSnapshot[];
    currentStep: number;
    sourceEvent?: string;
};

export type ProcedurePlanState = {
    goal: string;
    requests: ProcedurePlanRequest[];
    currentStep: number;
    sourceEvent?: string;
};

export type ProcedurePlanTaskStartFunctionSelector = (
    request: ProcedurePlanRequestSnapshot,
) => TaskStartFunction | null | undefined;

export type ProcedurePlanTaskErrorHandler = (
    request: ProcedurePlanRequestSnapshot,
    error: unknown,
) => void;

type ProcedurePlanChangeHandler = (plan: ProcedurePlanState | null) => void;

const defaultProcedurePlanErrorHandler: ProcedurePlanTaskErrorHandler = (request, error) => {
    console.error(`Procedure plan task '${request.title}' failed.`, error);
};

export class ProcedurePlanRunner extends AiToolComponentBase<ProcedurePlanSnapshot | null> {
    private plan: ProcedurePlanState | null = null;
    private activeRun: {
        controller: AbortController;
        token: symbol;
        taskStart: TaskStartFunction;
    } | null = null;
    private readonly onChange: ProcedurePlanChangeHandler;
    private readonly onError: ProcedurePlanTaskErrorHandler;

    constructor(componentName: string, onChange?: ProcedurePlanChangeHandler, onError?: ProcedurePlanTaskErrorHandler);
    constructor(onChange: ProcedurePlanChangeHandler, onError?: ProcedurePlanTaskErrorHandler);
    constructor(
        componentNameOrOnChange: string | ProcedurePlanChangeHandler,
        onChangeOrError?: ProcedurePlanChangeHandler | ProcedurePlanTaskErrorHandler,
        onError: ProcedurePlanTaskErrorHandler = defaultProcedurePlanErrorHandler,
    ) {
        const legacySignature = typeof componentNameOrOnChange === 'function';
        super(legacySignature ? 'procedure-plan' : componentNameOrOnChange, null);
        this.onChange = legacySignature
            ? componentNameOrOnChange
            : (onChangeOrError as ProcedurePlanChangeHandler | undefined) ?? (() => undefined);
        this.onError = legacySignature
            ? (onChangeOrError as ProcedurePlanTaskErrorHandler | undefined) ?? defaultProcedurePlanErrorHandler
            : onError;
    }

    get(): ProcedurePlanState | null {
        return this.plan;
    }

    replace(plan: ProcedurePlanState | null): void {
        this.abort();
        this.publish(plan?.requests.length ? plan : null);
        this.runNext();
    }

    clear(): void {
        this.replace(null);
    }

    advance(reason?: string): ProcedurePlanAdvanceResult {
        if (!this.plan) throw new Error('Cannot advance an empty procedure plan.');
        const result = advanceProcedurePlan(this.plan, reason);
        this.replace(result.status === 'complete' ? null : result.plan);
        return result;
    }

    abort(): void {
        const activeRun = this.activeRun;
        this.activeRun = null;
        activeRun?.controller.abort();
    }

    protected onDispose(): void {
        this.abort();
        this.plan = null;
    }

    private publish(plan: ProcedurePlanState | null): void {
        this.plan = plan;
        this.publishSnapshot(plan ? serializeProcedurePlan(plan) : null);
        this.onChange(plan);
        if (!plan) this.deleteComponentRef();
    }

    private settle(token: symbol, error?: unknown): void {
        const activeRun = this.activeRun;
        if (!activeRun || activeRun.token !== token || activeRun.controller.signal.aborted) return;
        this.activeRun = null;

        const current = this.plan;
        if (!current) return;
        const requestIndex = current.currentStep;
        const request = current.requests[requestIndex];
        if (!request || request.taskStart !== activeRun.taskStart) return;
        if (error !== undefined) {
            try {
                this.onError(serializeProcedurePlanRequest(request), error);
            } catch (handlerError) {
                console.error('Procedure plan task error handler failed.', handlerError);
            }
        }

        const requests = current.requests.filter((_item, index) => index !== requestIndex);
        this.publish(requests.length > 0 ? {
            ...current,
            requests,
            currentStep: Math.min(requestIndex, requests.length - 1),
        } : null);
        this.runNext();
    }

    private runNext(): void {
        if (!this.plan || this.activeRun) return;
        const request = this.plan.requests[this.plan.currentStep];
        if (!request || request.status !== 'pending') return;

        const controller = new AbortController();
        const token = Symbol(request.title);
        this.activeRun = { controller, token, taskStart: request.taskStart };
        this.publish({
            ...this.plan,
            requests: this.plan.requests.map((item, index) => (
                index === this.plan!.currentStep
                    ? { ...item, status: 'running' as const }
                    : item
            )),
        });

        try {
            Promise.resolve(request.taskStart(controller.signal)).then(
                () => this.settle(token),
                (error) => this.settle(token, error),
            );
        } catch (error) {
            this.settle(token, error);
        }
    }
}

export type ProcedurePlanAdvanceResult = {
    plan: ProcedurePlanState;
    status: 'advanced' | 'complete';
    current_request: number;
    current_step: number;
    request: ProcedurePlanRequest;
    step: string;
    reason?: string;
};

const PROCEDURE_PLAN_DONE_STATUSES: readonly ProcedurePlanStepStatus[] = ['complete', 'failed', 'skipped'];

const serializeProcedurePlanRequest = (
    request: ProcedurePlanRequest | ProcedurePlanRequestSnapshot,
): ProcedurePlanRequestSnapshot => ({
    type: request.type,
    title: request.title,
    ...(request.name !== undefined ? { name: request.name } : {}),
    status: request.status,
    ...(request.detail !== undefined ? { detail: request.detail } : {}),
    ...(request.method !== undefined ? { method: request.method } : {}),
    ...(request.url !== undefined ? { url: request.url } : {}),
    ...(request.payload !== undefined ? { payload: toStablePlanValue(request.payload) } : {}),
});

export const serializeProcedurePlan = (
    plan: ProcedurePlanState | ProcedurePlanSnapshot,
): ProcedurePlanSnapshot => ({
    goal: plan.goal,
    requests: plan.requests.map(serializeProcedurePlanRequest),
    currentStep: plan.currentStep,
    ...(plan.sourceEvent !== undefined ? { sourceEvent: plan.sourceEvent } : {}),
});

const toStablePlanValue = (value: unknown): unknown => {
    if (Array.isArray(value)) {
        return value.map(toStablePlanValue);
    }
    if (value && typeof value === 'object') {
        return Object.keys(value as Record<string, unknown>)
            .sort()
            .reduce<Record<string, unknown>>((acc, key) => {
                acc[key] = toStablePlanValue((value as Record<string, unknown>)[key]);
                return acc;
            }, {});
    }
    return value;
};

export const getProcedurePlanUpdateKey = (
    plan: ProcedurePlanState | ProcedurePlanSnapshot,
): string => JSON.stringify(serializeProcedurePlan(plan));

export const isProcedurePlanStartEvent = (sourceEvent?: string): boolean => (
    typeof sourceEvent === 'string'
    && (
        sourceEvent === 'procedure_plan_started'
        || sourceEvent.endsWith('_plan_started')
    )
);

const toNonEmptyString = (value: unknown): string | null => {
    if (typeof value !== 'string') return null;
    const trimmed = value.trim();
    return trimmed || null;
};

const toRecord = (value: unknown): Record<string, unknown> | null => (
    value && typeof value === 'object' && !Array.isArray(value)
        ? value as Record<string, unknown>
        : null
);

export const getProcedurePlanToolArguments = (
    request: ProcedurePlanRequestSnapshot,
): Record<string, unknown> => {
    const payload = toRecord(request.payload);
    if (!payload) return {};
    const nested = payload.arguments || payload.args || payload.parameters;
    return toRecord(nested) || payload;
};

export const getProcedurePlanToolRunKey = (
    plan: ProcedurePlanState | ProcedurePlanSnapshot,
    request: ProcedurePlanRequestSnapshot,
): string => `${plan.currentStep}:${request.name || ''}:${JSON.stringify(request.payload ?? null)}`;

const isProcedurePlanStepStatus = (value: unknown): value is ProcedurePlanStepStatus => (
    typeof value === 'string'
    && (PROCEDURE_PLAN_STEP_STATUSES as readonly string[]).includes(value)
);

const buildProcedurePlanRequestSnapshot = (value: unknown): ProcedurePlanRequestSnapshot | null => {
    const request = toRecord(value);
    if (!request) return null;

    const type = toNonEmptyString(request.type);
    const title = toNonEmptyString(request.title);
    const name = toNonEmptyString(request.name);
    if (!type || !title) return null;

    return {
        type,
        title,
        name: name || undefined,
        status: isProcedurePlanStepStatus(request.status) ? request.status : 'pending',
        detail: toNonEmptyString(request.detail) || undefined,
        method: toNonEmptyString(request.method) || undefined,
        url: toNonEmptyString(request.url) || undefined,
        payload: request.payload,
    };
};

const toProcedurePlanRequestSnapshots = (value: unknown): ProcedurePlanRequestSnapshot[] | null => {
    if (!Array.isArray(value) || value.length === 0) return null;
    const snapshots = value.map(buildProcedurePlanRequestSnapshot);
    if (snapshots.some((item) => !item)) return null;
    return snapshots as ProcedurePlanRequestSnapshot[];
};

export const isProcedurePlanRequestDone = (
    request: ProcedurePlanRequest | ProcedurePlanRequestSnapshot | undefined,
): boolean => (
    Boolean(request && PROCEDURE_PLAN_DONE_STATUSES.includes(request.status))
);

export const getSelfAdvancingProcedurePlan = (plan: ProcedurePlanState): ProcedurePlanState => {
    const requests = plan.requests
        .slice(Math.max(0, plan.currentStep))
        .filter((request) => !isProcedurePlanRequestDone(request))
        .map((request) => ({
            ...request,
            status: 'pending' as const,
        }));
    return { ...plan, requests, currentStep: 0 };
};

export const advanceProcedurePlan = (
    plan: ProcedurePlanState,
    reason?: string,
): ProcedurePlanAdvanceResult => {
    const completedRequest = plan.requests[plan.currentStep];
    if (!completedRequest) throw new Error('Cannot advance an empty procedure plan.');
    const requests = plan.requests.filter((_request, index) => index !== plan.currentStep);
    const nextPlan = getSelfAdvancingProcedurePlan({
        ...plan,
        requests,
        currentStep: Math.min(plan.currentStep, Math.max(0, requests.length - 1)),
    });
    const request = nextPlan.requests[nextPlan.currentStep] ?? completedRequest;

    return {
        plan: nextPlan,
        status: nextPlan.requests.length === 0 ? 'complete' : 'advanced',
        current_request: nextPlan.currentStep,
        current_step: nextPlan.currentStep,
        request,
        step: request.title,
        reason,
    };
};

export const isProcedurePlanOptOutRequest = (text: unknown): boolean => {
    if (typeof text !== 'string') return false;
    const normalized = text
        .toLowerCase()
        .replace(/['\u2019]/g, '')
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    if (!normalized) return false;

    const planTarget = '(?:the\\s+)?(?:procedure\\s+)?plan';
    const optOutVerb = '(?:cancel|clear|stop|end|exit|dismiss|hide|skip|drop|forget)';
    return [
        new RegExp(`\\b${optOutVerb}\\b.*\\b${planTarget}\\b`),
        new RegExp(`\\b${planTarget}\\b.*\\b${optOutVerb}\\b`),
        new RegExp(`\\bopt\\s*out\\b.*\\b${planTarget}\\b`),
        new RegExp(`\\b(?:dont|do\\s+not|no\\s+longer)\\s+(?:follow|use|respect)\\s+${planTarget}\\b`),
        new RegExp(`\\bignore\\s+${planTarget}\\b`),
    ].some((pattern) => pattern.test(normalized));
};

export const isProcedurePlanClearEvent = (sourceEvent?: string): boolean => (
    typeof sourceEvent === 'string'
    && (
        sourceEvent === 'procedure_plan_cleared'
        || sourceEvent === 'procedure_plan_terminated'
        || sourceEvent.endsWith('_plan_cleared')
        || sourceEvent.endsWith('_plan_terminated')
    )
);

export const buildProcedurePlan = (
    data: Record<string, unknown>,
    selectTaskStartFunction: ProcedurePlanTaskStartFunctionSelector,
): ProcedurePlanState | null => {
    const sourceEvent = toNonEmptyString(data.event);
    const snapshots = toProcedurePlanRequestSnapshots(data.requests);
    if (!snapshots) return null;

    const requestedStep = Math.floor(Number(data.current_request ?? 0));
    const currentStep = Number.isFinite(requestedStep)
        ? Math.max(0, Math.min(snapshots.length - 1, requestedStep))
        : 0;
    const runnableSnapshots = snapshots
        .slice(currentStep)
        .filter((request) => !isProcedurePlanRequestDone(request));
    if (runnableSnapshots.length === 0) return null;
    const requests = runnableSnapshots.map((request) => {
        const taskStart = selectTaskStartFunction(request);
        return typeof taskStart === 'function'
            ? { ...request, status: 'pending' as const, taskStart }
            : null;
    });
    if (requests.some((request) => request === null)) return null;

    return {
        goal: toNonEmptyString(data.goal) || snapshots[0].title,
        requests: requests as ProcedurePlanRequest[],
        currentStep: 0,
        sourceEvent: sourceEvent || undefined,
    };
};

export type ProcedurePlanProps = {
    plan: ProcedurePlanSnapshot;
    surface?: 'chat' | 'pill';
    onClear?: () => void;
};

const getProcedurePlanRequestMeta = (request: ProcedurePlanSnapshot['requests'][number]): string => {
    const parts = [
        request.type,
        request.status,
    ].filter((part): part is string => Boolean(part));
    return parts.join(' - ');
};

const ProcedurePlan: React.FC<ProcedurePlanProps> = ({
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

export default ProcedurePlan;
