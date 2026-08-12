import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AiToolComponentBase } from './AiToolComponentBase';
import type {
    NamedAiToolComponentHandle,
} from 'contexts/AiToolComponentRefContext';
import { useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import type { AiToolDispatcher } from './Goal';

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

export type ProcedurePlanRequest = ProcedurePlanRequestSnapshot;

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

export type ProcedurePlanTaskErrorHandler = (
    request: ProcedurePlanRequestSnapshot,
    error: unknown,
) => void;

type ProcedurePlanChangeHandler = (plan: ProcedurePlanState | null) => void;

export type ProcedurePlanTaskResult = {
    title: string;
    tool_name: string;
    status: 'completed' | 'failed';
    run_id: string;
    output?: unknown;
    error?: string;
};

export type ProcedurePlanRunResult = {
    status: 'complete' | 'failed' | 'advanced' | 'cleared';
    goal: string;
    current_request: number;
    request?: ProcedurePlanRequestSnapshot;
    run_id?: string;
    task_results: ProcedurePlanTaskResult[];
    request_count: number;
    reason?: string;
};

export interface ProcedurePlanHandle extends NamedAiToolComponentHandle {
    createProcedurePlan(plan: ProcedurePlanState): Promise<ProcedurePlanRunResult>;
    advancePlanStep(reason?: string): Promise<ProcedurePlanRunResult>;
    clearProcedurePlan(reason?: string): ProcedurePlanRunResult;
    getProcedurePlan(): ProcedurePlanState | null;
}

const defaultProcedurePlanErrorHandler: ProcedurePlanTaskErrorHandler = (request, error) => {
    console.error(`Procedure plan task '${request.title}' failed.`, error);
};

export class ProcedurePlanRunner extends AiToolComponentBase<ProcedurePlanSnapshot | null> {
    private plan: ProcedurePlanState | null = null;
    private active = false;
    private taskResults: ProcedurePlanTaskResult[] = [];
    private lastRunId: string | undefined;
    private readonly onChange: ProcedurePlanChangeHandler;
    private readonly onError: ProcedurePlanTaskErrorHandler;

    constructor(
        componentName: string,
        private readonly dispatchTool: AiToolDispatcher,
        onChange?: ProcedurePlanChangeHandler,
        onError: ProcedurePlanTaskErrorHandler = defaultProcedurePlanErrorHandler,
    ) {
        super(componentName, null);
        this.onChange = onChange ?? (() => undefined);
        this.onError = onError;
    }

    get(): ProcedurePlanState | null {
        return this.plan ? cloneProcedurePlanState(this.plan) : null;
    }

    async replace(plan: ProcedurePlanState | null): Promise<ProcedurePlanRunResult> {
        if (this.active) throw new Error('Cannot replace a running procedure plan.');
        this.taskResults = [];
        this.lastRunId = undefined;
        this.publish(plan?.requests.length ? cloneProcedurePlanState(plan) : null);
        if (!this.plan) return this.result('cleared');
        return this.runNext();
    }

    clear(reason?: string): ProcedurePlanRunResult {
        this.active = false;
        const cleared = this.result('cleared');
        this.publish(null);
        return { ...cleared, ...(reason ? { reason } : {}) };
    }

    async advance(reason?: string): Promise<ProcedurePlanRunResult> {
        if (!this.plan) throw new Error('Cannot advance an empty procedure plan.');
        if (this.active) throw new Error('Cannot advance while a procedure plan step is running.');
        const result = advanceProcedurePlan(this.plan, reason);
        if (result.status === 'complete') {
            const completed = this.result('complete');
            this.finish();
            return completed;
        }
        this.publish(result.plan);
        return this.runNext();
    }

    async retryFailedStep(): Promise<ProcedurePlanRunResult> {
        if (!this.plan || this.active) throw new Error('No failed procedure plan step is available to retry.');
        const request = this.plan.requests[this.plan.currentStep];
        if (!request || request.status !== 'failed') {
            throw new Error('No failed procedure plan step is available to retry.');
        }
        this.publish({
            ...this.plan,
            requests: this.plan.requests.map((item, index) => (
                index === this.plan!.currentStep ? { ...item, status: 'pending' } : item
            )),
        });
        return this.runNext();
    }

    protected onDispose(): void {
        this.active = false;
        this.plan = null;
    }

    private publish(plan: ProcedurePlanState | null): void {
        this.plan = plan;
        this.publishSnapshot(plan ? serializeProcedurePlan(plan) : null);
        this.onChange(plan ? cloneProcedurePlanState(plan) : null);
        if (!plan) this.deleteComponentRef();
    }

    private async runNext(): Promise<ProcedurePlanRunResult> {
        while (this.plan) {
            const request = this.plan.requests[this.plan.currentStep];
            if (!request) {
                const completed = this.result('complete');
                this.finish();
                return completed;
            }
            if (request.status === 'failed') return this.result('failed', request);
            if (request.status !== 'pending') {
                this.publish({ ...this.plan, currentStep: this.plan.currentStep + 1 });
                continue;
            }
            this.active = true;
            this.publish({
                ...this.plan,
                requests: this.plan.requests.map((item, index) => (
                    index === this.plan!.currentStep ? { ...item, status: 'running' } : item
                )),
            });
            const runId = `plan-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
            let output: Record<string, unknown> | null = null;
            let executionError: unknown;
            try {
                output = await this.dispatchTool(
                    request.name || '',
                    getProcedurePlanToolArguments(request),
                );
            } catch (error) {
                executionError = error;
            }
            this.active = false;
            if (!this.plan) return this.result('cleared');
            const taskResult = this.toTaskResult(request, runId, output, executionError);
            this.lastRunId = runId;
            this.taskResults.push(taskResult);
            if (executionError !== undefined) {
                this.onError(serializeProcedurePlanRequest(request), executionError);
                this.publish({
                    ...this.plan!,
                    requests: this.plan!.requests.map((item, index) => (
                        index === this.plan!.currentStep ? { ...item, status: 'failed' } : item
                    )),
                });
                return this.result('failed', this.plan.requests[this.plan.currentStep], runId);
            }
            const nextIndex = this.plan.currentStep + 1;
            this.publish({
                ...this.plan,
                currentStep: nextIndex,
                requests: this.plan.requests.map((item, index) => (
                    index === this.plan!.currentStep ? { ...item, status: 'complete' } : item
                )),
            });
        }
        return this.result('complete');
    }

    private toTaskResult(
        request: ProcedurePlanRequestSnapshot,
        runId: string,
        output: Record<string, unknown> | null,
        error: unknown,
    ): ProcedurePlanTaskResult {
        return {
            title: request.title,
            tool_name: request.name || '',
            status: error === undefined ? 'completed' : 'failed',
            run_id: runId,
            ...(error === undefined
                ? { output }
                : { error: error instanceof Error ? error.message : String(error) }),
        };
    }

    private result(
        status: ProcedurePlanRunResult['status'],
        request?: ProcedurePlanRequestSnapshot,
        runId?: string,
    ): ProcedurePlanRunResult {
        return {
            status,
            goal: this.plan?.goal ?? '',
            current_request: this.plan?.currentStep ?? 0,
            ...(request ? { request: serializeProcedurePlanRequest(request) } : {}),
            ...(runId || this.lastRunId ? { run_id: runId || this.lastRunId } : {}),
            task_results: this.taskResults.map((result) => ({ ...result })),
            request_count: this.plan?.requests.length ?? this.taskResults.length,
        };
    }

    private finish(): void {
        // The completed plan stays mounted until AI Chat replaces it.
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

const cloneProcedurePlanState = (plan: ProcedurePlanState): ProcedurePlanState => ({
    ...plan,
    requests: plan.requests.map((request) => serializeProcedurePlanRequest(request)),
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
    const requests = runnableSnapshots.map((request) => ({
        ...request,
        status: 'pending' as const,
    }));

    return {
        goal: toNonEmptyString(data.goal) || snapshots[0].title,
        requests,
        currentStep: 0,
        sourceEvent: sourceEvent || undefined,
    };
};

export type ProcedurePlanProps = {
    plan: ProcedurePlanSnapshot;
    surface?: 'chat' | 'pill';
    onClear?: () => void;
};

export type ProcedurePlanWorkflowProps = {
    name: string;
    dispatchTool: AiToolDispatcher;
    onSnapshotChange?: (plan: ProcedurePlanState | null) => void;
    surface?: 'chat' | 'pill';
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

export const ProcedurePlanWorkflow: React.FC<ProcedurePlanWorkflowProps> = ({
    name,
    dispatchTool,
    onSnapshotChange,
    surface = 'chat',
}) => {
    const [plan, setPlan] = useState<ProcedurePlanState | null>(null);
    const onSnapshotChangeRef = useRef(onSnapshotChange);
    onSnapshotChangeRef.current = onSnapshotChange;
    const runnerRef = useRef<ProcedurePlanRunner | null>(null);
    if (!runnerRef.current) {
        runnerRef.current = new ProcedurePlanRunner(name, dispatchTool, (next) => {
            setPlan(next);
            onSnapshotChangeRef.current?.(next);
        });
    }

    const handle = useMemo<ProcedurePlanHandle>(() => ({
        getComponentName: () => name,
        createProcedurePlan: (next) => runnerRef.current!.replace(next),
        advancePlanStep: (reason) => runnerRef.current!.advance(reason),
        clearProcedurePlan: (reason) => runnerRef.current!.clear(reason),
        getProcedurePlan: () => runnerRef.current!.get(),
    }), [name]);
    useRegisterAiToolComponentRef(name, handle);

    useEffect(() => () => runnerRef.current?.dispose(), []);

    return plan ? (
        <ProcedurePlan
            plan={serializeProcedurePlan(plan)}
            surface={surface}
            onClear={() => runnerRef.current?.clear()}
        />
    ) : null;
};

export default ProcedurePlan;
