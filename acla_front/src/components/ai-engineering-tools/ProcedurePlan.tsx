import React from 'react';
import { AiToolComponentBase } from './AiToolComponentBase';
import type {
    NamedAiToolComponentHandle,
} from 'contexts/AiToolComponentRefContext';
import type { AiToolDispatcher } from './Goal';
import {
    createControlledAiToolOperation,
    createAiToolOperationFrom,
    type ControlledAiToolOperation,
    type AiToolOperation,
} from './ai-tool-operation';
import {
    ProcedurePlanReplacedError,
    ProcedurePlanStepFailedError,
} from 'contexts/AiToolComponentError';
import { serializeError, type SerializedError } from 'errors/AiToolError';
import type {
    AiOverlayComponentHandle,
    AiOverlayRenderer,
    AiOverlayRendererEvent,
} from 'views/floating-chat/ai-overlay-types';
import {
    isOverlayNonEmptyString,
    isOverlayRecord,
} from 'views/floating-chat/overlay-renderer-validation';

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
    error?: SerializedError;
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

export interface ProcedurePlanHandle extends NamedAiToolComponentHandle, AiOverlayComponentHandle<ProcedurePlanSnapshot | null> {
    createProcedurePlan(plan: ProcedurePlanState): AiToolOperation<ProcedurePlanRunResult>;
    advancePlanStep(reason?: string): AiToolOperation<ProcedurePlanRunResult>;
    clearProcedurePlan(reason?: string): AiToolOperation<ProcedurePlanRunResult>;
    getProcedurePlan(): ProcedurePlanState | null;
}

type ActiveProcedurePlanOperation = {
    controller: ControlledAiToolOperation<
        ProcedurePlanRunResult,
        never,
        'complete' | 'failed' | 'cancelled' | 'replaced'
    >;
};

const defaultProcedurePlanErrorHandler: ProcedurePlanTaskErrorHandler = (request, error) => {
    console.error(`Procedure plan task '${request.title}' failed.`, error);
};

export class ProcedurePlanRunner
extends AiToolComponentBase<ProcedurePlanSnapshot | null>
implements ProcedurePlanHandle {
    private plan: ProcedurePlanState | null = null;
    private active = false;
    private taskResults: ProcedurePlanTaskResult[] = [];
    private lastRunId: string | undefined;
    private generation = 0;
    private activeOperation: ActiveProcedurePlanOperation | null = null;
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

    createProcedurePlan(plan: ProcedurePlanState): AiToolOperation<ProcedurePlanRunResult> {
        return this.replace(plan);
    }

    advancePlanStep(reason?: string): AiToolOperation<ProcedurePlanRunResult> {
        return this.advance(reason);
    }

    clearProcedurePlan(reason?: string): AiToolOperation<ProcedurePlanRunResult> {
        return this.clear(reason);
    }

    getProcedurePlan(): ProcedurePlanState | null {
        return this.get();
    }

    getComponentType(): string {
        return 'procedure_plan';
    }

    getOverlayBehavior(snapshot: ProcedurePlanSnapshot | null) {
        return {
            placement: 'flow' as const,
            requestedStatus: 'expanded' as const,
            remove: snapshot === null,
        };
    }

    getOverlayMetadata() {
        return {};
    }

    handleOverlayRendererEvent(_event: AiOverlayRendererEvent): void {
        // Procedure plan overlays have no renderer-originated events.
    }

    get(): ProcedurePlanState | null {
        return this.plan ? cloneProcedurePlanState(this.plan) : null;
    }

    replace(plan: ProcedurePlanState | null): AiToolOperation<ProcedurePlanRunResult> {
        return this.startOperation(() => this.runReplace(plan));
    }

    private async runReplace(plan: ProcedurePlanState | null): Promise<ProcedurePlanRunResult> {
        const generation = ++this.generation;
        this.active = false;
        this.taskResults = [];
        this.lastRunId = undefined;
        this.publish(plan?.requests.length ? cloneProcedurePlanState(plan) : null);
        if (!this.plan) return this.result('cleared');
        return this.runNext(generation);
    }

    clear(reason?: string): AiToolOperation<ProcedurePlanRunResult> {
        return createAiToolOperationFrom(() => this.runClear(reason), 'cleared');
    }

    private runClear(reason?: string): ProcedurePlanRunResult {
        this.generation += 1;
        this.active = false;
        this.cancelActiveOperation('cancelled', new ProcedurePlanReplacedError(
            this.getComponentName(),
            'The procedure plan was cleared.',
        ));
        const cleared = this.result('cleared');
        this.publish(null);
        return { ...cleared, ...(reason ? { reason } : {}) };
    }

    advance(reason?: string): AiToolOperation<ProcedurePlanRunResult> {
        return this.startOperation(() => this.runAdvance(reason));
    }

    private async runAdvance(reason?: string): Promise<ProcedurePlanRunResult> {
        if (!this.plan) throw new Error('Cannot advance an empty procedure plan.');
        if (this.active) throw new Error('Cannot advance while a procedure plan step is running.');
        const generation = ++this.generation;
        const result = advanceProcedurePlan(this.plan, reason);
        if (result.status === 'complete') {
            const completed = this.result('complete');
            this.finish();
            return completed;
        }
        this.publish(result.plan);
        return this.runNext(generation);
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
        return this.runNext(++this.generation);
    }

    protected onDispose(): void {
        this.generation += 1;
        this.active = false;
        this.cancelActiveOperation('cancelled', new ProcedurePlanReplacedError(
            this.getComponentName(),
            'The procedure plan was disposed.',
        ));
        this.plan = null;
    }

    protected cloneSnapshot(
        snapshot: ProcedurePlanSnapshot | null,
    ): ProcedurePlanSnapshot | null {
        return snapshot ? serializeProcedurePlan(snapshot) : null;
    }

    private startOperation(
        run: () => Promise<ProcedurePlanRunResult>,
    ): AiToolOperation<ProcedurePlanRunResult> {
        this.cancelActiveOperation('replaced', new ProcedurePlanReplacedError(
            this.getComponentName(),
            'The procedure plan operation was replaced.',
        ));
        const operation: ActiveProcedurePlanOperation = {
            controller: createControlledAiToolOperation<
                ProcedurePlanRunResult,
                never,
                'complete' | 'failed' | 'cancelled' | 'replaced'
            >(),
        };
        this.activeOperation = operation;
        void run().then(
            (result) => operation.controller.resolve('complete', result),
            (error) => operation.controller.reject(
                'failed',
                error instanceof Error ? error : new Error(String(error)),
            ),
        ).finally(() => {
            if (this.activeOperation === operation) this.activeOperation = null;
        });
        return operation.controller.operation;
    }

    private cancelActiveOperation(
        status: 'cancelled' | 'replaced',
        error: Error,
    ): void {
        const operation = this.activeOperation;
        if (!operation) return;
        this.activeOperation = null;
        operation.controller.reject(status, error);
    }

    private publish(plan: ProcedurePlanState | null): void {
        this.plan = plan;
        this.publishSnapshot(plan ? serializeProcedurePlan(plan) : null);
        this.onChange(plan ? cloneProcedurePlanState(plan) : null);
        if (!plan) this.deleteComponentRef();
    }

    private async runNext(generation: number): Promise<ProcedurePlanRunResult> {
        while (this.plan) {
            if (generation !== this.generation) {
                throw new ProcedurePlanReplacedError(
                    this.getComponentName(),
                    'The procedure plan operation was replaced.',
                );
            }
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
            let output: import('./Goal').NestedAiToolResult | null = null;
            let executionError: unknown;
            try {
                const operation = this.dispatchTool(
                    request.name || '',
                    getProcedurePlanToolArguments(request),
                );
                const termination = await new Promise<{
                    status: string;
                    result: import('./Goal').NestedAiToolResult | Error;
                }>((resolve) => operation.notifyTerminated(resolve));
                if (termination.result instanceof Error) throw termination.result;
                output = termination.result;
            } catch (error) {
                executionError = error;
            }
            this.active = false;
            if (generation !== this.generation) {
                throw new ProcedurePlanReplacedError(
                    this.getComponentName(),
                    'The procedure plan operation was replaced.',
                );
            }
            if (!this.plan) return this.result('cleared');
            const stepError = executionError === undefined
                ? undefined
                : new ProcedurePlanStepFailedError(
                    this.getComponentName(),
                    executionError instanceof Error && executionError.message
                        ? executionError.message
                        : 'The procedure plan step failed.',
                    { cause: executionError },
                );
            const taskResult = this.toTaskResult(request, runId, output, stepError);
            this.lastRunId = runId;
            this.taskResults.push(taskResult);
            if (stepError) {
                this.onError(serializeProcedurePlanRequest(request), stepError);
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
        output: import('./Goal').NestedAiToolResult | null,
        error: ProcedurePlanStepFailedError | undefined,
    ): ProcedurePlanTaskResult {
        return {
            title: request.title,
            tool_name: request.name || '',
            status: error === undefined ? 'completed' : 'failed',
            run_id: runId,
            ...(error === undefined
                ? { output }
                : { error: serializeError(error) }),
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

    const title = toNonEmptyString(request.title);
    const name = toNonEmptyString(request.name);
    if (!title) return null;
    const type = toNonEmptyString(request.type) || (name ? 'tool_call' : 'request');

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
};

const getProcedurePlanRequestMeta = (request: ProcedurePlanSnapshot['requests'][number]): string => {
    const parts = [
        request.type === 'tool_call' ? null : request.type,
        request.status,
    ].filter((part): part is string => Boolean(part));
    return parts.join(' - ');
};

const ProcedurePlan: React.FC<ProcedurePlanProps> = ({
    plan,
    surface = 'chat',
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

export const procedurePlanOverlayRenderer: AiOverlayRenderer<ProcedurePlanSnapshot> = {
    componentType: 'procedure_plan',
    validateSnapshot: (snapshot): snapshot is ProcedurePlanSnapshot => (
        isOverlayRecord(snapshot)
        && isOverlayNonEmptyString(snapshot.goal)
        && Array.isArray(snapshot.requests)
        && snapshot.requests.every((request) => (
            isOverlayRecord(request) && isOverlayNonEmptyString(request.title)
        ))
        && typeof snapshot.currentStep === 'number'
        && Number.isInteger(snapshot.currentStep)
    ),
    renderOverlay: (snapshot, status) => status === 'folded'
        ? snapshot.requests[snapshot.currentStep]?.title || snapshot.goal
        : <ProcedurePlan plan={snapshot} surface="pill" />,
    dimensions: {
        expanded: { width: 420, height: 220 },
        folded: { width: 320, height: 58 },
    },
};

export default ProcedurePlan;
