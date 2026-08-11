import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { NamedAiToolComponentHandle } from 'contexts/AiToolComponentRefContext';
import { useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import type { TaskStartFunction } from './task-start-function';

export const GOAL_COMPARISON_OPERATORS = [
    'eq',
    'neq',
    'lt',
    'lte',
    'gt',
    'gte',
] as const;

export type GoalComparisonOperator = typeof GOAL_COMPARISON_OPERATORS[number];
export type GoalStatus = 'running' | 'achieved' | 'missed' | 'error';
export type GoalStepStatus = 'pending' | 'running' | 'completed' | 'error';

export type GoalStep = {
    id: string;
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
    taskStart: TaskStartFunction;
};

export type GoalStepDescriptor = Omit<GoalStep, 'taskStart'>;

export type GoalComparison = {
    step_id: string;
    result_path: string;
    operator: GoalComparisonOperator;
    target: number;
    metric_label: string;
    unit?: string;
};

export type GoalStepSnapshot = GoalStepDescriptor & {
    status: GoalStepStatus;
    attempts: number;
    run_id?: string;
    error?: string;
};

export type GoalSourceResultMetadata = {
    step_id: string;
    tool_name: string;
    run_id: string;
    status: string;
    final: true;
};

export type GoalTaskResult = {
    step_id: string;
    tool_name: string;
    attempt: number;
    status: 'completed' | 'error';
    value: unknown;
    error?: string;
    source_result?: GoalSourceResultMetadata;
};

export type GoalSnapshot = {
    goal: string;
    status: GoalStatus;
    steps: GoalStepSnapshot[];
    comparison: GoalComparison | null;
    target: number | null;
    actual: number | null;
    completed_steps: string[];
    source_result: GoalSourceResultMetadata | null;
    failed_step?: string;
    error?: string;
};

export type GoalToolOutputEnvelope = {
    tool_name: string;
    run_id: string;
    status: string;
    output: unknown;
    final: boolean;
    message?: string;
};

export type GoalRunResult = Pick<
    GoalSnapshot,
    | 'goal'
    | 'status'
    | 'target'
    | 'actual'
    | 'completed_steps'
    | 'source_result'
    | 'failed_step'
    | 'error'
> & {
    comparison: GoalComparison | null;
    task_results: GoalTaskResult[];
};

export interface GoalHandle extends NamedAiToolComponentHandle {
    createGoal(input: GoalRequest): Promise<GoalRunResult>;
    retryFailedTask(): Promise<GoalRunResult>;
    acceptToolOutput(envelope: GoalToolOutputEnvelope): void;
    getSnapshot(): GoalSnapshot | null;
    clear(): void;
}

export type GoalDisplayProps = {
    snapshot: GoalSnapshot;
    surface?: 'chat' | 'pill';
};

export type GoalProps = {
    name: string;
    onSnapshotChange?: (snapshot: GoalSnapshot | null) => void;
    surface?: 'chat' | 'pill';
};

export type GoalRequest = {
    goal: string;
    steps: GoalStep[];
    comparison: GoalComparison;
};

export type GoalTaskStartFunctionSelector = (
    step: GoalStepDescriptor,
) => TaskStartFunction | null | undefined;

type FinalWaiter = {
    toolName: string;
    runId: string | null;
    resolve: (envelope: GoalToolOutputEnvelope) => void;
    reject: (error: Error) => void;
};

const RETRY_DELAY_MS = 1000;
const MAX_GOAL_ATTEMPTS = 2;
const UNSAFE_RESULT_PATH_SEGMENTS = new Set(['__proto__', 'prototype', 'constructor']);
const RESULT_PATH_SEGMENT_RE = /^(?:[A-Za-z_][A-Za-z0-9_]*|0|[1-9][0-9]*)$/;

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const isGoalToolOutputEnvelope = (value: unknown): value is GoalToolOutputEnvelope => (
    isRecord(value)
    && typeof value.tool_name === 'string'
    && typeof value.run_id === 'string'
    && typeof value.status === 'string'
    && typeof value.final === 'boolean'
    && Object.prototype.hasOwnProperty.call(value, 'output')
);

const normalizeTaskValue = (value: unknown): unknown => (
    value === undefined ? null : value
);

const normalizeTaskError = (value: unknown, fallback = 'goal_step_failed'): string => {
    if (value instanceof Error && value.message.trim()) return value.message.trim();
    if (typeof value === 'string' && value.trim()) return value.trim();
    if (isRecord(value)) {
        if (typeof value.error === 'string' && value.error.trim()) return value.error.trim();
        if (typeof value.message === 'string' && value.message.trim()) return value.message.trim();
    }
    if (value !== null && value !== undefined) {
        if (typeof value !== 'object') return String(value);
        try {
            const serialized = JSON.stringify(value);
            if (serialized && serialized !== '{}') return serialized;
        } catch {
            // Fall through to the stable generic error.
        }
    }
    return fallback;
};

const toNonEmptyString = (value: unknown): string | null => {
    if (typeof value !== 'string') return null;
    const normalized = value.trim();
    return normalized || null;
};

const isGoalComparisonOperator = (value: unknown): value is GoalComparisonOperator => (
    typeof value === 'string'
    && (GOAL_COMPARISON_OPERATORS as readonly string[]).includes(value)
);

export const isSafeGoalResultPath = (path: unknown): path is string => {
    const normalized = toNonEmptyString(path);
    if (!normalized) return false;
    const segments = normalized.split('.');
    return segments.every((segment) => (
        RESULT_PATH_SEGMENT_RE.test(segment)
        && !UNSAFE_RESULT_PATH_SEGMENTS.has(segment)
    ));
};

const parseGoalStepDescriptor = (value: unknown): GoalStepDescriptor | null => {
    const step = isRecord(value) ? value : null;
    if (!step) return null;
    const id = toNonEmptyString(step.id);
    const title = toNonEmptyString(step.title);
    const name = toNonEmptyString(step.name);
    if (!id || !title || !name) return null;
    if (step.arguments !== undefined && !isRecord(step.arguments)) return null;
    return {
        id,
        title,
        name,
        ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
    };
};

const parseGoalComparison = (value: unknown): GoalComparison | null => {
    const comparison = isRecord(value) ? value : null;
    if (!comparison) return null;
    const stepId = toNonEmptyString(comparison.step_id);
    const resultPath = toNonEmptyString(comparison.result_path);
    const metricLabel = toNonEmptyString(comparison.metric_label);
    const unit = comparison.unit === undefined ? undefined : toNonEmptyString(comparison.unit);
    if (
        !stepId
        || !resultPath
        || !metricLabel
        || !isSafeGoalResultPath(resultPath)
        || !isGoalComparisonOperator(comparison.operator)
        || typeof comparison.target !== 'number'
        || !Number.isFinite(comparison.target)
        || (comparison.unit !== undefined && !unit)
    ) {
        return null;
    }
    return {
        step_id: stepId,
        result_path: resultPath,
        operator: comparison.operator,
        target: comparison.target,
        metric_label: metricLabel,
        ...(unit ? { unit } : {}),
    };
};

export const validateGoalRequest = (
    value: unknown,
): { request: GoalRequest } | { error: string; goal?: string } => {
    const input = isRecord(value) ? value : null;
    const goal = toNonEmptyString(input?.goal);
    if (!input || !goal) return { error: 'invalid_goal_title' };
    if (!Array.isArray(input.steps) || input.steps.length === 0) {
        return { error: 'invalid_goal_steps', goal };
    }
    const steps = input.steps.map((step) => {
        const descriptor = parseGoalStepDescriptor(step);
        if (!descriptor || !isRecord(step) || typeof step.taskStart !== 'function') return null;
        return { ...descriptor, taskStart: step.taskStart as TaskStartFunction };
    });
    if (steps.some((step) => !step)) return { error: 'invalid_goal_steps', goal };
    const parsedSteps = steps as GoalStep[];
    const ids = new Set<string>();
    for (const step of parsedSteps) {
        if (ids.has(step.id)) return { error: 'duplicate_goal_step_id', goal };
        ids.add(step.id);
        if (step.name === 'create_goal' || step.name === 'retry_goal_task') {
            return { error: 'recursive_goal_step', goal };
        }
    }
    const comparison = parseGoalComparison(input.comparison);
    if (!comparison) return { error: 'invalid_goal_comparison', goal };
    if (
        !ids.has(comparison.step_id)
        || parsedSteps[parsedSteps.length - 1].id !== comparison.step_id
    ) {
        return { error: 'invalid_goal_comparison_step', goal };
    }
    return { request: { goal, steps: parsedSteps, comparison } };
};

export const buildGoalRequest = (
    value: unknown,
    selectTaskStartFunction: GoalTaskStartFunctionSelector,
): { request: GoalRequest } | { error: string; goal?: string } => {
    const input = isRecord(value) ? value : null;
    const goal = toNonEmptyString(input?.goal);
    if (!input || !goal) return { error: 'invalid_goal_title' };
    if (!Array.isArray(input.steps) || input.steps.length === 0) {
        return { error: 'invalid_goal_steps', goal };
    }

    const steps: GoalStep[] = [];
    for (const rawStep of input.steps) {
        const descriptor = parseGoalStepDescriptor(rawStep);
        if (!descriptor) return { error: 'invalid_goal_steps', goal };
        if (descriptor.name === 'create_goal' || descriptor.name === 'retry_goal_task') {
            return { error: 'recursive_goal_step', goal };
        }
        const taskStart = selectTaskStartFunction(descriptor);
        if (typeof taskStart !== 'function') {
            return { error: 'goal_step_task_unavailable', goal };
        }
        steps.push({ ...descriptor, taskStart });
    }

    return validateGoalRequest({ ...input, goal, steps });
};

export const extractGoalResultPath = (value: unknown, path: string): unknown => {
    if (!isSafeGoalResultPath(path)) return undefined;
    let current = value;
    for (const segment of path.split('.')) {
        if (Array.isArray(current) && /^\d+$/.test(segment)) {
            const index = Number(segment);
            if (!Object.prototype.hasOwnProperty.call(current, index)) return undefined;
            current = current[index];
            continue;
        }
        if (!isRecord(current) || !Object.prototype.hasOwnProperty.call(current, segment)) {
            return undefined;
        }
        current = current[segment];
    }
    return current;
};

export const compareGoalValues = (
    actual: number,
    operator: GoalComparisonOperator,
    target: number,
): boolean => {
    switch (operator) {
        case 'eq': return actual === target;
        case 'neq': return actual !== target;
        case 'lt': return actual < target;
        case 'lte': return actual <= target;
        case 'gt': return actual > target;
        case 'gte': return actual >= target;
        default: return false;
    }
};

const cloneSnapshot = (snapshot: GoalSnapshot): GoalSnapshot => ({
    ...snapshot,
    steps: snapshot.steps.map((step) => ({
        ...step,
        ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
    })),
    comparison: snapshot.comparison ? { ...snapshot.comparison } : null,
    completed_steps: [...snapshot.completed_steps],
    source_result: snapshot.source_result ? { ...snapshot.source_result } : null,
});

const cloneTaskResults = (taskResults: GoalTaskResult[]): GoalTaskResult[] => (
    taskResults.map((result) => ({
        ...result,
        ...(result.source_result ? { source_result: { ...result.source_result } } : {}),
    }))
);

const toRunResult = (
    snapshot: GoalSnapshot,
    taskResults: GoalTaskResult[],
): GoalRunResult => ({
    goal: snapshot.goal,
    status: snapshot.status,
    comparison: snapshot.comparison ? { ...snapshot.comparison } : null,
    target: snapshot.target,
    actual: snapshot.actual,
    completed_steps: [...snapshot.completed_steps],
    source_result: snapshot.source_result ? { ...snapshot.source_result } : null,
    task_results: cloneTaskResults(taskResults),
    ...(snapshot.failed_step ? { failed_step: snapshot.failed_step } : {}),
    ...(snapshot.error ? { error: snapshot.error } : {}),
});

type StepExecutionResult = {
    value: unknown;
    error?: string;
    source_result?: GoalSourceResultMetadata;
};

class GoalRunner {
    private snapshot: GoalSnapshot | null = null;
    private controller: AbortController | null = null;
    private generation = 0;
    private finalWaiter: FinalWaiter | null = null;
    private request: GoalRequest | null = null;
    private failedStepIndex: number | null = null;
    private stepAttempts: number[] = [];
    private taskResults: GoalTaskResult[] = [];
    private goalAttempt = 1;

    constructor(
        private readonly onChange: (snapshot: GoalSnapshot | null) => void,
    ) {}

    getSnapshot(): GoalSnapshot | null {
        return this.snapshot ? cloneSnapshot(this.snapshot) : null;
    }

    acceptToolOutput(envelope: GoalToolOutputEnvelope): void {
        const waiter = this.finalWaiter;
        if (!waiter || waiter.toolName !== envelope.tool_name) return;
        if (waiter.runId && waiter.runId !== envelope.run_id) return;
        if (!waiter.runId) waiter.runId = envelope.run_id;
        if (!envelope.final) return;
        this.finalWaiter = null;
        waiter.resolve(envelope);
    }

    async create(input: GoalRequest): Promise<GoalRunResult> {
        this.cancelActive('goal_replaced');
        const generation = ++this.generation;
        this.request = null;
        this.failedStepIndex = null;
        this.stepAttempts = [];
        this.taskResults = [];
        this.goalAttempt = 1;
        const validation = validateGoalRequest(input);
        if ('error' in validation) {
            const invalidSnapshot: GoalSnapshot = {
                goal: validation.goal || 'Goal',
                status: 'error',
                steps: [],
                comparison: null,
                target: null,
                actual: null,
                completed_steps: [],
                source_result: null,
                error: validation.error,
            };
            this.publish(invalidSnapshot);
            return toRunResult(invalidSnapshot, this.taskResults);
        }

        const request = validation.request;
        this.request = request;
        this.stepAttempts = request.steps.map(() => 0);
        const controller = new AbortController();
        this.controller = controller;
        this.publish(this.createRunningSnapshot(request));

        return this.runSteps(request, generation, controller, 0);
    }

    async retryFailedTask(): Promise<GoalRunResult> {
        const request = this.request;
        const failedStepIndex = this.failedStepIndex;
        if (
            !request
            || !this.snapshot
            || this.snapshot.status !== 'error'
            || failedStepIndex === null
            || this.snapshot.failed_step !== request.steps[failedStepIndex]?.id
        ) {
            return this.retryUnavailableResult();
        }

        const generation = ++this.generation;
        const controller = new AbortController();
        this.controller = controller;
        this.failedStepIndex = null;
        const {
            failed_step: _failedStep,
            error: _error,
            ...retrySnapshot
        } = this.snapshot;
        this.publish({
            ...retrySnapshot,
            status: 'running',
            actual: null,
            source_result: null,
        });

        return this.runSteps(request, generation, controller, failedStepIndex);
    }

    clear(): void {
        this.cancelActive('goal_cleared');
        this.generation += 1;
        this.snapshot = null;
        this.request = null;
        this.failedStepIndex = null;
        this.stepAttempts = [];
        this.taskResults = [];
        this.onChange(null);
    }

    dispose(): void {
        this.cancelActive('goal_disposed');
        this.snapshot = null;
        this.request = null;
        this.failedStepIndex = null;
    }

    private async runSteps(
        request: GoalRequest,
        generation: number,
        controller: AbortController,
        startIndex: number,
    ): Promise<GoalRunResult> {
        for (let index = startIndex; index < request.steps.length; index += 1) {
            const step = request.steps[index];
            if (!this.isCurrent(generation, controller)) {
                return this.cancelledResult(request, 'goal_replaced');
            }

            const attempt = (this.stepAttempts[index] ?? 0) + 1;
            this.stepAttempts[index] = attempt;
            this.updateStep(index, {
                status: 'running',
                attempts: attempt,
                error: undefined,
            });

            const execution = await this.executeStep(step, controller.signal);
            if (!this.isCurrent(generation, controller)) {
                return this.cancelledResult(request, 'goal_replaced');
            }

            let error = execution.error;
            let actual: number | null = null;
            if (!error && step.id === request.comparison.step_id) {
                const comparisonValue = extractGoalResultPath(
                    execution.value,
                    request.comparison.result_path,
                );
                if (typeof comparisonValue !== 'number' || !Number.isFinite(comparisonValue)) {
                    error = 'goal_comparison_value_not_numeric';
                } else {
                    actual = comparisonValue;
                }
            }

            const taskResult: GoalTaskResult = {
                step_id: step.id,
                tool_name: step.name,
                attempt,
                status: error ? 'error' : 'completed',
                value: execution.value,
                ...(error ? { error } : {}),
                ...(execution.source_result
                    ? { source_result: { ...execution.source_result } }
                    : {}),
            };
            this.taskResults.push(taskResult);

            if (error) {
                this.updateStep(index, { status: 'error', error });
                this.failedStepIndex = index;
                const errorSnapshot: GoalSnapshot = {
                    ...this.snapshot!,
                    status: 'error',
                    actual: null,
                    source_result: execution.source_result
                        ? { ...execution.source_result }
                        : null,
                    failed_step: step.id,
                    error,
                };
                this.publish(errorSnapshot);
                this.controller = null;
                return toRunResult(errorSnapshot, this.taskResults);
            }

            this.updateStep(index, { status: 'completed', error: undefined });
            const completed = this.snapshot!.steps
                .filter((item) => item.status === 'completed')
                .map((item) => item.id);
            this.publish({ ...this.snapshot!, completed_steps: completed });

            if (step.id === request.comparison.step_id) {
                const achieved = compareGoalValues(
                    actual!,
                    request.comparison.operator,
                    request.comparison.target,
                );
                const {
                    failed_step: _failedStep,
                    error: _error,
                    ...completedSnapshot
                } = this.snapshot!;
                const finalSnapshot: GoalSnapshot = {
                    ...completedSnapshot,
                    status: achieved ? 'achieved' : 'missed',
                    actual,
                    completed_steps: completed,
                    source_result: execution.source_result
                        ? { ...execution.source_result }
                        : null,
                };
                this.publish(finalSnapshot);
                if (achieved || this.goalAttempt === MAX_GOAL_ATTEMPTS) {
                    this.controller = null;
                    return toRunResult(finalSnapshot, this.taskResults);
                }
                try {
                    await this.retryDelay(controller.signal);
                } catch {
                    return this.cancelledResult(request, 'goal_replaced');
                }
                if (!this.isCurrent(generation, controller)) {
                    return this.cancelledResult(request, 'goal_replaced');
                }
                this.goalAttempt += 1;
                this.publish(this.createRunningSnapshot(request));
                index = -1;
            }
        }

        const invalidFinalSnapshot: GoalSnapshot = {
            ...this.snapshot!,
            status: 'error',
            error: 'goal_comparison_step_not_executed',
        };
        this.publish(invalidFinalSnapshot);
        this.controller = null;
        return toRunResult(invalidFinalSnapshot, this.taskResults);
    }

    private createRunningSnapshot(request: GoalRequest): GoalSnapshot {
        return {
            goal: request.goal,
            status: 'running',
            steps: request.steps.map((step, index) => ({
                id: step.id,
                title: step.title,
                name: step.name,
                ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
                status: 'pending',
                attempts: this.stepAttempts[index] ?? 0,
            })),
            comparison: request.comparison,
            target: request.comparison.target,
            actual: null,
            completed_steps: [],
            source_result: null,
        };
    }

    private publish(snapshot: GoalSnapshot): void {
        this.snapshot = cloneSnapshot(snapshot);
        this.onChange(this.getSnapshot());
    }

    private updateStep(index: number, update: Partial<GoalStepSnapshot>): void {
        if (!this.snapshot) return;
        this.publish({
            ...this.snapshot,
            steps: this.snapshot.steps.map((step, stepIndex) => (
                stepIndex === index ? { ...step, ...update } : step
            )),
        });
    }

    private isCurrent(generation: number, controller: AbortController): boolean {
        return (
            generation === this.generation
            && this.controller === controller
            && !controller.signal.aborted
        );
    }

    private cancelActive(reason: string): void {
        if (!this.controller) return;
        this.controller.abort();
        this.controller = null;
        this.finalWaiter?.reject(new Error(reason));
        this.finalWaiter = null;
    }

    private cancelledResult(request: GoalRequest, error: string): GoalRunResult {
        return {
            goal: request.goal,
            status: 'error',
            comparison: request.comparison,
            target: request.comparison.target,
            actual: null,
            completed_steps: this.snapshot?.completed_steps ?? [],
            source_result: null,
            task_results: cloneTaskResults(this.taskResults),
            error,
        };
    }

    private retryUnavailableResult(): GoalRunResult {
        return {
            goal: this.snapshot?.goal ?? 'Goal',
            status: 'error',
            comparison: this.snapshot?.comparison
                ? { ...this.snapshot.comparison }
                : null,
            target: this.snapshot?.target ?? null,
            actual: this.snapshot?.actual ?? null,
            completed_steps: [...(this.snapshot?.completed_steps ?? [])],
            source_result: this.snapshot?.source_result
                ? { ...this.snapshot.source_result }
                : null,
            task_results: cloneTaskResults(this.taskResults),
            error: 'goal_task_retry_unavailable',
        };
    }

    private async executeStep(
        step: GoalStep,
        signal: AbortSignal,
    ): Promise<StepExecutionResult> {
        let deliveredEnvelope: GoalToolOutputEnvelope | null = null;
        let waiter!: FinalWaiter;
        const finalPromise = new Promise<GoalToolOutputEnvelope>((resolve, reject) => {
            waiter = {
                toolName: step.name,
                runId: null,
                resolve: (envelope) => {
                    deliveredEnvelope = envelope;
                    resolve(envelope);
                },
                reject,
            };
            this.finalWaiter = waiter;
        });
        void finalPromise.catch(() => undefined);
        const abortFinalWait = () => {
            if (this.finalWaiter !== waiter) return;
            this.finalWaiter = null;
            waiter.reject(new Error('goal_cancelled'));
        };
        signal.addEventListener('abort', abortFinalWait, { once: true });

        try {
            const returned = await step.taskStart(signal);
            if (deliveredEnvelope) {
                return this.executionFromEnvelope(step, deliveredEnvelope);
            }
            if (isGoalToolOutputEnvelope(returned)) {
                if (returned.tool_name !== step.name) {
                    return {
                        value: normalizeTaskValue(returned.output),
                        error: 'goal_step_output_tool_mismatch',
                    };
                }
                if (returned.final) return this.executionFromEnvelope(step, returned);
                waiter.runId = returned.run_id;
                return this.executionFromEnvelope(step, await finalPromise);
            }
            if (returned === undefined && waiter.runId) {
                return this.executionFromEnvelope(step, await finalPromise);
            }

            return { value: normalizeTaskValue(returned) };
        } catch (error) {
            return { value: null, error: normalizeTaskError(error) };
        } finally {
            signal.removeEventListener('abort', abortFinalWait);
            if (this.finalWaiter === waiter) this.finalWaiter = null;
        }
    }

    private executionFromEnvelope(
        step: GoalStep,
        envelope: GoalToolOutputEnvelope,
    ): StepExecutionResult {
        const sourceResult: GoalSourceResultMetadata = {
            step_id: step.id,
            tool_name: envelope.tool_name,
            run_id: envelope.run_id,
            status: envelope.status,
            final: true,
        };
        return {
            value: normalizeTaskValue(envelope.output),
            source_result: sourceResult,
        };
    }

    private retryDelay(signal: AbortSignal): Promise<void> {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(resolve, RETRY_DELAY_MS);
            signal.addEventListener('abort', () => {
                clearTimeout(timer);
                reject(new Error('goal_cancelled'));
            }, { once: true });
        });
    }
}

const getComparisonText = (snapshot: GoalSnapshot): string => {
    const comparison = snapshot.comparison;
    if (!comparison) return snapshot.error || 'Invalid goal';
    const unit = comparison.unit ? ` ${comparison.unit}` : '';
    const actual = snapshot.actual === null ? '—' : `${snapshot.actual}${unit}`;
    return `${comparison.metric_label}: ${actual} / target ${comparison.target}${unit}`;
};

export const GoalDisplay: React.FC<GoalDisplayProps> = ({ snapshot, surface = 'chat' }) => (
    <section
        className={`ai-chat__goal ai-chat__goal--${surface} ai-chat__goal--${snapshot.status}`}
        aria-label="Goal"
        aria-live="polite"
    >
        <div className="ai-chat__goal-head">
            <div>
                <span className="ai-chat__goal-kicker">GOAL · {snapshot.status}</span>
                <div className="ai-chat__goal-title">{snapshot.goal}</div>
            </div>
            <span className="ai-chat__goal-metric">{getComparisonText(snapshot)}</span>
        </div>
        {snapshot.steps.length > 0 && (
            <ol className="ai-chat__goal-steps">
                {snapshot.steps.map((step) => (
                    <li
                        key={step.id}
                        className={`ai-chat__goal-step ai-chat__goal-step--${step.status}`}
                    >
                        <span className="ai-chat__goal-step-dot" aria-hidden="true" />
                        <span className="ai-chat__goal-step-copy">
                            <span>{step.title}</span>
                            <span>{step.status}{step.attempts > 1 ? ` · attempt ${step.attempts}` : ''}</span>
                            {step.error && <span className="ai-chat__goal-error">{step.error}</span>}
                        </span>
                    </li>
                ))}
            </ol>
        )}
        {snapshot.error && snapshot.steps.length === 0 && (
            <div className="ai-chat__goal-error">{snapshot.error}</div>
        )}
    </section>
);

const Goal: React.FC<GoalProps> = ({
    name,
    onSnapshotChange,
    surface = 'chat',
}) => {
    const [snapshot, setSnapshot] = useState<GoalSnapshot | null>(null);
    const onSnapshotChangeRef = useRef(onSnapshotChange);
    onSnapshotChangeRef.current = onSnapshotChange;
    const runnerRef = useRef<GoalRunner | null>(null);
    if (!runnerRef.current) {
        runnerRef.current = new GoalRunner((next) => {
            setSnapshot(next);
            onSnapshotChangeRef.current?.(next);
        });
    }

    const handle = useMemo<GoalHandle>(() => ({
        getComponentName: () => name,
        createGoal: (input) => runnerRef.current!.create(input),
        retryFailedTask: () => runnerRef.current!.retryFailedTask(),
        acceptToolOutput: (envelope) => runnerRef.current!.acceptToolOutput(envelope),
        getSnapshot: () => runnerRef.current!.getSnapshot(),
        clear: () => runnerRef.current!.clear(),
    }), [name]);
    useRegisterAiToolComponentRef(name, handle);

    useEffect(() => () => runnerRef.current?.dispose(), []);

    return snapshot ? <GoalDisplay snapshot={snapshot} surface={surface} /> : null;
};

export default Goal;
