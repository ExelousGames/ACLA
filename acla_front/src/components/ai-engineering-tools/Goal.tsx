import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { NamedAiToolComponentHandle } from 'contexts/AiToolComponentRefContext';
import { useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import {
    AiToolComponentErrorConstructor,
    DuplicateGoalStepIdError,
    GoalComponentError,
    GoalDeterminationFailedError,
    GoalDeterminationValueNotNumericError,
    GoalReplacedError,
    GoalStepFailedError,
    GoalTaskRetryUnavailableError,
    InvalidGoalDeterminationError,
    InvalidGoalNameError,
    InvalidGoalStepsError,
    RecursiveGoalDeterminationError,
    RecursiveGoalStepError,
} from 'contexts/AiToolComponentError';
import { AiToolComponentBase } from './AiToolComponentBase';
import {
    createAiToolDeferred,
    createAiToolOperation,
    createAiToolOperationFrom,
    mapAiToolOperation,
    type AiToolDeferred,
    type AiToolOperation,
} from './ai-tool-operation';

export interface NestedAiToolResult {
    [key: string]: unknown;
}
export interface NestedAiToolStatus {
    [key: string]: unknown;
}

export type AiToolDispatcher = (
    name: string,
    args?: Record<string, unknown>,
) => AiToolOperation<NestedAiToolResult, NestedAiToolStatus>;

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
export type GoalDeterminationStatus = GoalStepStatus;

export type GoalStepDescriptor = {
    id: string;
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalDeterminationTool = {
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalDetermination = {
    tool: GoalDeterminationTool;
    result_path: string;
    operator: GoalComparisonOperator;
    target: number;
};

export type GoalRequest = {
    name: string;
    steps: GoalStepDescriptor[];
    determination: GoalDetermination;
};

export type GoalTaskDescriptor = {
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalStepSnapshot = GoalStepDescriptor & {
    status: GoalStepStatus;
    attempts: number;
    run_id?: string;
    error?: string;
};

export type GoalSourceResultMetadata = {
    tool_name: string;
    run_id: string;
    status: string;
};

export type GoalStepSourceResultMetadata = GoalSourceResultMetadata & {
    step_id: string;
};

export type GoalTaskResult = {
    step_id: string;
    tool_name: string;
    attempt: number;
    status: 'completed' | 'error';
    value: unknown;
    error?: string;
    source_result?: GoalStepSourceResultMetadata;
};

export type GoalDeterminationResult = {
    tool_name: string;
    attempt: number;
    status: GoalDeterminationStatus;
    value: number | null;
    error?: string;
    source_result?: GoalSourceResultMetadata;
};

export type GoalSnapshot = {
    name: string;
    status: GoalStatus;
    steps: GoalStepSnapshot[];
    determination: GoalDetermination | null;
    determination_result: GoalDeterminationResult | null;
    target: number | null;
    actual: number | null;
    completed_steps: string[];
    failed_step?: string;
    error?: string;
};

export type GoalRunResult = Pick<
    GoalSnapshot,
    | 'name'
    | 'target'
    | 'actual'
    | 'completed_steps'
    | 'determination'
    | 'determination_result'
> & {
    status: 'achieved' | 'missed' | 'failed';
    task_results: GoalTaskResult[];
    failed_step?: string;
    error?: string;
};

export type GoalAiResult = Omit<GoalRunResult, 'name'> & { goal: string };

export type GoalProgressStatus = {
    workflow: 'goal';
    phase: 'step' | 'determination';
    id: string;
    tool_name: string;
    status: 'completed' | 'failed' | 'skipped';
    value?: unknown;
    error?: string;
    nested_statuses?: readonly object[];
};

export interface GoalHandle extends NamedAiToolComponentHandle {
    createGoal(input: GoalRequest): AiToolOperation<GoalAiResult, GoalProgressStatus>;
    retryFailedTask(): AiToolOperation<GoalAiResult, GoalProgressStatus>;
    getSnapshot(): GoalSnapshot | null;
    clear(): void;
}

export type GoalDisplayProps = {
    snapshot: GoalSnapshot;
    surface?: 'chat' | 'pill';
};

export type GoalProps = {
    name: string;
    dispatchTool: AiToolDispatcher;
    onSnapshotChange?: (snapshot: GoalSnapshot | null) => void;
    surface?: 'chat' | 'pill';
};

const RETRY_DELAY_MS = 1000;
const MAX_GOAL_ATTEMPTS = 2;
const RECURSIVE_GOAL_TOOL_NAMES = new Set(['create_goal', 'retry_goal_task']);
const UNSAFE_RESULT_PATH_SEGMENTS = new Set(['__proto__', 'prototype', 'constructor']);
const RESULT_PATH_SEGMENT_RE = /^(?:[A-Za-z_][A-Za-z0-9_]*|0|[1-9][0-9]*)$/;

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const hasOnlyKeys = (value: Record<string, unknown>, allowedKeys: readonly string[]): boolean => {
    const allowed = new Set(allowedKeys);
    return Object.keys(value).every((key) => allowed.has(key));
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
    if (!step || !hasOnlyKeys(step, ['id', 'title', 'name', 'arguments'])) return null;
    const id = toNonEmptyString(step.id);
    const title = toNonEmptyString(step.title);
    const name = toNonEmptyString(step.name);
    if (!id || !title || !name) return null;
    if (step.arguments !== undefined && !isRecord(step.arguments)) return null;
    return {
        id,
        title,
        name,
        ...(step.arguments !== undefined ? { arguments: { ...step.arguments } } : {}),
    };
};

const parseGoalDeterminationTool = (value: unknown): GoalDeterminationTool | null => {
    const tool = isRecord(value) ? value : null;
    if (!tool || !hasOnlyKeys(tool, ['name', 'arguments'])) return null;
    const name = toNonEmptyString(tool.name);
    if (!name || (tool.arguments !== undefined && !isRecord(tool.arguments))) return null;
    return {
        name,
        ...(tool.arguments !== undefined ? { arguments: { ...tool.arguments } } : {}),
    };
};

const parseGoalDetermination = (value: unknown): GoalDetermination | null => {
    const determination = isRecord(value) ? value : null;
    if (!determination || !hasOnlyKeys(
        determination,
        ['tool', 'result_path', 'operator', 'target'],
    )) return null;
    const tool = parseGoalDeterminationTool(determination.tool);
    const resultPath = toNonEmptyString(determination.result_path);
    if (
        !tool
        || !resultPath
        || !isSafeGoalResultPath(resultPath)
        || !isGoalComparisonOperator(determination.operator)
        || typeof determination.target !== 'number'
        || !Number.isFinite(determination.target)
    ) {
        return null;
    }
    return {
        tool,
        result_path: resultPath,
        operator: determination.operator,
        target: determination.target,
    };
};

export const validateGoalRequest = (
    value: unknown,
    componentName = 'goal',
): { request: GoalRequest } | { error: GoalComponentError; name?: string } => {
    const input = isRecord(value) ? value : null;
    const name = toNonEmptyString(input?.name);
    if (!input || !name || !hasOnlyKeys(input, ['name', 'steps', 'determination'])) {
        return {
            error: new InvalidGoalNameError(componentName, 'Provide a valid goal name.'),
            ...(name ? { name } : {}),
        };
    }
    if (!Array.isArray(input.steps) || input.steps.length === 0) {
        return {
            error: new InvalidGoalStepsError(componentName, 'Provide at least one valid goal step.'),
            name,
        };
    }
    const steps = input.steps.map(parseGoalStepDescriptor);
    if (steps.some((step) => !step)) {
        return {
            error: new InvalidGoalStepsError(componentName, 'Every goal step must have a valid id, title, name, and arguments object.'),
            name,
        };
    }
    const parsedSteps = steps as GoalStepDescriptor[];
    const ids = new Set<string>();
    for (const step of parsedSteps) {
        if (ids.has(step.id)) {
            return {
                error: new DuplicateGoalStepIdError(componentName, `Goal step id '${step.id}' is duplicated.`),
                name,
            };
        }
        ids.add(step.id);
        if (RECURSIVE_GOAL_TOOL_NAMES.has(step.name)) {
            return {
                error: new RecursiveGoalStepError(componentName, 'Goal steps cannot invoke goal-management tools.'),
                name,
            };
        }
    }
    const determination = parseGoalDetermination(input.determination);
    if (!determination) {
        return {
            error: new InvalidGoalDeterminationError(componentName, 'Provide a valid goal determination.'),
            name,
        };
    }
    if (RECURSIVE_GOAL_TOOL_NAMES.has(determination.tool.name)) {
        return {
            error: new RecursiveGoalDeterminationError(componentName, 'Goal determination cannot invoke a goal-management tool.'),
            name,
        };
    }
    return { request: { name, steps: parsedSteps, determination } };
};

export const buildGoalRequest = (
    value: unknown,
    componentName = 'goal',
): { request: GoalRequest } | { error: GoalComponentError; name?: string } => (
    validateGoalRequest(value, componentName)
);

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

const cloneDeterminationTool = (tool: GoalDeterminationTool): GoalDeterminationTool => ({
    ...tool,
    ...(tool.arguments ? { arguments: { ...tool.arguments } } : {}),
});

const cloneDetermination = (determination: GoalDetermination): GoalDetermination => ({
    ...determination,
    tool: cloneDeterminationTool(determination.tool),
});

const cloneDeterminationResult = (
    result: GoalDeterminationResult | null,
): GoalDeterminationResult | null => result ? ({
    ...result,
    ...(result.source_result ? { source_result: { ...result.source_result } } : {}),
}) : null;

const cloneSnapshot = (snapshot: GoalSnapshot): GoalSnapshot => ({
    ...snapshot,
    steps: snapshot.steps.map((step) => ({
        ...step,
        ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
    })),
    determination: snapshot.determination
        ? cloneDetermination(snapshot.determination)
        : null,
    determination_result: cloneDeterminationResult(snapshot.determination_result),
    completed_steps: [...snapshot.completed_steps],
});

const cloneTaskResults = (taskResults: GoalTaskResult[]): GoalTaskResult[] => (
    taskResults.map((result) => ({
        ...result,
        ...(result.source_result ? { source_result: { ...result.source_result } } : {}),
    }))
);

const toRunResult = (
    snapshot: GoalSnapshot & { status: GoalRunResult['status'] },
    taskResults: GoalTaskResult[],
): GoalRunResult => ({
    name: snapshot.name,
    status: snapshot.status,
    determination: snapshot.determination
        ? cloneDetermination(snapshot.determination)
        : null,
    determination_result: cloneDeterminationResult(snapshot.determination_result),
    target: snapshot.target,
    actual: snapshot.actual,
    completed_steps: [...snapshot.completed_steps],
    task_results: cloneTaskResults(taskResults),
});

type RuntimeTaskExecutionResult = {
    value: unknown;
    error?: GoalComponentError;
    source_result?: GoalSourceResultMetadata;
    nested_statuses: object[];
};

type ActiveGoalOperation = {
    result: AiToolDeferred<GoalRunResult>;
    statuses: Map<string, AiToolDeferred<GoalProgressStatus>>;
};

const getOutputStatus = (output: Record<string, unknown>): string => {
    const status = output.status;
    return typeof status === 'string' && status ? status : 'complete';
};

const toGoalAiResult = (result: GoalRunResult): GoalAiResult => {
    const { name, ...safeResult } = result;
    return { ...safeResult, goal: name };
};

export class GoalRunner extends AiToolComponentBase<GoalSnapshot | null> {
    private currentSnapshot: GoalSnapshot | null = null;
    private request: GoalRequest | null = null;
    private failedStepIndex: number | null = null;
    private determinationFailed = false;
    private stepAttempts: number[] = [];
    private determinationAttempts = 0;
    private taskResults: GoalTaskResult[] = [];
    private goalAttempt = 1;
    private generation = 0;
    private activeOperation: ActiveGoalOperation | null = null;

    constructor(
        componentName: string,
        private readonly dispatchTool: AiToolDispatcher,
        private readonly onChange?: (snapshot: GoalSnapshot | null) => void,
    ) {
        super(componentName, null);
    }

    getSnapshot(): GoalSnapshot | null {
        return this.currentSnapshot ? cloneSnapshot(this.currentSnapshot) : null;
    }

    create(input: GoalRequest): AiToolOperation<GoalRunResult, GoalProgressStatus> {
        const validation = validateGoalRequest(input, this.getComponentName());
        if ('error' in validation) {
            return createAiToolOperationFrom(() => this.runCreate(input));
        }
        return this.startOperation(validation.request, () => this.runCreate(input));
    }

    private async runCreate(input: GoalRequest): Promise<GoalRunResult> {
        const validation = validateGoalRequest(input, this.getComponentName());
        if ('error' in validation) {
            const snapshot: GoalSnapshot = {
                name: validation.name || 'Goal',
                status: 'error',
                steps: [],
                determination: null,
                determination_result: null,
                target: null,
                actual: null,
                completed_steps: [],
                error: validation.error.message,
            };
            this.publish(snapshot);
            throw validation.error;
        }

        this.generation += 1;
        this.request = validation.request;
        this.failedStepIndex = null;
        this.determinationFailed = false;
        this.stepAttempts = validation.request.steps.map(() => 0);
        this.determinationAttempts = 0;
        this.taskResults = [];
        this.goalAttempt = 1;
        this.publish(this.createRunningSnapshot(validation.request));
        return this.runPreparation(validation.request, this.generation, 0);
    }

    retryFailedTask(): AiToolOperation<GoalRunResult, GoalProgressStatus> {
        const request = this.request;
        if (!request) return createAiToolOperationFrom(() => this.runRetryFailedTask());
        return this.startOperation(request, () => this.runRetryFailedTask());
    }

    private async runRetryFailedTask(): Promise<GoalRunResult> {
        const request = this.request;
        if (!request || !this.currentSnapshot || this.currentSnapshot.status !== 'error') {
            throw new GoalTaskRetryUnavailableError(
                this.getComponentName(),
                'The failed goal task could not be retried.',
            );
        }
        const generation = ++this.generation;
        if (this.failedStepIndex !== null) {
            const index = this.failedStepIndex;
            this.failedStepIndex = null;
            const { failed_step: _failedStep, error: _error, ...snapshot } = this.currentSnapshot;
            this.publish({ ...snapshot, status: 'running', actual: null });
            return this.runPreparation(request, generation, index);
        }
        if (this.determinationFailed) {
            this.determinationFailed = false;
            const { failed_step: _failedStep, error: _error, ...snapshot } = this.currentSnapshot;
            this.publish({
                ...snapshot,
                status: 'running',
                actual: null,
                determination_result: this.pendingDeterminationResult(request),
            });
            return this.runDetermination(request, generation);
        }
        throw new GoalTaskRetryUnavailableError(
            this.getComponentName(),
            'The failed goal task could not be retried.',
        );
    }

    clear(): void {
        this.cancelActiveOperation(new GoalReplacedError(
            this.getComponentName(),
            'The goal run was cleared.',
        ));
        this.generation += 1;
        this.currentSnapshot = null;
        this.request = null;
        this.failedStepIndex = null;
        this.determinationFailed = false;
        this.onChange?.(null);
        this.publishSnapshot(null);
    }

    protected onDispose(): void {
        this.cancelActiveOperation(new GoalReplacedError(
            this.getComponentName(),
            'The goal run was disposed.',
        ));
        this.generation += 1;
        this.currentSnapshot = null;
        this.request = null;
    }

    private startOperation(
        request: GoalRequest,
        run: () => Promise<GoalRunResult>,
    ): AiToolOperation<GoalRunResult, GoalProgressStatus> {
        this.cancelActiveOperation(new GoalReplacedError(
            this.getComponentName(),
            'The goal run was replaced.',
        ));
        const operation: ActiveGoalOperation = {
            result: createAiToolDeferred<GoalRunResult>(),
            statuses: new Map([
                ...request.steps.map((step) => [
                    `step:${step.id}`,
                    createAiToolDeferred<GoalProgressStatus>(),
                ] as const),
                ['determination', createAiToolDeferred<GoalProgressStatus>()] as const,
            ]),
        };
        this.activeOperation = operation;
        void run().then(
            (result) => operation.result.resolve(result),
            (error) => operation.result.reject(error),
        ).finally(() => {
            operation.statuses.forEach((status, key) => {
                if (status.settled) return;
                const stepId = key.startsWith('step:') ? key.slice(5) : 'determination';
                const step = request.steps.find((candidate) => candidate.id === stepId);
                status.resolve({
                    workflow: 'goal',
                    phase: key === 'determination' ? 'determination' : 'step',
                    id: stepId,
                    tool_name: key === 'determination'
                        ? request.determination.tool.name
                        : step?.name ?? '',
                    status: 'skipped',
                });
            });
            if (this.activeOperation === operation) this.activeOperation = null;
        });
        return createAiToolOperation(
            operation.result.promise,
            Array.from(operation.statuses.values()).map((status) => status.promise),
        );
    }

    private cancelActiveOperation(error: Error): void {
        const operation = this.activeOperation;
        if (!operation) return;
        this.activeOperation = null;
        operation.statuses.forEach((status) => {
            if (!status.settled) status.resolve({
                workflow: 'goal',
                phase: 'step',
                id: 'cancelled',
                tool_name: '',
                status: 'skipped',
                error: error.message,
            });
        });
        operation.result.reject(error);
    }

    private resolveProgress(key: string, status: GoalProgressStatus): void {
        this.activeOperation?.statuses.get(key)?.resolve(status);
    }

    private async runPreparation(
        request: GoalRequest,
        generation: number,
        startIndex: number,
    ): Promise<GoalRunResult> {
        for (let index = startIndex; index < request.steps.length; index += 1) {
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
            }
            const step = request.steps[index];
            const attempt = (this.stepAttempts[index] ?? 0) + 1;
            this.stepAttempts[index] = attempt;
            this.updateStep(index, {
                status: 'running',
                attempts: attempt,
                run_id: undefined,
                error: undefined,
            });
            const execution = await this.executeTask(
                step.name,
                step.arguments,
                GoalStepFailedError,
                'The goal step failed.',
            );
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
            }
            const sourceResult = execution.source_result
                ? { ...execution.source_result, step_id: step.id }
                : undefined;
            this.taskResults.push({
                step_id: step.id,
                tool_name: step.name,
                attempt,
                status: execution.error ? 'error' : 'completed',
                value: execution.value,
                ...(execution.error ? { error: execution.error.message } : {}),
                ...(sourceResult ? { source_result: sourceResult } : {}),
            });
            if (execution.error) {
                this.resolveProgress(`step:${step.id}`, {
                    workflow: 'goal',
                    phase: 'step',
                    id: step.id,
                    tool_name: step.name,
                    status: 'failed',
                    error: execution.error.message,
                    nested_statuses: execution.nested_statuses,
                });
                this.failedStepIndex = index;
                this.determinationFailed = false;
                this.updateStep(index, {
                    status: 'error',
                    run_id: sourceResult?.run_id,
                    error: execution.error.message,
                });
                const snapshot: GoalSnapshot = {
                    ...this.currentSnapshot!,
                    status: 'error',
                    actual: null,
                    failed_step: step.id,
                    error: execution.error.message,
                };
                this.publish(snapshot);
                return this.failedRunResult(snapshot);
            }
            this.updateStep(index, {
                status: 'completed',
                run_id: sourceResult?.run_id,
                error: undefined,
            });
            this.resolveProgress(`step:${step.id}`, {
                workflow: 'goal',
                phase: 'step',
                id: step.id,
                tool_name: step.name,
                status: 'completed',
                value: execution.value,
                nested_statuses: execution.nested_statuses,
            });
            this.publish({
                ...this.currentSnapshot!,
                completed_steps: this.currentSnapshot!.steps
                    .filter((item) => item.status === 'completed')
                    .map((item) => item.id),
            });
        }
        return this.runDetermination(request, generation);
    }

    private async runDetermination(
        request: GoalRequest,
        generation: number,
    ): Promise<GoalRunResult> {
        const attempt = ++this.determinationAttempts;
        this.publish({
            ...this.currentSnapshot!,
            determination_result: {
                tool_name: request.determination.tool.name,
                attempt,
                status: 'running',
                value: null,
            },
        });
        const execution = await this.executeTask(
            request.determination.tool.name,
            request.determination.tool.arguments,
            GoalDeterminationFailedError,
            'The goal determination failed.',
        );
        if (generation !== this.generation) {
            throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
        }
        let error = execution.error;
        let actual: number | null = null;
        if (!error) {
            const value = extractGoalResultPath(
                execution.value,
                request.determination.result_path,
            );
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                error = new GoalDeterminationValueNotNumericError(
                    this.getComponentName(),
                    `Goal determination path '${request.determination.result_path}' did not resolve to a finite number.`,
                );
            } else actual = value;
        }
        if (error) {
            this.resolveProgress('determination', {
                workflow: 'goal',
                phase: 'determination',
                id: 'determination',
                tool_name: request.determination.tool.name,
                status: 'failed',
                error: error.message,
                nested_statuses: execution.nested_statuses,
            });
            this.failedStepIndex = null;
            this.determinationFailed = true;
            const snapshot: GoalSnapshot = {
                ...this.currentSnapshot!,
                status: 'error',
                actual: null,
                error: error.message,
                determination_result: {
                    tool_name: request.determination.tool.name,
                    attempt,
                    status: 'error',
                    value: null,
                    error: error.message,
                    ...(execution.source_result
                        ? { source_result: { ...execution.source_result } }
                        : {}),
                },
            };
            this.publish(snapshot);
            return this.failedRunResult(snapshot);
        }

        this.failedStepIndex = null;
        this.determinationFailed = false;
        const achieved = compareGoalValues(
            actual!,
            request.determination.operator,
            request.determination.target,
        );
        const snapshot: GoalSnapshot & { status: 'achieved' | 'missed' } = {
            ...this.currentSnapshot!,
            status: achieved ? 'achieved' : 'missed',
            actual,
            determination_result: {
                tool_name: request.determination.tool.name,
                attempt,
                status: 'completed',
                value: actual,
                ...(execution.source_result
                    ? { source_result: { ...execution.source_result } }
                    : {}),
            },
        };
        this.resolveProgress('determination', {
            workflow: 'goal',
            phase: 'determination',
            id: 'determination',
            tool_name: request.determination.tool.name,
            status: 'completed',
            value: actual,
            nested_statuses: execution.nested_statuses,
        });
        this.publish(snapshot);
        if (!achieved && this.goalAttempt < MAX_GOAL_ATTEMPTS) {
            await this.retryDelay();
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
            }
            this.goalAttempt += 1;
            this.publish(this.createRunningSnapshot(request));
            return this.runPreparation(request, generation, 0);
        }
        this.finish();
        return toRunResult(snapshot, this.taskResults);
    }

    private async executeTask(
        toolName: string,
        argumentsValue: Record<string, unknown> | undefined,
        FailureError: AiToolComponentErrorConstructor<GoalComponentError>,
        fallbackMessage: string,
    ): Promise<RuntimeTaskExecutionResult> {
        const runId = `goal-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        try {
            const operation = this.dispatchTool(toolName, argumentsValue);
            const nestedStatuses = (await Promise.allSettled(operation.statuses)).map((status) => {
                if (status.status === 'fulfilled') return status.value;
                const message = status.reason instanceof Error
                    ? status.reason.message
                    : String(status.reason);
                console.error(`Nested goal tool '${toolName}' status failed.`, status.reason);
                return { status: 'status_failed', error: message };
            });
            const output = await operation.result;
            if (output instanceof Error) throw output;
            return {
                value: output,
                nested_statuses: nestedStatuses,
                source_result: {
                    tool_name: toolName,
                    run_id: runId,
                    status: getOutputStatus(output),
                },
            };
        } catch (error) {
            return {
                value: null,
                nested_statuses: [],
                source_result: {
                    tool_name: toolName,
                    run_id: runId,
                    status: 'failed',
                },
                error: new FailureError(
                    this.getComponentName(),
                    error instanceof Error && error.message ? error.message : fallbackMessage,
                    { cause: error },
                ),
            };
        }
    }

    private createRunningSnapshot(request: GoalRequest): GoalSnapshot {
        return {
            name: request.name,
            status: 'running',
            steps: request.steps.map((step, index) => ({
                ...step,
                ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
                status: 'pending',
                attempts: this.stepAttempts[index] ?? 0,
            })),
            determination: cloneDetermination(request.determination),
            determination_result: this.pendingDeterminationResult(request),
            target: request.determination.target,
            actual: null,
            completed_steps: [],
        };
    }

    private pendingDeterminationResult(request: GoalRequest): GoalDeterminationResult {
        return {
            tool_name: request.determination.tool.name,
            attempt: this.determinationAttempts,
            status: 'pending',
            value: null,
        };
    }

    private failedRunResult(snapshot: GoalSnapshot): GoalRunResult {
        return {
            name: snapshot.name,
            status: 'failed',
            determination: snapshot.determination
                ? cloneDetermination(snapshot.determination)
                : null,
            determination_result: cloneDeterminationResult(snapshot.determination_result),
            target: snapshot.target,
            actual: snapshot.actual,
            completed_steps: [...snapshot.completed_steps],
            task_results: cloneTaskResults(this.taskResults),
            ...(snapshot.failed_step ? { failed_step: snapshot.failed_step } : {}),
            ...(snapshot.error ? { error: snapshot.error } : {}),
        };
    }

    private publish(snapshot: GoalSnapshot): void {
        this.currentSnapshot = cloneSnapshot(snapshot);
        this.publishSnapshot(this.getSnapshot());
        this.onChange?.(this.getSnapshot());
    }

    private updateStep(index: number, update: Partial<GoalStepSnapshot>): void {
        if (!this.currentSnapshot) return;
        this.publish({
            ...this.currentSnapshot,
            steps: this.currentSnapshot.steps.map((step, stepIndex) => (
                stepIndex === index ? { ...step, ...update } : step
            )),
        });
    }

    private finish(): void {
        // The completed goal stays mounted until AI Chat replaces it.
    }

    private retryDelay(): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }
}

const getComparisonText = (snapshot: GoalSnapshot): string => {
    const determination = snapshot.determination;
    if (!determination) return snapshot.error || 'Invalid goal';
    const actual = snapshot.actual === null ? '—' : String(snapshot.actual);
    return `${actual} ${determination.operator} ${determination.target}`;
};

export const GoalDisplay: React.FC<GoalDisplayProps> = ({ snapshot, surface = 'chat' }) => {
    const determinationResult = snapshot.determination_result;
    return (
        <section
            className={`ai-chat__goal ai-chat__goal--${surface} ai-chat__goal--${snapshot.status}`}
            aria-label="Goal"
            aria-live="polite"
        >
            <div className="ai-chat__goal-head">
                <div>
                    <span className="ai-chat__goal-kicker">GOAL · {snapshot.status}</span>
                    <div className="ai-chat__goal-title">{snapshot.name}</div>
                </div>
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
                            </span>
                        </li>
                    ))}
                </ol>
            )}
            {snapshot.determination && determinationResult && (
                <div
                    className={`ai-chat__goal-determination ai-chat__goal-determination--${determinationResult.status}`}
                    aria-label="Determination"
                >
                    <span className="ai-chat__goal-step-dot" aria-hidden="true" />
                    <span className="ai-chat__goal-step-copy">
                        <span>Determination · {determinationResult.tool_name}</span>
                        <span>
                            {determinationResult.status}
                            {determinationResult.attempt > 0
                                ? ` · attempt ${determinationResult.attempt}`
                                : ''}
                        </span>
                        <span className="ai-chat__goal-metric">{getComparisonText(snapshot)}</span>
                    </span>
                </div>
            )}
            {snapshot.error && snapshot.steps.length === 0 && (
                <div className="ai-chat__goal-error">{snapshot.error}</div>
            )}
        </section>
    );
};

const Goal: React.FC<GoalProps> = ({
    name,
    dispatchTool,
    onSnapshotChange,
    surface = 'chat',
}) => {
    const [snapshot, setSnapshot] = useState<GoalSnapshot | null>(null);
    const onSnapshotChangeRef = useRef(onSnapshotChange);
    onSnapshotChangeRef.current = onSnapshotChange;
    const runnerRef = useRef<GoalRunner | null>(null);
    if (!runnerRef.current) {
        runnerRef.current = new GoalRunner(name, dispatchTool, (next) => {
            setSnapshot(next);
            onSnapshotChangeRef.current?.(next);
        });
    }

    const handle = useMemo<GoalHandle>(() => ({
        getComponentName: () => name,
        createGoal: (input) => mapAiToolOperation(
            runnerRef.current!.create(input),
            toGoalAiResult,
        ),
        retryFailedTask: () => mapAiToolOperation(
            runnerRef.current!.retryFailedTask(),
            toGoalAiResult,
        ),
        getSnapshot: () => runnerRef.current?.getSnapshot() ?? null,
        clear: () => runnerRef.current?.clear(),
    }), [name]);
    useRegisterAiToolComponentRef(name, handle);

    useEffect(() => () => runnerRef.current?.dispose(), []);

    return snapshot ? <GoalDisplay snapshot={snapshot} surface={surface} /> : null;
};

export default Goal;
